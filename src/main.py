import os
import logging
import argparse
from typing import List
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vit import DinoFeatureClassifier

# distribute
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb

from pipeline_utils import PathsAndLabels, HotspotDataset, get_dataloaders


def step(model, optimizer, scheduler, criterion, dataloader, device, rank, device_count, threshold, average="weighted"):
    running_loss = torch.tensor(0.0, device=device)
    true = []
    preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)


        if optimizer is not None: optimizer.zero_grad()
        output = model(images)['y_pixel'].squeeze(2).squeeze(2)

        loss = criterion(output, labels.view(-1))

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        
        softmax_output = torch.nn.functional.softmax(output.data, dim=1)
        predicted = (softmax_output[:, 1] > threshold).long()

        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    dist.reduce(running_loss, dst=0)
    if rank == 0:
        running_loss /= device_count

    epoch_loss = running_loss.item() / len(dataloader.dataset)

    true_tensor = torch.tensor(true, device=device)
    preds_tensor = torch.tensor(preds, device=device)

    gathered_true_tensors = [torch.zeros_like(true_tensor) for _ in range(device_count)]
    gathered_preds_tensors = [torch.zeros_like(preds_tensor) for _ in range(device_count)]

    dist.all_gather(gathered_true_tensors, true_tensor)
    dist.all_gather(gathered_preds_tensors, preds_tensor)

    if rank == 0:
        gathered_trues = torch.cat(gathered_true_tensors, dim=0).cpu().numpy()
        gathered_preds = torch.cat(gathered_preds_tensors, dim=0).cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(gathered_trues, gathered_preds, average=average, zero_division=0.0)
        return epoch_loss, precision, recall, f1

    return None, None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--optimizer_index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--scheduler", type=int, choices=[0, 1])
    parser.add_argument("--crop_size", type=lambda c: int(c) if int(c) > 0 and int(c) % 16 == 0 else argparse.ArgumentTypeError(f"{c} crop size has to be multiple of 16"))
    parser.add_argument("--models_dir", type=str, default='~/models')
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--batch_size", type=lambda x: int(x) if int(x) > 0 else argparse.ArgumentTypeError(f"{x} is an invalid batch size"))
    parser.add_argument("--best_val_loss", type=float, default=float('inf'))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--t", type=float, default=0.5)

    args = parser.parse_args()

    run_name = f'n_model_shift_{args.shift}_opt_{args.optimizer_index}_crop_{args.crop_size}_batch_size_{args.batch_size}_scheduler_{args.scheduler}_t_{args.t}'

    dist.init_process_group(backend='nccl', init_method='env://')

    rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:

        logging.basicConfig(level = logging.INFO, filemode='a', filename=run_name)
        logging.info("Creating Model ...")

        wandb.init(
                project="ViT Oversampling No Decision Threshold",
                group=run_name,
                config= {
                    "learning_rate": args.lr,
                    "architecture": "ViT",
                    "dataset": "Tumor Budding Hotspots",
                    }
                )

    device_count = torch.cuda.device_count()

    device = torch.device(f'cuda:{rank}') if args.device == 'cuda' else torch.device('cpu')

    if rank == 0:
        logging.info(f'Using device: {device}')

    # load model if it's already present
    model_filename = f'{run_name}.pth'
    model_path = f'{args.models_dir}/{model_filename}'

    if os.path.isfile(model_path):
        logging.info(f'Loading existing model from {model_path}')
        _ = DinoFeatureClassifier()
        model = torch.load(model_path)
    else:
        logging.info(f'No existing model found. Creating a new one.')
        model = DinoFeatureClassifier()

    model = DinoFeatureClassifier()
    model = model.to(device)

    if device_count > 1:
        if rank == 0:
            logging.info(f'running distributed on {torch.cuda.device_count()} GPUs')
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    if args.optimizer_index == 0:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.2)
    elif args.optimizer_index == 1:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer_index == 2:
        optimizer = Adam(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 1 else None


    if rank == 0:
        logging.info("Loading Data ...")

    train_dataloader, val_dataloader, test_dataloader, class_weights = get_dataloaders(args.shift, args.data_dir, args.crop_size, args.batch_size)

    weights = torch.tensor(class_weights).to(device)

    if rank == 0:
        logging.info(f'Class weights {class_weights}')

    criterion = nn.CrossEntropyLoss(weight=weights)

    if rank == 0:
        logging.info("Start Training ...")

    best_val_loss = args.best_val_loss 
    no_loss_improvement = 0

    flag_tensor = torch.zeros(1).to(device)

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_precision, train_recall, train_f1 = step(model, optimizer, scheduler, criterion, train_dataloader, device, rank, device_count, args.t)
        model.eval()
        val_loss, val_precision, val_recall, val_f1 = step(model, None, None, criterion, val_dataloader, device, rank, device_count, args.t, average=None)
        if rank == 0:
            wandb.log({"train_loss": train_loss, "train_prec": train_precision, "train_recall": train_recall, "train_f1": train_f1, "val_loss": val_loss, "val_prec_0": val_precision[0], "val_prec_1": val_precision[1], "val_recall_0": val_recall[0], "val_recall_1": val_recall[1], "val_f1_0": val_f1[0], "val_f1_1": val_f1[1]})
            logging.info(f'Epoch: {epoch + 1}\nTrain:\nLoss: {train_loss}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}\nValidation:\nLoss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')

            if  best_val_loss > val_loss:
                best_val_loss = val_loss 
                no_loss_improvement = 0
                logging.info(f'Best Val loss improved to {val_loss}, saving model...')
                torch.save(model, model_path)
                logging.info(f'Model saved to {model_path}') 
            else:
                no_loss_improvement += 1
                if no_loss_improvement >= 40:
                    logging.info("No Val Loss improvement for 40 Epoch, exiting training")
                    flag_tensor += 1

        dist.all_reduce(flag_tensor)

        if flag_tensor == 1:
            break;

    # test values
    model.eval() 
    test_loss, test_precision, test_recall, test_f1 = step(model, None, None, criterion, test_dataloader, device, rank, device_count)
    if rank == 0:
        wandb.finish()
        logging.info(f'Training completed. Best Val loss = {best_val_loss}\nTest:\nLoss: {test_loss}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f11}')


if __name__ == "__main__":
    main()

