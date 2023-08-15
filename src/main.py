import os
import logging
import argparse
from typing import List, Optional
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

import re

# distribute
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb

from pipeline_utils import PathsAndLabels, HotspotDataset, get_dataloaders


def step(model, optimizer, scheduler, criterion, dataloader, device, rank, device_count, average="weighted"):
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

        predicted = torch.max(output.data, 1)[1]
        
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


# validation
def check_batch_size(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def validate_model_and_extract(s):
    # Check for vit[*] format
    vit_match = re.match(r'^(vit)\[(\d+)\]$', s)
    if vit_match:
        number = int(vit_match.group(2))
        if number > 0 and number % 16 == 0:
            return {'type': 'vit', 'value': number}
        else:
            raise ValueError(f"Invalid value for 'vit': {number}. It must be a positive integer divisible by 16.")

    # Check for stream[m, n] format
    stream_match = re.match(r'^stream\[(vit|cnn|resnet), (\d+)\]$', s)
    if stream_match:
        m = stream_match.group(1)
        n = int(stream_match.group(2))

        if n <= 0:
            raise ValueError(f"Invalid value for 'n': {n}. It must be a positive integer.")
        
        if m == "vit" and n % 16 != 0:
            raise ValueError(f"Invalid combination: stream[{m}, {n}]. When 'm' is 'vit', 'n' must be divisible by 16.")
        
        return {'type': 'stream', 'subtype': m, 'value': n}

    raise ValueError("Model type not available\nPossible types:\n- vit[<tile_size>]\n- stream[<cnn | vit | resnet>, <patch_size>]")


def create_model(param, lr: float, epochs: int, load_path: str):

    if param['type'] == 'vit':
        model = DinoFeatureClassifier()
        if os.path.isfile(load_path):
            logging.info('Loading stored ViT')
            checkpoint = torch.load(load_path)

            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}

            model.load_state_dict(model_state_dict)

            optimizer = AdamW(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer'])

            scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
            scheduler.load_state_dict(checkpoint['scheduler'])

            return model, optimizer, scheduler, float(checkpoint['best_f1']), float(checkpoint['best_loss']), int(checkpoint['epoch']), True
        else:
            logging.info('Creating New ViT')
            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
            return model, optimizer, scheduler, 0.0, float('inf'), 0, False
    else:
        raise ValueError('Not Imlemented')




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--type", type=str, default="vit[2048]")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--batch_size", type=check_batch_size, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--oversample", type=float, default=0.5)
    parser.add_argument("--test_only", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    model_type = validate_model_and_extract(args.type)
    test_only = args.test_only == 1

    run_name = f'{args.type}_shift_{args.shift}_batch_size_{args.batch_size}_oversample_{args.oversample}'
    checkpoint_filepath = f'{args.models_dir}/{run_name}.pth'

    dist.init_process_group(backend='nccl', init_method='env://')

    rank = int(os.environ["LOCAL_RANK"])


    device_count = torch.cuda.device_count()

    device = torch.device(f'cuda:{rank}') if args.device == 'cuda' else torch.device('cpu')

    if rank == 0:
        logging.basicConfig(level = logging.INFO, filemode='a', filename=run_name)
        logging.info(f'Using device: {device}')


    model, optimizer, scheduler, best_f1, best_loss, epoch, is_resume =  create_model(model_type, float(args.lr), int(args.epochs), checkpoint_filepath)


    if rank == 0:
        if not test_only:
            wandb.init(
                    project=f'pT1',
                    group=f'{args.type}',
                    name = f'cv{args.shift}',
                    config= {
                        "learning_rate": args.lr,
                        "architecture": f'{args.type}',
                        "dataset": "Tumor Budding Hotspots",
                        },
                    resume=is_resume
                    )
    assert model is not None

    model = model.to(device)

    if rank == 0:
        logging.info(f'running distributed on {torch.cuda.device_count()} GPUs')
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) 

    if rank == 0:
        logging.info("Loading Data ...")

    train_dataloader, val_dataloader, test_dataloader, class_weights = get_dataloaders(args.shift, args.data_dir, args.batch_size, args.oversample, model_type)

    weights = torch.from_numpy(class_weights).float().to(device)

    if rank == 0:
        logging.info(f'Class weights {class_weights}')

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)

    if rank == 0:
        logging.info("Start Training ...")

    no_improvement = 0

    flag_tensor = torch.zeros(1).to(device)

    epochs = 0 if test_only else 10000 

    while epoch < epochs:
        model.train()
        train_loss, train_precision, train_recall, train_f1 = step(model, optimizer, scheduler, criterion, train_dataloader, device, rank, device_count)
        model.eval()
        val_loss, val_precision, val_recall, val_f1 = step(model, None, None, criterion, val_dataloader, device, rank, device_count, average=None)
        if rank == 0:
            wandb.log({"train_loss": train_loss, "train_prec": train_precision, "train_recall": train_recall, "train_f1": train_f1, "val_loss": val_loss, "val_prec_0": val_precision[0], "val_prec_1": val_precision[1], "val_recall_0": val_recall[0], "val_recall_1": val_recall[1], "val_f1_0": val_f1[0], "val_f1_1": val_f1[1]})
            logging.info(f'Epoch: {epoch + 1}\nTrain:\nLoss: {train_loss}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}\nValidation:\nLoss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')

            average_f1 = (val_f1[0] + val_f1[1]) / 2

            if  best_loss > val_loss or average_f1 > best_f1: 

                best_loss = val_loss if best_loss > val_loss else best_loss
                best_f1 = average_f1 if average_f1 > best_f1 else best_f1

                no_improvement = 0
                model_path = f'{args.models_dir}/{run_name}.pth'
                logging.info(f'Saving model to {model_path}')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'best_loss': best_loss,
                    },
                    model_path 
                    )
                wandb.save(model_path)
            else:
                no_improvement += 1
                if no_improvement >= 40:
                    logging.info("No Loss And F1 improvement for 40 Epoch, exiting training")
                    flag_tensor += 1

        dist.all_reduce(flag_tensor)

        if flag_tensor == 1:
            break;

    # test values
    model = model.to(device)
    model.eval() 
    test_loss, test_precision, test_recall, test_f1 = step(model, None, None, criterion, test_dataloader, device, rank, device_count, args.t, average=None)
    if rank == 0:
        wandb.finish()
        logging.info(f'Best Val loss = {best_loss}\nTest:\nLoss: {test_loss}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')


if __name__ == "__main__":
    main()

