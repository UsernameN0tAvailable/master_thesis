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

from scnn import StreamingCNN
from cnn import Net, StreamNet


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

        predicted = (torch.nn.functional.softmax(output.data, dim=1)[:, 1] > threshold).long() if threshold != 0.5 else torch.max(output.data, 1)[1]
        
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    dist.reduce(running_loss, dst=0)
    running_loss /= device_count

    epoch_loss = running_loss.item() / len(dataloader.dataset)

    true_tensor = torch.tensor(true, device=device)
    preds_tensor = torch.tensor(preds, device=device)

    gathered_true_tensors = [torch.zeros_like(true_tensor) for _ in range(device_count)]
    gathered_preds_tensors = [torch.zeros_like(preds_tensor) for _ in range(device_count)]

    dist.all_gather(gathered_true_tensors, true_tensor)
    dist.all_gather(gathered_preds_tensors, preds_tensor)

    precision, recall, f1, _ = precision_recall_fscore_support(np.ndarray(true) ,np.ndarray(preds) , average=average, zero_division=0.0)
    return epoch_loss, precision, recall, f1

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
    parser.add_argument("--best_f1", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--oversample", type=float, required=True, default=0.0)
    parser.add_argument("--augmentations", type=int, default=1)

    args = parser.parse_args()

    run_name = f'stream_model_shift_{args.shift}_opt_{args.optimizer_index}_crop_{args.crop_size}_batch_size_{args.batch_size}_scheduler_{args.scheduler}_t_{args.t}_s_{args.oversample}_a_{args.augmentations}_rand'

    logging.basicConfig(level = logging.INFO, filemode='a', filename=run_name)
    logging.info("Creating Model ...")

    wandb.init(
            project=f'sCNN {args.crop_size} New',
            group=run_name,
            config= {
                "learning_rate": args.lr,
                "architecture": "sCNN",
                "dataset": "Tumor Budding Hotspots",
                }
            )

    device = torch.device(f'cuda') if args.device == 'cuda' else torch.device('cpu')

    logging.info(f'Using device: {device}')

    # load model if it's already present
    model_filename = f'{run_name}.pth'
    model_path = f'{args.models_dir}/{model_filename}'

    stream_net = StreamNet().to(device)
    net = Net().to(device)

    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
            mod.bias.data.fill_(0)

    params = list(stream_net.parameters()) + list(net.parameters())

    sCNN = StreamingCNN(stream_net, tile_shape=(1, 3, 32, 32), deterministic=True, verbose=True)

    sCNN.verbose = False

    if args.optimizer_index == 0:
        optimizer = SGD(params, lr=args.lr, momentum=0.2)
    elif args.optimizer_index == 1:
        optimizer = AdamW(params, lr=args.lr)
    elif args.optimizer_index == 2:
        optimizer = Adam(params, lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 1 else None

    logging.info("Loading Data ...")

    train_dataloader, val_dataloader, test_dataloader, class_weights = get_dataloaders(args.shift, args.data_dir, args.crop_size, args.batch_size, args.oversample, args.augmentations, False)

    weights = torch.from_numpy(class_weights).float().to(device)

    logging.info(f'Class weights {class_weights}')

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)

    logging.info("Start Training ...")

    best_val_loss = args.best_val_loss 
    best_f1 = args.best_f1

    no_improvement = 0

    sCNN.enable()

    for epoch in range(epochs):
        running_loss = 0.0
        i = 0
        
        for images, labels in tqdm(train_dataloader):
            with torch.no_grad():
                first_output = sCNN.forward(images)
            first_output.requires_grad = True

            labels = labels.to(device)
            
            # inference final part of network
            second_output = net(first_output)

            # backpropagation through final network
            loss = criterion(second_output, labels)
            loss.backward()

            # backpropagation through first network using checkpointing / streaming
            sCNN.backward(images, first_output.grad)

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            i += 1

                    
        stream_losses.append(running_loss / i)

        running_loss = 0
        i = 0
        accurate = 0
        with torch.no_grad():
            for images, labels in valloader:
                first_output = sCNN.forward(images.cuda())
                second_output = net(first_output)

                loss = criterion(second_output, labels.cuda())
                running_loss += loss.item()

                i += 1
                accurate += (torch.argmax(torch.softmax(second_output, dim=1), dim=1).cpu() == labels).sum() / float(len(images))
            
        stream_val_accuracy.append(accurate / float(i))
            
        stream_val_losses.append(running_loss / i)


if __name__ == "__main__":
    main()

