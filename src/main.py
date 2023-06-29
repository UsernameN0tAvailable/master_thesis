import os
import json
import logging
import argparse
from typing import List
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
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



class HotspotDataset(Dataset):
    def __init__(self, image_dir: str, splits: List[str], labels: List[int], crop_size: int):
        self.image_dir = image_dir
        self.splits = splits
        self.labels = labels
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        img_name = self.splits[idx]
        img_path = os.path.join(self.image_dir, f'{img_name}.png')
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label


def get_dataloaders(shift, data_dir, crop_size):
    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)

    image_dir = os.path.join(data_dir, 'hotspots-png')

    train_image_paths = []
    train_image_labels = []

    validation_image_paths = []
    validation_image_labels = []

    for split in range(5):
        if (split - shift) % 5 < 3:
            # Training split
            for label in ['0', '1']:
                for img_name in splits[str(split)][label]:
                    img_path = os.path.join(image_dir, f'{img_name}.png')
                    if os.path.exists(img_path):
                        train_image_paths.append(img_name)
                        train_image_labels.append(int(label))
                    else:
                        logging.info(f'Image {img_path} not found.')
        else:
            # Validation split
            for label in ['0', '1']:
                for img_name in splits[str(split)][label]:
                    img_path = os.path.join(image_dir, f'{img_name}.png')
                    if os.path.exists(img_path):
                        validation_image_paths.append(img_name)
                        validation_image_labels.append(int(label))
                    else:
                        logging.info(f'Image {img_path} not found.')

    return [HotspotDataset(image_dir, train_image_paths, train_image_labels, crop_size), HotspotDataset(image_dir, validation_image_paths, validation_image_labels, crop_size)]


def train(model, optimizer, scheduler, criterion, dataloader, device, rank, device_count):
    model.train()  # set the model to training mode
    running_loss = torch.tensor(0.0, device=device)
    true = []
    preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)['y_pixel'].squeeze(2).squeeze(2)

        loss = criterion(outputs, labels.view(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs.data, 1)
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

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
        f1 = f1_score(gathered_trues, gathered_preds, average='weighted')
    else:
        f1 = None

    return epoch_loss, f1

def validate(model, criterion, dataloader, device, rank, device_count):
    model.eval()  # set the model to training mode
    running_loss = torch.tensor(0.0, device=device)
    true = []
    preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)['y_pixel'].squeeze(2).squeeze(2)

        loss = criterion(outputs, labels.view(-1))

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs.data, 1)
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

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
        f1 = f1_score(gathered_trues, gathered_preds, average='weighted')
    else:
        f1 = None

    return epoch_loss, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--optimizer_index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--crop_size", type=lambda c: int(c) if int(c) > 0 and int(c) % 16 == 0 else argparse.ArgumentTypeError(f"{c} crop size has to be multiple of 16"))
    parser.add_argument("--models_dir", type=str, default='~/models')
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--batch_size", type=lambda x: int(x) if int(x) > 0 else argparse.ArgumentTypeError(f"{x} is an invalid batch size"))
    parser.add_argument("--best_val_loss", type=float, default=float('inf'))

    args = parser.parse_args()

    run_name = f'model_shift{args.shift}_opt{args.optimizer_index}_crop{args.crop_size}_batch_size{args.batch_size}'

    logging.basicConfig(level = logging.INFO, filemode='a', filename=run_name)

    logging.info("Creating Model ...")

    dist.init_process_group(backend='nccl', init_method='env://')

    rank = int(os.environ["LOCAL_RANK"])
    device_count = torch.cuda.device_count()

    device = torch.device(f'cuda:{rank}') if args.device == 'cuda' else torch.device('cpu')

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
        logging.info(f'running distributed on {torch.cuda.device_count()} GPUs')
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    if args.optimizer_index == 0:
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer_index == 1:
        optimizer = AdamW(model.parameters(), lr=0.01)
    elif args.optimizer_index == 2:
        optimizer = Adam(model.parameters(), lr=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    logging.info("Loading Data ...")

    train_dataset, val_dataset = get_dataloaders(args.shift, args.data_dir, args.crop_size)
    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    logging.info("Start Training ...")

    best_val_loss = args.best_val_loss 

    for epoch in range(args.epochs):
        train_loss, train_f1 = train(model, optimizer, scheduler, criterion, train_dataloader, device, rank, device_count)
        val_loss, val_f1 = validate(model, criterion, val_dataloader, device, rank, device_count)
        if rank == 0:
            logging.info(f'Epoch: {epoch + 1}, Train Loss: {train_loss}, Train F1: {train_f1}, Val Loss: {val_loss}, Val F1: {val_f1}')

            if  best_val_loss > val_loss:
                best_val_loss = val_loss 
                logging.info(f'Best Val loss improved to {val_loss}, saving model...')
                torch.save(model, model_path)
                logging.info(f'Model saved to {model_path}') 

    logging.info(f'Training completed. Best Val loss = {best_val_loss}')


if __name__ == "__main__":
    main()

