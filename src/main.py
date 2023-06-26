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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vit import DinoFeatureClassifier



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


def get_data(shift, data_dir, crop_size):
    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)
        
    image_dir = os.path.join(data_dir, 'hotspots-png')
    
    image_paths = []
    labels = []
    
    for split in range(5):
        if (split - shift) % 5 < 3:
            # Training split
            for label in ['0', '1']:
                for img_name in splits[str(split)][label]:
                    img_path = os.path.join(image_dir, f'{img_name}.png')
                    if os.path.exists(img_path):
                        image_paths.append(img_name)
                        labels.append(int(label))
                    else:
                        logging.info(f'Image {img_path} not found.')
                        
    return HotspotDataset(image_dir, image_paths, labels, crop_size)


def train(model, optimizer, criterion, dataloader, device):
    model.train()  # set the model to training mode
    running_loss = 0.0
    true = []
    preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)['y_pixel'].squeeze()

        try:
            loss = criterion(outputs, labels.view(-1))


            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            true.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
        except:
            logging.info('Error in training batch.')

    epoch_loss = running_loss / len(dataloader.dataset)
    f1 = f1_score(true, preds)

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

    args = parser.parse_args()

    run_name = f'model_shift{args.shift}_opt{args.optimizer_index}_crop{args.crop_size}_batch_size{args.batch_size}'

    logging.basicConfig(level = logging.INFO, filemode='a', filename=run_name)

    logging.info("Creating Model ...")

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

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

    if args.optimizer_index == 0:
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer_index == 1:
        optimizer = AdamW(model.parameters(), lr=0.01)
    elif args.optimizer_index == 2:
        optimizer = Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    logging.info("Loading Data ...")

    dataset = get_data(args.shift, args.data_dir, args.crop_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logging.info("Start Training ...")

    best_loss = float('inf') 

    for epoch in range(args.epochs):
        epoch_loss, f1 = train(model, optimizer, criterion, dataloader, device)
        logging.info(f'Epoch: {epoch + 1}, Loss: {epoch_loss}, F1: {f1}')

        if  best_loss > epoch_loss:
            best_loss = epoch_loss 
            logging.info(f'Best loss improved to {epoch_loss}, saving model...')
            torch.save(model, model_path)
            logging.info(f'Model saved to {model_path}') 

    logging.info(f'Training completed. Best loss = {best_loss}')


if __name__ == "__main__":
    main()
 
