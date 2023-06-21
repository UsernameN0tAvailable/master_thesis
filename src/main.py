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

logging.basicConfig(level = logging.INFO)


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
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs.data, 1)
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    f1 = f1_score(true, preds)

    return epoch_loss, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--optimizer_index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--crop_size", type=int, default=128)

    args = parser.parse_args()

    logging.info("Creating Model ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DinoFeatureClassifier().to(device)

    if args.optimizer_index == 0:
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer_index == 1:
        optimizer = AdamW(model.parameters(), lr=0.01)
    elif args.optimizer_index == 2:
        optimizer = Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    logging.info("Loading Data ...")

    dataset = get_data(args.shift, args.data_dir, args.crop_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    logging.info("Start Training ...")

    for epoch in range(args.epochs):
        epoch_loss, f1 = train(model, optimizer, criterion, dataloader, device)
        logging.info(f'Epoch: {epoch + 1}, Loss: {epoch_loss}, F1: {f1}')


if __name__ == "__main__":
    main()
 
