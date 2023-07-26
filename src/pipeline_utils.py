from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, List, Dict
from torchvision import transforms
from PIL import Image
import os
import logging
import json

class HotspotDataset(Dataset):
    def __init__(self, image_dir: str, splits: List[str], labels: List[int], transform):
        self.image_dir = image_dir
        self.splits = splits
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.splits)

    def get_class_weights(self):

        labels_length = len(self.labels)

        positive_count = 0

        for l in self.labels:
            positive_count += l

        weights = [(labels_length - positive_count) / labels_length, positive_count / labels_length]

        return torch.tensor(weights)


    def __getitem__(self, idx):
        img_name = self.splits[idx]
        img_path = os.path.join(self.image_dir, f'{img_name}.png')
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label

class PathsAndLabels():
    def __init__(self, data_dir: str):
        self.paths = []
        self.labels = []
        self.data_dir = data_dir

    def append(self, split: Dict[str, List[str]]):
        image_dir = os.path.join(self.data_dir, 'hotspots-png')

        for label in ['0', '1']:
            for img_name in split[label]:
                img_path = os.path.join(image_dir, f'{img_name}.png')
                if os.path.exists(img_path):
                    self.paths.append(img_name)
                    self.labels.append(int(label))
                else:
                    logging.info(f'Image {img_path} not found.')

    def get_dataset(self, transform) -> HotspotDataset:
        HotspotDataset(self.data_dir, self.paths, self.labels, transform)

def get_dataloaders(shift, data_dir, crop_size):
    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)

    image_dir = os.path.join(data_dir, 'hotspots-png')

    train_data =  PathsAndLabels(image_dir)
    validation_data =  PathsAndLabels(image_dir)
    test_data =  PathsAndLabels(image_dir)

    for split_n in range(5):
        split_index = (split_n - shift) % 5

        split = splits[str(split_n)]

        if split_index < 3: # Training split
            train_data.append(split)
        elif split_index == 3: # Validation Split
            validation_data.append(split)
        else: 
            test_data.append(split)

    return [
            train_data.get_dataset(
                transforms.Compose([
                    transforms.RandomCrop(crop_size),
                    transforms.ToTensor()
                    ])), 
                validation_data.get_dataset(transforms.Compose([
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor()
                    ])),
                train_data.get_dataset(transforms.Compose([
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor()
                    ])),
                ]


