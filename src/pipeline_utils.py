from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, List, Dict, Optional
from torchvision import transforms
from PIL import Image
import os
import logging
import json
import random
import numpy as np

from torch.utils.data.distributed import DistributedSampler

class HotspotDataset(Dataset):
    def __init__(self, image_dir: str, splits: List[str], labels: List[int], transform):
        self.image_dir = image_dir
        self.splits = splits
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.splits)

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
        self.label_distribution = [0, 0]

    def add_sample(self, img_name: str, label: str):
        img_path = os.path.join(self.data_dir, f'{img_name}.png')
        if os.path.exists(img_path):
            self.paths.append(img_name)
            self.labels.append(int(label))
        else:
            logging.info(f'Image {img_path} not found.')

    def get_weights(self):
        if self.label_distribution[0] == 0 and self.label_distribution[1] == 0:
            raise ValueError("Class Frequencies not counted!")
        else:
            tot_unique_samples = self.label_distribution[0] + self.label_distribution[1]
            return 1 / (np.array(self.label_distribution, dtype=float) * 2)

    def push(self, split: Dict[str, List[str]], oversample: Optional[float]):

        if oversample not None:
            negatives = split['0']
            positives = split['1']
            neg_l = len(negatives)
            pos_l = len(positives)

            self.label_distribution[0] += neg_l
            self.label_distribution[1] += pos_l

            for img_name in negatives:
                self.add_sample(img_name, '0')



            oversample_l = ((neg_l / (( 1.0 - oversample) * 10) ) * 10 - neg_l)

            # over sampling
            for n_i in range(int(oversample_l)):
                pos_i = n_i % pos_l
                self.add_sample(positives[pos_i], '1')

 
        else:
            for label in ['0', '1']:
                self.label_distribution[int(label)] += len(split[label])
                for img_name in split[label]:
                    self.add_sample(img_name, label)

    def __len__(self):
        return len(self.paths)

    def get_dataset(self, batch_size: int, transform) -> DataLoader:
        dataset = HotspotDataset(self.data_dir, self.paths, self.labels, transform)
        sampler = DistributedSampler(dataset, shuffle=False)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def get_dataloaders(shift: int, data_dir: str, crop_size: int, batch_size: int, oversample: Optional[float]):
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
            train_data.push(split, oversample)
        elif split_index == 3: # Validation Split
            validation_data.push(split)
        else: 
            test_data.push(split)

    logging.info(f'SAMPLES COUNT:\ntrain: {len(train_data)}, val: {len(validation_data)}, test: {len(test_data)}')

    weights = train_data.get_weights()

    return [
            train_data.get_dataset(
                batch_size,
                transforms.Compose([
                    transforms.RandomCrop(crop_size),
                    Random90Rotation(),
                    transforms.ColorJitter(brightness=0.2),
                    transforms.ToTensor(),
                    RandomNoise(std=0.05),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])), 
                validation_data.get_dataset(
                    batch_size,
                    transforms.Compose([
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor()
                    ])),
                test_data.get_dataset(
                    batch_size,
                    transforms.Compose([
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor()
                    ])),
                weights
                ]


class RandomNoise:
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

class Random90Rotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle)
