from enum import Enum
from numpy.typing import NDArray
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Optional, Iterator, Any, Tuple, TypeVar, Callable
from torch.nn import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import os
import logging
import json
import random
import numpy as np

import random

from torch.utils.data.distributed import DistributedSampler
from torch import Tensor, device

FULL_IMAGE_SIZE=3504

class Optimizer(): 

    def __init__(self, optimizer: AdamW, scheduler: CosineAnnealingLR): 
        self.optimizer = optimizer
        self.scheduler = scheduler

    @classmethod
    def new(cls, model_paramters: Iterator[Parameter], lr: float, epochs: int, state_dict: Optional[Dict[str, Any]]) -> 'Optimizer':
        optimizer: AdamW = AdamW(model_paramters, lr=lr)
        if state_dict is not None and 'optimizer' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])

        scheduler: CosineAnnealingLR = CosineAnnealingLR(optimizer, T_max=epochs)
        if state_dict is not None and 'scheduler' in state_dict:
            scheduler.load_state_dict(state_dict['scheduler'])
        return Optimizer(optimizer, scheduler)

class HotspotDataset(Dataset):
    def __init__(self, image_dir: str, splits: List[str], labels: List[int], transform, clinical_data, mlp_header_only: bool = False):
        self.mlp_header_only =  mlp_header_only
        self.image_dir = image_dir
        self.splits = splits
        self.labels = labels
        self.clinical_data = clinical_data
        self.transform = transform

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, NDArray, str] :
        img_name = self.splits[idx]
        img_path = os.path.join(self.image_dir, f'{img_name}.png')
        label = self.labels[idx]

        if not self.mlp_header_only:
            image = Image.open(img_path)
            image = self.transform(image)
        else:
            image = torch.empty(0)

        if self.clinical_data is not None and img_name in self.clinical_data:
            clinical_data = self.clinical_data[img_name]
        else:
            clinical_data = np.zeros(4)

        return image, label, clinical_data, img_name

class PathsAndLabels():
    def __init__(self, data_dir: str, header_only: bool = False):
        self.header_only = header_only;
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
            return 1 / (np.array(self.label_distribution, dtype=float) * 2)

    def push(self, split: Dict[str, List[str]], oversample: float = 0.0):

        is_oversample = len(split['0']) != 0 and len(split['1']) != 0 and oversample > len(split['1']) / (len(split['1']) + len(split['0']))

        if is_oversample:

            random.seed(3.0)

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
            for _ in range(int(oversample_l)):
                pos_i = int(random.uniform(0.0, float(pos_l - 1)))
                self.add_sample(positives[pos_i], '1')

 
        else:
            for label in ['0', '1']:
                self.label_distribution[int(label)] += len(split[label])
                for img_name in split[label]:
                    self.add_sample(img_name, label)

    def __len__(self):
        return len(self.paths)

    def get_dataset(self, batch_size: int, transform: transforms.Compose, pin_memory: bool, clinical_data: Optional[Dict[str, np.ndarray]] = None) -> DataLoader:
        dataset = HotspotDataset(self.data_dir, self.paths, self.labels, transform, clinical_data=clinical_data, mlp_header_only=self.header_only)
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)

def get_dataloaders(shift: int, data_dir: str, batch_size: int, oversample: float, model_type, is_pre_training: bool, pin_memory: bool, clinical_data: Optional[Dict[str, np.ndarray]], header_only: bool = False):
    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)

    image_dir = os.path.join(data_dir, 'hotspots-png')

    train_data =  PathsAndLabels(image_dir, header_only=header_only)
    validation_data =  PathsAndLabels(image_dir, header_only=header_only)
    test_data = PathsAndLabels(image_dir, header_only=header_only)

    for split_n in range(5):
        split_index = (split_n - shift) % 5

        split = splits[str(split_n)]

        if split_index < 3: # Training split
            train_data.push(split, oversample)
        elif split_index == 3: # Validation Split
            validation_data.push(split)
        else: 
            test_data.push(split)

    RunTime.log(f'\nLoaders:\nTrain: {len(train_data)}\nVal: {len(validation_data)}, Test: {len(test_data)}')

    weights = train_data.get_weights()

    val_input_img_size = int(model_type["value"]) if model_type["type"] == "vit" else FULL_IMAGE_SIZE

    train_input_img_size = val_input_img_size
    #crop_val_fn = transforms.CenterCrop(val_input_img_size) 
    crop_val_fn = CustomCrop(2736, 2736, 3648, 3648)
    train_batch_size = batch_size
    val_batch_size = batch_size 

    if is_pre_training:
        train_input_img_size = int(model_type["value"])
        train_input_img_size = train_input_img_size * 2 
        train_batch_size = int(((FULL_IMAGE_SIZE ** 2) / (train_input_img_size ** 2)) * batch_size)
        crop_val_fn = transforms.RandomCrop(train_input_img_size)
        val_batch_size = train_batch_size
        val_input_img_size = train_batch_size

    crop_train_fn = transforms.RandomCrop(train_input_img_size) 

    RunTime.log(f'Train Sampler:\nBatch Size: {train_batch_size}, Image Size: {train_input_img_size}\nVal Sampler:\nBatch Size: {val_batch_size}, Image Size {val_input_img_size}', Rank.Local)
    

    return [
            train_data.get_dataset(
                train_batch_size,
                transforms.Compose([
                    crop_train_fn,
                    Random90Rotation(), 
                    transforms.RandomHorizontalFlip(), 
                    transforms.RandomVerticalFlip(), 
                    transforms.ColorJitter(brightness=0.2), 
                    transforms.ToTensor(),RandomNoise(std=0.05), 
                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                ),
                pin_memory,
                clinical_data=clinical_data
                ), 
                validation_data.get_dataset(
                    val_batch_size,
                    transforms.Compose([
                    crop_val_fn,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]),
                    pin_memory,
                    clinical_data=clinical_data
                    ),
                test_data.get_dataset(
                    val_batch_size,
                    transforms.Compose([
                    crop_val_fn,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]),
                    pin_memory,
                    clinical_data=clinical_data
                    ),
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

class CustomCrop:
    def __init__(self, start_width, start_height, end_width, end_height):
        self.start_height = start_height
        self.start_width = start_width
        self.end_height = end_height
        self.end_width = end_width

    def __call__(self, img):
        return transforms.functional.crop(img, self.start_height, self.start_width, self.end_height - self.start_height, self.end_width - self.start_width)

T = TypeVar('T')
A = TypeVar('A')

class Rank(Enum):
    Main = 0,
    Local = 1

class RunTime():
    main_rank: int = 0
    device_count: int = 0
    device_type: str
    @staticmethod
    def rank() -> int:
        return int(os.environ["LOCAL_RANK"])

    @staticmethod
    def init(filename: str, main_rank: int, device_count: int, device_type: str, level =logging.INFO):
        RunTime.main_rank = main_rank
        RunTime.device_count = device_count
        RunTime.device_type = device_type
        logging.basicConfig(level=level, filemode='a', filename=filename)

    @staticmethod
    def call_executable(closure: Callable[[], T], target_rank: Rank = Rank.Main, local_rank=rank()) -> Optional[T]:
        if target_rank == Rank.Main and local_rank != RunTime.main_rank:
            return None

        return closure()

    @staticmethod
    def execute(closure: Callable[..., T], *args, target_rank: Rank = Rank.Main, **kwargs) -> Optional[T]:
        RunTime.call_executable(lambda: closure(*args, **kwargs), target_rank=target_rank, local_rank=RunTime.rank())

    @staticmethod
    def log(val: str, target_rank: Rank = Rank.Main):
        local_rank = RunTime.rank()
        def fn(val: str) -> None:
            if target_rank == Rank.Local:
                val = f'Device: {local_rank}: {val}'
            logging.info(val)

        RunTime.call_executable(lambda: fn(val), target_rank=target_rank, local_rank=local_rank)

    @staticmethod
    def device() -> device:
        return torch.device(f'cuda:{RunTime.rank()}') if RunTime.device_type == 'cuda' else torch.device('cpu')


        











