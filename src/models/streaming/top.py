#!/usr/bin/env python
# coding: utf-8

# # Emperical experiment for StreamingCNN

# To evaluate whether a neural network using streaming trains equivalently to the conventional training, we can train a CNN on small images using both methods, starting from the same initialization. We used a subset of the ImageNet dataset, [ImageNette](https://github.com/fastai/imagenette), using 100 examples of 10 ImageNet classes (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

import torch
from torch import Tensor
from typing import Optional
from numpy import ndarray

# # Model definition

class TopCNN(torch.nn.Module):
    def __init__(self, has_clinical_data: bool):
        super(TopCNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = torch.nn.Linear(260 if has_clinical_data else 256, 2)

    def forward(self, x: Tensor, clinical_data: Tensor):
        x: Tensor = self.layers(x)
        x: Tensor = x.view(x.shape[0], -1)
        x = torch.cat((x, clinical_data), dim=1)
        x = self.classifier(x)
        return x
