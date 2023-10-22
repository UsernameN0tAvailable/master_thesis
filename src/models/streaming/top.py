#!/usr/bin/env python
# coding: utf-8

# # Emperical experiment for StreamingCNN

# To evaluate whether a neural network using streaming trains equivalently to the conventional training, we can train a CNN on small images using both methods, starting from the same initialization. We used a subset of the ImageNet dataset, [ImageNette](https://github.com/fastai/imagenette), using 100 examples of 10 ImageNet classes (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

import torch

from typing import Optional

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

    def forward(self, x, clinical_data: Optional[ndarray]):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        if clinical_data is not None:
            x = torch.cat((x, torch.from_numpy(clinical_data)), dim=0)
        x = self.classifier(x)
        return x
