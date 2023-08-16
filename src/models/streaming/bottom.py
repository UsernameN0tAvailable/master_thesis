import torch
from torch import nn
import numpy as np
from typing import Optional
import re

from torchvision.models import ResNet50_Weights, resnet50

class BottomCNN(nn.Module):  # Not subclassing from nn.Sequential anymore
    def __init__(self):
        super(BottomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

 
        return x


class BottomNet(nn.Module):
    def __init__(
            self,
            backbone
    ):
        super().__init__()

        self.backbone = backbone 

        parameters = list(self.backbone.parameters())

        for param in parameters:
            param.requires_grad = False

        last_layer_size = parameters[-1].shape[0]

        self.resize_net = torch.nn.Sequential( 
            torch.nn.Conv2d(last_layer_size, 32, kernel_size=3, stride=4, padding=1),  # This convolution will reduce the spatial dimensions by 4.
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, height: int, width: int):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return self.resize_net(x)



class BottomVit(BottomNet):
    def __init__(self):
        super(BottomVit, self).__init__(torch.hub.load('facebookresearch/dino:main', 'dino_vits16'))

    def forward(self, img, n: Optional[int] = 1):

        height = img.shape[2];
        width = img.shape[3]
        feat_h = height  // 16 
        feat_w = width // 16
        feat = self.backbone.get_intermediate_layers(x=img, n=n)

        feat_tokens = [f[:, 1:, :].reshape(img.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2) for f in feat]
        out = [f[:, 0, :] for f in feat]

        if len(feat_tokens) == 1:
            feat_tokens = feat_tokens[0]
            out = out[0]

        return super(BottomVit, self).forward(out, height, width)


class BottomResNet(BottomNet):
    def __init__(self):
        super(BottomResNet, self).__init__(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))

    def forward(self, img):
        height = img.shape[2]
        width = img.shape[3]

        out = self.backbone(img)
        return super(BottomResNet, self).forward(out, height, width)
