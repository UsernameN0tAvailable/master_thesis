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


class BottomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.resize_net = torch.nn.Sequential( 
            torch.nn.Conv2d(1000, 32, kernel_size=3, stride=4, padding=1),  # This convolution will reduce the spatial dimensions by 4.
            torch.nn.ReLU(inplace=True),
        )


    def forward(self, img, n: Optional[int] = 1):

        height = img.shape[2]
        width = img.shape[3]

        out = self.backbone(img)
        out = out.unsqueeze(-1).unsqueeze(-1)
        out = torch.nn.functional.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)

        return self.resize_net(out)




class BottomVit(nn.Module):
    def __init__(
            self,
            dino_arch: str = "vits16",
            freeze: str = 'backbone.head',
    ):
        super().__init__()

        self.arch = dino_arch
        self.patch_size = int(re.findall('[0-9][0-9]|[0-9]', self.arch)[0])
        self.arch = "dino_{}".format(self.arch)
        self.backbone = torch.hub.load('facebookresearch/dino:main', self.arch)
        self.n_feats = self.backbone.norm.bias.shape[0]

        self.resize_net = torch.nn.Sequential( 
            torch.nn.Conv2d(self.n_feats, 32, kernel_size=3, stride=4, padding=1),  # This convolution will reduce the spatial dimensions by 4.
            torch.nn.ReLU(inplace=True),
        )

        self._freeze(freeze=freeze)

    def _freeze(self, freeze: str = 'backbone.head'):

        named_modules = np.array([n for n, _ in self.named_modules()])

        # No freezing option, freeze none
        if freeze not in named_modules:
            return
        else:
            # Freeze up to layer (not included)
            id_layer = np.nonzero(named_modules == freeze)[0][0]
            frozen_layers = named_modules[:id_layer]

            for name, module in self.named_modules():
                if name in frozen_layers:
                    # Set the module in evaluation mode
                    module.eval()
                    # Disable gradient tracking
                    module.requires_grad_(False)
                else:
                    module.train()
                    # Enable gradient tracking
                    module.requires_grad_(True)

    def forward(self, img, n: Optional[int] = 1):

        height = img.shape[2];
        width = img.shape[3]
        # Inference for ViT like arch
        b = img.shape[0]
        feat_h = height  // self.patch_size
        feat_w = width // self.patch_size
        # get selected layer activations
        feat = self.backbone.get_intermediate_layers(x=img, n=n)

        feat_tokens = [f[:, 1:, :].reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2) for f in feat]
        out = [f[:, 0, :] for f in feat]

        if len(feat_tokens) == 1:
            feat_tokens = feat_tokens[0]
            out = out[0]

        out = out.unsqueeze(-1).unsqueeze(-1)
        out = torch.nn.functional.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)
        res = self.resize_net(out)

        return res

