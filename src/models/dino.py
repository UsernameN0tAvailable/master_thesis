import torch
from torch import nn
import numpy as np
from torch.nn.functional import normalize
from typing import Optional
import re
import logging
from pipeline_utils import Optimizer

def make_nonlinear_clusterer(in_channels: int, out_channels: int, bias: Optional[bool] = False):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels, out_channels, (1, 1), bias=bias)
    )

def make_linear_clusterer(in_channels: int, out_channels: int, bias: Optional[bool] = False):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, (1, 1), bias=bias)
    )


class DinoFeature(nn.Module):
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

        # Inference for ViT like arch
        b = img.shape[0]
        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size
        # get selected layer activations
        feat = self.backbone.get_intermediate_layers(x=img, n=n)

        feat_tokens = [f[:, 1:, :].reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2) for f in feat]
        feat_cls_token = [f[:, 0, :] for f in feat]

        if len(feat_tokens) == 1:
            feat_tokens = feat_tokens[0]
            feat_cls_token = feat_cls_token[0]

        return feat_cls_token, feat_tokens



class DinoFeatureClassifier(DinoFeature):

    def __init__(
            self,
            dino_arch: str = "vits16",
            n_cls: int = 2,
            dropout: float = 0.1,
            freeze: str = 'backbone.head',
            clinical_data: bool = False
    ):
        super(DinoFeatureClassifier, self).__init__(
            dino_arch=dino_arch, freeze=freeze
        )

        self.n_cls = n_cls
        self.dropout = torch.nn.Dropout2d(p=dropout)

        tot_in_channels =  self.n_feats + 4 if clinical_data else self.n_feats

        self.cluster = make_nonlinear_clusterer(
            in_channels=tot_in_channels,
            out_channels=self.n_cls
            )

    def forward(self, img, n: Optional[int] = 1, clinical_data: Optional[np.ndarray] = None):

        # Compute embedding
        z, _ = super(DinoFeatureClassifier, self).forward(img, n=n)
        z: Tensor = z.unsqueeze(-1).unsqueeze(-1)

        # Use either cls token or avg of tokens
        y_pixel: Tensor = self.dropout(z)

        # insert additional clinical data before classifier
        if clinical_data is not None:
            add_tensor = torch.from_numpy(clinical_data)
            y_pixel = torch.cat((y_pixel, add_tensor), dim=0)
        
        y_pixel = self.cluster(y_pixel)

        return y_pixel.squeeze(2).squeeze(2)

    def step(self, images, labels, criterion, optimizer: Optional[Optimizer]):
        output = self.forward(images, 1, )
        loss = criterion(output, labels.view(-1))

        if optimizer is not None:
            loss.backward()

        return output, loss

