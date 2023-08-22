import torch
from torch import nn
import numpy as np
from torch.nn.functional import normalize
from typing import Optional
import re
import logging

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
            type_cls: str = 'nonlinear',
    ):
        super(DinoFeatureClassifier, self).__init__(
            dino_arch=dino_arch, freeze=freeze
        )

        self.n_cls = n_cls
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.type_cls = type_cls

        if type_cls == 'linear':
            self.cluster = make_linear_clusterer(
                in_channels=self.n_feats,
                out_channels=self.n_cls,
            )
        elif type_cls == 'nonlinear':
            self.cluster = make_nonlinear_clusterer(
                in_channels=self.n_feats,
                out_channels=self.n_cls,
            )
        else:
            raise NotImplementedError('Unknown classifier type: {}'.format(type_cls))

    def forward(self, img, n: Optional[int] = 1):

        # Compute embedding
        z, _ = super(DinoFeatureClassifier, self).forward(img, n=n)
        z = z.unsqueeze(-1).unsqueeze(-1)

        # Use either cls token or avg of tokens
        y_pixel = self.dropout(z)
        y_pixel = self.cluster(y_pixel)

        return y_pixel.squeeze(2).squeeze(2)

    def step(self, images, labels, criterion, optimizer):
        output = self.forward(images)
        loss = criterion(output, labels.view(-1))

        if optimizer is not None:
            loss.backward()

        return output, loss

