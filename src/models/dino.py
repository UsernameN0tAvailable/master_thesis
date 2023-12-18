import torch
from torch import nn, Tensor
import numpy as np
from torch.nn.functional import normalize
from typing import Optional
import re
import logging
from pipeline_utils import Optimizer

def make_nonlinear_clusterer(in_channels: int, out_channels: int):
    return torch.nn.Sequential(
        torch.nn.Linear(in_channels, in_channels),
        torch.nn.ReLU(),
        torch.nn.Linear(in_channels, out_channels)
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
    ):
        super(DinoFeatureClassifier, self).__init__(
            dino_arch=dino_arch, freeze=freeze
        )

        self._attention_maps: Optional[Tensor] = None

        self.n_cls = n_cls
        self.dropout = torch.nn.Dropout2d(p=dropout)

        tot_in_channels =  self.n_feats + 4

        self.cluster = make_nonlinear_clusterer(
            in_channels=tot_in_channels,
            out_channels=self.n_cls
            )

    def forward(self, img, n: Optional[int], clinical_data: Tensor):

        # Compute embedding
        z, _ = super(DinoFeatureClassifier, self).forward(img, n=n)
        z: Tensor = z.unsqueeze(-1).unsqueeze(-1)

        # Use either cls token or avg of tokens
        y_pixel: Tensor = self.dropout(z).squeeze(2).squeeze(2)

        # insert additional clinical data before classifier
        y_pixel = torch.cat((y_pixel, clinical_data), dim=1)  
        y_pixel = self.cluster(y_pixel)

        return y_pixel

    def step(self, images, labels, criterion, optimizer: Optional[Optimizer], clinical_data: Tensor, store_feature_maps: bool = False, store_activation_maps: bool = False):
        output = self.forward(images, 1, clinical_data=clinical_data)

        if store_activation_maps:
            _, _, height, width = images.shape
            activation_maps = self.backbone.get_last_selfattention(images).cpu().detach()
            activation_maps = torch.nn.functional.interpolate(activation_maps, size=(height, width), mode='bilinear', align_corners=False)
            activation_maps = activation_maps.transpose(1, -1)
            activation_maps = torch.nn.functional.avg_pool2d(activation_maps, kernel_size=(1, activation_maps.shape[3]//3))
            activation_maps = activation_maps.transpose(1, -1)
            activation_maps = torch.nn.functional.normalize(activation_maps, p=2, dim=0)            
            self._attention_maps = activation_maps 

        loss = criterion(output, labels.view(-1))

        if optimizer is not None:
            loss.backward()

        return output, loss

    def get_activation_maps(self) -> Tensor:
        if self._attention_maps is None:
            raise ValueError("Cannot get maps when they're not gathered!")
        return self._attention_maps

