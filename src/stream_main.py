import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse

from pipeline_utils import get_dataloaders

def get_tiles(image, kernel_size, stride, padding):
    unfolded_image = image.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    unfolded_image = unfolded_image.permute(0, 2, 3, 1, 4, 5)
    return unfolded_image.contiguous().view(-1, image.size(1), kernel_size, kernel_size)

class SplitModel(nn.Module):
    def __init__(self, base_model, top_model):
        super(SplitModel, self).__init__()
        self.base_model = base_model
        self.top_model = top_model

    def forward(self, x, compute_activations=False):
        batch_size, num_tiles, c, h, w = x.size()
        x = x.view(batch_size * num_tiles, c, h, w)  # Reshaping to (batch_size * num_tiles, channels, height, width)
        x = self.base_model(x)
        if compute_activations:
            x = x.view(batch_size, num_tiles, *x.shape[1:])  # Reshaping back to (batch_size, num_tiles, other_dimensions)
            return x
        x = x.view(batch_size, num_tiles, *x.shape[1:])
        x = self.top_model(x)
        return x
 

def combine_tiles(tiles, output_size, kernel_size, stride, padding):
    folded_tiles = tiles.view(tiles.size(0), tiles.size(1), int(math.sqrt(tiles.size(2))), int(math.sqrt(tiles.size(2))), tiles.size(3), tiles.size(4))
    folded_tiles = folded_tiles.permute(0, 1, 4, 2, 5, 3).contiguous()
    folded_tiles = folded_tiles.view(tiles.size(0), tiles.size(1)*kernel_size*kernel_size, -1, int(math.sqrt(tiles.size(2)))*kernel_size)
    image = F.fold(folded_tiles, output_size, kernel_size, stride=stride, padding=padding)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--optimizer_index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--scheduler", type=int, choices=[0, 1])
    parser.add_argument("--crop_size", type=lambda c: int(c) if int(c) > 0 and int(c) % 16 == 0 else argparse.ArgumentTypeError(f"{c} crop size has to be multiple of 16"))
    parser.add_argument("--models_dir", type=str, default='~/models')
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--batch_size", type=lambda x: int(x) if int(x) > 0 else argparse.ArgumentTypeError(f"{x} is an invalid batch size"))
    parser.add_argument("--best_val_loss", type=float, default=float('inf'))
    parser.add_argument("--best_f1", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--oversample", type=float, required=True, default=0.0)
    parser.add_argument("--augmentations", type=int, default=1)

    args = parser.parse_args()

    train_loader, valid_loader, train_loader, weights = get_dataloaders(args.shift, args.data_dir, args.crop_size, args.batch_size, args.oversample, args.augmentations, False)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    base_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    target_layer = '4'  # Adjust this according to your model architecture.
    base_layers = list(base_model.children())[:int(target_layer)+1]
    top_layers = list(base_model.children())[int(target_layer)+1:]
    base = nn.Sequential(*base_layers).to(device)
    top = nn.Sequential(*top_layers).to(device)

    model = SplitModel(base, top).to(device)
    criterion = nn.CrossEntropyLoss(weight= torch.from_numpy(weights))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    kernel_size = 3
    stride = 1
    padding = 1

    print("start training")

    for epoch in range(args.epochs):
        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            tiles = get_tiles(images, kernel_size, stride, padding)
            activation_maps = model(tiles, compute_activations=True)
            composite_map = combine_tiles(activation_maps, images.shape[2:], kernel_size, stride, padding)
            outputs = model(composite_map.view(images.size(0), -1, *composite_map.shape[1:]))  # Adjust for batch size
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}')

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(valid_loader):
                images, labels = images.to(device), labels.to(device)
                tiles = get_tiles(images, kernel_size, stride, padding)
                activation_maps = model(tiles, compute_activations=True)
                composite_map = combine_tiles(activation_maps, images.shape[2:], kernel_size, stride, padding)
                outputs = model(composite_map.view(images.size(0), -1, *composite_map.shape[1:]))  # Adjust for batch size
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_loss/i}, Validation Accuracy: {100 * correct / total}%')
        model.train()

if __name__ == "__main__":
    main()
