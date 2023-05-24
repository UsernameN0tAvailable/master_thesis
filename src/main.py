from vit import DinoFeatureClassifier
import os
import json
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import argparse

from typing import Optional;


def get_optimizer(model, id: Optional[int]):
    if id is None:
        raise ValueError("Should specify an optimizer")

    match id:
        case 0:
            return torch.optim.SGD(model.params(), lr=0.01)
        case 1:
            return torch.optim.RMSprop(model.params(), lr=0.01)
        case 2: 
            return torch.optim.Adam(model.params(), lr=0.01)
        case other:
            raise ValueError("optimizer value not allowed")

def f1_score(true_labels, predicted_labels):
    true_positives = (true_labels * predicted_labels).sum().float()
    false_positives = ((1 - true_labels) * predicted_labels).sum().float()
    false_negatives = (true_labels * (1 - predicted_labels)).sum().float()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1




if __name__ == "__main__":

    # get current split
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift', type=int, default=0, help='the shift to apply to the data splits')
    parser.add_argument('--epochs', type=int, default=100, help='number of epoch')
    parser.add_argument('--data', type=str, help="data directory")
    parser.add_argument('--optimizer', type=int, default=0, help="optimizer for the mlp, 0 => SGD, 1 => RMSProp, 2 => Adam")

    args = parser.parse_args()

    # dunno if this check is needed
    data_dir = args.data
    if data_dir is None or len(data_dir) == 0:
        raise ValueError("No data directory given!")

    shift = args.shift
    if shift is None:
        raise ValueError("No shift given!")

    num_epochs = args.epochs
    if num_epochs is None:
        raise ValueError("No epochs number given!")


    splits_dir = data_dir + '/splits.json' 

    # Load the json file
    with open(splits_dir, "r") as f:
        data = json.load(f)

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((3600, 3600)),  # Resize to 3600x3600 pixels
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] for RGB
        ])

    image_dir = data_dir + '/hotspots-png'

    # Load all images into memory and apply transformations
    all_images = []
    all_labels = []
    for split in data.values():
        for label, images in split.items():
            for image_name in images:
                image_path = os.path.join(image_dir, image_name + '.png')
                image = Image.open(image_path)
                image = transform(image)
                all_images.append(image)
                all_labels.append(int(label))

    # Convert lists to tensors
    all_images = torch.stack(all_images)
    all_labels = torch.Tensor(all_labels)

    # Create a Dataset
    dataset = TensorDataset(all_images, all_labels)

    # Create indices for each split
    indices = []
    for split in data.values():
        split_indices = []
        for images in split.values():
            split_indices.extend([image_name for image_name in images])
        indices.append(split_indices)

    # Number of splits
    num_splits = len(indices)

    # Init Model Init Model
    model = DinoFeatureClassifier()  

    # binary output true or false
    criterion = nn.BCELoss()

    # stochastic gradient descent opt
    optimizer = get_optimizer(model, args.optimizer)


    print(f'Starting ... Optimizer: {args.optimizer}, Shift: {shift}')

    # Training and validation loop
    for epoch in range(num_epochs):
        # Define training and validation splits for this epoch
        train_splits = indices[shift: shift + 3]
        validation_splits = indices[shift + 3: shift + 5]

        # Flatten the lists of splits
        train_indices = [item for sublist in train_splits for item in sublist]
        validation_indices = [item for sublist in validation_splits for item in sublist]

        # Create DataLoaders for this epoch
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=32, shuffle=True)
        validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=32, shuffle=True)

        # Train
        model.train()  # Set the model to training mode
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Compute the gradients
            optimizer.step()  # Update the weights

        # Validate
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0
        correct = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():  # Do not compute gradients in this block
            for images, labels in validation_loader:
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                # Append the labels and predictions for F1 score computation
                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())

        # Compute the F1 score
        f1 = f1_score(all_labels, all_predictions)

        print(f'Epoch {epoch + 1}, Train Loss: {loss.item()}, Validation Loss: {validation_loss / len(validation_loader)}, Validation Accuracy: {correct / len(validation_indices)}, Validation F1 Score: {f1}')
