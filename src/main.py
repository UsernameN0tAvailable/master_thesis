from vit import DinoFeatureClassifier
import os
import json
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
from torchvision import transforms
from PIL import Image
import argparse
from typing import Optional;

best_validation_loss = float("inf")
 
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def get_optimizer(model, id: Optional[int]):
    if id is None:
        raise ValueError("Should specify an optimizer")

    match id:
        case 0:
            return torch.optim.SGD(model.parameters(), lr=0.01)
        case 1:
            return torch.optim.RMSprop(model.parameters(), lr=0.01)
        case 2: 
            return torch.optim.Adam(model.parameters(), lr=0.01)
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

    image_dir = data_dir + '/hotspots-png'

    print("loading images...")

    # Load all images into memory and apply transformations
    all_images_paths = []
    all_labels = []
    for split in data.values():
        for label, images in split.items():
            for image_name in images:
                image_path = os.path.join(image_dir, image_name + '.png')
                if os.path.isfile(image_path):
                    all_images_paths.append(image_path)
                    all_labels.append(int(label))
                else:
                    print(f'{image_path} Not Found')

    transform = transforms.Compose([
        transforms.Resize((3584, 3584)),  # vits16 requires multiples of 16 
        transforms.ToTensor() 
        ])

    dataset = CustomImageDataset(all_images_paths, all_labels, transform=transform)

    # Set the indices for the train and validation sets
    train_indices = [idx for idx in range(len(dataset)) if idx % 5 != shift]
    validation_indices = [idx for idx in range(len(dataset)) if idx % 5 == shift]

    # Create the DataLoaders
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=2, shuffle=False)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                                                                                      
    # Init Model
    model = DinoFeatureClassifier().to(device)
    # binary output true or false
    criterion = nn.BCELoss()
    #optimizer
    optimizer = get_optimizer(model, args.optimizer)

    print(f'Starting ... Optimizer: {args.optimizer}, Shift: {shift}, device: {device}')

    # Training and validation loop
    for epoch in range(num_epochs):

        # Train
        model.train()  
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval() 
        validation_loss = 0
        correct = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())

        f1 = f1_score(all_labels, all_predictions)

        tot_validation_loss = validation_loss / len(validation_loader)

        if tot_validation_loss  < best_validation_loss:
           best_validation_loss  = tot_validation_loss
           torch.save(model.state_dict(), f"vit_opt_{args.optimizer}_shift_{args.shift}.pth")


        print(f'Epoch {epoch + 1}, Train Loss: {loss.item()}, Validation Loss: {validation_loss / len(validation_loader)}, Validation Accuracy: {correct / len(validation_indices)}, Validation F1 Score: {f1}')
