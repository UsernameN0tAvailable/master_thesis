import torch
from torch import nn, Tensor
from typing import Optional
from pipeline_utils import Optimizer

class MLPHeader(nn.Module):
    def __init__(self):
        super(MLPHeader, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(4, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)  # Output two values for softmax

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)  # Softmax applied here

    def step(self, _, labels, criterion, optimizer: Optional[Optimizer], clinical_data: Tensor):
        output = self.forward(clinical_data)
        loss = criterion(output, labels.view(-1))

        if optimizer is not None:
            loss.backward()

        return output, loss
