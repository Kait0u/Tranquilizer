import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, num_layers=20):
        super(DnCNN, self).__init__()
        layers = []
        # First convolutional layer
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))

        # Last convolutional layer
        layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)