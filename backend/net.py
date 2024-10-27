import torch
import torch.nn as nn

class DnCNN(nn.Module):
    """
    A class representing a DnCNN network.
    For more information about DnCNN, check out the following link: https://ieeexplore.ieee.org/document/7839189.
    """
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, num_layers=20):
        """
        Initializes a DnCNN network.
        :param in_channels: Number of input channels (Default: 3, another likely value: 1).
        :param out_channels: Number of output channels (Default: 3, another likely value: 1). Usually equal to in_channels.
        :param num_filters: Number of filters (Default: 64).
        :param num_layers: Number of layers (Default: 20).
        """

        super(DnCNN, self).__init__()
        layers = []
        # First convolutional layer
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)) # Convolution
        layers.append(nn.ReLU(inplace=True)) # ReLU for non-linearity.

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters)) # Normalize the layers over a batch.
            layers.append(nn.ReLU(inplace=True)) # ReLU for non-linearity.

        # Last convolutional layer
        layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        :param x: An input tensor (image).
        :return: A residual noise tensor.
        """
        return self.model(x)