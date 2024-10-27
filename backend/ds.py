import os
import random

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class DenoisingDataset(Dataset):
    """
    Represents a dataset that is made up of images for training.
    """
    def __init__(self, folder_path: str, gsc: bool = False, noise_level: int = 25, limit: int =0):
        """
        Initializes the dataset.
        :param folder_path: The path to the folder containing the images.
        :param gsc: Whether to turn the channel-split / grayscale mode on (True) or off (False, default).
        :param noise_level: The noise level for the to-be-applied Gaussian noise (default = 25)
        :param limit: The maximum number of images to load (default = 0 - no limit).
        """

        self.folder_path = folder_path
        self.noise_level = noise_level
        # Filter the files to only extract images
        self.image_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")
        ]
        # If the limit is 0 or lower, all the files are loaded. If it's more than that, only that number of files is loaded.
        if limit > 0:
            self.image_files = random.sample(self.image_files, min(limit, len(self.image_files)))

        # The transformations that will apply to any image loaded by this dataset.
        self.transform = transforms.Compose([
            transforms.ToTensor(),          # Converts a PIL Image to a PyTorch tensor
            transforms.CenterCrop(256)      # Crops the central 256x256 square to save on computing
                                            # without losing the essence of a picture
        ])
        self.gsc = gsc

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Find and open ain image file
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name)
        if self.gsc:
            image = image.convert("L")  # Convert to grayscale if requested

        # Apply the requested transformations
        image = self.transform(image)

        # Add Gaussian noise
        noise = torch.randn(image.size()) * (self.noise_level / 255.0)
        noisy_image = torch.clamp(image + noise, 0.0, 1.0)  # Clip to [0, 1]

        return noisy_image, noise