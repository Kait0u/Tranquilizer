import os
import random

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class DenoisingDataset(Dataset):
    def __init__(self, folder_path, gsc=False, noise_level=25, limit=0, resid=False):
        self.folder_path = folder_path
        self.noise_level = noise_level
        self.image_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")
        ]
        if limit > 0:
            self.image_files = random.sample(self.image_files, min(limit, len(self.image_files)))

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)])
        self.gsc = gsc

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name)
        if self.gsc:
            image = image.convert("L")  # Convert to grayscale
        image = self.transform(image)

        # Add Gaussian noise
        noise = torch.randn(image.size()) * (self.noise_level / 255.0)
        noisy_image = torch.clamp(image + noise, 0.0, 1.0)  # Clip to [0, 1]

        # transforms.ToPILImage()(noisy_image).show()
        # transforms.ToPILImage()(noise).show()
        # transforms.ToPILImage()(noisy_image - noise).show() - this is how to denoise an image
        # input()

        return noisy_image, noise