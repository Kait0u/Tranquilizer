import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from backend.ds import DenoisingDataset
from backend.net import DnCNN
from backend.train import train_dncnn


# Main Function
def main(image_path, dataset_path, num_epochs=20, noise_level=25, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = DnCNN()
    model.to(device)

    # Check if model exists
    if os.path.exists('dncnn_model.pth'):
        print("Loading existing model...")
        model.load_state_dict(torch.load('dncnn_model.pth'))
    else:
        print("No model found. Training a new one...")
        dataset = DenoisingDataset(dataset_path, noise_level=noise_level)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        train_dncnn(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # Load and process the input image
    input_image = Image.open(image_path)
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)  # Add batch dimension

    # Move to device
    input_tensor = input_tensor.to(device)

    # Perform denoising
    with torch.no_grad():
        denoised_image = model(input_tensor)

    # Compute residual noise
    residual_noise = input_tensor - denoised_image

    # Convert tensors to numpy for plotting
    input_image_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    # input_image_np = (maxmin_scale(input_image_np) * 255).astype(np.uint8)

    denoised_image_np = denoised_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    # denoised_image_np = (maxmin_scale(denoised_image_np) * 255).astype(np.uint8)

    residual_noise_np = residual_noise.squeeze().cpu().numpy().transpose(1, 2, 0)
    # residual_noise_np = (maxmin_scale(residual_noise_np) * 255).astype(np.uint8)

    # Image.fromarray(denoised_image_np).save("result.png")

    # Display the images
    plt.figure(figsize=(10, 15))

    plt.subplot(3, 1, 1)
    plt.title('Original Noisy Image')
    plt.imshow(input_image_np)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.title('Residual Noise')
    plt.imshow(residual_noise_np)
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised_image_np.clip(0, 1))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main_gsc(image_path, dataset_path, num_epochs=20, noise_level=25, learning_rate=0.001):
    print("GSC mode on.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = DnCNN(in_channels=1, out_channels=1)
    model.to(device)

    # Check if model exists
    if os.path.exists("../dncnn_model_gsc.pth"):
        print("Loading existing model...")
        model.load_state_dict(torch.load("../dncnn_model_gsc.pth"))
    else:
        print("No model found. Training a new one...")
        dataset = DenoisingDataset(dataset_path, noise_level=noise_level, gsc=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        train_dncnn(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, device=device, gsc=True)

    # Load and process the input image
    input_image = Image.open(image_path)
    input_image_channels = list(map(lambda im: im.convert("L"), input_image.split()))

    plt.figure(figsize=(15, 15))
    output_images = []
    cmaps = ["Reds_r", "Greens_r", "Blues_r"]

    for idx, channel_image in enumerate(input_image_channels, start=1):
        input_tensor = transforms.ToTensor()(channel_image).unsqueeze(0)  # Add batch dimension

        # Move to device
        input_tensor = input_tensor.to(device)

        # Perform denoising
        with torch.no_grad():
            denoised_image = model(input_tensor)

        # Compute residual noise
        residual_noise = input_tensor - denoised_image

        # Convert tensors to numpy for plotting
        input_image_np = input_tensor.squeeze().cpu().numpy()

        denoised_image_np = denoised_image.squeeze().cpu().numpy()
        output_images.append(denoised_image_np)

        residual_noise_np = residual_noise.squeeze().cpu().numpy()

        # Display the images
        plt.subplot(4, 3, idx)
        plt.title('Original Noisy Image')
        plt.imshow(input_image_np, cmap=cmaps[idx - 1])
        plt.axis('off')

        plt.subplot(4, 3, idx + 3)
        plt.title('Residual Noise')
        plt.imshow(residual_noise_np, cmap="gray")
        plt.axis('off')

        plt.subplot(4, 3, idx + 6)
        plt.title('Denoised Image')
        plt.imshow(denoised_image_np, cmap=cmaps[idx - 1])
        plt.axis('off')

    merged_image_np = np.stack(output_images, axis=-1)
    plt.subplot(4, 3, 11)
    plt.title('Merged Result')
    plt.imshow(merged_image_np)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = r"C:\Users\jjaws\Downloads\noisy.png"
    dataset_path = r"/datasets/pexels-110k-512p-min-jpg/images"
    main(image_path, dataset_path, num_epochs=20)