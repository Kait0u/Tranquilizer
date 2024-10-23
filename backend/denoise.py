import numpy as np
import torch
import io
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms


def denoise(model: nn.Module, input_image: Image.Image, gsc: bool = False) -> tuple[Image.Image, Image.Image]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gsc:
        return __denoise_gsc(model, input_image, device)
    else:
        return __denoise_rgb(model, input_image, device)

# ----------------------------------------------------------------------------------------------------------------------

def __denoise_rgb(model: nn.Module, input_image: Image.Image, device) -> tuple[Image.Image, Image.Image]:
    input_image = input_image.convert("RGB")
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)

    # Move to device
    input_tensor = input_tensor.to(device)

    # Perform denoising
    with torch.no_grad():
        denoised_image = model(input_tensor)

    # Compute residual noise
    residual_noise = input_tensor - denoised_image

    # Convert tensors to numpy for plotting
    input_image_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    denoised_image_np = denoised_image.squeeze().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    residual_noise_np = residual_noise.squeeze().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    img_denoised = Image.fromarray((denoised_image_np * 255).astype(np.uint8))

    # Create details
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

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    img_summary = Image.open(buf)
    plt.close()

    return img_denoised, img_summary


def __denoise_gsc(model: nn.Module, input_image: Image.Image, device) -> tuple[Image.Image, Image.Image]:
    input_image = input_image.convert("RGB")
    input_image_channels = list(map(lambda im: im.convert("L"), input_image.split()))
    output_images = []

    plt.figure(figsize=(15, 15))
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
        input_image_np = input_tensor.squeeze().cpu().numpy().clip(0, 1)

        denoised_image_np = denoised_image.squeeze().cpu().numpy().clip(0, 1)
        output_images.append(denoised_image_np)

        residual_noise_np = residual_noise.squeeze().cpu().numpy().clip(0, 1)

        # Create details
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
    img_denoised = Image.fromarray((merged_image_np * 255).astype(np.uint8))

    plt.subplot(4, 3, 11)
    plt.title('Merged Result')
    plt.imshow(merged_image_np)
    plt.axis('off')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    img_summary = Image.open(buf)
    plt.close()

    return img_denoised, img_summary