from datetime import datetime

import torch
from torch import nn, optim

from tqdm import tqdm


def train_dncnn(model, dataloader, num_epochs=50, learning_rate=0.001, device='cpu', gsc=False, with_checkpoints = True, save_every_epochs: int = 10, name: str|None = None):
    # For saving:
    name = name if name is not None else str(int(datetime.now().timestamp()))

    # Actual training:

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the appropriate device
    model.to(device)
    print("Starting training sequence...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1} / {num_epochs}", unit="batch") as pbar:
            for batch_idx, (noisy_images, result_images) in enumerate(dataloader):
                # Move data to the appropriate device
                noisy_images, result_images = noisy_images.to(device), result_images.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(noisy_images)

                # Compute loss
                loss = criterion(outputs, result_images)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.item()

                # Update the progress bar
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))
                pbar.update(1)


        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)

        # Print loss for the current epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Save a checkpoint
        if with_checkpoints and epoch + 1 < num_epochs and (epoch + 1) % save_every_epochs == 0:
            file_name = __build_filename(epoch + 1, gsc, name, checkpoint=True)
            torch.save(model.state_dict(), file_name)
            print(f"Checkpoint! Model saved as \"{file_name}\".")

    # Save the trained model
    file_name = __build_filename(num_epochs, gsc, name)
    torch.save(model.state_dict(), file_name)
    print(f"Success! Model saved as \"{file_name}\".")


def __build_filename(epochs: int, gsc: bool, name: str | None = None, checkpoint = False):
    name_str = name if name is not None else str(int(datetime.now().timestamp()))
    mode_str = "gsc" if gsc else "rgb"
    cp_str = "-cp" if checkpoint else ""
    return f"models/{mode_str}/{mode_str}_{name_str}{cp_str}_{epochs}e.pth"