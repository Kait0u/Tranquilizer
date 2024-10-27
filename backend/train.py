from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm


def train_dncnn(model: nn.Module,
                dataloader: DataLoader,
                num_epochs: int = 50,
                learning_rate: float = 0.001,
                device: str = "cpu",
                gsc: bool = False,
                with_checkpoints: bool = True,
                save_every_epochs: int = 10,
                name: str | None = None) -> None:

    """
    Train a DnCNN model.
    :param model: The model to train.
    :param dataloader: The dataloader to get training data from.
    :param num_epochs: The number of epochs to train the model for.
    :param learning_rate: The learning rate for the training process.
    :param device: The device to train on.
    :param gsc: Whether to use the GSC / Split-Channel mode (True) or RGB (False, default).
    :param with_checkpoints: Whether to save checkpoints. (Default: True)
    :param save_every_epochs: The number of epochs necessary to complete to save a checkpoint. (Default: 10)
    :param name: The name of the model. (Default: None - a UNIX timestamp will become the model's name)
    """

    # For saving: set up a name.
    name = name if name is not None else str(int(datetime.now().timestamp()))

    # Actual training --------------------------------------------------------------------------------------------------

    # Define the loss function and optimizer
    criterion = nn.MSELoss()                                        # A common choice for regression tasks.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # A common choice for an optimizer.
                                                                    # SGD is also valid.

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

        # Save a checkpoint if requested, and if it is time.
        if with_checkpoints and epoch + 1 < num_epochs and (epoch + 1) % save_every_epochs == 0:
            file_name = __build_filename(epoch + 1, gsc, name, checkpoint=True)
            torch.save(model.state_dict(), file_name)
            print(f"Checkpoint! Model saved as \"{file_name}\".")

    # Save the trained model.
    file_name = __build_filename(num_epochs, gsc, name)
    torch.save(model.state_dict(), file_name)
    print(f"Success! Model saved as \"{file_name}\".")


def __build_filename(epochs: int, gsc: bool, name: str | None = None, checkpoint = False):
    """
    Builds a filename for the model.
    :param epochs: The number of epochs to train the model for.
    :param gsc: Whether to use the GSC mode (True) or RGB (False).
    :param name: The name of the model. (Default: None - a UNIX timestamp will be the model's name)
    :param checkpoint: Whether the saved model is a checkpoint and not a fully-trained model.
    :return: A filename generated from the provided data.
    """

    name_str = name if name is not None else str(int(datetime.now().timestamp()))
    mode_str = "gsc" if gsc else "rgb"
    cp_str = "-cp" if checkpoint else ""
    return f"models/{mode_str}/{mode_str}_{name_str}{cp_str}_{epochs}e.pth"