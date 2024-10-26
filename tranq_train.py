import argparse
import os

from slugify import slugify

from shared.init_activities import initialization_activities


def __train(name: str | None, epochs: int, batch_size: int, data_dir: str, data_limit: int, grayscale: bool, save_every=10):
    import torch
    from torch.utils.data import DataLoader
    from backend.ds import DenoisingDataset
    from backend.net import DnCNN
    from backend.train import train_dncnn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")
    print()

    initialization_activities()

    ch = 1 if grayscale else 3
    model = DnCNN(in_channels=ch, out_channels=ch)
    model.to(device)

    dataset = DenoisingDataset(data_dir, noise_level=25, limit=data_limit, gsc=grayscale)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    train_dncnn(
        model=model,
        dataloader=dataloader,
        device=device,
        num_epochs=epochs,
        gsc=grayscale,
        name=name,
        save_every_epochs=save_every,
    )


def main():
    parser = argparse.ArgumentParser(description="A CLI app for training a model for the Tranquilizer app.")

    parser.add_argument("--name", type=str, help="Optional name for the training run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (minimum: 1, default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches (default: 32)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory. It must contain images.")
    parser.add_argument("--data_limit", type=int, default=0, help="Limit on the amount of data (default: 0 - unlimited)")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save a checkpoint every n epochs (default: 10)")
    parser.add_argument("--grayscale", action="store_true", help="Split images into 3 1-channel images instead of processing them as 3-channel images")


    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("epochs must be at least 1.")

    if args.checkpoint_every < 1:
        parser.error("\"Checkpoint every\" must be positive.")

    if not os.path.isdir(args.data_dir):
        parser.error(f"The data directory '{args.data_dir}' does not exist.")
    if len(os.listdir(args.data_dir)) < 1:
        parser.error(f"The data directory '{args.data_dir}' is empty.")

    legal_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    if not any([any([f.lower().endswith(ext) for f in os.listdir(args.data_dir) if os.path]) for ext in legal_extensions]):
        parser.error(f"The data directory '{args.data_dir}' does not contain any images.")

    args.name = slugify(args.name) if args.name is not None else None

    print(f"Name: {args.name if args.name is not None else '<Will become a timestamp>'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data directory: {args.data_dir}")
    print(f"Data limit: {args.data_limit}" + (" (unlimited)" if args.data_limit == 0 else ""))
    print(f"Checkpoint every: {args.checkpoint_every}")
    print("Channel Split Mode: " + ("YES" if args.grayscale else "NO"))
    print()
    __train(
        name=args.name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        data_limit=args.data_limit,
        grayscale=args.grayscale,
        save_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()