"""Training script for brainwave-to-Processing regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from .brainwave_dataset import (
    BrainwaveProcessingDataset,
    RandomTimeCrop,
    collate_pad_batch,
    z_score_normalise,
)
from .model import BrainwaveToProcessingModel, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the dataset root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where checkpoints and logs will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples per optimisation step.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay regularisation applied by AdamW.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Width of the temporal encoder hidden representation.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of residual blocks in the encoder.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied in the prediction head.",
    )
    parser.add_argument(
        "--crop-length",
        type=int,
        default=0,
        help="Randomly crop each sample to this many timesteps (0 disables cropping).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device identifier, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Clip gradient norm to this value (0 disables clipping).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision training.",
    )
    return parser.parse_args()


def create_transforms(args: argparse.Namespace):
    transforms = [z_score_normalise]
    if args.crop_length > 0:
        transforms.append(RandomTimeCrop(args.crop_length))

    def apply_all(tensor: torch.Tensor) -> torch.Tensor:
        for transform in transforms:
            tensor = transform(tensor)
        return tensor

    return apply_all


def build_dataloaders(
    dataset: BrainwaveProcessingDataset,
    validation_split: float,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Optional[DataLoader]]:
    if validation_split <= 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_pad_batch,
        )
        return loader, None

    validation_size = int(len(dataset) * validation_split)
    training_size = len(dataset) - validation_size
    train_dataset, val_dataset = random_split(dataset, [training_size, validation_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_pad_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_pad_batch,
    )
    return train_loader, val_loader


def evaluate(
    model: BrainwaveToProcessingModel,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    state: Dict[str, Any],
    output_dir: Path,
    epoch: int,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(state, best_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    transforms = create_transforms(args)
    dataset = BrainwaveProcessingDataset(args.dataset, transform=transforms)
    parameter_dim = dataset.parameter_size()

    train_loader, val_loader = build_dataloaders(
        dataset,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    example_features, _ = dataset[0]
    input_shape = (example_features.shape[0], example_features.shape[1])

    model = create_model(
        input_shape=input_shape,
        parameter_dim=parameter_dim,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.SmoothL1Loss()
    optimiser = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(features)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimiser)
            scaler.update()

            batch_size = features.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

        epoch_loss /= max(epoch_samples, 1)

        log_message = {
            "epoch": epoch,
            "train_loss": epoch_loss,
        }

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            log_message["val_loss"] = val_loss
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
        else:
            val_loss = None
            is_best = False

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "parameter_names": dataset.parameter_names,
            },
            args.output_dir,
            epoch,
            is_best=is_best,
        )

        print(json.dumps(log_message))

    print("Training complete.")


if __name__ == "__main__":
    main()

