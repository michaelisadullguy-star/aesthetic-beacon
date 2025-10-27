"""Training script for the artwork-to-Processing parameter predictor."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from .processing_features import ProcessingParameterPredictor, prepare_image_tensor


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class ArtworkSample:
    image_path: Path
    target: torch.Tensor


class ArtworkParameterDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset pairing artwork images with Processing parameter vectors."""

    def __init__(self, root: Path | str, *, resize_long_edge: int | None = 512) -> None:
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Expected an 'images' directory inside '{self.root}'."
            )

        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file '{metadata_path}' not found. Provide Processing parameters for each artwork."
            )

        with metadata_path.open("r", encoding="utf8") as handle:
            metadata = json.load(handle)

        if isinstance(metadata, dict):
            item_list: List[Tuple[str, Dict[str, float]]] = [
                (str(key), value) for key, value in metadata.items()
            ]
        elif isinstance(metadata, list):
            item_list = []
            for entry in metadata:
                if not isinstance(entry, dict) or "parameters" not in entry:
                    raise ValueError(
                        "List metadata entries must be objects with an 'id' and 'parameters' key."
                    )
                sample_id = entry.get("id") or entry.get("image")
                if not sample_id:
                    raise ValueError("Metadata entries must include an 'id' or 'image' field.")
                params = entry["parameters"]
                if not isinstance(params, dict):
                    raise ValueError("'parameters' must be a mapping from name to value.")
                item_list.append((str(sample_id), params))
        else:
            raise ValueError("metadata.json must contain either an object or a list of objects.")

        samples: List[ArtworkSample] = []
        parameter_names: Sequence[str] | None = None
        targets: List[torch.Tensor] = []

        for sample_id, params in item_list:
            if isinstance(params, dict):
                current_names = tuple(params.keys())
            else:
                raise ValueError("Parameter definitions must be dictionaries.")

            if parameter_names is None:
                parameter_names = current_names
            elif current_names != tuple(parameter_names):
                raise ValueError(
                    "Inconsistent parameter ordering detected. Ensure all entries share identical keys."
                )

            candidate_path = self.images_dir / sample_id
            if candidate_path.exists():
                image_path = candidate_path
            else:
                found = None
                for ext in SUPPORTED_EXTS:
                    tentative = self.images_dir / f"{sample_id}{ext}"
                    if tentative.exists():
                        found = tentative
                        break
                if found is None:
                    raise FileNotFoundError(
                        f"Unable to locate an image for sample '{sample_id}' inside {self.images_dir}."
                    )
                image_path = found

            target_tensor = torch.tensor(
                [float(params[name]) for name in parameter_names], dtype=torch.float32
            )
            samples.append(ArtworkSample(image_path=image_path, target=target_tensor))
            targets.append(target_tensor)

        if not samples:
            raise ValueError("No artwork samples found. Check metadata.json and the images directory.")

        self.samples = samples
        self.parameter_names = tuple(parameter_names or ())
        self.resize_long_edge = resize_long_edge
        self.targets_tensor = torch.stack(targets, dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_tensor = prepare_image_tensor(sample.image_path, resize_long_edge=self.resize_long_edge)
        return image_tensor, sample.target

    @property
    def target_mean(self) -> torch.Tensor:
        return self.targets_tensor.mean(dim=0)

    @property
    def target_std(self) -> torch.Tensor:
        return self.targets_tensor.std(dim=0).clamp_min(1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the artwork dataset root directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processing_features.pt"),
        help="Destination for the trained model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=512,
        help="Resize artwork so the longest edge matches this value (0 disables).",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base channel width for the convolutional encoder.",
    )
    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=128,
        help="Hidden dimension for the regression head.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout applied inside the regression head.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device identifier.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience measured in validation epochs.",
    )
    return parser.parse_args()


def build_loaders(
    dataset: ArtworkParameterDataset,
    validation_split: float,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader | None]:
    if validation_split <= 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader, None

    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(
    model: ProcessingParameterPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            norm_targets = (targets - target_mean) / target_std
            preds = model(images)
            loss = criterion(preds, norm_targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    resize = None if args.resize_long_edge <= 0 else args.resize_long_edge
    dataset = ArtworkParameterDataset(args.dataset, resize_long_edge=resize)
    train_loader, val_loader = build_loaders(
        dataset,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ProcessingParameterPredictor(
        parameter_count=len(dataset.parameter_names),
        base_channels=args.base_channels,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )
    model.to(device)

    optimiser = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    target_mean = dataset.target_mean.to(device)
    target_std = dataset.target_std.to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()

            norm_targets = (targets - target_mean) / target_std
            predictions = model(images)
            loss = criterion(predictions, norm_targets)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * images.size(0)
            running_samples += images.size(0)

        train_loss = running_loss / max(running_samples, 1)

        if val_loader is not None:
            val_loss = evaluate(
                model,
                val_loader,
                criterion,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            val_loss = train_loss
            improved = train_loss < best_val_loss
            if improved:
                best_val_loss = train_loss

        if patience_counter > args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}"
            + (f", val_loss={val_loss:.4f}" if val_loader is not None else "")
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "parameter_names": list(dataset.parameter_names),
        "model_config": model.config(),
        "target_normaliser": {
            "mean": dataset.target_mean.tolist(),
            "std": dataset.target_std.tolist(),
        },
        "training_args": vars(args),
    }
    torch.save(checkpoint, args.output)
    print(f"Saved Processing parameter model to {args.output}")


if __name__ == "__main__":
    main()
