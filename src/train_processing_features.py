"""Training script for the artwork-to-Processing parameter predictor.

Optional: Step 5 can derive artwork parameters by direct image analysis.  Train
this CNN only if you have curated parameters for a set of images and want new
artworks to be mapped the way you mapped those::

    python -m src.train_processing_features /path/to/artwork_dataset --output processing_features.pt

Dataset layout: ``images/`` with the artworks and ``metadata.json`` with the
parameters of each (see ``examples/parameter_metadata_example.json``).
"""
from __future__ import annotations

if __package__ in (None, ""):  # executed as a script
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    __package__ = "src"
    __import__("src")

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from .brainwave_dataset import load_parameter_metadata
from .parameters import PARAMETER_NAMES, ParameterCodec, canonicalise_parameters
from .processing_features import (
    SUPPORTED_IMAGE_SUFFIXES,
    ProcessingParameterPredictor,
    prepare_image_tensor,
)

SUPPORTED_EXTS = SUPPORTED_IMAGE_SUFFIXES


@dataclass
class ArtworkSample:
    image_path: Path
    target: torch.Tensor


class ArtworkParameterDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset pairing artwork images with encoded Processing parameter vectors."""

    def __init__(self, root: Path | str, *, resize_long_edge: int | None = 256) -> None:
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / "images"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Expected an 'images' directory inside '{self.root}'.")
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file '{metadata_path}' not found. Provide Processing parameters for each artwork."
            )

        self.codec = ParameterCodec()
        self.parameter_names = PARAMETER_NAMES
        self.resize_long_edge = resize_long_edge
        self.samples: List[ArtworkSample] = []
        for sample_id, parameters in load_parameter_metadata(metadata_path).items():
            image_path = next(
                (
                    self.images_dir / f"{sample_id}{ext}"
                    for ext in SUPPORTED_EXTS
                    if (self.images_dir / f"{sample_id}{ext}").is_file()
                ),
                None,
            )
            if image_path is None:
                raise FileNotFoundError(
                    f"Unable to locate an image for sample '{sample_id}' inside {self.images_dir}."
                )
            canonical, _ = canonicalise_parameters(parameters)
            self.samples.append(ArtworkSample(image_path, torch.from_numpy(self.codec.encode(canonical))))
        if not self.samples:
            raise ValueError("No artwork samples found. Check metadata.json and the images directory.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = prepare_image_tensor(sample.image_path, resize_long_edge=self.resize_long_edge)
        if self.resize_long_edge:
            # Pad to a square so images of different shapes can share a batch.
            _, height, width = image.shape
            size = self.resize_long_edge
            image = nn.functional.pad(image, (0, max(0, size - width), 0, max(0, size - height)))[:, :size, :size]
        return image, sample.target


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Path to the artwork dataset root directory.")
    parser.add_argument("--output", type=Path, default=Path("processing_features.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--resize-long-edge", type=int, default=256, help="Images are scaled and padded to this size.")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _epoch(model, loader, criterion, device, optimiser=None) -> float:
    model.train(optimiser is not None)
    total, count = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimiser is not None):
            loss = criterion(model(images), targets)
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = ArtworkParameterDataset(args.dataset, resize_long_edge=args.resize_long_edge or 256)
    val_size = int(len(dataset) * args.validation_split) if len(dataset) >= 5 else 0
    if val_size:
        train_set, val_set = random_split(
            dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        train_set, val_set = dataset, None
    # Batch normalisation needs more than one image per training batch.
    drop_last = len(train_set) > args.batch_size and len(train_set) % args.batch_size == 1
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers) if val_set else None
    if len(train_set) < 2:
        print("At least two annotated artworks are needed to train the CNN.", file=sys.stderr)
        return 1

    model = ProcessingParameterPredictor(
        parameter_count=dataset.codec.dim,
        base_channels=args.base_channels,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)
    optimiser = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    best_loss, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = _epoch(model, train_loader, criterion, device, optimiser)
        monitored = _epoch(model, val_loader, criterion, device) if val_loader else train_loss
        if monitored < best_loss - 1e-6:
            best_loss, best_state, stale = monitored, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}"
            + (f", val_loss={monitored:.4f}" if val_loader else "")
        )
        if val_loader and stale > args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break
    model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 2,
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "parameter_names": list(dataset.parameter_names),
        "model_config": model.config(),
        "codec": dataset.codec.to_dict(),
        "training_args": {key: str(value) for key, value in vars(args).items()},
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    print(f"Saved Processing parameter model to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
