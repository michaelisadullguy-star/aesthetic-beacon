#!/usr/bin/env python
"""Export automatically derived artwork parameters as an editable metadata.json.

Step 5 derives the Processing parameters of each artwork automatically.  This
tool writes those values to a file so that you can review and adjust them by
hand; a ``metadata.json`` placed in the dataset folder then takes precedence.

Usage:
    # parameters from direct image analysis (no trained model needed)
    python tools/export_cnn_predictions_as_metadata.py \
        --artworks-dir data/step5_image_training/artworks \
        --output data/step5_image_training/metadata.json

    # parameters predicted by a CNN trained with src/train_processing_features.py
    python tools/export_cnn_predictions_as_metadata.py \
        --artworks-dir path/to/artworks \
        --processing-weights processing_features.pt \
        --output metadata.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing_features import SUPPORTED_IMAGE_SUFFIXES, resolve_artwork_parameters  # noqa: E402

SUPPORTED_EXTS = SUPPORTED_IMAGE_SUFFIXES


def find_artwork_files(artworks_dir: Path) -> List[Path]:
    """Find all supported artwork image files in the directory."""

    return sorted(p for p in artworks_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def export_predictions(
    artworks_dir: Path,
    processing_weights: Optional[Path],
    output_path: Path,
    device: str = "cpu",
    resize_long_edge: int = 512,
) -> int:
    """Write the parameters of every artwork to ``output_path``; returns how many."""

    artwork_files = find_artwork_files(artworks_dir)
    if not artwork_files:
        raise ValueError(
            f"No artwork files found in '{artworks_dir}'. Supported extensions: {', '.join(SUPPORTED_EXTS)}"
        )

    metadata = {}
    for index, artwork_path in enumerate(artwork_files, 1):
        try:
            parameters, source = resolve_artwork_parameters(
                artwork_path,
                weights_path=processing_weights,
                resize_long_edge=resize_long_edge,
                device=device,
            )
        except Exception as error:
            print(f"[{index}/{len(artwork_files)}] {artwork_path.name}: failed ({error})")
            continue
        metadata[artwork_path.stem] = {name: round(float(value), 5) for name, value in parameters.items()}
        print(f"[{index}/{len(artwork_files)}] {artwork_path.name}: ok ({source})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nExported {len(metadata)} parameter sets to {output_path}")
    print("Edit the values as you see fit; Step 5 uses this file when it is named")
    print("metadata.json and sits next to the 'artworks' folder.")
    return len(metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artworks-dir", type=Path, required=True, help="Directory containing artwork images")
    parser.add_argument("--processing-weights", type=Path, default=None, help="Trained artwork CNN (optional)")
    parser.add_argument("--output", type=Path, default=Path("metadata.json"), help="Output file")
    parser.add_argument("--device", type=str, default="cpu", help="Device for CNN inference")
    parser.add_argument("--resize-long-edge", type=int, default=512, help="Resize before CNN inference")
    args = parser.parse_args()

    if not args.artworks_dir.is_dir():
        parser.error(f"Artworks directory does not exist: {args.artworks_dir}")
    if args.processing_weights is not None and not args.processing_weights.is_file():
        parser.error(f"Processing weights file does not exist: {args.processing_weights}")

    try:
        count = export_predictions(args.artworks_dir, args.processing_weights, args.output, args.device, args.resize_long_edge)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
