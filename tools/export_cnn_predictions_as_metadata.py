#!/usr/bin/env python
"""Export CNN predictions as ground truth metadata template.

This utility script helps migrate from CNN-based training to ground truth training.
It processes all artwork images through the trained CNN model and exports the
predictions as a metadata.json file, which can then be manually reviewed and adjusted
before being used for ground truth training.

Usage:
    python tools/export_cnn_predictions_as_metadata.py \
        --artworks-dir path/to/artworks \
        --processing-weights processing_features.pt \
        --output metadata.json

The output metadata.json can be:
1. Reviewed and manually adjusted to match desired aesthetic outcomes
2. Placed in the dataset directory for ground truth training
3. Used with the --parameter-metadata flag in src.train
"""

import argparse
import json
from pathlib import Path
from typing import List

from src.processing_features import compute_processing_parameters


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def find_artwork_files(artworks_dir: Path) -> List[Path]:
    """Find all supported artwork image files in the directory."""
    artwork_files = []
    for ext in SUPPORTED_EXTS:
        artwork_files.extend(artworks_dir.glob(f"*{ext}"))
    return sorted(artwork_files)


def export_predictions(
    artworks_dir: Path,
    processing_weights: Path,
    output_path: Path,
    device: str = "cpu",
    resize_long_edge: int = 512,
) -> None:
    """Export CNN parameter predictions for all artworks as metadata JSON."""

    artwork_files = find_artwork_files(artworks_dir)

    if not artwork_files:
        raise ValueError(
            f"No artwork files found in '{artworks_dir}'. "
            f"Supported extensions: {', '.join(SUPPORTED_EXTS)}"
        )

    print(f"Found {len(artwork_files)} artwork files")
    print(f"Processing with CNN model from: {processing_weights}")

    metadata = {}

    for i, artwork_path in enumerate(artwork_files, 1):
        sample_id = artwork_path.stem
        print(f"[{i}/{len(artwork_files)}] Processing {sample_id}...", end=" ")

        try:
            parameter_dict = compute_processing_parameters(
                artwork_path,
                weights_path=processing_weights,
                resize_long_edge=resize_long_edge,
                device=device,
            )

            # Convert OrderedDict to regular dict for JSON serialization
            metadata[sample_id] = dict(parameter_dict)
            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Exported {len(metadata)} parameter sets to: {output_path}")
    print("\nNext steps:")
    print("1. Review and manually adjust the parameters in the metadata file")
    print("2. Copy the metadata.json to your dataset root directory")
    print("3. Train with: python -m src.train <dataset> --parameter-metadata metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artworks-dir",
        type=Path,
        required=True,
        help="Directory containing artwork images",
    )
    parser.add_argument(
        "--processing-weights",
        type=Path,
        required=True,
        help="Path to the trained CNN model checkpoint (processing_features.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metadata.json"),
        help="Output path for the metadata JSON file (default: metadata.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for CNN inference (default: cpu)",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=512,
        help="Resize images to this long edge before inference (default: 512)",
    )

    args = parser.parse_args()

    if not args.artworks_dir.exists():
        parser.error(f"Artworks directory does not exist: {args.artworks_dir}")

    if not args.processing_weights.exists():
        parser.error(f"Processing weights file does not exist: {args.processing_weights}")

    export_predictions(
        artworks_dir=args.artworks_dir,
        processing_weights=args.processing_weights,
        output_path=args.output,
        device=args.device,
        resize_long_edge=args.resize_long_edge,
    )


if __name__ == "__main__":
    main()
