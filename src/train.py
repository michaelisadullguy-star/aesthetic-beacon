"""Stand-alone training of the EEG → Processing parameter model (Step 5).

Equivalent to ``workflow_orchestrator.py --step 5`` but with every option on
the command line::

    python -m src.train data/step5_image_training --output-dir runs/exp01 --epochs 150
"""
from __future__ import annotations

if __package__ in (None, ""):  # executed as a script
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    __package__ = "src"
    __import__("src")

import argparse
import json
import logging
import sys
from pathlib import Path

from .eeg_features import DEFAULT_FEATURE_CONFIG
from .model import save_checkpoint
from .pipeline import PipelineError, train_processing_parameter_model


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Dataset root with 'artworks/' and 'brainwaves/'.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where the model is written.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many epochs without improvement.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda or auto.")
    parser.add_argument(
        "--parameter-metadata",
        type=Path,
        default=None,
        help="Curated artwork parameters (default: metadata.json in the dataset, if present).",
    )
    parser.add_argument(
        "--processing-weights",
        type=Path,
        default=None,
        help="Artwork CNN from train_processing_features.py, used for artworks without curated parameters.",
    )
    # Options of earlier versions, accepted so existing command lines keep working.
    for flag in ("--crop-length", "--grad-clip", "--num-workers", "--expected-channels", "--expected-samples"):
        parser.add_argument(flag, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        model, history, names = train_processing_parameter_model(
            args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_split=args.validation_split,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            parameter_metadata=args.parameter_metadata,
            processing_weights=args.processing_weights,
        )
    except (PipelineError, FileNotFoundError, ValueError) as error:
        print(f"Cannot train: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = save_checkpoint(
        args.output_dir / "best_model.pt",
        kind="parameter",
        model=model,
        feature_config=DEFAULT_FEATURE_CONFIG,
        extra={"metrics": history.metrics},
    )
    with open(args.output_dir / "history.json", "w", encoding="utf8") as handle:
        json.dump(history.to_dict(), handle, indent=2)
    print(f"Saved the model for {len(names)} parameters to {path}")
    print(json.dumps(history.to_dict()["metrics"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
