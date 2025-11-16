"""
Complete 6-Step Workflow Orchestrator for Aesthetic Beacon

This script orchestrates the complete workflow from EEG data collection
through real-time visual feedback generation.

Usage:
    # Run individual steps
    python workflow_orchestrator.py --step 1 --input data/step1_hex_eeg
    python workflow_orchestrator.py --step 2 --train

    # Run complete workflow
    python workflow_orchestrator.py --all

    # Real-time feedback mode
    python workflow_orchestrator.py --step 6 --realtime
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Import pipeline functions
from .pipeline import (
    # Step 1
    decode_hex_eeg_segment,
    iter_brainwave_segments,
    # Step 2
    ViewerSample,
    ViewerEnjoymentDataset,
    train_viewer_model,
    # Step 3
    evaluate_enjoyment_windows,
    # Step 4
    CreatorAnnotation,
    CreatorStateDataset,
    train_creator_model,
    # Step 5
    train_processing_parameter_model,
    # Step 6
    run_feedback_loop,
)

# Import OSC bridge
from .osc_bridge import create_bridge, OSCConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WorkflowConfig:
    """Configuration for the complete workflow."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from file or defaults."""
        self.config = self._load_defaults()

        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)

    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            # Paths
            "data_root": "data",
            "model_dir": "data/step6_realtime/models",
            "output_dir": "data/step6_realtime/outputs",

            # Step 1: Hex EEG conversion
            "step1": {
                "hex_dir": "data/step1_hex_eeg",
                "output_dir": "data/step1_hex_eeg/converted",
                "sample_rate": 640,
                "num_channels": 8,
            },

            # Step 2: Viewer model training
            "step2": {
                "eeg_dir": "data/step2_viewer_training/eeg",
                "audio_dir": "data/step2_viewer_training/audio",
                "annotations": "data/step2_viewer_training/annotations/annotations.json",
                "model_path": "data/step6_realtime/models/viewer_model.pt",
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 0.0001,
            },

            # Step 3: Article reading evaluation
            "step3": {
                "eeg_dir": "data/step3_article_reading/eeg",
                "results_dir": "data/step3_article_reading/results",
                "viewer_model": "data/step6_realtime/models/viewer_model.pt",
                "window_size": 5.0,
                "stride": 1.0,
            },

            # Step 4: Creator model training
            "step4": {
                "eeg_dir": "data/step4_creator_training/eeg",
                "annotations": "data/step4_creator_training/annotations/annotations.json",
                "model_path": "data/step6_realtime/models/creator_model.pt",
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 0.0001,
            },

            # Step 5: Processing parameter model training
            "step5": {
                "dataset_root": "data/step5_image_training",
                "model_path": "data/step6_realtime/models/parameter_model.pt",
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
            },

            # Step 6: Real-time feedback
            "step6": {
                "creator_model": "data/step6_realtime/models/creator_model.pt",
                "parameter_model": "data/step6_realtime/models/parameter_model.pt",
                "creatable_threshold": 0.6,
                "update_interval": 5.0,
                "use_osc": True,
                "osc_host": "127.0.0.1",
                "osc_port": 12000,
            },

            # Device
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def save(self, path: Path):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)


class WorkflowOrchestrator:
    """Orchestrates the complete 6-step workflow."""

    def __init__(self, config: WorkflowConfig):
        """Initialize orchestrator with configuration."""
        self.config = config
        self.device = torch.device(config.get('device'))
        logger.info(f"Using device: {self.device}")

    def run_step1(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Step 1: Convert hexadecimal EEG dumps to .npy arrays.

        Args:
            input_dir: Directory containing .hex files
            output_dir: Directory to save .npy files
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Converting Hexadecimal EEG to NumPy Arrays")
        logger.info("=" * 60)

        input_dir = Path(input_dir or self.config.get('step1.hex_dir'))
        output_dir = Path(output_dir or self.config.get('step1.output_dir'))
        output_dir.mkdir(parents=True, exist_ok=True)

        hex_files = list(input_dir.glob("*.hex"))
        if not hex_files:
            logger.warning(f"No .hex files found in {input_dir}")
            return

        logger.info(f"Found {len(hex_files)} hex files to convert")

        for hex_file in tqdm(hex_files, desc="Converting hex files"):
            with open(hex_file, 'r') as f:
                hex_data = f.read().strip()

            # Convert all 5-second segments
            segments = list(iter_brainwave_segments(hex_data))
            logger.info(f"  {hex_file.name}: {len(segments)} segments")

            for i, segment in enumerate(segments):
                output_file = output_dir / f"{hex_file.stem}_segment_{i:04d}.npy"
                np.save(output_file, segment)

        logger.info(f"✓ Converted all segments to {output_dir}")

    def run_step2_train(self):
        """
        Step 2: Train the "to viewer" model (aesthetic enjoyment prediction).
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Training Viewer Enjoyment Model")
        logger.info("=" * 60)

        # Load annotations
        annotations_path = Path(self.config.get('step2.annotations'))
        if not annotations_path.exists():
            logger.error(f"Annotations file not found: {annotations_path}")
            return

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Create dataset
        samples = []
        for item in annotations.get('samples', []):
            eeg_path = Path(self.config.get('data_root')) / item['eeg_file']
            audio_path = Path(self.config.get('data_root')) / item['audio_file']

            samples.append(ViewerSample(
                brainwave_path=eeg_path,
                audio_path=audio_path,
                enjoyed=item['enjoyed']
            ))

        logger.info(f"Loaded {len(samples)} training samples")

        dataset = ViewerEnjoymentDataset(samples)

        # Train model
        model, history = train_viewer_model(
            dataset=dataset,
            val_dataset=None,  # TODO: Add validation split
            epochs=self.config.get('step2.epochs'),
            batch_size=self.config.get('step2.batch_size'),
            lr=self.config.get('step2.learning_rate'),
            device=self.device
        )

        # Save model
        model_path = Path(self.config.get('step2.model_path'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"✓ Model saved to {model_path}")

        # Save training history
        history_path = model_path.with_suffix('.history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def run_step3(self, eeg_file: Optional[Path] = None):
        """
        Step 3: Evaluate article reading EEG for aesthetic enjoyment.

        Args:
            eeg_file: Path to EEG recording during article reading
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Evaluating Article Reading EEG")
        logger.info("=" * 60)

        if eeg_file is None:
            eeg_dir = Path(self.config.get('step3.eeg_dir'))
            eeg_files = list(eeg_dir.glob("*.npy"))
            if not eeg_files:
                logger.warning(f"No EEG files found in {eeg_dir}")
                return
            eeg_file = eeg_files[0]  # Process first file
            logger.info(f"Processing: {eeg_file}")

        # Load viewer model
        model_path = Path(self.config.get('step3.viewer_model'))
        if not model_path.exists():
            logger.error(f"Viewer model not found: {model_path}")
            logger.info("Please run --step 2 --train first")
            return

        from .model import create_model
        from .pipeline import ViewerEnjoymentModel

        model = ViewerEnjoymentModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        # Load EEG
        eeg_data = np.load(eeg_file)
        logger.info(f"EEG shape: {eeg_data.shape}")

        # Evaluate windows
        windows = evaluate_enjoyment_windows(
            model=model,
            brainwave_array=eeg_data,
            window_size=self.config.get('step3.window_size'),
            stride=self.config.get('step3.stride'),
            audio_supplier=None,  # No audio for article reading
            device=self.device
        )

        logger.info(f"Evaluated {len(windows)} windows")

        # Save results
        results_dir = Path(self.config.get('step3.results_dir'))
        results_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_dir / f"{eeg_file.stem}_enjoyment.json"
        results = {
            "session_id": eeg_file.stem,
            "eeg_file": str(eeg_file),
            "windows": [
                {
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "enjoyment_score": float(w.enjoyment_score)
                }
                for w in windows
            ]
        }

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to {result_file}")

        # Show top enjoyment moments
        top_windows = sorted(windows, key=lambda w: w.enjoyment_score, reverse=True)[:5]
        logger.info("\nTop 5 enjoyment moments:")
        for i, w in enumerate(top_windows, 1):
            logger.info(f"  {i}. {w.start_time:.1f}s - {w.end_time:.1f}s: {w.enjoyment_score:.3f}")

    def run_step4_train(self):
        """
        Step 4: Train the "to creator" model (creatable state detection).
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Training Creator State Model")
        logger.info("=" * 60)

        # Load annotations
        annotations_path = Path(self.config.get('step4.annotations'))
        if not annotations_path.exists():
            logger.error(f"Annotations file not found: {annotations_path}")
            return

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Create annotations list
        creator_annotations = []
        for passage in annotations.get('excellent_passages', []):
            eeg_path = Path(self.config.get('data_root')) / annotations.get('eeg_file', '')

            creator_annotations.append(CreatorAnnotation(
                brainwave_path=eeg_path,
                start_time=passage['start_time'],
                end_time=passage['end_time'],
                is_creatable=(passage['label'] == 'creatable')
            ))

        logger.info(f"Loaded {len(creator_annotations)} annotations")

        # Create dataset
        dataset = CreatorStateDataset(creator_annotations, sample_rate=640)

        # Train model
        model, history = train_creator_model(
            dataset=dataset,
            val_dataset=None,  # TODO: Add validation split
            epochs=self.config.get('step4.epochs'),
            batch_size=self.config.get('step4.batch_size'),
            lr=self.config.get('step4.learning_rate'),
            device=self.device
        )

        # Save model
        model_path = Path(self.config.get('step4.model_path'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"✓ Model saved to {model_path}")

        # Save training history
        history_path = model_path.with_suffix('.history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def run_step5_train(self):
        """
        Step 5: Train the EEG → Processing parameters model.
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Training Processing Parameter Model")
        logger.info("=" * 60)

        dataset_root = Path(self.config.get('step5.dataset_root'))
        if not dataset_root.exists():
            logger.error(f"Dataset not found: {dataset_root}")
            return

        # Train model
        model, history, param_names = train_processing_parameter_model(
            dataset_root=dataset_root,
            val_dataset_root=None,  # TODO: Add validation split
            epochs=self.config.get('step5.epochs'),
            batch_size=self.config.get('step5.batch_size'),
            lr=self.config.get('step5.learning_rate'),
            device=self.device
        )

        # Save model
        model_path = Path(self.config.get('step5.model_path'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'param_names': param_names
        }, model_path)
        logger.info(f"✓ Model saved to {model_path}")

        # Save training history
        history_path = model_path.with_suffix('.history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def run_step6_realtime(self):
        """
        Step 6: Real-time feedback loop (production mode).
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Real-Time Feedback Loop")
        logger.info("=" * 60)

        # Load models
        creator_model_path = Path(self.config.get('step6.creator_model'))
        parameter_model_path = Path(self.config.get('step6.parameter_model'))

        if not creator_model_path.exists():
            logger.error(f"Creator model not found: {creator_model_path}")
            logger.info("Please run --step 4 --train first")
            return

        if not parameter_model_path.exists():
            logger.error(f"Parameter model not found: {parameter_model_path}")
            logger.info("Please run --step 5 --train first")
            return

        # Load creator model
        from .pipeline import CreatorStateModel
        creator_model = CreatorStateModel()
        creator_model.load_state_dict(torch.load(creator_model_path, map_location=self.device))
        creator_model = creator_model.to(self.device)
        creator_model.eval()

        # Load parameter model
        checkpoint = torch.load(parameter_model_path, map_location=self.device)
        from .model import BrainwaveToProcessingModel
        parameter_model = BrainwaveToProcessingModel(
            num_params=len(checkpoint['param_names'])
        )
        parameter_model.load_state_dict(checkpoint['model_state'])
        parameter_model = parameter_model.to(self.device)
        parameter_model.eval()
        param_names = checkpoint['param_names']

        logger.info("Models loaded successfully")

        # Create OSC bridge
        osc_config = OSCConfig(
            host=self.config.get('step6.osc_host'),
            port=self.config.get('step6.osc_port')
        )
        bridge = create_bridge(
            use_osc=self.config.get('step6.use_osc'),
            config=osc_config
        )

        logger.info("OSC bridge initialized")
        logger.info("=" * 60)
        logger.info("READY FOR REAL-TIME PROCESSING")
        logger.info("Waiting for EEG data...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        # Define EEG supplier (mock for now - replace with actual hardware interface)
        def mock_eeg_supplier():
            """Mock EEG supplier - replace with actual hardware interface."""
            # In production, this would read from your EEG headset
            # For now, return synthetic data
            return np.random.randn(8, 3200).astype(np.float32)

        # Define parameter update callback
        def update_processing(parameters: Dict[str, float], is_creatable: bool):
            """Callback to update Processing sketch."""
            # Send state change
            bridge.send_state_change(is_creatable, confidence=1.0)

            # Send parameters if creatable
            if is_creatable:
                bridge.send_parameters(parameters)

        # Run feedback loop
        try:
            run_feedback_loop(
                creator_model=creator_model,
                parameter_model=parameter_model,
                parameter_names=param_names,
                eeg_supplier=mock_eeg_supplier,
                update_callback=update_processing,
                creatable_threshold=self.config.get('step6.creatable_threshold'),
                update_interval=self.config.get('step6.update_interval'),
                device=self.device
            )
        except KeyboardInterrupt:
            logger.info("\nStopping feedback loop...")
            bridge.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aesthetic Beacon: EEG-Based Creative Feedback Workflow"
    )

    parser.add_argument('--config', type=Path, help='Configuration file path')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5, 6], help='Run specific step')
    parser.add_argument('--train', action='store_true', help='Training mode (for steps 2, 4, 5)')
    parser.add_argument('--realtime', action='store_true', help='Real-time mode (for step 6)')
    parser.add_argument('--all', action='store_true', help='Run all steps sequentially')
    parser.add_argument('--input', type=Path, help='Input directory/file')
    parser.add_argument('--output', type=Path, help='Output directory')

    args = parser.parse_args()

    # Load configuration
    config = WorkflowConfig(args.config)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(config)

    # Run requested step(s)
    if args.all:
        logger.info("Running complete workflow (Steps 1-6)")
        orchestrator.run_step1()
        orchestrator.run_step2_train()
        orchestrator.run_step3()
        orchestrator.run_step4_train()
        orchestrator.run_step5_train()
        orchestrator.run_step6_realtime()

    elif args.step == 1:
        orchestrator.run_step1(args.input, args.output)

    elif args.step == 2:
        if args.train:
            orchestrator.run_step2_train()
        else:
            logger.error("Step 2 requires --train flag")

    elif args.step == 3:
        orchestrator.run_step3(args.input)

    elif args.step == 4:
        if args.train:
            orchestrator.run_step4_train()
        else:
            logger.error("Step 4 requires --train flag")

    elif args.step == 5:
        if args.train:
            orchestrator.run_step5_train()
        else:
            logger.error("Step 5 requires --train flag")

    elif args.step == 6:
        if args.realtime:
            orchestrator.run_step6_realtime()
        else:
            logger.error("Step 6 requires --realtime flag")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
