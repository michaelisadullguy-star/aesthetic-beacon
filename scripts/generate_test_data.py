"""
Generate Synthetic Test Data for the 6-Step Workflow

This script creates synthetic data for testing the aesthetic beacon pipeline.
Useful for development and testing when real EEG data is not available.

Usage:
    python scripts/generate_test_data.py --output data/
    python scripts/generate_test_data.py --step 2 --num-samples 50
"""

import argparse
import json
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def generate_hex_eeg(num_seconds: int = 30, sample_rate: int = 640, num_channels: int = 8) -> str:
    """
    Generate synthetic hexadecimal EEG data.

    Args:
        num_seconds: Duration in seconds
        sample_rate: Sampling rate (Hz)
        num_channels: Number of EEG channels

    Returns:
        Hexadecimal string representation of EEG data
    """
    num_samples = num_seconds * sample_rate

    # Generate synthetic EEG-like data with different frequency components
    time = np.linspace(0, num_seconds, num_samples)
    eeg_data = np.zeros((num_channels, num_samples))

    for ch in range(num_channels):
        # Mix of different frequency bands
        freq1 = 1.0 + ch * 0.5  # Delta (0.5-4 Hz)
        freq2 = 8.0 + ch * 1.0  # Alpha (8-13 Hz)
        freq3 = 20.0 + ch * 2.0  # Beta (13-30 Hz)

        signal = (
            0.5 * np.sin(2 * np.pi * freq1 * time) +
            0.3 * np.sin(2 * np.pi * freq2 * time) +
            0.2 * np.sin(2 * np.pi * freq3 * time)
        )

        # Add noise
        signal += np.random.normal(0, 0.1, num_samples)

        # Convert to int16 range
        signal = (signal * 1000).astype(np.int16)
        eeg_data[ch] = signal

    # Convert to hex (little-endian int16)
    hex_parts = []
    for i in range(num_samples):
        for ch in range(num_channels):
            value = int(eeg_data[ch, i])
            hex_bytes = struct.pack('<h', value)
            hex_parts.append(hex_bytes.hex())

    return ''.join(hex_parts)


def generate_eeg_segment(duration: float = 5.0, sample_rate: int = 640, num_channels: int = 8) -> np.ndarray:
    """
    Generate a single 5-second EEG segment as numpy array.

    Args:
        duration: Duration in seconds
        sample_rate: Sampling rate (Hz)
        num_channels: Number of EEG channels

    Returns:
        EEG array of shape (num_channels, num_samples)
    """
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    eeg_data = np.zeros((num_channels, num_samples), dtype=np.float32)

    for ch in range(num_channels):
        # Simulate different frequency bands
        freq1 = 2.0 + ch * 0.3
        freq2 = 10.0 + ch * 1.5
        freq3 = 25.0 + ch * 3.0

        signal = (
            0.4 * np.sin(2 * np.pi * freq1 * time) +
            0.35 * np.sin(2 * np.pi * freq2 * time) +
            0.25 * np.sin(2 * np.pi * freq3 * time)
        )

        # Add realistic noise
        signal += np.random.normal(0, 0.15, num_samples)

        # Normalize to [-1, 1]
        signal = signal / (np.abs(signal).max() + 1e-6)
        eeg_data[ch] = signal

    return eeg_data


def generate_audio_segment(duration: float = 5.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate synthetic audio segment.

    Args:
        duration: Duration in seconds
        sample_rate: Audio sample rate (Hz)

    Returns:
        Audio array of shape (num_samples,)
    """
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)

    # Generate musical tones
    frequencies = [220, 277, 330, 392, 440]  # Musical notes
    audio = np.zeros(num_samples)

    for freq in frequencies:
        audio += 0.2 * np.sin(2 * np.pi * freq * time)

    # Add envelope
    envelope = np.exp(-time * 0.3)
    audio *= envelope

    # Normalize to int16 range
    audio = (audio * 32767).astype(np.int16)

    return audio


def generate_synthetic_image(width: int = 512, height: int = 512, style: str = 'gradient') -> Image.Image:
    """
    Generate a synthetic image for testing.

    Args:
        width: Image width
        height: Image height
        style: Image style ('gradient', 'circles', 'noise')

    Returns:
        PIL Image
    """
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    if style == 'gradient':
        # Gradient background
        for y in range(height):
            r = int(255 * (y / height))
            g = int(255 * (1 - y / height))
            b = 128
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    elif style == 'circles':
        # Random circles
        for _ in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            r = np.random.randint(20, 100)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    elif style == 'noise':
        # Random noise
        pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)

    return img


def generate_processing_parameters() -> dict:
    """Generate random Processing parameters."""
    return {
        "canvasWidth": np.random.randint(600, 1200),
        "canvasHeight": np.random.randint(400, 1000),
        "cellSize": np.random.randint(15, 50),
        "cellScaleJitter": np.random.uniform(0.1, 0.6),
        "cellRotationJitter": np.random.uniform(0.2, 0.8),
        "cellOffsetJitter": np.random.uniform(0.1, 0.5),
        "cornerRadiusJitter": np.random.uniform(1.0, 5.0),
        "palettePhaseJitter": np.random.uniform(0.1, 0.4),

        "noiseScale": np.random.uniform(0.003, 0.02),
        "noiseSpeed": np.random.uniform(0.1, 0.5),
        "noiseOctaves": np.random.randint(3, 6),
        "noiseFalloff": np.random.uniform(0.4, 0.6),

        "colorCycleSpeed": np.random.uniform(0.1, 0.8),
        "paletteSpread": np.random.uniform(40, 120),
        "saturationJitter": np.random.uniform(10, 40),
        "brightnessJitter": np.random.uniform(10, 40),

        "textureDepth": np.random.uniform(0.4, 1.0),
        "fillAlpha": np.random.uniform(40, 80),
        "cornerRadius": np.random.uniform(2.0, 8.0),

        "accentThreshold": np.random.uniform(0.6, 0.85),
        "accentLayers": np.random.randint(2, 5),
        "accentScale": np.random.uniform(1.5, 3.0),
        "accentStrokeWeight": np.random.uniform(1.0, 3.0),
        "accentAlpha": np.random.uniform(30, 60),
        "accentJitter": np.random.uniform(0.2, 0.5),
        "accentDrift": np.random.uniform(0.2, 0.6),
        "accentPhaseJitter": np.random.uniform(0.2, 0.5),
        "accentRadiusJitter": np.random.uniform(0.2, 0.5),

        "overlayBands": np.random.randint(3, 8),
        "overlayNoiseScale": np.random.uniform(0.002, 0.008),
        "overlayAmplitude": np.random.uniform(20, 60),
        "overlayAlpha": np.random.uniform(15, 40),
        "overlaySpeed": np.random.uniform(0.2, 0.6),
        "overlayStep": np.random.randint(8, 24),
        "overlayStrokeScale": np.random.uniform(2.0, 5.0),
        "overlayAlphaJitter": np.random.uniform(0.2, 0.5),

        "backgroundHue": np.random.uniform(0, 360),
        "backgroundSaturation": np.random.uniform(10, 30),
        "backgroundBrightness": np.random.uniform(5, 15),
        "backgroundAlpha": 100.0,

        "targetFrameRate": 60,
        "smoothingSamples": 4,

        "palette": [
            {"h": np.random.uniform(0, 360), "s": np.random.uniform(50, 90), "b": np.random.uniform(80, 100)},
            {"h": np.random.uniform(0, 360), "s": np.random.uniform(50, 90), "b": np.random.uniform(80, 100)},
            {"h": np.random.uniform(0, 360), "s": np.random.uniform(50, 90), "b": np.random.uniform(80, 100)},
            {"h": np.random.uniform(0, 360), "s": np.random.uniform(50, 90), "b": np.random.uniform(80, 100)},
        ]
    }


def generate_step1_data(output_dir: Path, num_files: int = 3):
    """Generate test data for Step 1 (hex EEG conversion)."""
    print(f"Generating Step 1 data: {num_files} hex files")
    output_dir = output_dir / "step1_hex_eeg"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_files):
        hex_data = generate_hex_eeg(num_seconds=30)
        file_path = output_dir / f"session_{i+1:03d}.hex"
        with open(file_path, 'w') as f:
            f.write(hex_data)
        print(f"  Created {file_path}")


def generate_step2_data(output_dir: Path, num_samples: int = 20):
    """Generate test data for Step 2 (viewer model training)."""
    print(f"Generating Step 2 data: {num_samples} EEG-audio pairs")

    eeg_dir = output_dir / "step2_viewer_training" / "eeg"
    audio_dir = output_dir / "step2_viewer_training" / "audio"
    anno_dir = output_dir / "step2_viewer_training" / "annotations"

    eeg_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    annotations = {"samples": []}

    for i in range(num_samples):
        # Generate EEG
        eeg = generate_eeg_segment()
        eeg_file = eeg_dir / f"sample_{i+1:04d}.npy"
        np.save(eeg_file, eeg)

        # Generate audio (save as .wav would require scipy/soundfile)
        # For now, just save as .npy
        audio = generate_audio_segment()
        audio_file = audio_dir / f"sample_{i+1:04d}.npy"
        np.save(audio_file, audio)

        # Random enjoyment label
        enjoyed = np.random.random() > 0.5

        annotations["samples"].append({
            "eeg_file": f"step2_viewer_training/eeg/{eeg_file.name}",
            "audio_file": f"step2_viewer_training/audio/{audio_file.name}",
            "enjoyed": enjoyed
        })

    # Save annotations
    anno_file = anno_dir / "annotations.json"
    with open(anno_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"  Created {num_samples} samples in {eeg_dir} and {audio_dir}")
    print(f"  Annotations: {anno_file}")


def generate_step3_data(output_dir: Path, num_sessions: int = 2):
    """Generate test data for Step 3 (article reading evaluation)."""
    print(f"Generating Step 3 data: {num_sessions} article reading sessions")

    eeg_dir = output_dir / "step3_article_reading" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_sessions):
        # Generate long EEG recording (60 seconds)
        duration = 60.0
        sample_rate = 640
        num_samples = int(duration * sample_rate)

        # Generate as concatenated 5-second segments
        segments = []
        for _ in range(int(duration / 5)):
            seg = generate_eeg_segment()
            segments.append(seg)

        eeg = np.concatenate(segments, axis=1)
        eeg_file = eeg_dir / f"participant_{i+1:03d}_article.npy"
        np.save(eeg_file, eeg)
        print(f"  Created {eeg_file} (shape: {eeg.shape})")


def generate_step4_data(output_dir: Path, num_sessions: int = 2):
    """Generate test data for Step 4 (creator model training)."""
    print(f"Generating Step 4 data: {num_sessions} creator sessions")

    eeg_dir = output_dir / "step4_creator_training" / "eeg"
    anno_dir = output_dir / "step4_creator_training" / "annotations"

    eeg_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_sessions):
        # Generate creator EEG (120 seconds)
        duration = 120.0
        segments = []
        for _ in range(int(duration / 5)):
            seg = generate_eeg_segment()
            segments.append(seg)

        eeg = np.concatenate(segments, axis=1)
        eeg_file = eeg_dir / f"creator_session_{i+1:03d}.npy"
        np.save(eeg_file, eeg)

        # Generate annotations
        annotations = {
            "session_id": f"creator_session_{i+1:03d}",
            "eeg_file": f"step4_creator_training/eeg/{eeg_file.name}",
            "keystroke_log": f"creator_session_{i+1:03d}_keystrokes.json",
            "excellent_passages": []
        }

        # Random excellent passages
        num_passages = np.random.randint(5, 10)
        for j in range(num_passages):
            start = np.random.uniform(0, duration - 5)
            annotations["excellent_passages"].append({
                "start_time": float(start),
                "end_time": float(start + 5.0),
                "content": f"Example passage {j+1}",
                "label": "creatable" if np.random.random() > 0.3 else "not_creatable"
            })

        anno_file = anno_dir / f"creator_session_{i+1:03d}.json"
        with open(anno_file, 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"  Created {eeg_file} with {len(annotations['excellent_passages'])} annotations")


def generate_step5_data(output_dir: Path, num_samples: int = 50):
    """Generate test data for Step 5 (EEG to Processing parameters)."""
    print(f"Generating Step 5 data: {num_samples} EEG-image pairs")

    dataset_root = output_dir / "step5_image_training"
    brainwave_dir = dataset_root / "brainwaves"
    artwork_dir = dataset_root / "artworks"

    brainwave_dir.mkdir(parents=True, exist_ok=True)
    artwork_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}

    styles = ['gradient', 'circles', 'noise']

    for i in range(num_samples):
        # Generate EEG
        eeg = generate_eeg_segment()
        eeg_file = brainwave_dir / f"sample_{i+1:04d}.npy"
        np.save(eeg_file, eeg)

        # Generate image
        style = styles[i % len(styles)]
        img = generate_synthetic_image(style=style)
        img_file = artwork_dir / f"sample_{i+1:04d}.png"
        img.save(img_file)

        # Generate parameters
        params = generate_processing_parameters()
        metadata[img_file.name] = params

    # Save metadata
    metadata_file = dataset_root / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Created {num_samples} samples")
    print(f"  Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument('--output', type=Path, default=Path('data'), help='Output directory')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5], help='Generate data for specific step only')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Synthetic Test Data")
    print("=" * 60)

    if args.step is None or args.step == 1:
        generate_step1_data(output_dir, num_files=3)

    if args.step is None or args.step == 2:
        generate_step2_data(output_dir, num_samples=args.num_samples)

    if args.step is None or args.step == 3:
        generate_step3_data(output_dir, num_sessions=2)

    if args.step is None or args.step == 4:
        generate_step4_data(output_dir, num_sessions=2)

    if args.step is None or args.step == 5:
        generate_step5_data(output_dir, num_samples=args.num_samples)

    print("=" * 60)
    print("✓ Test data generation complete!")
    print(f"Data saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
