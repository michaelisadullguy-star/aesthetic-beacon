# Aesthetic Beacon

**EEG-Based Creative Feedback System**

Aesthetic Beacon is a complete workflow for using EEG brainwave data to create real-time creative feedback. The system learns from participants' aesthetic responses to generate dynamic visuals that respond to a creator's brainwave patterns.

## Quick Links

- **[Complete Workflow Guide](WORKFLOW.md)** - Detailed step-by-step instructions
- **[Data Directory Structure](data/README.md)** - Understanding the data organization
- **[Example Configuration](config.example.json)** - Template for custom configs

## What This System Does

1. **Measures aesthetic enjoyment** from EEG during music listening
2. **Identifies excellent passages** in written content based on reader EEG
3. **Detects creative states** in the creator's brainwaves
4. **Maps brainwaves to visuals** using Processing parameters
5. **Provides real-time feedback** by updating visuals when creator is in flow state

## Quick Start

```bash
# 1. Generate test data
python scripts/generate_test_data.py --output data/

# 2. Train models
python src/workflow_orchestrator.py --step 2 --train  # Viewer model
python src/workflow_orchestrator.py --step 4 --train  # Creator model
python src/workflow_orchestrator.py --step 5 --train  # Parameter model

# 3. Run real-time feedback
# First, open scripts/processing_templates/dynamic_texture_osc.pde in Processing
# Then:
python src/workflow_orchestrator.py --step 6 --realtime
```

For detailed instructions, see **[WORKFLOW.md](WORKFLOW.md)**.

## The 6-Step Workflow

This project provides a PyTorch-based pipeline with six integrated steps:

1. **Hex EEG Conversion** - Convert raw hexadecimal EEG to numpy arrays
2. **Viewer Model Training** - Learn aesthetic enjoyment patterns from EEG + audio
3. **Article Evaluation** - Identify best passages using reader EEG responses
4. **Creator Model Training** - Detect "creatable" brainwave states
5. **Parameter Model Training** - Map EEG to Processing visual parameters
6. **Real-Time Feedback** - Update visuals based on creator's brainwave state

## Project Structure

### Core Pipeline (`src/`)
- `pipeline.py` - Complete 6-step workflow implementation (all steps integrated)
- `workflow_orchestrator.py` - Command-line orchestrator for running any step
- `osc_bridge.py` - Python → Processing communication via OSC
- `brainwave_dataset.py` - Dataset utilities for EEG data loading
- `model.py` - Neural network architectures (temporal encoding, regression)
- `processing_features.py` - Image → Processing parameter prediction

### Training Scripts (`src/`)
- `train.py` - Train brainwave → Processing parameters model
- `train_processing_features.py` - Train artwork → Processing parameters model

### Utilities (`scripts/`)
- `generate_test_data.py` - Generate synthetic data for testing
- `processing_templates/dynamic_texture_template.pde` - Base Processing sketch
- `processing_templates/dynamic_texture_osc.pde` - OSC-enabled Processing sketch

### Data Organization (`data/`)
```
data/
├── step1_hex_eeg/           # Raw hex EEG files
├── step2_viewer_training/   # EEG + audio + enjoyment labels
├── step3_article_reading/   # Article reading EEG sessions
├── step4_creator_training/  # Creator EEG + annotations
├── step5_image_training/    # EEG + images + parameters
└── step6_realtime/
    ├── models/              # Trained model checkpoints
    └── outputs/             # Generated parameters
```

## Preparing the dataset

Organise the data with the following layout:

```
dataset_root/
├── brainwaves/
│   ├── sample_0001.npy
│   ├── sample_0002.npy
│   └── ...
└── artworks/
    ├── sample_0001.jpg
    ├── sample_0002.jpg
    └── ...
```

- Each `brainwaves/*.npy` file should contain a floating-point tensor with shape
  `(8, 3200)` representing eight frequency bands sampled at 640 Hz over five
  seconds. Values are interpreted directly with no hexadecimal decoding step.
- `artworks/` contains colour reference images whose file names (without extension)
  match the brainwave recording they are paired with. Supported formats include PNG,
  JPEG, and BMP. Processing descriptors are predicted by the trained artwork model
  referenced via the `--processing-weights` flag when launching `src/train.py`.

> **Tip:** the artwork preprocessing pipeline resizes images whose longest edge
> exceeds 512 pixels during both model training and inference. Adjust this with
> the `--resize-long-edge` option in `train_processing_features.py` or by passing
> `processing_resize_long_edge` to `BrainwaveProcessingDataset`.

## Training the artwork parameter model

Before mapping brainwaves to Processing controls, train the artwork predictor on
reference images paired with manually curated Processing parameters. Arrange the
artwork dataset as follows:

```
artwork_dataset/
├── images/
│   ├── sample_0001.png
│   ├── sample_0002.png
│   └── ...
└── metadata.json
```

`metadata.json` may be either an object mapping image identifiers to parameter
dictionaries or a list of objects with `id`/`image` and `parameters` keys:

```json
[
  {
    "id": "sample_0001",
    "parameters": {
      "background_r": 0.12,
      "background_g": 0.08,
      "background_b": 0.15,
      "stroke_weight": 4.0
      // ... additional parameters ...
    }
  }
]
```

Train the predictor and store the resulting checkpoint:

```bash
python -m src.train_processing_features /path/to/artwork_dataset \
  --output processing_features.pt \
  --epochs 120 \
  --batch-size 16
```

The generated `.pt` file contains the learned weights, parameter ordering, and
normalisation statistics required for downstream inference.

## Training

Install the dependencies:

```bash
pip install torch torchvision torchaudio pillow --index-url https://download.pytorch.org/whl/cu118
```

(Optional) create a virtual environment before installing packages.

Run the brainwave-to-Processing training script and provide the artwork weights:

```bash
python -m src.train /path/to/dataset \
  --output-dir runs/exp01 \
  --batch-size 32 \
  --epochs 150 \
  --learning-rate 5e-4 \
  --crop-length 2560 \
  --expected-channels 8 \
  --expected-samples 3200 \
  --processing-weights /path/to/processing_features.pt \
  --amp
```

During training, JSON logs are printed every epoch and checkpoints are stored in the
specified output directory. The best-performing model on the validation split is also
saved as `best_model.pt`.

## Inference

After training, load the model and use it to predict Processing parameters for new
brainwave segments:

```python
import torch
from src.model import create_model

checkpoint = torch.load("runs/exp01/best_model.pt", map_location="cpu")
parameter_names = checkpoint["parameter_names"]
model_state = checkpoint["model_state"]

model = create_model(input_shape=(8, 3200), parameter_dim=len(parameter_names))
model.load_state_dict(model_state)
model.eval()

with torch.no_grad():
    brainwave_tensor = torch.randn(1, 8, 3200)  # replace with real data
    prediction = model(brainwave_tensor)

print(dict(zip(parameter_names, prediction.squeeze(0).tolist())))
```

The resulting dictionary can be fed into your Processing sketch to reproduce the learned
texture and colour dynamics.

## Post-merge sanity check

To verify that no merge conflict markers remain after integrating branches, run the
helper script:

```bash
python scripts/check_merge_conflicts.py
```

The command scans all tracked files and reports any unresolved Git conflict markers so
they can be addressed before committing.

