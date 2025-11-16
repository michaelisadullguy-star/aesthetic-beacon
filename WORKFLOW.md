# Aesthetic Beacon: Complete Workflow Guide

This guide walks you through the complete 6-step EEG-based creative feedback workflow.

## Overview

The Aesthetic Beacon system uses EEG data to create a real-time creative feedback loop:

1. **Step 1**: Convert hexadecimal EEG data to numpy arrays
2. **Step 2**: Train a "to viewer" model (predicts aesthetic enjoyment from EEG + audio)
3. **Step 3**: Evaluate participants' EEG during article reading to identify best passages
4. **Step 4**: Train a "to creator" model (detects creatable brainwave states)
5. **Step 5**: Train EEG → Processing parameters model (maps brainwaves to visuals)
6. **Step 6**: Real-time feedback loop (update visuals based on creator's EEG)

---

## Prerequisites

### Software Requirements

```bash
# Python dependencies
pip install torch numpy pillow tqdm python-osc

# Processing (for visualization)
# Download from: https://processing.org/download
# Install oscP5 library: Sketch → Import Library → Manage Libraries → oscP5
```

### Hardware Requirements

- EEG headset (8 channels, 640 Hz sampling rate)
- Computer capable of running PyTorch models in real-time
- Display for Processing visualization

---

## Quick Start (with Test Data)

### 1. Generate Test Data

```bash
# Generate synthetic test data for all steps
python scripts/generate_test_data.py --output data/

# Or generate data for a specific step
python scripts/generate_test_data.py --step 5 --num-samples 100
```

### 2. Run Individual Steps

```bash
# Step 1: Convert hex EEG to numpy
python src/workflow_orchestrator.py --step 1 --input data/step1_hex_eeg

# Step 2: Train viewer model
python src/workflow_orchestrator.py --step 2 --train

# Step 3: Evaluate article reading
python src/workflow_orchestrator.py --step 3

# Step 4: Train creator model
python src/workflow_orchestrator.py --step 4 --train

# Step 5: Train parameter model
python src/workflow_orchestrator.py --step 5 --train

# Step 6: Real-time feedback
python src/workflow_orchestrator.py --step 6 --realtime
```

### 3. Run Complete Workflow

```bash
# Execute all steps sequentially
python src/workflow_orchestrator.py --all
```

---

## Detailed Step-by-Step Instructions

### Step 1: Hexadecimal EEG Conversion

**Purpose**: Convert raw EEG data from hex format to 8×3200 numpy arrays.

**Input**:
- Hex files in `data/step1_hex_eeg/*.hex`
- Format: Plain text hexadecimal (little-endian int16)

**Output**:
- Numpy files in `data/step1_hex_eeg/converted/*.npy`
- Shape: (8, 3200) for each 5-second segment

**Run**:
```bash
python src/workflow_orchestrator.py --step 1
```

**Python API**:
```python
from pipeline import decode_hex_eeg_segment, iter_brainwave_segments

# Load hex data
with open('session.hex', 'r') as f:
    hex_data = f.read()

# Convert all segments
for i, segment in enumerate(iter_brainwave_segments(hex_data)):
    print(f"Segment {i}: shape {segment.shape}")  # (8, 3200)
    np.save(f'segment_{i}.npy', segment)
```

---

### Step 2: Train "To Viewer" Model

**Purpose**: Learn which EEG patterns (during music listening) indicate aesthetic enjoyment.

**Input**:
- EEG segments: `data/step2_viewer_training/eeg/*.npy`
- Audio clips: `data/step2_viewer_training/audio/*.wav`
- Annotations: `data/step2_viewer_training/annotations/annotations.json`

**Annotation Format**:
```json
{
  "samples": [
    {
      "eeg_file": "step2_viewer_training/eeg/sample_001.npy",
      "audio_file": "step2_viewer_training/audio/sample_001.wav",
      "enjoyed": true
    }
  ]
}
```

**Output**:
- Trained model: `data/step6_realtime/models/viewer_model.pt`
- Training history: `data/step6_realtime/models/viewer_model.history.json`

**Run**:
```bash
python src/workflow_orchestrator.py --step 2 --train
```

**Python API**:
```python
from pipeline import ViewerSample, ViewerEnjoymentDataset, train_viewer_model

# Create samples
samples = [
    ViewerSample(
        brainwave_path='eeg/sample_001.npy',
        audio_path='audio/sample_001.wav',
        enjoyed=True
    ),
    # ... more samples
]

# Train
dataset = ViewerEnjoymentDataset(samples)
model, history = train_viewer_model(
    dataset=dataset,
    epochs=50,
    batch_size=16,
    lr=0.0001
)

# Save
torch.save(model.state_dict(), 'viewer_model.pt')
```

---

### Step 3: Evaluate Article Reading EEG

**Purpose**: Use the viewer model to find which article passages delivered best aesthetic enjoyment.

**Input**:
- Long EEG recordings: `data/step3_article_reading/eeg/*.npy`
- Trained viewer model: `data/step6_realtime/models/viewer_model.pt`

**Output**:
- Enjoyment scores per time window: `data/step3_article_reading/results/*_enjoyment.json`

**Result Format**:
```json
{
  "session_id": "participant_001_article",
  "windows": [
    {"start_time": 0.0, "end_time": 5.0, "enjoyment_score": 0.87},
    {"start_time": 1.0, "end_time": 6.0, "enjoyment_score": 0.92}
  ]
}
```

**Run**:
```bash
python src/workflow_orchestrator.py --step 3 --input data/step3_article_reading/eeg/participant_001.npy
```

**Python API**:
```python
from pipeline import evaluate_enjoyment_windows

# Load model
model = ViewerEnjoymentModel()
model.load_state_dict(torch.load('viewer_model.pt'))

# Load EEG
eeg = np.load('participant_001_reading.npy')

# Evaluate
windows = evaluate_enjoyment_windows(
    model=model,
    brainwave_array=eeg,
    window_size=5.0,  # seconds
    stride=1.0,       # seconds
    device='cuda'
)

# Find best moments
top_windows = sorted(windows, key=lambda w: w.enjoyment_score, reverse=True)[:5]
for w in top_windows:
    print(f"{w.start_time:.1f}s - {w.end_time:.1f}s: {w.enjoyment_score:.3f}")
```

---

### Step 4: Train "To Creator" Model

**Purpose**: Learn which EEG patterns indicate the creator is in a "creatable" state.

**Input**:
- Creator EEG: `data/step4_creator_training/eeg/*.npy`
- Annotations: `data/step4_creator_training/annotations/*.json`

**Annotation Format**:
```json
{
  "session_id": "creator_session_001",
  "eeg_file": "step4_creator_training/eeg/creator_session_001.npy",
  "excellent_passages": [
    {
      "start_time": 120.5,
      "end_time": 125.5,
      "content": "The passage written during this time",
      "label": "creatable"
    }
  ]
}
```

**Output**:
- Trained model: `data/step6_realtime/models/creator_model.pt`

**Run**:
```bash
python src/workflow_orchestrator.py --step 4 --train
```

**Python API**:
```python
from pipeline import CreatorAnnotation, CreatorStateDataset, train_creator_model

# Create annotations
annotations = [
    CreatorAnnotation(
        brainwave_path='creator_eeg.npy',
        start_time=120.5,
        end_time=125.5,
        is_creatable=True
    ),
    # ... more annotations
]

# Train
dataset = CreatorStateDataset(annotations, sample_rate=640)
model, history = train_creator_model(
    dataset=dataset,
    epochs=50,
    batch_size=16
)

# Save
torch.save(model.state_dict(), 'creator_model.pt')
```

---

### Step 5: Train EEG → Processing Parameters Model

**Purpose**: Learn mapping from EEG (viewing images) to Processing visual parameters.

**Input**:
- EEG responses: `data/step5_image_training/brainwaves/*.npy`
- Artwork images: `data/step5_image_training/artworks/*.png`
- Metadata: `data/step5_image_training/metadata.json`

**Metadata Format**:
```json
{
  "artwork_001.png": {
    "canvasWidth": 800,
    "canvasHeight": 800,
    "noiseScale": 0.015,
    "colorCycleSpeed": 0.002,
    "cellSize": 40,
    ...
  }
}
```

**Important**: Filenames must match!
- `brainwaves/sample_001.npy` ↔ `artworks/sample_001.png`

**Output**:
- Trained model: `data/step6_realtime/models/parameter_model.pt`

**Run**:
```bash
python src/workflow_orchestrator.py --step 5 --train
```

**Python API**:
```python
from pipeline import train_processing_parameter_model

# Train
model, history, param_names = train_processing_parameter_model(
    dataset_root='data/step5_image_training',
    epochs=100,
    batch_size=8,
    lr=0.001
)

# Save
torch.save({
    'model_state': model.state_dict(),
    'param_names': param_names
}, 'parameter_model.pt')
```

---

### Step 6: Real-Time Feedback Loop

**Purpose**: Update Processing visuals in real-time based on creator's EEG state.

**Prerequisites**:
1. Trained creator model (Step 4)
2. Trained parameter model (Step 5)
3. Processing sketch running with OSC enabled

**Setup**:

**A. Start Processing Sketch**:
```processing
// Open: scripts/processing_templates/dynamic_texture_osc.pde
// In Processing IDE: Sketch → Run
// Press 'O' to show OSC status overlay
```

**B. Start Python Feedback Loop**:
```bash
python src/workflow_orchestrator.py --step 6 --realtime
```

**How it Works**:
1. Python reads EEG every 5 seconds (from your EEG headset)
2. Creator model predicts if you're in "creatable" state
3. If creatable: Parameter model generates Processing parameters from EEG
4. Parameters sent to Processing via OSC
5. Visuals update in real-time

**Python API**:
```python
from pipeline import run_feedback_loop
from osc_bridge import create_bridge

# Load models
creator_model = CreatorStateModel()
creator_model.load_state_dict(torch.load('creator_model.pt'))

parameter_model = BrainwaveToProcessingModel(num_params=40)
checkpoint = torch.load('parameter_model.pt')
parameter_model.load_state_dict(checkpoint['model_state'])
param_names = checkpoint['param_names']

# Create OSC bridge
bridge = create_bridge(use_osc=True)

# Define EEG supplier (replace with your hardware interface)
def get_eeg():
    # Read from your EEG headset
    # Return: numpy array of shape (8, 3200)
    return my_eeg_headset.read_segment()

# Define callback
def update_processing(parameters, is_creatable):
    bridge.send_state_change(is_creatable, confidence=1.0)
    if is_creatable:
        bridge.send_parameters(parameters)

# Run loop
run_feedback_loop(
    creator_model=creator_model,
    parameter_model=parameter_model,
    parameter_names=param_names,
    eeg_supplier=get_eeg,
    update_callback=update_processing,
    creatable_threshold=0.6,
    update_interval=5.0
)
```

---

## Configuration

### Using Configuration Files

Create a custom configuration file:

```json
{
  "device": "cuda",
  "step5": {
    "epochs": 200,
    "batch_size": 16,
    "learning_rate": 0.0005
  },
  "step6": {
    "creatable_threshold": 0.7,
    "update_interval": 3.0,
    "osc_port": 12000
  }
}
```

Run with custom config:
```bash
python src/workflow_orchestrator.py --config my_config.json --step 5 --train
```

---

## Data Collection Guidelines

### Collecting Real EEG Data

**For Step 2 (Viewer Model)**:
1. Participants listen to music while wearing EEG headset
2. Record 5-second EEG segments + corresponding audio
3. After each clip, ask: "Did you enjoy this?" (yes/no)
4. Save EEG as `.npy`, audio as `.wav`, annotation as JSON

**For Step 3 (Article Reading)**:
1. Participants read your article while wearing EEG headset
2. Record continuous EEG throughout reading session
3. Save as single `.npy` file (shape: 8 × N samples)

**For Step 4 (Creator Model)**:
1. Wear EEG headset while writing/creating
2. Record continuous EEG + keystroke log with timestamps
3. After session, manually review and mark 5-second windows where "excellent" content was produced
4. Label as "creatable" or "not_creatable"

**For Step 5 (Processing Parameters)**:
1. Participants view images while wearing EEG headset
2. Record 5-second EEG for each image
3. Use `processing_features.py` to compute Processing parameters from each image
4. Save: EEG as `.npy`, image as `.png`, parameters in `metadata.json`

---

## Troubleshooting

### Common Issues

**"OSC not available"**:
```bash
pip install python-osc
```

**"No .hex files found"**:
- Check input directory path
- Ensure files have `.hex` extension
- Generate test data: `python scripts/generate_test_data.py`

**"Annotations file not found"**:
- Create annotations file following format in this guide
- Or generate test data with: `python scripts/generate_test_data.py --step 2`

**Processing sketch not receiving OSC**:
- Check OSC port matches (default: 12000)
- Ensure oscP5 library is installed in Processing
- Press 'O' in Processing to show OSC status
- Check firewall settings

**CUDA out of memory**:
- Reduce batch size in config
- Use CPU instead: `"device": "cpu"`

---

## Advanced Usage

### Custom EEG Hardware Integration

Replace the mock EEG supplier in Step 6:

```python
# Example: OpenBCI integration
import brainflow

def get_eeg_from_openbci():
    board = brainflow.BoardShim(...)
    board.prepare_session()
    board.start_stream()

    while True:
        # Read 5 seconds of data
        data = board.get_board_data()
        # Convert to (8, 3200) format
        eeg_segment = preprocess_data(data)
        yield eeg_segment
```

### Custom Parameter Mapping

Modify Processing parameters based on specific EEG features:

```python
def custom_parameter_mapping(eeg_segment):
    # Extract features
    alpha_power = compute_band_power(eeg_segment, freq_band='alpha')

    # Map to parameters
    params = {
        'colorCycleSpeed': alpha_power * 0.01,
        'noiseScale': 0.01 + (beta_power * 0.005),
        # ... custom mappings
    }
    return params
```

---

## Performance Optimization

### GPU Acceleration

```python
# Use mixed precision for faster training
model, history = train_viewer_model(
    dataset=dataset,
    epochs=50,
    use_amp=True,  # Automatic Mixed Precision
    device='cuda'
)
```

### Batch Processing

Process multiple files in parallel:

```python
from multiprocessing import Pool

def process_file(file_path):
    # Process single file
    pass

with Pool(8) as p:
    p.map(process_file, file_paths)
```

---

## Citation

If you use this workflow in your research, please cite:

```bibtex
@software{aesthetic_beacon,
  title={Aesthetic Beacon: EEG-Based Creative Feedback System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/aesthetic-beacon}
}
```

---

## Support

For issues and questions:
- Check the troubleshooting section above
- Review example code in this guide
- Check the codebase documentation
- Generate test data to verify your setup

---

## Next Steps

1. **Generate test data**: `python scripts/generate_test_data.py`
2. **Train models**: Run Steps 2, 4, 5 with `--train`
3. **Test real-time feedback**: Run Step 6 with Processing sketch
4. **Collect real data**: Follow data collection guidelines
5. **Refine models**: Re-train with your real EEG data

Good luck with your creative experiments! 🎨🧠
