# Data Directory Structure

This directory contains all data for the 6-step aesthetic feedback workflow.

## Directory Organization

### `step1_hex_eeg/`
Raw hexadecimal EEG dumps before processing.
- **Format**: Plain text hex files (e.g., `session_001.hex`)
- **Content**: Hexadecimal representation of EEG data (8 channels × 640 Hz)
- **Usage**: Input for `decode_hex_eeg_segment()` in pipeline Step 1

### `step2_viewer_training/`
Training data for the "to viewer" model (aesthetic enjoyment prediction).

- **`eeg/`**: 5-second EEG segments during music listening (`.npy` files, shape 8×3200)
- **`audio/`**: Corresponding 5-second audio clips (`.wav` files, 44.1 kHz recommended)
- **`annotations/`**: JSON files mapping samples to enjoyment labels
  ```json
  {
    "samples": [
      {"eeg_file": "eeg/sample_001.npy", "audio_file": "audio/sample_001.wav", "enjoyed": true},
      {"eeg_file": "eeg/sample_002.npy", "audio_file": "audio/sample_002.wav", "enjoyed": false}
    ]
  }
  ```

### `step3_article_reading/`
Data from participants reading articles.

- **`eeg/`**: Full EEG recordings during article reading (`.npy` files, shape 8×N)
- **`results/`**: Output JSON files with enjoyment scores per time window
  ```json
  {
    "session_id": "participant_001_article_reading",
    "windows": [
      {"start_time": 0.0, "end_time": 5.0, "enjoyment_score": 0.87},
      {"start_time": 1.0, "end_time": 6.0, "enjoyment_score": 0.92}
    ]
  }
  ```

### `step4_creator_training/`
Training data for the "to creator" model (creatable state detection).

- **`eeg/`**: Creator's EEG during writing/creating (`.npy` files, shape 8×N)
- **`annotations/`**: JSON files marking "excellent" passages
  ```json
  {
    "session_id": "creator_session_001",
    "keystroke_log": "creator_session_001_keystrokes.json",
    "excellent_passages": [
      {
        "start_time": 120.5,
        "end_time": 125.5,
        "content": "The sunset painted the sky in hues of amber and rose...",
        "label": "creatable"
      },
      {
        "start_time": 300.0,
        "end_time": 305.0,
        "content": "A mundane passage...",
        "label": "not_creatable"
      }
    ]
  }
  ```

### `step5_image_training/`
Training data for EEG → Processing parameters mapping.

- **`brainwaves/`**: EEG responses to viewing images (`.npy` files, shape 8×3200)
- **`artworks/`**: Corresponding images that participants viewed (`.png`, `.jpg`)
- **`metadata.json`**: Processing parameters for each artwork
  ```json
  {
    "artwork_001.png": {
      "canvasWidth": 800,
      "canvasHeight": 800,
      "noiseScale": 0.015,
      "colorCycleSpeed": 0.002,
      ...
    }
  }
  ```

**Note**: Filenames must match: `brainwaves/sample_001.npy` ↔ `artworks/sample_001.png`

### `step6_realtime/`
Real-time feedback loop resources.

- **`models/`**: Trained PyTorch model checkpoints
  - `creator_model.pt` (Step 4 output)
  - `parameter_model.pt` (Step 5 output)
  - `viewer_model.pt` (Step 2 output, optional)
- **`outputs/`**: Generated parameter files for Processing
  - `current_parameters.json` (latest parameters)
  - `parameter_history/` (timestamped snapshots)

## Data Format Specifications

### EEG Data (.npy files)
- **Shape**: `(8, N)` where N = samples
  - For 5-second segments: N = 3200 (at 640 Hz)
  - For longer recordings: N = arbitrary length
- **Data type**: `float32` or `float64`
- **Range**: Normalized to approximately [-1, 1]
- **Channels**: 8 EEG frequency bands

### Audio Data (.wav files)
- **Sample rate**: 44,100 Hz (or 22,050 Hz)
- **Channels**: Mono or stereo
- **Duration**: 5 seconds (for training pairs)
- **Format**: Uncompressed WAV

### Hexadecimal EEG
- **Encoding**: Little-endian int16 by default
- **Structure**: Continuous hex stream (e.g., `a3f201cd...`)
- **Channels**: 8 channels interleaved

## Example Data Generation

To generate synthetic test data:

```bash
python scripts/generate_test_data.py --output data/
```

This will create sample files in the appropriate directories for testing the pipeline.
