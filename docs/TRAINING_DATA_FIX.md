# Training Data Mismatch Fix

## Problem

The original training methodology had a fundamental **static-to-dynamic mismatch**:

### Original Flow (Problematic)
```
Static Training Images → CNN Model (dynamic inference) → Parameter Predictions → EEG Training Targets
```

**Issues:**
1. **No Ground Truth**: Training targets were CNN predictions, not actual parameter values
2. **Dynamic Generation**: Same static image could produce different parameters depending on CNN state
3. **Error Propagation**: CNN Stage 1 errors cascaded into EEG Stage 2 training
4. **Misaligned Learning**: EEG model learned to reproduce CNN feature extraction, not EEG-to-parameter mapping

### Code Location of Original Issue

In `src/brainwave_dataset.py` (lines 142-147), parameters were generated dynamically:

```python
parameter_dict = compute_processing_parameters(
    image_path,
    weights_path=self.processing_weights,  # CNN checkpoint
    device=self.processing_device,
)
```

This meant the EEG model was training to match CNN predictions, not ground truth parameters.

## Solution

### New Flow (Fixed)
```
Static Training Images + Ground Truth Parameters (from metadata.json) → EEG Training Targets
```

**Benefits:**
1. ✅ **Static-to-Static Mapping**: Fixed images map to fixed parameters
2. ✅ **Ground Truth Training**: EEG model learns actual EEG-to-parameter relationships
3. ✅ **No Error Propagation**: No dependency on CNN accuracy
4. ✅ **Backward Compatible**: Falls back to CNN if metadata not provided

## Usage

### Ground Truth Training (Recommended)

#### 1. Create Parameter Metadata

Create a `metadata.json` file with ground truth parameter annotations for your artworks:

**Format Option 1 - Dictionary:**
```json
{
  "sample_001": {
    "canvasWidth": 800,
    "canvasHeight": 600,
    "cellSize": 32,
    "noiseScale": 0.01,
    "noiseSpeed": 1.5,
    "colorCycleSpeed": 0.5,
    "backgroundHue": 200,
    "backgroundSaturation": 50,
    "backgroundBrightness": 80,
    ...
  },
  "sample_002": {
    "canvasWidth": 1024,
    "canvasHeight": 768,
    ...
  }
}
```

**Format Option 2 - Array:**
```json
[
  {
    "id": "sample_001",
    "parameters": {
      "canvasWidth": 800,
      "canvasHeight": 600,
      "cellSize": 32,
      ...
    }
  },
  {
    "id": "sample_002",
    "parameters": {
      "canvasWidth": 1024,
      ...
    }
  }
]
```

#### 2. Dataset Directory Structure

```
dataset_root/
├── brainwaves/
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
├── artworks/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── metadata.json  # Ground truth parameters
```

#### 3. Train with Ground Truth

```bash
python -m src.train \
    path/to/dataset \
    --parameter-metadata path/to/dataset/metadata.json \
    --output-dir outputs/ground_truth_training \
    --batch-size 16 \
    --epochs 100 \
    --learning-rate 1e-3
```

**Note:** When using `--parameter-metadata`, the `--processing-weights` argument is optional.

### CNN-Based Training (Legacy)

For backward compatibility, you can still use CNN-generated parameters:

```bash
python -m src.train \
    path/to/dataset \
    --processing-weights processing_features.pt \
    --output-dir outputs/cnn_based_training \
    --batch-size 16 \
    --epochs 100
```

## Parameter Validation

All parameters (whether from ground truth or CNN) are validated against constraints defined in `src/processing_features.py`:

- **Range Constraints**: Parameters are clamped to valid ranges (e.g., `canvasWidth: 64-4096`)
- **Integer Parameters**: Automatically rounded (e.g., `cellSize`, `noiseOctaves`)
- **Complete List**: See `_RANGE_CONSTRAINTS` and `_INT_PARAMETER_NAMES` in `src/processing_features.py`

## Migration Guide

### From CNN-Based to Ground Truth Training

1. **Extract Current CNN Predictions** (optional, for reference):
   ```python
   from src.processing_features import compute_processing_parameters
   import json

   metadata = {}
   for sample_id in sample_ids:
       image_path = f"artworks/{sample_id}.png"
       params = compute_processing_parameters(
           image_path,
           weights_path="processing_features.pt"
       )
       metadata[sample_id] = dict(params)

   with open("metadata.json", "w") as f:
       json.dump(metadata, f, indent=2)
   ```

2. **Review and Adjust Parameters**: Manually review the generated parameters and adjust to match desired aesthetic outcomes

3. **Retrain with Ground Truth**: Use the new metadata file as shown above

## Implementation Details

### Changes to `src/brainwave_dataset.py`

- Added `parameter_metadata` parameter to `BrainwaveProcessingDataset.__init__()`
- Added ground truth parameter loading logic
- Made `processing_weights` optional when `parameter_metadata` is provided
- Added logging to indicate which parameter source is being used

### Changes to `src/train.py`

- Added `--parameter-metadata` command-line argument
- Made `--processing-weights` optional
- Added validation to ensure at least one parameter source is provided
- Updated dataset instantiation to pass metadata path

### No Changes to `src/processing_features.py`

**Important:** As requested, no changes were made to preset parameters, constraints, or parameter definitions in `src/processing_features.py`. All existing parameter validation logic remains intact.

## Verification

After loading the dataset, you'll see a confirmation message:

**Ground Truth Mode:**
```
Loaded 150 samples using ground truth parameters from metadata.
```

**CNN Mode:**
```
Loaded 150 samples using CNN-generated parameters.
```

## Recommended Workflow

1. **Initial Training**: Use CNN-based approach to generate initial parameter predictions
2. **Export & Review**: Export CNN predictions as metadata.json
3. **Manual Curation**: Adjust parameters based on aesthetic judgment and user testing
4. **Ground Truth Training**: Retrain EEG model using curated ground truth parameters
5. **Iterate**: Refine parameters based on live testing and user feedback

This approach ensures the EEG model learns meaningful relationships between brain activity and desired visual parameters, rather than simply reproducing CNN feature extraction.
