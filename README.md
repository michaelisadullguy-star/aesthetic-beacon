# Aesthetic Beacon

This project provides a PyTorch-based training pipeline for mapping five-second brainwave
segments to Processing sketch parameters that capture artwork texture and colour
characteristics.

## Project structure

- `src/brainwave_dataset.py` &ndash; Dataset utilities for loading EEG recordings stored as
  8&times;3200 tensors (`.npy` files) and deriving Processing parameters from paired
  artwork images.
- `src/model.py` &ndash; Neural network modules that encode the temporal signals and regress
  Processing-ready descriptors.
- `src/train.py` &ndash; Command-line training entry point that ties the dataset and model
  together.
- `src/processing_features.py` &ndash; Image analysis helpers plus the
  `ProcessingParameterPredictor`, a trainable convolutional model that maps
  artworks to Processing-friendly parameter vectors.
- `src/train_processing_features.py` &ndash; Command-line utility used to train the
  artwork-to-Processing predictor on curated image/parameter pairs.

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

## Processing sketch template

The repository also includes a ready-to-run Processing template at
`processing/DynamicTextureTemplate.pde`. The sketch renders a layered, animated texture
with continuously shifting colours and relies on the helper function `loadParamsJSON()`
to load its configuration from `processing/data/params.json` without recompilation.

### Parameter groups

The JSON schema is designed so you can tweak a broad range of stylistic controls:

| Group | Keys | Description |
| --- | --- | --- |
| `canvas` | `width`, `height`, `smooth`, `renderer` | Sets sketch dimensions, smoothing level, and renderer (`P2D` or default). |
| `animation` | `timeScale`, `noiseSpeed`, `driftSpeed` | Adjusts playback speed and the motion of the noise-driven layers. |
| `texture` | `noiseScale`, `turbulence`, `lineFrequency`, `lineCurvature`, `contrast`, `brightness`, `vignette`, `grain` | Controls multi-octave noise sampling, linear wave interference, brightness, vignette strength, and film-like grain. |
| `background` | `hue`, `saturation`, `brightness`, `alpha` | Defines the base colour wash behind the animated layers. |
| `overlay` | `hueShift`, `alpha` | Adds a dynamic overlay tint that subtly shifts hue over time. |
| `palette` | array of objects with `h`, `s`, `b`, `a` (plus optional `name`) | Declares the HSB palette used to blend multiple colour bands across the texture. |

Press `r` while the sketch is running to reload the JSON file and experiment with new
looks instantly. An example configuration is provided in `processing/data/params.json` to
illustrate how the parameters combine to produce rich, non-repetitive colour dynamics.

## Post-merge sanity check

To verify that no merge conflict markers remain after integrating branches, run the
helper script:

```bash
python scripts/check_merge_conflicts.py
```

The command scans all tracked files and reports any unresolved Git conflict markers so
they can be addressed before committing.

