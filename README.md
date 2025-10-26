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
- `src/processing_features.py` &ndash; Image analysis helpers that convert artwork texture
  and colour statistics into Processing-friendly parameter vectors using a lightweight
  convolutional neural network tailored to Processing controls.

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
  match the brainwave recording they are paired with. Supported formats include PNG, JPEG,
  and BMP. When the dataset class is instantiated it will automatically extract
  Processing-ready descriptors. Dominant palette values are obtained directly from RGB
  statistics, while texture-oriented controls such as gradient strength, orientation
  variance, `stroke_weight`, `pattern_scale`, and `noise_scale` are inferred by a fixed
  convolutional neural network that emulates classical edge and Gabor filter banks.

> **Tip:** the feature extraction step resizes images whose longest edge exceeds 512
> pixels to keep preprocessing fast while still capturing global texture cues. Adjust
> this by passing `resize_long_edge` to
> `processing_features.compute_processing_parameters` if you customise the dataset.

## Training

Install the dependencies:

```bash
pip install torch torchvision torchaudio pillow --index-url https://download.pytorch.org/whl/cu118
```

(Optional) create a virtual environment before installing packages.

Run the training script:

```bash
python -m src.train /path/to/dataset \
  --output-dir runs/exp01 \
  --batch-size 32 \
  --epochs 150 \
  --learning-rate 5e-4 \
  --crop-length 2560 \
  --expected-channels 8 \
  --expected-samples 3200 \
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

