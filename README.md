# Aesthetic Beacon

This project provides a PyTorch-based training pipeline for mapping five-second brainwave
segments to Processing sketch parameters that capture artwork texture and colour
characteristics.

## Project structure

- `src/brainwave_dataset.py` &ndash; Dataset utilities for loading EEG recordings stored as
  8&times;3200 tensors (`.npy` files) alongside Processing parameter metadata.
- `src/model.py` &ndash; Neural network modules that encode the temporal signals and regress
  Processing-ready descriptors.
- `src/train.py` &ndash; Command-line training entry point that ties the dataset and model
  together.

## Preparing the dataset

Organise the data with the following layout:

```
dataset_root/
├── metadata.json
└── brainwaves/
    ├── sample_0001.npy
    ├── sample_0002.npy
    └── ...
```

- Each `brainwaves/*.npy` file must contain an `8 × T` floating-point array, where `T`
  is the number of samples captured at 640 Hz (3200 for a five-second recording).
- `metadata.json` is a dictionary mapping each sample ID to a JSON object of numeric
  Processing parameters. The keys define the ordering of the regression targets.

Example `metadata.json` snippet:

```json
{
  "sample_0001": {
    "stroke_weight": 2.5,
    "color_hue": 180,
    "noise_scale": 0.8
  },
  "sample_0002": {
    "stroke_weight": 1.7,
    "color_hue": 240,
    "noise_scale": 0.5
  }
}
```

## Training

Install the dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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

