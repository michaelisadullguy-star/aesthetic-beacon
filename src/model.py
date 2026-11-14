"""Neural networks of the pipeline and their self-describing checkpoints.

Three models share one temporal encoder:

* :class:`ViewerEnjoymentModel` — *to viewer*: was this window of a listener's
  or reader's EEG enjoyed?  Uses the accompanying audio when there is some and
  works on EEG alone when there is none (reading).
* :class:`CreatorStateModel` — *to creator*: is the creator in a "creatable"
  state right now?
* :class:`BrainwaveToProcessingModel` — maps a window of EEG to the Processing
  parameters of the display.

All of them take the ``(batch, 8, 3200)`` band matrix produced by
:mod:`src.eeg_features`.  A checkpoint stores the constructor arguments, the
feature configuration and the parameter codec next to the weights, so loading
never depends on defaults that may have changed since training.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .eeg_features import FeatureConfig
from .parameters import PARAMETER_NAMES, ParameterCodec

CHECKPOINT_VERSION = 2


def _group_norm(channels: int) -> nn.GroupNorm:
    """Normalisation that behaves identically for any batch size, including one."""

    return nn.GroupNorm(math.gcd(8, channels), channels)


class ResidualBlock(nn.Module):
    """Dilated residual block for 1-D sequences."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = _group_norm(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = _group_norm(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class TemporalFeatureEncoder(nn.Module):
    """Encode a band matrix into a fixed-size vector.

    Band envelopes change slowly compared with the 640 Hz they are sampled at,
    so the input is first averaged down by ``input_stride``.  This loses no
    information the bands carry and makes training and live inference several
    times cheaper.  The output concatenates average and maximum pooling over
    time, so both sustained levels and brief events are represented.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_blocks: int,
        input_stride: int = 8,
    ) -> None:
        super().__init__()
        if input_stride < 1:
            raise ValueError("input_stride must be at least 1")
        self.input_stride = int(input_stride)
        layers: List[nn.Module] = [
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            _group_norm(hidden_channels),
            nn.SiLU(),
        ]
        for block_index in range(num_blocks):
            layers.append(ResidualBlock(hidden_channels, kernel_size=5, dilation=2**block_index))
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_channels * 2

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("Expected EEG shaped (batch, channels, samples).")
        if self.input_stride > 1 and x.size(-1) >= self.input_stride:
            x = nn.functional.avg_pool1d(x, self.input_stride)
        x = self.network(x)
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


class AudioEncoder(nn.Module):
    """CNN that embeds log-mel spectrograms into a fixed-size vector."""

    def __init__(self, input_channels: int = 64, hidden_channels: int = 32) -> None:
        super().__init__()
        self.input_channels = input_channels
        half = max(8, hidden_channels // 2)
        self.network = nn.Sequential(
            nn.Conv2d(1, half, kernel_size=3, stride=2, padding=1),
            _group_norm(half),
            nn.SiLU(),
            nn.Conv2d(half, hidden_channels, kernel_size=3, stride=2, padding=1),
            _group_norm(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _group_norm(hidden_channels),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = hidden_channels

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # (batch, 1, mels, frames)
        x = self.network(x)
        return self.pool(x).flatten(1)


class _CalibratedClassifier(nn.Module):
    """Shared behaviour of the two binary classifiers."""

    temperature: Tensor

    def _init_calibration(self) -> None:
        self.register_buffer("temperature", torch.ones(()))

    def set_temperature(self, value: float) -> None:
        self.temperature.fill_(float(min(max(value, 1.0), 20.0)))


class ViewerEnjoymentModel(_CalibratedClassifier):
    """Predicts aesthetic enjoyment from EEG, optionally together with audio.

    During training the audio branch is replaced by a learned "no audio"
    embedding for a random share of the samples.  The network therefore learns
    to judge enjoyment from EEG alone as well, which is what Step 3 relies on
    when it scores the EEG of people reading an article.
    """

    def __init__(
        self,
        eeg_channels: int = 8,
        eeg_length: int = 3200,
        audio_mels: int = 64,
        audio_frames: int = 345,
        hidden_channels: int = 128,
        input_stride: int = 8,
        audio_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.eeg_channels = eeg_channels
        self.eeg_length = eeg_length
        self.audio_mels = audio_mels
        self.audio_frames = audio_frames
        self.hidden_channels = hidden_channels
        self.input_stride = input_stride
        self.audio_dropout = float(audio_dropout)

        self.eeg_encoder = TemporalFeatureEncoder(
            input_channels=eeg_channels,
            hidden_channels=hidden_channels,
            num_blocks=3,
            input_stride=input_stride,
        )
        self.audio_encoder = AudioEncoder(input_channels=audio_mels)
        self.no_audio = nn.Parameter(torch.zeros(self.audio_encoder.output_dim))
        fusion_dim = self.eeg_encoder.output_dim + self.audio_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )
        self.register_buffer("audio_mean", torch.zeros(()))
        self.register_buffer("audio_std", torch.ones(()))
        self._init_calibration()

    def config(self) -> Dict[str, Any]:
        return {
            "eeg_channels": self.eeg_channels,
            "eeg_length": self.eeg_length,
            "audio_mels": self.audio_mels,
            "audio_frames": self.audio_frames,
            "hidden_channels": self.hidden_channels,
            "input_stride": self.input_stride,
            "audio_dropout": self.audio_dropout,
        }

    def forward(self, eeg: Tensor, audio: Optional[Tensor] = None) -> Tensor:
        eeg_features = self.eeg_encoder(eeg)
        batch = eeg_features.size(0)
        missing = self.no_audio.unsqueeze(0).expand(batch, -1)
        if audio is None:
            audio_features = missing
        else:
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)
            audio_features = self.audio_encoder((audio - self.audio_mean) / self.audio_std)
            if self.training and self.audio_dropout > 0:
                drop = torch.rand(batch, 1, device=eeg_features.device) < self.audio_dropout
                audio_features = torch.where(drop, missing, audio_features)
        return self.classifier(torch.cat([eeg_features, audio_features], dim=1))

    @torch.no_grad()
    def predict_proba(self, eeg: Tensor, audio: Optional[Tensor] = None) -> Tensor:
        """Calibrated probability of enjoyment, shape ``(batch,)``."""

        return torch.sigmoid(self.forward(eeg, audio).squeeze(-1) / self.temperature)


class CreatorStateModel(_CalibratedClassifier):
    """Classifies whether a window of the creator's EEG is a "creatable" state."""

    def __init__(
        self,
        channels: int = 8,
        samples: int = 3200,
        hidden_channels: int = 128,
        input_stride: int = 8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.hidden_channels = hidden_channels
        self.input_stride = input_stride
        self.encoder = TemporalFeatureEncoder(
            input_channels=channels,
            hidden_channels=hidden_channels,
            num_blocks=4,
            input_stride=input_stride,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )
        self._init_calibration()

    def config(self) -> Dict[str, Any]:
        return {
            "channels": self.channels,
            "samples": self.samples,
            "hidden_channels": self.hidden_channels,
            "input_stride": self.input_stride,
        }

    def forward(self, eeg: Tensor) -> Tensor:
        return self.classifier(self.encoder(eeg))

    @torch.no_grad()
    def predict_proba(self, eeg: Tensor) -> Tensor:
        """Calibrated probability of the creatable state, shape ``(batch,)``."""

        return torch.sigmoid(self.forward(eeg).squeeze(-1) / self.temperature)


class BrainwaveToProcessingModel(nn.Module):
    """Regression model predicting Processing parameters from brainwave segments.

    ``forward`` returns the encoded parameter vector (see
    :class:`~src.parameters.ParameterCodec`).  Use :meth:`predict_parameters`
    to obtain named, range-checked parameter values.
    """

    def __init__(
        self,
        input_channels: int,
        parameter_dim: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
        input_stride: int = 8,
        codec: Optional[ParameterCodec] = None,
    ) -> None:
        super().__init__()
        if codec is not None and codec.dim != parameter_dim:
            raise ValueError(
                f"parameter_dim is {parameter_dim} but the codec encodes {codec.dim} values."
            )
        self.input_channels = input_channels
        self.parameter_dim = parameter_dim
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.input_stride = input_stride
        self.codec = codec
        self.encoder = TemporalFeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            input_stride=input_stride,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, parameter_dim),
        )

    def config(self) -> Dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "parameter_dim": self.parameter_dim,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout,
            "input_stride": self.input_stride,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def predict_parameters(self, x: Tensor) -> List[Dict[str, float]]:
        """Named parameter values for each window in the batch, clamped to safe ranges."""

        codec = self.codec or ParameterCodec()
        outputs = self.forward(x).detach().cpu().numpy()
        return [codec.decode(row) for row in outputs]


def create_model(
    input_shape: Tuple[int, int],
    parameter_dim: Optional[int] = None,
    hidden_channels: int = 128,
    num_blocks: int = 4,
    dropout: float = 0.1,
    input_stride: int = 8,
    codec: Optional[ParameterCodec] = None,
) -> BrainwaveToProcessingModel:
    """Convenience factory for the EEG → Processing parameter model."""

    channels, _ = input_shape
    if codec is None and parameter_dim is None:
        codec = ParameterCodec()
    if parameter_dim is None:
        assert codec is not None
        parameter_dim = codec.dim
    return BrainwaveToProcessingModel(
        input_channels=channels,
        parameter_dim=parameter_dim,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        input_stride=input_stride,
        codec=codec,
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

_MODEL_KINDS = {
    "viewer": ViewerEnjoymentModel,
    "creator": CreatorStateModel,
    "parameter": BrainwaveToProcessingModel,
}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, damaged or of the wrong kind."""


def _plain(value: Any) -> Any:
    """Reduce ``value`` to types that load with ``torch.load(weights_only=True)``."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, type(None))):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Tensor):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_checkpoint(
    path: Path | str,
    *,
    kind: str,
    model: nn.Module,
    feature_config: FeatureConfig,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a checkpoint atomically (a crash never leaves a half-written file)."""

    if kind not in _MODEL_KINDS:
        raise ValueError(f"Unknown model kind '{kind}'.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "kind": kind,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": _plain(model.config()),  # type: ignore[operator]
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_config": _plain(feature_config.to_dict()),
    }
    if kind == "parameter":
        codec = getattr(model, "codec", None) or ParameterCodec()
        payload["codec"] = _plain(codec.to_dict())
        payload["parameter_names"] = list(codec.names)
    if extra:
        payload.update(_plain(extra))
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: Path | str,
    *,
    kind: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a model and its metadata; the model is returned in evaluation mode."""

    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Model file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"Could not read model file {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"{path} was not written by this version of the pipeline. Re-run the training step."
        )
    found = payload.get("kind")
    if kind is not None and found != kind:
        raise CheckpointError(f"{path} holds a '{found}' model but a '{kind}' model is required.")
    if found not in _MODEL_KINDS:
        raise CheckpointError(f"{path} has unknown model kind '{found}'.")

    config = dict(payload["model_config"])
    if found == "parameter":
        codec = ParameterCodec.from_dict(payload.get("codec"))
        model: nn.Module = BrainwaveToProcessingModel(codec=codec, **config)
    else:
        model = _MODEL_KINDS[found](**config)
    try:
        model.load_state_dict(payload["model_state"])
    except Exception as error:
        raise CheckpointError(f"{path} does not match its recorded architecture: {error}") from error
    model.to(device).eval()
    return model, payload


def checkpoint_feature_config(payload: Mapping[str, Any]) -> FeatureConfig:
    """Feature configuration a loaded model was trained with."""

    return FeatureConfig.from_dict(payload.get("feature_config"))


__all__ = [
    "AudioEncoder",
    "BrainwaveToProcessingModel",
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "CreatorStateModel",
    "PARAMETER_NAMES",
    "ResidualBlock",
    "TemporalFeatureEncoder",
    "ViewerEnjoymentModel",
    "checkpoint_feature_config",
    "create_model",
    "load_checkpoint",
    "save_checkpoint",
]
