"""Neural network models for brainwave-to-Processing parameter regression."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A simple residual block for 1D convolution."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return self.act(x)


class TemporalFeatureEncoder(nn.Module):
    """Encode brainwave time-series into a fixed-size vector."""

    def __init__(self, input_channels: int, hidden_channels: int, num_blocks: int) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
        ]

        for block_index in range(num_blocks):
            dilation = 2 ** block_index
            layers.append(ResidualBlock(hidden_channels, kernel_size=5, dilation=dilation))

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.pool(x)
        return x.squeeze(-1)


class BrainwaveToProcessingModel(nn.Module):
    """Regression model predicting Processing parameters from brainwave segments."""

    def __init__(
        self,
        input_channels: int,
        parameter_dim: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TemporalFeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, parameter_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape: (batch, channels, samples)
        features = self.encoder(x)
        return self.head(features)


def create_model(
    input_shape: Tuple[int, int],
    parameter_dim: int,
    hidden_channels: int = 128,
    num_blocks: int = 4,
    dropout: float = 0.1,
) -> BrainwaveToProcessingModel:
    """Convenience factory for instantiating the model."""

    channels, _ = input_shape
    return BrainwaveToProcessingModel(
        input_channels=channels,
        parameter_dim=parameter_dim,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dropout=dropout,
    )

