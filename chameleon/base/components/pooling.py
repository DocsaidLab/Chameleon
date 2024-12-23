from typing import Union

import torch
from torch import nn
from torch.nn.modules.pooling import (AdaptiveAvgPool1d, AdaptiveAvgPool2d,
                                      AdaptiveAvgPool3d, AdaptiveMaxPool1d,
                                      AdaptiveMaxPool2d, AdaptiveMaxPool3d,
                                      AvgPool1d, AvgPool2d, AvgPool3d,
                                      MaxPool1d, MaxPool2d, MaxPool3d)

from ...registry import COMPONENTS

__all__ = [
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'MaxPool1d',
    'MaxPool2d', 'MaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d', 'GAP', 'GMP',
]


class GAP(nn.Module):
    """Global Average Pooling layer."""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global average pooling on the input tensor."""
        return self.pool(x)


class GMP(nn.Module):
    """Global Max Pooling layer."""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global max pooling on the input tensor."""
        return self.pool(x)


for k in __all__:
    COMPONENTS.register_module(name=k, module=globals()[k])
