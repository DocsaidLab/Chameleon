import torch
import torch.nn as nn

from ...registry import BLOCKS, LAYERS
from ..power_module import PowerModule

__all__ = ['SELayer']


@LAYERS.register_module()
class SELayer(PowerModule):

    def __init__(self, in_channels: int, reduction: int = 4):
        """
        Initializes the Squeeze-and-Excitation (SE) layer.

        Args:
            in_channels (int): Number of input channels.
            reduction (int):
                Reduction ratio for the number of channels in the SE block.
                Default is 4, meaning the output will have 1/4 of the input channels.
        """
        super().__init__()

        # Compute the number of channels in the middle layer of the SE block.
        # If the number of input channels is less than reduction, set it to 1.
        mid_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = BLOCKS.build(
            {
                'name': 'Conv2dBlock',
                'in_channels': in_channels,
                'out_channels': mid_channels,
                'kernel': 1,
                'stride': 1,
                'padding': 0,
                'act': {'name': 'ReLU', 'inplace': True},
            }
        )
        self.fc2 = BLOCKS.build(
            {
                'name': 'Conv2dBlock',
                'in_channels': mid_channels,
                'out_channels': in_channels,
                'kernel': 1,
                'stride': 1,
                'padding': 0,
                'act': {'name': 'Sigmoid'},
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.fc2(y)
        return x * y
