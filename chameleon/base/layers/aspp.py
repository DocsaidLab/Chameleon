
import torch
import torch.nn as nn

from ...registry import BLOCKS, COMPONENTS, LAYERS
from ..power_module import PowerModule

__all__ = ['ASPP']

__doc__ = """
    REFERENCES: DeepLab: Semantic Image Segmentation with Deep Convolutional
                Nets, Atrous Convolution, and Fully Connected CRFs
    URL: https://arxiv.org/pdf/1606.00915.pdf
"""


@LAYERS.register_module()
class ASPP(PowerModule):

    ARCHS = {
        # ksize, stride, padding, dilation, is_use_hs
        'dilate_layer1': [3, 1, 1, 1, True],
        'dilate_layer2': [3, 1, 2, 2, True],
        'dilate_layer3': [3, 1, 4, 4, True],
        'dilate_layer4': [3, 1, 8, 8, True],
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_act: dict = {'name': 'ReLU', 'inplace': True},
    ):
        """
        Constructor for the ASPP class.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            out_act (dict, optional):
                Activation function for the output layer. Defaults to ReLU.
        """
        super().__init__()
        for name, cfg in self.ARCHS.items():
            ksize, stride, padding, dilation, use_hs = cfg
            layer = BLOCKS.build(
                {
                    'name': 'Conv2dBlock',
                    'in_channels': in_channels,
                    'out_channels': in_channels,
                    'kernel': ksize,
                    'stride': stride,
                    'padding': padding,
                    'dilation': dilation,
                    'norm': COMPONENTS.build({'name': 'BatchNorm2d', 'num_features': in_channels}),
                    'act': COMPONENTS.build({'name': 'Hswish' if use_hs else 'ReLU'}),
                }
            )
            self.add_module(name, layer)

        self.output_layer = BLOCKS.build(
            {
                'name': 'Conv2dBlock',
                'in_channels': in_channels * len(self.ARCHS),
                'out_channels': out_channels,
                'kernel': 1,
                'stride': 1,
                'padding': 0,
                'norm': COMPONENTS.build({'name': 'BatchNorm2d', 'num_features': out_channels}),
                'act':  COMPONENTS.build(out_act),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [getattr(self, name)(x) for name in self.ARCHS.keys()]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.output_layer(outputs)
        return outputs
