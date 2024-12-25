from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...registry import BLOCKS, COMPONENTS
from ..power_module import PowerModule


@BLOCKS.register_module()
class SeparableConv2dBlock(PowerModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        inner_norm: Optional[dict] = None,
        inner_act: Optional[dict] = None,
        norm: Optional[dict] = None,
        act: Optional[dict] = None,
        init_type: str = 'normal',
    ):
        """
        A separable convolution block consisting of a depthwise convolution and a pointwise convolution.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int, optional):
                Number of output channels. If not provided, defaults to `in_channels`.
            kernel (int or Tuple[int, int], optional):
                Size of the convolution kernel. Defaults to 3.
            stride (int or Tuple[int, int], optional):
                Stride of the convolution. Defaults to 1.
            padding (int or Tuple[int, int], optional):
                Padding added to all four sides of the input. Defaults to 1.
            bias (bool):
                Whether to include a bias term in the convolutional layer.
                Noted: if normalization layer is not None, bias will always be set to False.
                Defaults to False.
            inner_norm (dict, optional):
                Configuration of normalization layer between dw and pw layer. Defaults to None.
            inner_act (dict, optional):
                Configuration of activation layer between dw and pw layer. Defaults to None.
            norm (dict, optional):
                Configuration of normalization layer after pw layer. Defaults to None.
            act (dict, optional):
                Configuration of activation layer after pw layer. Defaults to None.
            init_type (str, optional):
                Initialization method for the model parameters. Defaults to 'normal'.
        """
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        bias = False if norm is not None else bias

        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        if inner_norm is not None:
            self.inner_norm = COMPONENTS.build(inner_norm) if isinstance(inner_norm, dict) else inner_norm
        if inner_act is not None:
            self.inner_act = COMPONENTS.build(inner_act) if isinstance(inner_act, dict) else inner_act
        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        if norm is not None:
            self.norm = COMPONENTS.build(norm) if isinstance(norm, dict) else norm
        if act is not None:
            self.act = COMPONENTS.build(act) if isinstance(act, dict) else act
        self.initialize_weights_(init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.inner_norm(x) if hasattr(self, 'inner_norm') else x
        x = self.inner_act(x) if hasattr(self, 'inner_act') else x
        x = self.pw_conv(x)
        x = self.norm(x) if hasattr(self, 'norm') else x
        x = self.act(x) if hasattr(self, 'act') else x
        return x


@BLOCKS.register_module()
class Conv2dBlock(PowerModule):
    def __init__(
        self,
        in_channels: Union[float, int],
        out_channels: Union[float, int],
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',
        norm: Optional[dict] = None,
        act: Optional[dict] = None,
        init_type: str = 'normal',
    ):
        """
        This class is used to build a 2D convolutional neural network cell.

        Args:
            in_channels (int or float):
                Number of input channels.
            out_channels (int or float):
                Number of output channels.
            kernel (int or tuple, optional):
                Size of the convolutional kernel. Defaults to 3.
            stride (int or tuple, optional):
                Stride size. Defaults to 1.
            padding (int or tuple, optional):
                Padding size. Defaults to 1.
            dilation (int, optional):
                Spacing between kernel elements. Defaults to 1.
            groups (int, optional):
                Number of blocked connections from input channels to output
                channels. Defaults to 1.
            bias (bool, optional):
                Whether to include a bias term in the convolutional layer.
                If bias = None, bias would be set as Ture when normalization layer is None and
                False when normalization layer is not None.
                Defaults to None.
            padding_mode (str, optional):
                Options = {'zeros', 'reflect', 'replicate', 'circular'}.
                Defaults to 'zeros'.
            norm (Optional[dict], optional):
                normalization layer or a dictionary of arguments for building a
                normalization layer. Default to None.
            act (Optional[dict], optional):
                Activation function or a dictionary of arguments for building an
                activation function. Default to None.
            pool (Optional[dict], optional):
                pooling layer or a dictionary of arguments for building a pooling
                layer. Default to None.
            init_type (str):
                Method for initializing model parameters. Default to 'normal'.
                Options = {'normal', 'uniform'}

        Examples for using norm, act, and pool:
            1. conv_block = Conv2dBlock(
                    in_channels=3,
                    out_channels=12,
                    norm=nn.BatchNorm2d(12),
                    act=nn.ReLU(),
                    pool=nn.AdaptiveAvgPool2d(1)
                )
            2. conv_block = Conv2dBlock(
                    in_channels=3,
                    out_channels=12,
                    norm={'name': 'BatchNorm2d', 'num_features': 12},
                    act={'name': 'ReLU', 'inplace': True},
                )

        Attributes:
            block (nn.Module): a model block.
        """
        super().__init__()
        bias = False if norm is not None else bias

        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        if norm is not None:
            self.norm = COMPONENTS.build(norm) if isinstance(norm, dict) else norm
        if act is not None:
            self.act = COMPONENTS.build(act) if isinstance(act, dict) else act

        self.initialize_weights_(init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x) if hasattr(self, 'norm') else x
        x = self.act(x) if hasattr(self, 'act') else x
        return x
