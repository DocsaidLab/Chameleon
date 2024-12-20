# from typing import List, Optional, Union

# import torch.nn as nn

# from ..nn import CNN2Dcell, SeparableConvBlock
# from .fpn import FPN


# class PAFPN(FPN):

#     def __init__(
#         self,
#         in_channels_list: List[int],
#         out_channels: int,
#         extra_layers: int = 0,
#         out_indices: Optional[List[int]] = None,
#         norm: Optional[Union[dict, nn.Module]] = None,
#         act: Optional[Union[dict, nn.Module]] = None,
#         upsample_mode: str = 'bilinear',
#         use_dwconv: bool = False,
#     ):
#         """
#         Feature Pyramid Network (FPN) module.

#         Args:
#             in_channels_list (List[int]):
#                 A list of integers representing the number of channels in each
#                 input feature map.
#             out_channels (int):
#                 The number of output channels for all feature maps.
#             extra_layers (int, optional):
#                 The number of extra down-sampling layers to add. Defaults to 0.
#             out_indices (Optional[List[int]], optional):
#                 A list of integers indicating the indices of the feature maps to
#                 output. If None, all feature maps are output. Defaults to None.
#             norm Optional[Union[dict, nn.Module]]:
#                 Optional normalization module or dictionary of its parameters.
#                 Defaults to None.
#             act Optional[Union[dict, nn.Module]]:
#                 Optional activation function or dictionary of its parameters.
#                 Defaults to None.
#             upsample_mode (str, optional):
#                 The type of upsampling method to use, which can be 'bilinear' or
#                 'nearest'. Bilinear upsampling is recommended in most cases for
#                 its better performance. Nearest neighbor upsampling may be useful
#                 when input feature maps have a small spatial resolution.
#                 Defaults to 'bilinear'.
#             use_dwconv (bool, optional):
#                 Whether to use depth-wise convolution in each Conv2d block.
#                 Depth-wise convolution can reduce the number of parameters and
#                 improve computation efficiency. However, it may also degrade the
#                 quality of feature maps due to its low capacity.
#                 Defaults to False.

#         Raises:
#             ValueError: If the number of input feature maps does not match the length of `in_channels_list`.
#                 Or if `extra_layers` is negative.
#         """
#         super().__init__(
#             in_channels_list=in_channels_list,
#             out_channels=out_channels,
#             extra_layers=extra_layers,
#             out_indices=out_indices,
#             norm=norm,
#             act=act,
#             upsample_mode=upsample_mode,
#             use_dwconv=use_dwconv,
#         )
#         conv2d = SeparableConvBlock if use_dwconv else CNN2Dcell

#         self.downsample_convs = nn.ModuleList()
#         self.pafpn_convs = nn.ModuleList()
#         for i in range(self.start_level + 1, self.backbone_end_level):
#             d_conv = conv2d(
#                 out_channels,
#                 out_channels,
#                 kernel=3,
#                 stride=2,
#                 padding=1,
#                 norm=deepcopy(norm),
#                 act=deepcopy(act),
#             )
#             pafpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False,
#             )
#             self.downsample_convs.append(d_conv)
#             self.pafpn_convs.append(pafpn_conv)
