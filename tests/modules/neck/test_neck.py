import pytest
import torch

from chameleon import build_neck

INPUT1 = [
    torch.rand(1, 16, 80, 80),
    torch.rand(1, 32, 40, 40),
    torch.rand(1, 64, 20, 20),
]

data = [
    (
        INPUT1,
        {'name': 'FPN', 'in_channels_list': [16, 32, 64], 'out_channels': 24},
        {'out_shapes': [
            torch.Size((1, 24, 80, 80)),
            torch.Size((1, 24, 40, 40)),
            torch.Size((1, 24, 20, 20)),
        ]}
    ),
    (
        INPUT1,
        {'name': 'BiFPN', 'in_channels_list': [16, 32, 64], 'out_channels': 24, 'extra_layers': 2},
        {'out_shapes': [
            torch.Size((1, 24, 80, 80)),
            torch.Size((1, 24, 40, 40)),
            torch.Size((1, 24, 20, 20)),
            torch.Size((1, 24, 10, 10)),
            torch.Size((1, 24, 5, 5)),
        ]}
    ),
    (
        INPUT1,
        {'name': 'BiFPN', 'in_channels_list': [16, 32, 64], 'out_channels': 24, 'out_indices': [0, 1, 2]},
        {'out_shapes': [
            torch.Size((1, 24, 80, 80)),
            torch.Size((1, 24, 40, 40)),
            torch.Size((1, 24, 20, 20)),
        ]}
    ),
]


@ pytest.mark.parametrize('in_tensor,build_kwargs,expected', data)
def test_build_backbone(in_tensor, build_kwargs, expected):
    model = build_neck(**build_kwargs)
    outs = model(in_tensor)
    if isinstance(outs, (list, tuple)):
        out_shapes = [x.shape for x in outs]
    else:
        out_shapes = outs.shape
    assert out_shapes == expected['out_shapes']
