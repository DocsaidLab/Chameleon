import pytest
import torch
import torch.nn as nn

from chameleon.base.blocks import Conv2dBlock, SeparableConv2dBlock


@pytest.fixture
def cnn_arch():
    return [
        {'in_channels': 3, 'out_channels': 32, 'kernel': 3},
        {'in_channels': 32, 'out_channels': 64, 'kernel': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel': 3},
    ]


@pytest.fixture
def fc_arch():
    return [
        {'in_channels': 3, 'out_channels': 32},
        {'in_channels': 32, 'out_channels': 64},
        {'in_channels': 64, 'out_channels': 128},
    ]


def test_SeparableConv2dBlock_forward():
    # Test input and output shapes
    in_channels = 64
    out_channels = 128
    block = SeparableConv2dBlock(in_channels, out_channels)
    x = torch.randn(1, in_channels, 64, 64)
    output = block(x)
    assert output.shape == (1, out_channels, 64, 64)

    # Test with different kernel size and padding
    kernel_size = (5, 3)
    padding = (1, 2)
    block = SeparableConv2dBlock(in_channels, out_channels, kernel=kernel_size, padding=padding)
    output = block(x)
    assert output.shape == (1, out_channels, 62, 66)

    # Test with different stride
    stride = 2
    block = SeparableConv2dBlock(in_channels, out_channels, stride=stride)
    output = block(x)
    assert output.shape == (1, out_channels, 32, 32)

    # Test with different output channels
    out_channels = 32
    block = SeparableConv2dBlock(in_channels, out_channels)
    output = block(x)
    assert output.shape == (1, out_channels, 64, 64)

    # Test without normalization and activation
    block = SeparableConv2dBlock(in_channels, out_channels, norm=None, act=None)
    output = block(x)
    assert output.shape == (1, out_channels, 64, 64)


def test_SeparableConv2dBlock_build_component():
    # Test build_component() function with different activation functions
    activation_fns = [
        {'name': 'ReLU'},
        {'name': 'Sigmoid'},
        {'name': 'Tanh'},
        {'name': 'LeakyReLU', 'negative_slope': 0.2}
    ]
    for act in activation_fns:
        block = SeparableConv2dBlock(64, 64, act=act)
        assert isinstance(block.act, nn.Module)


def test_SeparableConv2dBlock_build_component():
    # Test build_component() function with different normalization layers
    norm_layers = [
        {'name': 'BatchNorm2d', 'num_features': 64},
        {'name': 'InstanceNorm2d', 'num_features': 64},
        {'name': 'GroupNorm', 'num_groups': 8, 'num_channels': 64},
    ]
    tgt_norms = [
        nn.BatchNorm2d,
        nn.InstanceNorm2d,
        nn.GroupNorm
    ]
    for norm, tgt in zip(norm_layers, tgt_norms):
        block = SeparableConv2dBlock(64, 64, norm=norm)
        assert isinstance(block.block['norm'], tgt)


@pytest.fixture
def input_tensor():
    return torch.randn((2, 3, 32, 32))


@pytest.fixture
def output_shape():
    return (2, 16, 32, 32)


def test_Conv2dBlock_forward(input_tensor, output_shape):
    model = Conv2dBlock(in_channels=3, out_channels=16)
    output = model(input_tensor)
    assert output.shape == output_shape


def test_Conv2dBlock_with_activation(input_tensor, output_shape):
    model = Conv2dBlock(in_channels=3, out_channels=16, act={'name': 'ReLU', 'inplace': True})
    output = model(input_tensor)
    assert output.shape == output_shape
    assert torch.all(output >= 0)


def test_Conv2dBlock_with_batch_norm(input_tensor, output_shape):
    model = Conv2dBlock(in_channels=3, out_channels=16, norm={'name': 'BatchNorm2d', 'num_features': 16})
    output = model(input_tensor)
    assert output.shape == output_shape
    assert torch.allclose(output.mean(dim=(0, 2, 3)), torch.zeros(16), rtol=1e-3, atol=1e-5)
    assert torch.allclose(output.var(dim=(0, 2, 3)), torch.ones(16), rtol=1e-3, atol=1e-5)


def test_Conv2dBlock_init_type(input_tensor):
    model = Conv2dBlock(in_channels=3, out_channels=16, init_type='uniform')
    output1 = model(input_tensor)
    model = Conv2dBlock(in_channels=3, out_channels=16, init_type='normal')
    output2 = model(input_tensor)
    assert not torch.allclose(output1, output2, rtol=1e-3, atol=1e-5)


def test_Conv2dBlock_all_together(input_tensor):
    model = Conv2dBlock(in_channels=3, out_channels=16,
                        kernel=5, stride=2, padding=2, dilation=2, groups=1,
                        bias=True, padding_mode='reflect',
                        norm={'name': 'BatchNorm2d', 'num_features': 16, 'momentum': 0.5},
                        act={'name': 'LeakyReLU', 'negative_slope': 0.1, 'inplace': True},
                        init_type='uniform')
    output = model(input_tensor)
    assert output.shape == (2, 16, 14, 14)
