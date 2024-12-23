import torch
from torchmetrics.metric import Metric

from chameleon import ASPP, FPN, AdamW, AWingLoss, Conv2dBlock, GPUNet
from chameleon.registry import (BACKBONES, BLOCKS, COMPONENTS, LAYERS, METRICS,
                                NECKS, OPS, OPTIMIZERS)


def test_COMPONENTS():
    loss = COMPONENTS.build({'name': 'AWingLoss'})
    assert isinstance(loss, AWingLoss)


def test_OPTIMIZERS():
    model = BLOCKS.build({'name': 'Conv2dBlock', 'in_channels': 3,
                         'out_channels': 3, 'kernel': 3, 'stride': 1, 'padding': 1})
    optimizer = OPTIMIZERS.build({'name': 'AdamW', 'params': model.parameters(), 'lr': 1e-3})
    assert isinstance(optimizer, AdamW)


def test_BLOCKS():
    model = BLOCKS.build({'name': 'Conv2dBlock', 'in_channels': 3,
                         'out_channels': 3, 'kernel': 3, 'stride': 1, 'padding': 1})
    assert isinstance(model, Conv2dBlock)


def test_LAYERS():
    model = LAYERS.build({'name': 'ASPP', 'in_channels': 3, 'out_channels': 3})
    assert isinstance(model, ASPP)


def test_METRICS():
    model = METRICS.build({'name': 'NormalizedLevenshteinSimilarity'})
    assert isinstance(model, Metric)


def test_BACKBONES():
    model = BACKBONES.build({'name': 'GPUNet_0'})
    assert isinstance(model, GPUNet)


def test_NECKS():
    model = NECKS.build({'name': 'FPN', 'in_channels_list': [3, 3, 3], 'out_channels': 3})
    assert isinstance(model, FPN)


def test_OPS():
    func = OPS.build({'name': 'sinusoidal_positional_encoding_1d'})
    assert isinstance(func(10, 2), torch.Tensor)


def test_add_to_registry():

    @COMPONENTS.register_module()
    class Test:
        pass
    assert 'Test' in COMPONENTS
