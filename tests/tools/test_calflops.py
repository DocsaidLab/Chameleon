import pytest

from chameleon.modules import build_backbone
from chameleon.tools import calculate_flops


def test_calcualte_flops():
    model = build_backbone('resnet50')
    flops, macs, params = calculate_flops(model, (1, 3, 224, 224))
    assert flops == '8.21 GFLOPS'
    assert macs == '4.09 GMACs'
    assert params == '25.56 M'
