import pytest

from chameleon import BACKBONES
from chameleon.tools import calculate_flops


def test_calcualte_flops():
    model = BACKBONES.build({'name': 'timm_resnet50'})
    flops, macs, params = calculate_flops(model, (1, 3, 224, 224))
    assert flops == '8.21 GFLOPS'
    assert macs == '4.09 GMACs'
    assert params == '25.56 M'
