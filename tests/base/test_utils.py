import pytest
import torch
import torch.nn as nn

from chameleon.base.utils import (has_children, initialize_weights_,
                                  replace_module, replace_module_attr_value)


def test_has_children():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )
    )
    assert has_children(model)


def test_replace_module():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
    )
    replace_module(model, nn.ReLU, nn.Sigmoid())
    assert model[1].__class__ == nn.Sigmoid


def test_replace_module_attr_value():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
    )
    replace_module_attr_value(model, nn.ReLU, 'inplace', True)
    assert model[1].inplace == True


def test_initialize_weights_():
    model = nn.Conv2d(3, 64, 3)
    model.weight.data.fill_(0)
    initialize_weights_(model, 'normal')
    for param in model.parameters():
        assert not torch.isnan(param).any()
