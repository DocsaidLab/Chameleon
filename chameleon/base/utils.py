from typing import Any, Union

import torch.nn as nn

from ..registry import COMPONENTS

__all__ = ['has_children', 'replace_module', 'replace_module_attr_value', 'initialize_weights_']


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def replace_module(
    model: nn.Module,
    target: Union[type, str],
    dst_module: Union[nn.Module, dict]
) -> None:
    """
    Function to replace modules.

    Args:
        model (nn.Module):
            NN module.
        target (Union[type, str]):
            The type of module you want to replace.
        dst_module (Union[nn.Module, dict]):
            The module you want to use after replacement.
    """
    if not isinstance(dst_module, (nn.Module, dict)):
        raise ValueError(f'dst_module = {dst_module} should be an instance of Module or dict.')

    target = COMPONENTS.get(target) if isinstance(target, str) else target
    dst_module = COMPONENTS.build(dst_module) if isinstance(dst_module, dict) else dst_module

    for name, m in model.named_children():
        if has_children(m):
            replace_module(m, target, dst_module)
        else:
            if isinstance(m, target):
                setattr(model, name, dst_module)


def replace_module_attr_value(
    model: nn.Module,
    target: Union[type, str],
    attr_name: str,
    attr_value: Any
) -> None:
    """
    Function to replace attr's value in target module

    Args:
        model (nn.Module): NN module.
        target (Union[type, str]): The type of module you want to modify.
        attr_name (str): The name of the attribute you want to modify.
        attr_value (Any): The new value of the attribute.
    """
    target = COMPONENTS.get(target) if isinstance(target, str) else target
    for module in model.modules():
        if isinstance(module, target):
            setattr(module, attr_name, attr_value)


def initialize_weights_(
    model: nn.Module,
    init_type: str = 'normal',
) -> None:
    """
    Initialize the weights in the given model.

    Args:
        model (nn.Module):
            The model to initialize.
        init_type (str, optional):
            The initialization method to use. Supported options are 'uniform'
            and 'normal'. Defaults to 'normal'.

    Raises:
        TypeError: If init_type is not supported.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f'model must be an instance of nn.Module, but got {type(model)}')

    init_functions = {
        'uniform': nn.init.kaiming_uniform_,
        'normal': nn.init.kaiming_normal_
    }

    if init_type not in init_functions:
        raise TypeError(f'init_type {init_type} is not supported.')
    nn_init = init_functions[init_type]

    def _recursive_init(m):
        if has_children(m):
            for child in m.children():
                _recursive_init(child)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.affine:
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    _recursive_init(model)
