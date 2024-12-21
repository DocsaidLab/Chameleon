from typing import List, Union

import torch.nn as nn

from .utils import initialize_weights_


class PowerModule(nn.Module):
    """
    A module that provides additional functionality for weight initialization,
    freezing and melting layers.
    """

    def initialize_weights_(self, init_type: str = 'normal') -> None:
        """
        Initialize the weights of the module.

        Args:
            init_type (str): The type of initialization. Can be 'normal' or 'uniform'.
        """
        initialize_weights_(self, init_type)

    def freeze(self, part_names: Union[str, List[str]] = 'all', verbose: bool = False) -> None:
        """
        Freeze the parameters of specified layers.

        Args:
            part_names (Union[str, List[str]]): The names of the layers to freeze.
                If 'all', all layers are frozen.
            verbose (bool): Whether to print messages indicating which layers were frozen.
        """
        if part_names == 'all':
            for name, params in self.named_parameters():
                if verbose:
                    print(f'Freezing layer {name}')
                params.requires_grad_(False)
        elif part_names is None:
            return
        else:
            part_names = [part_names] if isinstance(part_names, str) \
                else part_names
            for layer_name in part_names:
                module = self
                for attr in layer_name.split('.'):
                    module = getattr(module, attr)
                for name, param in module.named_parameters():
                    if verbose:
                        print(f'Freezing layer {layer_name}.{name}')
                    param.requires_grad_(False)

    def melt(self, part_names: Union[str, List[str]] = 'all', verbose: bool = False) -> None:
        """
        Unfreeze the parameters of specified layers.

        Args:
            part_names (Union[str, List[str]]): The names of the layers to unfreeze.
                If 'all', all layers are unfrozen.
            verbose (bool): Whether to print messages indicating which layers were unfrozen.
        """
        if part_names == 'all':
            for name, params in self.named_parameters():
                if verbose:
                    print(f'Unfreezing layer {name}')
                params.requires_grad_(True)
        elif part_names is None:
            return
        else:
            part_names = [part_names] if isinstance(part_names, str) \
                else part_names
            for layer_name in part_names:
                module = self
                for attr in layer_name.split('.'):
                    module = getattr(module, attr)
                for name, param in module.named_parameters():
                    if verbose:
                        print(f'Unfreezing layer {layer_name}.{name}')
                    param.requires_grad_(True)
