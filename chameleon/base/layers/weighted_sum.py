from typing import List, Optional

import torch
import torch.nn as nn

from ...registry import COMPONENTS, LAYERS


@LAYERS.register_module()
class WeightedSum(nn.Module):
    def __init__(
        self,
        input_size: int,
        act: Optional[dict] = None,
        requires_grad: bool = True,
    ) -> None:
        """
        Initializes a WeightedSum module.

        Args:
            input_size (int):
                The number of inputs to be summed.
            act Optional[dict]:
                Optional activation function or dictionary of its parameters.
                Defaults to None.
            requires_grad (bool, optional):
                Whether to require gradients for the weights. Defaults to True.
        """
        super().__init__()
        self.input_size = input_size
        self.weights = nn.Parameter(
            torch.ones(input_size, dtype=torch.float32),
            requires_grad=requires_grad
        )
        self.weights_relu = nn.ReLU()
        if act is None:
            self.relu = nn.Identity()
        else:
            self.relu = act if isinstance(act, nn.Module) else COMPONENTS.build(act)
        self.epsilon = 1e-4

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != self.input_size:
            raise ValueError('Invalid input size not equal to weight size.')
        weights = self.weights_relu(self.weights)
        weights = weights / (
            torch.sum(weights, dim=0, keepdim=True) + self.epsilon)
        weighted_x = torch.einsum(
            'i,i...->...', weights, torch.stack(x, dim=0))
        weighted_x = self.relu(weighted_x)
        return weighted_x
