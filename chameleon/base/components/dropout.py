from torch.nn import AlphaDropout, Dropout, Dropout2d, Dropout3d

from ...registry import COMPONENTS

__all__ = [
    'Dropout', 'Dropout2d', 'Dropout3d', 'AlphaDropout',
]


for k in __all__:
    COMPONENTS.register_module(name=k, module=globals()[k])
