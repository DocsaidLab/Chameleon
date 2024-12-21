import fnmatch

from .aspp import ASPP
from .grl import GradientReversalLayer
from .selayer import SELayer
from .vae import VAE
from .weighted_sum import WeightedSum


def build_layer(name, **options):
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'Layer named {name} is not support.')
    return cls(**options)


def list_layers(filter=''):
    layer_list = [k for k in globals().keys() if 'Layer' in k]
    if len(filter):
        return fnmatch.filter(layer_list, filter)  # include these layers
    else:
        return layer_list
