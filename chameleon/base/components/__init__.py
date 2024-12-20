import fnmatch

from .activation import *
from .dropout import *
from .loss import *
from .norm import *
from .pooling import *


def build_component_cls(name):
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'Component named {name} is not support.')
    return cls


def build_component(name, **options):
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'Component named {name} is not support.')
    return cls(**options)


def list_components(filter=''):
    component_list = [k for k in globals().keys() if 'Component' in k]
    if len(filter):
        return fnmatch.filter(component_list, filter)  # include these components
    else:
        return component_list
