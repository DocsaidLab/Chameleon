from functools import partial

import timm

from ...registry import BACKBONES
from .gpunet import GPUNet

timm_models = timm.list_models()
for name in timm_models:
    create_func = partial(timm.create_model, model_name=name)
    BACKBONES.register_module(f'timm_{name}', module=create_func)


__all__ = ['GPUNet']
