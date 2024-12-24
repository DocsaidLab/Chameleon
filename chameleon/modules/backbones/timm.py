from functools import partial

import timm
import torch.nn as nn

from ...registry import BACKBONES

models = timm.list_models()

for name in models:
    BACKBONES.register_module(
        f'timm_{name}',
        module=partial(timm.create_model, model_name=name),
        is_model_builder=True,
    )
