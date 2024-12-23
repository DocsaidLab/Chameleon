import timm
import torch.nn as nn

from ...registry import BACKBONES


class Timm:
    @staticmethod
    def build_model(*args, **kwargs) -> nn.Module:
        return timm.create_model(*args, **kwargs)


timm_models = timm.list_models()
for name in timm_models:
    BACKBONES.register_module(f'timm_{name}', module=Timm.build_model)
