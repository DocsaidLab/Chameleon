from torch.optim import (ASGD, LBFGS, SGD, Adadelta, Adagrad, Adam, Adamax,
                         AdamW, RMSprop, Rprop, SparseAdam)
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LambdaLR, MultiStepLR,
                                      OneCycleLR, ReduceLROnPlateau, StepLR)

from ...registry import OPTIMIZERS
from .polynomial_lr_warmup import *
from .warm_up import *

__all__ = [
    'ASGD', 'LBFGS', 'SGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'AdamW',
    'RMSprop', 'Rprop', 'SparseAdam', 'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts', 'CyclicLR', 'ExponentialLR', 'LambdaLR',
    'MultiStepLR', 'OneCycleLR', 'ReduceLROnPlateau', 'StepLR',
]


for k in __all__:
    OPTIMIZERS.register_module(name=k, force=True, module=globals()[k])


__all__ += ['PolynomialLRWarmup', 'WrappedLRScheduler']
