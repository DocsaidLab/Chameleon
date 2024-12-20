import fnmatch

from torch.optim import (ASGD, LBFGS, SGD, Adadelta, Adagrad, Adam, Adamax,
                         AdamW, RMSprop, Rprop, SparseAdam)
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LambdaLR, MultiStepLR,
                                      OneCycleLR, ReduceLROnPlateau, StepLR)

from .polynomial_lr_warmup import PolynomialLRWarmup
from .warm_up import *


def build_optimizer(model_params, name, **optim_options):
    cls_ = globals().get(name, None)
    if cls_ is None:
        raise ValueError(f'{name} is not supported optimizer.')
    return cls_(model_params, **optim_options)


def build_lr_scheduler(optimizer, name, **lr_scheduler_options):
    cls_ = globals().get(name, None)
    if cls_ is None:
        raise ValueError(f'{name} is not supported lr scheduler.')
    return cls_(optimizer, **lr_scheduler_options)


def list_optimizers(filter=''):
    optimizer_list = [k for k in globals().keys() if k[0].isupper()]
    if len(filter):
        return [o for o in optimizer_list if filter in o.lower()]
    else:
        return optimizer_list


def list_lr_schedulers(filter=''):
    lr_scheduler_list = [k for k in globals().keys() if 'LR' in k]
    if len(filter):
        return [o for o in lr_scheduler_list if filter in o.lower()]
    else:
        return lr_scheduler_list
