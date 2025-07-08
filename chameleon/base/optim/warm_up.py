from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

# Project-specific registry (keep as-is or remove if unused)
from ...registry import OPTIMIZERS

__all__ = ["WrappedLRScheduler", "MultiStepLRWarmUp"]


@OPTIMIZERS.register_module()
class WrappedLRScheduler(LRScheduler):
    """
    Gradual warmup scheduler.

    During the first `milestone` steps (or epochs), the learning rate
    increases linearly from 0 (or base_lr) up to base_lr * multiplier.
    After warmup completes, scheduling is delegated to `after_scheduler`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestone (int): Number of steps (or epochs) for warmup; must be > 0.
        multiplier (float, optional): Final LR = base_lr * multiplier.
            - If multiplier == 1.0, warmup goes from 0 -> base_lr.
            - If multiplier > 1.0, warmup goes from base_lr -> base_lr * multiplier.
        after_scheduler (LRScheduler, optional): Scheduler to use after warmup.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for
            each update. Default: False.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestone: int,
        multiplier: float = 1.0,
        after_scheduler: Optional[LRScheduler] = None,
        last_epoch: int = -1
    ):
        if milestone <= 0:
            raise ValueError("milestone must be > 0.")
        if multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0.")

        self.milestone = milestone
        self.multiplier = multiplier
        self.after_scheduler = after_scheduler
        self.finished = False

        # Initialize base class with optimizer, last_epoch, and verbose
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # During warmup phase
        if self.last_epoch <= self.milestone:
            if self.multiplier == 1.0:
                # Linear increase: 0 -> base_lr
                return [
                    base_lr * (self.last_epoch / self.milestone)
                    for base_lr in self.base_lrs
                ]
            else:
                # Linear increase: base_lr -> base_lr * multiplier
                return [
                    base_lr * ((self.multiplier - 1.0) *
                               self.last_epoch / self.milestone + 1.0)
                    for base_lr in self.base_lrs
                ]

        # After warmup completes
        if self.after_scheduler is not None:
            # On first transition, reset the after_scheduler's base_lrs
            if not self.finished:
                self.after_scheduler.base_lrs = [
                    base_lr * self.multiplier for base_lr in self.base_lrs
                ]
                self.finished = True
            # Delegate to after_scheduler
            return self.after_scheduler.get_last_lr()

        # No after_scheduler: keep LR at base_lr * multiplier
        return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """
        Update the learning rate.

        If warmup is finished and an after_scheduler is provided,
        delegate the step to after_scheduler. Otherwise, call the
        base class step() to continue warmup.

        Args:
            epoch (int, optional): Current epoch or step index.
            metrics (float, optional): Metric for ReduceLROnPlateau.
        """
        if self.finished and self.after_scheduler is not None:
            # If using ReduceLROnPlateau (metric-based), pass metrics first
            if metrics is not None and "plateau" in type(self.after_scheduler).__name__.lower():
                self.after_scheduler.step(
                    metrics, epoch - self.milestone if epoch is not None else None)
            else:
                # Standard scheduler.step(epoch)
                self.after_scheduler.step(
                    epoch - self.milestone if epoch is not None else None)
            # Sync the last learning rates
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            # Still in warmup or no after_scheduler: use base class logic
            super().step(epoch)


@OPTIMIZERS.register_module(is_model_builder=True)
def MultiStepLRWarmUp(
    optimizer: Optimizer,
    milestones: List[int],
    warmup_milestone: int,
    gamma: float = 0.1,
    last_epoch: int = -1
) -> WrappedLRScheduler:
    """
    Factory function to create a warmup + MultiStepLR scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (List[int]): List of epoch indices where LR is decayed by gamma.
        warmup_milestone (int): Number of epochs for linear warmup.
        gamma (float, optional): Multiplicative LR decay factor for MultiStepLR. Default: 0.1.
        last_epoch (int, optional): Index of last epoch. Default: -1 (start from scratch).

    Returns:
        WrappedLRScheduler: Scheduler that linearly warms up for `warmup_milestone`
                            epochs, then delegates to MultiStepLR.
    """
    # 1) create the MultiStepLR scheduler that will run *after* warmup
    multi_step = MultiStepLR(
        optimizer=optimizer,
        milestones=milestones,
        gamma=gamma,
        last_epoch=last_epoch,
    )

    # 2) wrap it with linear warmup
    return WrappedLRScheduler(
        optimizer=optimizer,
        milestone=warmup_milestone,
        multiplier=1.0,           # warmup from 0 -> base_lr
        after_scheduler=multi_step,
        last_epoch=last_epoch,
    )
