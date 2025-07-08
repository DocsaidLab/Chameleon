from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class PolynomialLRWarmup(LRScheduler):
    """
    Scheduler with an initial linear warm-up followed by polynomial decay.

    - For the first `warmup_iters` steps, LR increases linearly
      from 0 -> base_lr.
    - For steps `warmup_iters < step <= total_iters`, LR decays as
      base_lr * (1 - (step - warmup_iters) / (total_iters - warmup_iters))^power.
    - After `total_iters`, LR is held at the final decayed value.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): Number of steps for linear warm-up; must be ≥ 0.
        total_iters (int): Total number of steps for warm-up + decay; must be ≥ warmup_iters.
        power (float): Exponent for polynomial decay. Default: 1.0 (linear).
        last_epoch (int): The index of last step. Default: -1 (start from step 0).
        verbose (bool): If True, prints a message for each LR update.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        total_iters: int,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        # input validation
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
        if total_iters < warmup_iters:
            raise ValueError(
                f"total_iters ({total_iters}) must be >= warmup_iters ({warmup_iters})")
        if power < 0:
            raise ValueError(f"power must be non-negative, got {power}")

        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.power = power

        super().__init__(optimizer, last_epoch)

    def get_closed_form(self) -> List[float]:
        """
        Compute the learning rate for the current `last_epoch` in closed form.
        Called by the base class when you use the chainable API: scheduler.step().
        """
        # Clamp epoch to [0, total_iters]
        epoch = min(max(self.last_epoch, 0), self.total_iters)

        # 1) Warm-up phase
        if epoch <= self.warmup_iters:
            return [
                base_lr *
                (epoch / self.warmup_iters if self.warmup_iters > 0 else 1.0)
                for base_lr in self.base_lrs
            ]

        # 2) Polynomial decay phase
        decay_steps = epoch - self.warmup_iters
        decay_total = self.total_iters - self.warmup_iters
        factor = (1.0 - decay_steps / decay_total) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

    def get_lr(self) -> List[float]:
        """
        Legacy step API. If you’re still calling scheduler.step(epoch),
        this will be invoked instead of get_closed_form().
        """
        return self.get_closed_form()
