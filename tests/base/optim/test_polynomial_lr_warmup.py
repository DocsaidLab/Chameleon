import math

import pytest
import torch
from chameleon.base.optim.polynomial_lr_warmup import PolynomialLRWarmup
from torch.optim import SGD
from torch.optim.lr_scheduler import LRScheduler


def make_optimizer(lr: float = 0.1):
    """Create an optimizer with a single parameter and given base LR."""
    p = torch.nn.Parameter(torch.zeros(1))
    return SGD([p], lr=lr)


def test_init_invalid_warmup_iters():
    """warmup_iters < 0 should raise ValueError."""
    opt = make_optimizer()
    with pytest.raises(ValueError):
        PolynomialLRWarmup(opt, warmup_iters=-1, total_iters=5)
    with pytest.raises(ValueError):
        PolynomialLRWarmup(opt, warmup_iters=5, total_iters=4)


def test_init_invalid_power():
    """power < 0 should raise ValueError."""
    opt = make_optimizer()
    with pytest.raises(ValueError):
        PolynomialLRWarmup(opt, warmup_iters=0, total_iters=5, power=-0.1)


@pytest.mark.parametrize("warmup, total, epoch, expected_factor", [
    # warmup only, no decay
    (5, 10, 0, 0.0 / 5),
    (5, 10, 1, 1.0 / 5),
    (5, 10, 5, 5.0 / 5),
])
def test_linear_warmup_phase(warmup, total, epoch, expected_factor):
    """Test linear increase from 0 -> base_lr over warmup_iters."""
    base_lr = 0.2
    opt = make_optimizer(base_lr)
    sched = PolynomialLRWarmup(
        opt, warmup_iters=warmup, total_iters=total, power=1.0)
    sched.last_epoch = epoch
    lr_closed = sched.get_closed_form()[0]
    lr_legacy = sched.get_lr()[0]
    expected_lr = base_lr * expected_factor
    assert math.isclose(lr_closed, expected_lr, rel_tol=1e-6)
    assert math.isclose(lr_legacy, expected_lr, rel_tol=1e-6)


def test_warmup_zero_iters():
    """warmup_iters=0 should immediately use decay formula starting at epoch 0."""
    base_lr = 0.3
    opt = make_optimizer(base_lr)
    # with warmup_iters=0, get_closed_form multiplies by 1.0 in warmup branch
    sched = PolynomialLRWarmup(opt, warmup_iters=0, total_iters=5, power=1.0)
    for epoch in range(0, 3):
        sched.last_epoch = epoch
        # decay_total = total_iters - 0 = 5
        expected = base_lr * (1.0 - epoch / 5) ** 1.0
        assert math.isclose(sched.get_closed_form()[0], expected, rel_tol=1e-6)


def test_polynomial_decay_phase():
    """Test polynomial decay after warmup_iters up to total_iters."""
    base_lr = 0.4
    opt = make_optimizer(base_lr)
    warmup = 2
    total = 8
    power = 2.0
    sched = PolynomialLRWarmup(
        opt, warmup_iters=warmup, total_iters=total, power=power)
    # test a few epochs in decay
    for epoch in [3, 5, 8]:
        sched.last_epoch = epoch
        # compute expected factor
        decay_steps = min(epoch, total) - warmup
        decay_total = total - warmup
        factor = (1.0 - decay_steps / decay_total) ** power
        expected_lr = base_lr * factor
        assert math.isclose(sched.get_closed_form()[
                            0], expected_lr, rel_tol=1e-6)


def test_after_total_iters_clamps_to_final():
    """Epochs > total_iters should clamp and hold lr at final decayed value."""
    base_lr = 0.5
    opt = make_optimizer(base_lr)
    sched = PolynomialLRWarmup(opt, warmup_iters=3, total_iters=6, power=1.0)
    # compute final lr at epoch=total_iters
    sched.last_epoch = 6
    final_lr = sched.get_closed_form()[0]
    # at epoch 9 (> total), should be equal to final_lr
    sched.last_epoch = 9
    assert math.isclose(sched.get_closed_form()[0], final_lr, rel_tol=1e-6)


def test_scheduler_chainable_api():
    """
    Ensure that using the modern .step() API after optimizer.step()
    produces the same lr as get_closed_form, up to the one-step offset
    inherent in the chainable scheduler design.
    """
    base_lr = 0.25

    opt = make_optimizer(base_lr)
    sched = PolynomialLRWarmup(opt, warmup_iters=2, total_iters=4, power=1.0)
    seen = []
    for _ in range(6):
        opt.step()
        sched.step()
        seen.append(opt.param_groups[0]["lr"])

    opt2 = make_optimizer(base_lr)
    sched2 = PolynomialLRWarmup(opt2, warmup_iters=2, total_iters=4, power=1.0)
    manual = [sched2.get_closed_form()[0]]  # 初始 last_epoch = -1 -> clamp to 0
    for epoch in range(1, 6):
        sched2.last_epoch = epoch
        manual.append(sched2.get_closed_form()[0])

    max_idx = len(manual) - 1
    for i, lr in enumerate(seen):
        expected = manual[min(i+1, max_idx)]
        assert math.isclose(
            lr, expected, rel_tol=1e-6
        ), f"At step {i}: chainable={lr}, expected closed_form at epoch {min(i+1, max_idx)} = {expected}"
