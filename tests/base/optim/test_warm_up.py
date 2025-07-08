import math

import pytest
import torch
from chameleon.base.optim import MultiStepLRWarmUp, WrappedLRScheduler
from torch.optim.lr_scheduler import MultiStepLR


def make_optimizer(lr: float = 0.1):
    """Helper to create a simple optimizer with one Linear parameter."""
    model = torch.nn.Linear(4, 2, bias=False)
    return torch.optim.SGD(model.parameters(), lr=lr)


class DummyScheduler(MultiStepLR):
    """A dummy MultiStepLR that exposes the base_lrs for testing."""
    pass


def test_wrapped_scheduler_invalid_args():
    """WrappedLRScheduler should reject non-positive milestone or multiplier < 1.0."""
    opt = make_optimizer()
    with pytest.raises(ValueError):
        WrappedLRScheduler(opt, milestone=0)
    with pytest.raises(ValueError):
        WrappedLRScheduler(opt, milestone=-1)
    with pytest.raises(ValueError):
        WrappedLRScheduler(opt, milestone=5, multiplier=0.5)


def test_warmup_linear_increase_to_base_lr():
    """
    Test that with multiplier=1.0 and no after_scheduler,
    LR increases linearly from 0 to base_lr over `milestone` steps,
    then stays flat.
    """
    base_lr = 0.2
    opt = make_optimizer(lr=base_lr)
    milestone = 5
    scheduler = WrappedLRScheduler(
        opt, milestone=milestone, multiplier=1.0, after_scheduler=None)

    lrs = []
    # simulate epochs 0 through milestone+2
    for epoch in range(milestone + 3):
        scheduler.step(epoch)
        lrs.append(scheduler.get_last_lr()[0])

    # At epoch=0 LR should be 0
    assert math.isclose(lrs[0], 0.0, rel_tol=1e-6)
    # At epoch=milestone LR should reach base_lr
    assert math.isclose(lrs[milestone], base_lr, rel_tol=1e-6)
    # After milestone, LR should remain flat at base_lr
    assert all(math.isclose(lr, base_lr, rel_tol=1e-6)
               for lr in lrs[milestone + 1:])


def test_warmup_with_multiplier_greater_than_one():
    """
    Test that with multiplier > 1.0 and no after_scheduler,
    LR increases linearly from base_lr to base_lr*multiplier over `milestone` steps.
    """
    base_lr = 0.1
    multiplier = 2.0
    opt = make_optimizer(lr=base_lr)
    milestone = 4
    scheduler = WrappedLRScheduler(
        opt, milestone=milestone, multiplier=multiplier, after_scheduler=None)

    # simulate epochs 0 through milestone
    for epoch in range(milestone + 1):
        scheduler.step(epoch)
        lr = scheduler.get_last_lr()[0]
        expected = base_lr * ((multiplier - 1.0) * epoch / milestone + 1.0)
        assert math.isclose(
            lr, expected, rel_tol=1e-6), f"epoch={epoch}: got {lr}, expected {expected}"


def test_wrapped_scheduler_without_after_scheduler_freezes_after_milestone():
    """
    When there is no after_scheduler,
    after warm-up milestone the LR should remain at base_lr * multiplier forever.
    """
    base_lr = 0.15
    multiplier = 1.5
    opt = make_optimizer(lr=base_lr)
    milestone = 3
    scheduler = WrappedLRScheduler(
        opt, milestone=milestone, multiplier=multiplier, after_scheduler=None)

    # advance to beyond milestone
    for epoch in range(milestone + 5):
        scheduler.step(epoch)

    # last_lr should always equal base_lr * multiplier
    assert math.isclose(scheduler.get_last_lr()[
                        0], base_lr * multiplier, rel_tol=1e-6)


def test_multisteplr_warmup_factory_and_errors():
    """
    Test that MultiStepLRWarmUp returns a WrappedLRScheduler
    and that invalid warmup_milestone raises an error.
    """
    opt = make_optimizer()
    # Valid factory call
    sched = MultiStepLRWarmUp(
        opt, milestones=[2, 4], warmup_milestone=2, gamma=0.5)
    assert isinstance(sched, WrappedLRScheduler)

    # Invalid warmup_milestone propagates ValueError from WrappedLRScheduler
    with pytest.raises(ValueError):
        MultiStepLRWarmUp(
            opt, milestones=[2, 4], warmup_milestone=0, gamma=0.5)


def test_multisteplr_warmup_delegates_to_multisteplr():
    """
    Test end-to-end behavior of MultiStepLRWarmUp:
    - Warm-up for `warmup_milestone` epochs (0,1,2)
    - Then apply MultiStepLR at epoch >= warmup_milestone+1
    """
    base_lr = 0.1
    opt = make_optimizer(lr=base_lr)
    wrapped = MultiStepLRWarmUp(
        optimizer=opt,
        milestones=[3, 6],
        warmup_milestone=2,
        gamma=0.1,
        last_epoch=-1
    )

    lrs = []
    for epoch in range(8):
        wrapped.step(epoch)
        lrs.append(wrapped.get_last_lr()[0])

    # epochs 0,1,2: warm-up 0 -> base_lr
    assert math.isclose(lrs[0], 0.0, rel_tol=1e-6)
    assert math.isclose(lrs[1], base_lr * 1/2, rel_tol=1e-6)
    assert math.isclose(lrs[2], base_lr,      rel_tol=1e-6)

    # epochs 3,4: still no decay
    assert all(math.isclose(lrs[e], base_lr, rel_tol=1e-6) for e in [3, 4])

    # epochs 5,6,7: decay by gamma once
    expected_decay_lr = base_lr * 0.1
    assert all(math.isclose(lrs[e], expected_decay_lr, rel_tol=1e-6)
               for e in [5, 6, 7])
