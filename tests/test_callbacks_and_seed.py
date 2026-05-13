"""Tests for the LossHistoryCallback and the ``fix_seed`` reproducibility
helper.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from aestetik.callbacks.callbacks import LossHistoryCallback
from aestetik.utils.utils_grid import fix_seed


def test_loss_history_callback_appends_only_dict_losses() -> None:
    cb = LossHistoryCallback()
    cb.on_train_batch_end(
        trainer=None,
        pl_module=None,
        outputs={"loss": torch.tensor(0.5)},
        batch=None,
        batch_idx=0,
    )
    cb.on_train_batch_end(
        trainer=None,
        pl_module=None,
        outputs=None,  # not a dict -> ignored
        batch=None,
        batch_idx=1,
    )
    cb.on_train_batch_end(
        trainer=None,
        pl_module=None,
        outputs={"no_loss_here": torch.tensor(0.1)},
        batch=None,
        batch_idx=2,
    )
    assert cb.losses == [0.5]


def test_fix_seed_produces_reproducible_streams() -> None:
    fix_seed(123)
    a = (np.random.randn(5), torch.randn(5))
    fix_seed(123)
    b = (np.random.randn(5), torch.randn(5))
    np.testing.assert_array_equal(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_fix_seed_sets_python_hashseed() -> None:
    fix_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"
