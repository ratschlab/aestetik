"""Regression test for issue #8: train_dataloader must iterate train_dataset.

When validation_split > 0, setup() splits self.dataset into self.train_dataset
and self.val_dataset via random_split. train_dataloader must wrap
self.train_dataset; otherwise validation samples leak into training.
"""
import importlib.util
import os
import sys
import types

import torch
from torch.utils.data import TensorDataset, random_split


def _load_data_module():
    """Import data_module.py without triggering aestetik package side effects.

    data_module.py imports `prepare_input_for_model` and `CustomDataset` at
    module load time. Both are only invoked by setup(), which the tests
    bypass. We register lightweight stub modules so the imports resolve
    without dragging in plotnine, anndata grid utilities, etc.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.normpath(os.path.join(here, "..", "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Stub the heavy imports referenced at module load time.
    pkg = types.ModuleType("aestetik")
    pkg.__path__ = [os.path.join(src_path, "aestetik")]
    sys.modules.setdefault("aestetik", pkg)

    utils_pkg = types.ModuleType("aestetik.utils")
    utils_pkg.__path__ = [os.path.join(src_path, "aestetik", "utils")]
    sys.modules.setdefault("aestetik.utils", utils_pkg)

    utils_data = types.ModuleType("aestetik.utils.utils_data")
    utils_data.prepare_input_for_model = lambda *a, **k: (1.0, 1.0)
    sys.modules["aestetik.utils.utils_data"] = utils_data

    dataloader_stub = types.ModuleType("aestetik.dataloader")
    dataloader_stub.CustomDataset = object  # only referenced inside setup()
    sys.modules["aestetik.dataloader"] = dataloader_stub

    path = os.path.join(src_path, "aestetik", "data_modules", "data_module.py")
    spec = importlib.util.spec_from_file_location("aestetik_data_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_module(dm_cls, full_dataset, train_dataset, val_dataset):
    """Build an AESTETIKDataModule bypassing __init__ and setup heavy paths."""
    dm = dm_cls.__new__(dm_cls)
    # LightningDataModule.__init__ does light bookkeeping; safe to skip and
    # populate the attributes train_dataloader / val_dataloader read.
    dm.dataset = full_dataset
    dm.train_dataset = train_dataset
    dm.val_dataset = val_dataset
    dm.dataloader_params = {"batch_size": 1, "num_workers": 0}
    return dm


def test_train_dataloader_uses_train_split_subset():
    dm_module = _load_data_module()
    full = TensorDataset(torch.arange(10).unsqueeze(1).float())
    train_subset, val_subset = random_split(
        full, [7, 3], generator=torch.Generator().manual_seed(0)
    )
    dm = _make_module(dm_module.AESTETIKDataModule, full, train_subset, val_subset)

    loader = dm.train_dataloader()
    n_train_items = sum(1 for _ in loader)

    assert n_train_items == 7, (
        f"train_dataloader yielded {n_train_items} batches; expected 7 "
        "(the train split). The full dataset has 10 items, so yielding 10 "
        "means validation data leaks into training."
    )


def test_val_dataloader_uses_val_split_subset():
    """Sanity check: val_dataloader already wraps val_dataset; pin behavior."""
    dm_module = _load_data_module()
    full = TensorDataset(torch.arange(10).unsqueeze(1).float())
    train_subset, val_subset = random_split(
        full, [7, 3], generator=torch.Generator().manual_seed(0)
    )
    dm = _make_module(dm_module.AESTETIKDataModule, full, train_subset, val_subset)

    loader = dm.val_dataloader()
    n_val_items = sum(1 for _ in loader)
    assert n_val_items == 3
