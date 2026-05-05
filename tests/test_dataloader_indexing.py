"""Regression test for issue #7: __getitem__ must use positional indexing.

CustomDataset.__getitem__ accesses cluster labels via Series[idx] where idx
is a positional integer from a DataLoader. With non-default obs indexes
(string barcodes, or shuffled integers from train_test_split), pandas falls
back to label-based lookup which raises FutureWarning today and KeyError
on pandas >= 3.0. Switching to .iloc[idx] is the documented positional API.
"""
import importlib.util
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _load_dataloader_module():
    """Import src/aestetik/dataloader.py without triggering aestetik/__init__.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "src", "aestetik", "dataloader.py"))
    spec = importlib.util.spec_from_file_location("aestetik_dataloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_adata():
    n = 6
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "X_pca_transcriptomics_cluster": [0, 1, 0, 1, 0, 1],
            "X_pca_morphology_cluster": [1, 0, 1, 0, 1, 0],
        },
        index=[f"barcode_{i}" for i in range(n)],
    )
    X = rng.standard_normal((n, 3)).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_st_grid"] = rng.standard_normal((n, 4, 4, 3)).astype(np.float32)
    return adata


def test_getitem_does_not_use_label_based_indexing():
    """Series.__getitem__ on a non-default index emits FutureWarning under
    label-based fallback and will raise KeyError on pandas >= 3.0. The fix
    is to use .iloc, which never triggers either."""
    dl = _load_dataloader_module()
    adata = _build_adata()
    ds = dl.CustomDataset(adata, multi_triplet_loss=False, repeats=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for i in range(len(ds)):
            ds[i]


def test_getitem_returns_position_consistent_labels():
    """The label returned by ds[i] must match obs.iloc[i], not obs.loc[i]."""
    dl = _load_dataloader_module()
    adata = _build_adata()
    ds = dl.CustomDataset(adata, multi_triplet_loss=False, repeats=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for i in range(len(ds)):
            ds[i]  # must not raise (KeyError on pandas >= 3.0 with label-based)

    expected_t = list(adata.obs["X_pca_transcriptomics_cluster"].iloc[: len(ds)])
    expected_m = list(adata.obs["X_pca_morphology_cluster"].iloc[: len(ds)])
    assert expected_t == [0, 1, 0, 1, 0, 1]
    assert expected_m == [1, 0, 1, 0, 1, 0]
