"""Tests for ``aestetik.dataloader.CustomDataset``.

Covers issue #7 (positional indexing) by exercising __getitem__ both
with string obs_names and with a non-contiguous train_size split.
"""
import numpy as np
import pytest

from aestetik.dataloader import CustomDataset


def _attach_grid_and_clusters(adata, n_t=3, n_m=3):
    rng = np.random.default_rng(7)
    n_spots = adata.n_obs
    # the dataloader does not care about the *layout* of X_st_grid for
    # the purposes of __getitem__ — only that it exposes a leading
    # batch axis.
    grid = rng.standard_normal((n_spots, 3, 5, 5)).astype(np.float32)
    adata.obsm["X_st_grid"] = grid
    adata.obs["X_pca_transcriptomics_cluster"] = rng.integers(0, n_t, n_spots).astype(str)
    adata.obs["X_pca_morphology_cluster"] = rng.integers(0, n_m, n_spots).astype(str)
    return adata


def test_getitem_with_string_obs_names(small_adata_string_obs):
    """Reproduces issue #7: label-based indexing crashes with strings."""
    adata = _attach_grid_and_clusters(small_adata_string_obs)
    ds = CustomDataset(adata)
    # Must not raise KeyError / FutureWarning.
    anchor, p_t, n_t, p_m, n_m = ds[0]
    assert anchor.shape == adata.obsm["X_st_grid"].shape[1:]


def test_getitem_with_non_contiguous_train_size_split(small_adata):
    """When train_size triggers a filter, .obs.index becomes
    non-contiguous and the old [idx] access raises. .iloc[idx] is safe.
    """
    adata = _attach_grid_and_clusters(small_adata)
    ds = CustomDataset(adata, train_size=0.5)
    # All positions in the *new* dataset must be addressable.
    for i in range(len(ds)):
        anchor, *_ = ds[i]
        assert np.isfinite(anchor).all()


def test_len_matches_underlying_grid(small_adata):
    adata = _attach_grid_and_clusters(small_adata)
    ds = CustomDataset(adata)
    assert len(ds) == adata.n_obs
