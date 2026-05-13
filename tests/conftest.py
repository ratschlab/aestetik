"""Shared pytest fixtures.

The fixtures here construct tiny synthetic AnnData objects so the tests
do not depend on the heavyweight files in `test_data/`.
"""
from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest


def _make_grid_adata(n_x: int = 6, n_y: int = 6,
                     transcriptomics_dim: int = 4,
                     morphology_dim: int = 4,
                     n_clusters: int = 3,
                     batch_label: str | None = None,
                     obs_names: str = "int") -> anndata.AnnData:
    """Build an AnnData on a regular ``n_x x n_y`` grid.

    Parameters
    ----------
    obs_names : "int" or "string"
        Controls whether ``obs_names`` are integers (as strings) or
        true barcode-like strings. The latter is the standard Visium
        layout and exercises the label-vs-positional indexing bug
        (issue #7).
    """
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="xy")
    xs = xs.ravel()
    ys = ys.ravel()
    n_spots = xs.size

    X = rng.standard_normal((n_spots, max(transcriptomics_dim, 5))).astype(np.float32)
    obs = pd.DataFrame({
        "x_array": xs.astype(int),
        "y_array": ys.astype(int),
        "ground_truth": rng.integers(0, n_clusters, n_spots).astype(str),
    })
    if obs_names == "string":
        obs.index = [f"AAACAA-{i:04d}" for i in range(n_spots)]
    else:
        obs.index = [str(i) for i in range(n_spots)]

    if batch_label is not None:
        obs[batch_label] = rng.integers(0, 2, n_spots).astype(str)

    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsm["X_pca_transcriptomics"] = rng.standard_normal((n_spots, transcriptomics_dim)).astype(np.float32)
    adata.obsm["X_pca_morphology"] = rng.standard_normal((n_spots, morphology_dim)).astype(np.float32)
    return adata


@pytest.fixture
def small_adata():
    return _make_grid_adata()


@pytest.fixture
def small_adata_string_obs():
    return _make_grid_adata(obs_names="string")


@pytest.fixture
def small_adata_with_batch():
    return _make_grid_adata(batch_label="sample_id")
