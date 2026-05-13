"""Shared pytest fixtures.

We provide two kinds of fixtures:

* Synthetic-grid fixtures (``small_adata`` etc.) — fast, deterministic,
  used for unit tests that only need the right shapes and a few clusters.
* Real-data fixtures (``simulated_adata``, ``dlpfc_adata``,
  ``dlpfc_image_path``) — derived from the files committed under
  ``test_data/``. They exercise the full transcriptomics / morphology
  pipeline and are skipped automatically if the data is not present.
"""
from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest


TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


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


# ----------------------------- Real fixtures -----------------------------
# We avoid re-reading the .h5ad file on every test by caching the load at
# session scope and handing out a fresh ``.copy()`` to each test.


def _require_data(name: str) -> Path:
    path = TEST_DATA_DIR / name
    if not path.exists():
        pytest.skip(f"required fixture file {path} not found")
    return path


@pytest.fixture(scope="session")
def simulated_adata_path() -> Path:
    """Path to the small (1000 spot) simulated dataset."""
    return _require_data("A.h5ad")


@pytest.fixture(scope="session")
def _simulated_adata_cached(simulated_adata_path: Path) -> anndata.AnnData:
    return anndata.read_h5ad(simulated_adata_path)


@pytest.fixture
def simulated_adata(_simulated_adata_cached: anndata.AnnData) -> anndata.AnnData:
    """Per-test copy of the simulated dataset (1000 spots, X_pca + image obsm)."""
    return _simulated_adata_cached.copy()


@pytest.fixture(scope="session")
def dlpfc_adata_path() -> Path:
    """Path to the LIBD DLPFC sample 151676 raw-count fixture."""
    return _require_data("151676.h5ad")


@pytest.fixture(scope="session")
def dlpfc_image_path() -> Path:
    """Path to the LIBD DLPFC sample 151676 H&E image."""
    return _require_data("151676.png")


@pytest.fixture(scope="session")
def dlpfc_json_path() -> Path:
    return _require_data("151676.json")
