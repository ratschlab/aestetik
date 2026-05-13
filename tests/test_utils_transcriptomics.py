"""Tests for ``aestetik.utils.utils_transcriptomics``."""
from __future__ import annotations

import anndata
import pytest

from aestetik.utils.utils_transcriptomics import preprocess_adata

# preprocess_adata uses scanpy's seurat_v3 HVG flavor, which requires
# scikit-misc. Skip the whole module if it's not available.
pytest.importorskip("skmisc", reason="scikit-misc required for seurat_v3 HVG flavor")


@pytest.fixture
def dlpfc_adata(dlpfc_adata_path) -> anndata.AnnData:
    """The LIBD DLPFC sample 151676. Re-read per test so .raw / .obs
    modifications by ``preprocess_adata`` are not shared.
    """
    adata = anndata.read_h5ad(dlpfc_adata_path)
    adata.var_names_make_unique()
    return adata


def test_preprocess_adata_produces_X_pca(dlpfc_adata: anndata.AnnData) -> None:
    out = preprocess_adata(dlpfc_adata, pca_n_comps=15)
    assert "X_pca" in out.obsm
    assert out.obsm["X_pca"].shape[1] == 15
    # spot count is preserved
    assert out.n_obs == dlpfc_adata.n_obs
    # genes get filtered
    assert out.n_vars < dlpfc_adata.n_vars


def test_preprocess_adata_does_not_mutate_input(dlpfc_adata: anndata.AnnData) -> None:
    n_vars_before = dlpfc_adata.n_vars
    obsm_keys_before = set(dlpfc_adata.obsm.keys())
    preprocess_adata(dlpfc_adata)
    assert dlpfc_adata.n_vars == n_vars_before
    assert set(dlpfc_adata.obsm.keys()) == obsm_keys_before


def test_preprocess_adata_keeps_raw(dlpfc_adata: anndata.AnnData) -> None:
    out = preprocess_adata(dlpfc_adata, pca_n_comps=10)
    assert out.raw is not None
    # raw should carry the un-normalised, pre-filter expression
    assert out.raw.n_vars >= out.n_vars


def test_preprocess_adata_respects_pca_n_comps(dlpfc_adata: anndata.AnnData) -> None:
    out = preprocess_adata(dlpfc_adata, pca_n_comps=5)
    assert out.obsm["X_pca"].shape == (dlpfc_adata.n_obs, 5)
