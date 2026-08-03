"""Tests for ``aestetik.utils.utils_clustering``.

Covers kmeans/bgm/leiden/louvain paths, the resolution-search helper,
and the spatial cluster-refinement code path including the multi-batch
branch.
"""
from __future__ import annotations

import numpy as np
import pytest

from aestetik.utils.utils_clustering import (
    _refine,
    _refine_cluster,
    clustering,
    search_res,
)


def test_kmeans_clustering_assigns_labels(small_adata):
    clustering(
        adata=small_adata,
        num_cluster=3,
        used_obsm="X_pca_transcriptomics",
        method="kmeans",
        refine_cluster=False,
    )
    col = "X_pca_transcriptomics_cluster"
    assert col in small_adata.obs.columns
    assert small_adata.obs[col].nunique() == 3


def test_bgm_clustering_creates_proba_column(small_adata):
    clustering(
        adata=small_adata,
        num_cluster=3,
        used_obsm="X_pca_transcriptomics",
        method="bgm",
        refine_cluster=False,
    )
    assert "X_pca_transcriptomics_cluster_proba" in small_adata.obs.columns


def test_refine_cluster_keeps_label_dtype(small_adata):
    clustering(
        adata=small_adata,
        num_cluster=3,
        used_obsm="X_pca_transcriptomics",
        method="kmeans",
        refine_cluster=True,
        n_neighbors=3,
    )
    col = "X_pca_transcriptomics_cluster"
    # pandas 2 reports object, pandas 3 reports str; both are label-like.
    assert small_adata.obs[col].dtype.name in {"object", "category", "str", "string"}


def test_leiden_clustering_with_float_resolution(small_adata):
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    # nCluster as float -> directly used as resolution, no search.
    clustering(
        adata=small_adata,
        num_cluster=0.5,
        used_obsm="X_pca_transcriptomics",
        method="leiden",
        refine_cluster=False,
    )
    col = "X_pca_transcriptomics_cluster"
    assert col in small_adata.obs.columns
    # leiden writes a categorical
    assert small_adata.obs[col].nunique() >= 1


def test_louvain_clustering_with_float_resolution(small_adata):
    pytest.importorskip("igraph")
    pytest.importorskip("louvain")
    clustering(
        adata=small_adata,
        num_cluster=0.5,
        used_obsm="X_pca_transcriptomics",
        method="louvain",
        refine_cluster=False,
    )
    col = "X_pca_transcriptomics_cluster"
    assert col in small_adata.obs.columns


def test_search_res_finds_resolution_for_target_count(small_adata):
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    res = search_res(
        small_adata,
        n_clusters=2,
        use_rep="X_pca_transcriptomics",
        method="leiden",
        start=0.05,
        end=2.0,
        increment=0.05,
    )
    assert 0.05 <= res <= 2.0


def test_refine_helper_returns_none_for_single_class(small_adata):
    df = small_adata.obs.copy()
    df["cluster"] = "0"  # single class
    out = _refine(
        df,
        cluster_col="cluster",
        proba_col="missing_proba_col",
        conf_proba=0.9,
        n_neighbors=3,
        spatial_cols=["x_array", "y_array"],
    )
    assert out is None


def test_refine_cluster_handles_per_batch(small_adata_with_batch):
    rng = np.random.default_rng(0)
    small_adata_with_batch.obs["X_pca_transcriptomics_cluster"] = rng.integers(
        0, 3, small_adata_with_batch.n_obs
    ).astype(str)
    _refine_cluster(
        adata=small_adata_with_batch,
        used_obsm="X_pca_transcriptomics",
        used_obs_batch="sample_id",
        n_neighbors=3,
        conf_proba=0.5,
    )
    # the column is still present after per-batch refinement
    assert "X_pca_transcriptomics_cluster" in small_adata_with_batch.obs.columns
