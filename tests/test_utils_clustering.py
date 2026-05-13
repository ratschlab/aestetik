"""Tests for clustering utilities."""
import numpy as np

from aestetik.utils.utils_clustering import clustering


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
    # refined labels should still be string-typed
    assert small_adata.obs[col].dtype == object or small_adata.obs[col].dtype.name == "category"
