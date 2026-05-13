"""Tests for ``aestetik.utils.utils_data`` (prepare_input_for_model,
build_grid, calibrate_transcriptomics_morphology_ratio).
"""
from __future__ import annotations

import numpy as np
import pytest

from aestetik.utils.utils_data import (
    build_grid,
    calibrate_transcriptomics_morphology_ratio,
    prepare_input_for_model,
)


def test_build_grid_writes_X_st_grid(small_adata):
    build_grid(
        small_adata,
        used_obsm_transcriptomics="X_pca_transcriptomics",
        used_obsm_morphology="X_pca_morphology",
        window_size=3,
        n_jobs=1,
    )
    assert "X_st_grid" in small_adata.obsm
    grid = small_adata.obsm["X_st_grid"]
    # shape: (n_spots, t_dim + m_dim, window_size, window_size)
    assert grid.shape == (
        small_adata.n_obs,
        small_adata.obsm["X_pca_transcriptomics"].shape[1]
        + small_adata.obsm["X_pca_morphology"].shape[1],
        3,
        3,
    )
    assert np.isfinite(grid).all()


def test_build_grid_with_batch(small_adata_with_batch):
    build_grid(
        small_adata_with_batch,
        used_obsm_transcriptomics="X_pca_transcriptomics",
        used_obsm_morphology="X_pca_morphology",
        window_size=3,
        n_jobs=1,
        used_obs_batch="sample_id",
    )
    assert small_adata_with_batch.obsm["X_st_grid"].shape[0] == small_adata_with_batch.n_obs


def test_prepare_input_for_model_returns_weights(small_adata):
    t_w, m_w = prepare_input_for_model(
        adata=small_adata,
        nCluster=3,
        used_obsm_transcriptomics="X_pca_transcriptomics",
        used_obsm_morphology="X_pca_morphology",
        used_obsm_combined="X_pca",
        clustering_method="kmeans",
        n_neighbors=3,
        window_size=3,
        n_jobs=1,
        total_weight=3.0,
        morphology_weight=1.5,
    )
    assert t_w + m_w == pytest.approx(3.0)
    # clustering side effects:
    for col in (
        "X_pca_transcriptomics_cluster",
        "X_pca_morphology_cluster",
    ):
        assert col in small_adata.obs.columns
    assert "X_pca" in small_adata.obsm
    assert "X_st_grid" in small_adata.obsm


def test_calibrate_zeroes_out_collapsed_modality(small_adata):
    # Make morphology clustering degenerate -> 1 unique cluster.
    small_adata.obs["X_pca_morphology_cluster"] = "0"
    small_adata.obs["X_pca_transcriptomics_cluster"] = np.array(
        ["0", "1", "2"] * (small_adata.n_obs // 3 + 1)
    )[: small_adata.n_obs]
    t_w, m_w = calibrate_transcriptomics_morphology_ratio(
        adata=small_adata,
        nCluster=3,
        used_obsm_transcriptomics="X_pca_transcriptomics",
        used_obsm_morphology="X_pca_morphology",
        total_weight=3.0,
        morphology_weight=1.5,
    )
    assert m_w == 0
    # transcriptomics keeps its share (total - original morphology)
    assert t_w == pytest.approx(1.5)
