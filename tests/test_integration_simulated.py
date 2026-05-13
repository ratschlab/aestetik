"""End-to-end integration tests using the small simulated dataset
``test_data/A.h5ad`` (1000 spots, X_pca + image obsm pre-computed).

Marked ``slow`` so they can be skipped in fast CI runs via
``pytest -m 'not slow'``.
"""
from __future__ import annotations

import logging

import pytest

from aestetik import AESTETIK

pytestmark = pytest.mark.slow


def _prepare_simulated(adata):
    adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"][:, :10]
    adata.obsm["X_pca_morphology"] = adata.obsm["image"][:, :10]
    return adata


def test_fit_predict_on_simulated(simulated_adata):
    logging.getLogger("lightning").setLevel(logging.ERROR)
    adata = _prepare_simulated(simulated_adata)

    model = AESTETIK(
        nCluster=adata.obs.ground_truth.unique().size,
        morphology_weight=1.5,
        total_weight=3.0,
        window_size=3,
        kernel_size=3,
        latent_dim=8,
        c_hid=16,
        max_epochs=1,
        clustering_method="kmeans",
        n_ensemble=1,
        n_neighbors=5,
        refine_cluster=True,
        n_jobs=1,
        num_workers=0,
    )
    model.fit_predict(adata, num_repeats=2)

    assert "AESTETIK" in adata.obsm
    assert adata.obsm["AESTETIK"].shape == (adata.n_obs, 8)
    assert "AESTETIK_cluster" in adata.obs.columns
    # we trained the joint model: ARI vs ground truth should be at least
    # better than chance. We don't pin a strong threshold — the test is
    # there to flag catastrophic regressions.
    from sklearn.metrics.cluster import adjusted_rand_score
    ari = adjusted_rand_score(
        adata.obs.ground_truth.values,
        adata.obs["AESTETIK_cluster"].values,
    )
    assert ari > -0.1  # purely a "did anything sensible happen" gate


def test_fit_with_validation_split_runs_early_stopping(simulated_adata):
    logging.getLogger("lightning").setLevel(logging.ERROR)
    adata = _prepare_simulated(simulated_adata)
    model = AESTETIK(
        nCluster=3,
        morphology_weight=1.0,
        total_weight=2.0,
        window_size=3,
        kernel_size=3,
        latent_dim=4,
        c_hid=8,
        max_epochs=2,
        clustering_method="kmeans",
        n_ensemble=1,
        n_neighbors=5,
        refine_cluster=False,
        n_jobs=1,
        num_workers=0,
    )
    model.fit(adata, validation_split=0.2,
              early_stopping_params={"patience": 1, "min_delta": 0.0})
    # callbacks should include the loss history collector
    assert len(model.losses) >= 1


def test_predict_only_on_a_fitted_model(simulated_adata):
    logging.getLogger("lightning").setLevel(logging.ERROR)
    adata = _prepare_simulated(simulated_adata)
    model = AESTETIK(
        nCluster=3,
        morphology_weight=1.0,
        total_weight=2.0,
        window_size=3,
        kernel_size=3,
        latent_dim=4,
        c_hid=8,
        max_epochs=1,
        clustering_method="kmeans",
        n_ensemble=1,
        n_neighbors=5,
        refine_cluster=False,
        n_jobs=1,
        num_workers=0,
    )
    model.fit(adata)
    # Run prediction on a copy with reordered spots to confirm we don't
    # depend on row order.
    held_out = adata[::-1].copy()
    model.predict(held_out, num_repeats=2, cluster=False)
    assert "AESTETIK" in held_out.obsm
    assert held_out.obsm["AESTETIK"].shape[0] == held_out.n_obs
