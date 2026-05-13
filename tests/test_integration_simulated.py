"""End-to-end integration tests modelled after the getting-started
notebooks, using the small simulated dataset ``test_data/A.h5ad``
(1000 spots, ``X_pca`` + ``image`` obsm pre-computed, ``ground_truth``
labels available).

These tests exercise the sklearn-style public API exactly the way the
notebooks do. Marked ``slow`` so they can be skipped via
``pytest -m 'not slow'``.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.metrics.cluster import adjusted_rand_score

from aestetik import AESTETIK
from aestetik.utils.utils_grid import fix_seed


pytestmark = pytest.mark.slow

ARI_THRESHOLD = 0.4
SEED = 2023

# Tests use a deliberately tiny configuration so the slow suite
# finishes within ~5 min on CPU. The simulated dataset is well
# separated; the ARI gate is comfortable at this budget.
FAST_CFG = dict(
    max_epochs=10,
    latent_dim=8,
    c_hid=32,
    n_ensemble=1,
    num_repeats_predict=20,
)
# Subsample for speed. With 100 stratified spots the three simulated
# clusters are still well-separated.
N_SPOTS = 100


def _prepare_simulated(adata, n_components: int = 15, n_spots: int = N_SPOTS):
    """Mirror the notebook obsm setup, then subsample to ``n_spots``
    spots stratified by ground_truth for test-suite speed.
    """
    if n_spots is not None and n_spots < adata.n_obs:
        rng = np.random.default_rng(SEED)
        groups = adata.obs.ground_truth.values
        idx = []
        per_group = n_spots // len(np.unique(groups))
        for g in np.unique(groups):
            members = np.where(groups == g)[0]
            pick = rng.choice(members, size=min(per_group, len(members)), replace=False)
            idx.extend(pick)
        adata = adata[np.array(idx)].copy()
    adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"][:, :n_components]
    adata.obsm["X_pca_morphology"] = adata.obsm["image"][:, :n_components]
    return adata


def _ari(ground_truth, labels) -> float:
    return float(adjusted_rand_score(ground_truth, labels))


def test_simulated_notebook_fit_predict(simulated_adata):
    """Exact reproduction of the simulated-data notebook recipe via
    the sklearn ``fit_predict`` surface (ClusterMixin).
    """
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(SEED)
    adata = _prepare_simulated(simulated_adata)

    model = AESTETIK(
        n_cluster=adata.obs.ground_truth.unique().size,
        morphology_weight=1.5,
        refine_cluster=True,
        window_size=3,
        clustering_method="kmeans",
        random_state=SEED,
        n_jobs=1,
        num_workers=0,
        **FAST_CFG,
    )
    labels = model.fit_predict(adata)

    assert labels.shape == (adata.n_obs,)
    assert model.embedding_.shape == (adata.n_obs, FAST_CFG["latent_dim"])
    np.testing.assert_array_equal(labels, model.labels_)
    ari = _ari(adata.obs.ground_truth.values, labels)
    assert ari > ARI_THRESHOLD, (
        f"ARI vs ground_truth dropped to {ari:.3f} (threshold {ARI_THRESHOLD}). "
        "Joint multi-modal training regressed."
    )


def test_simulated_notebook_fit_transform(simulated_adata):
    """``fit_transform`` returns the embedding array (TransformerMixin)."""
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(SEED)
    adata = _prepare_simulated(simulated_adata)

    model = AESTETIK(
        n_cluster=adata.obs.ground_truth.unique().size,
        morphology_weight=1.5,
        refine_cluster=True,
        window_size=3,
        clustering_method="kmeans",
        random_state=SEED,
        n_jobs=1,
        num_workers=0,
        **FAST_CFG,
    )
    embedding = model.fit_transform(adata)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (adata.n_obs, FAST_CFG["latent_dim"])
    # ``model.embedding_`` is cached during fit. ``fit_transform`` runs
    # an *independent* stochastic forward pass via the encoder
    # (dropout is enabled by predict_step's model.train()), so the two
    # arrays correlate but are not byte-equal. Just check the shape +
    # finite-ness here.
    assert model.embedding_.shape == embedding.shape
    assert np.isfinite(embedding).all()


def test_transcriptomics_only_recipe(simulated_adata):
    """The DLPFC notebook uses ``morphology_weight=0`` so only
    transcriptomics drives the embedding. We exercise that pathway
    end-to-end on the simulated dataset.
    """
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(SEED)
    adata = _prepare_simulated(simulated_adata)

    model = AESTETIK(
        n_cluster=adata.obs.ground_truth.unique().size,
        morphology_weight=0,
        refine_cluster=True,
        window_size=7,
        clustering_method="kmeans",
        random_state=SEED,
        n_jobs=1,
        num_workers=0,
        **FAST_CFG,
    )
    model.fit(adata)
    labels = model.predict(adata)
    ari = _ari(adata.obs.ground_truth.values, labels)
    assert ari > ARI_THRESHOLD, (
        f"transcriptomics-only ARI dropped to {ari:.3f} (threshold {ARI_THRESHOLD})."
    )


def test_validation_split_runs_early_stopping(simulated_adata):
    """validation_split > 0 enables EarlyStopping; the loss history
    should still be populated.
    """
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(SEED)
    adata = _prepare_simulated(simulated_adata)
    model = AESTETIK(
        n_cluster=3,
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
        validation_split=0.2,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        random_state=SEED,
        n_jobs=1,
        num_workers=0,
    )
    model.fit(adata)
    assert len(model.losses_) >= 1


def test_fit_does_not_mutate_input(simulated_adata):
    """sklearn convention: X is read-only across fit / transform / predict."""
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(SEED)
    adata = _prepare_simulated(simulated_adata)

    obsm_before = set(adata.obsm.keys())
    obs_before = set(adata.obs.columns)

    model = AESTETIK(
        n_cluster=3,
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
        random_state=SEED,
        n_jobs=1,
        num_workers=0,
    )
    model.fit(adata)
    model.transform(adata)
    model.predict(adata)

    assert set(adata.obsm.keys()) == obsm_before
    assert set(adata.obs.columns) == obs_before
