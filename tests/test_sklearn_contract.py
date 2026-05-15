"""Tests asserting that ``AESTETIK`` honours the scikit-learn estimator
contract: clone round-trip, get/set_params symmetry, fit_predict ==
labels_, and the modality-only cluster attributes added in 0.3.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.base import clone

from aestetik import AESTETIK
from aestetik.utils.utils_grid import fix_seed


def _make_model(**overrides) -> AESTETIK:
    params = dict(
        n_cluster=3,
        morphology_weight=1.5,
        window_size=3,
        kernel_size=3,
        latent_dim=4,
        c_hid=8,
        max_epochs=1,
        clustering_method="kmeans",
        n_ensemble=1,
        n_neighbors=3,
        refine_cluster=False,
        n_jobs=1,
        num_workers=0,
        num_repeats_predict=2,
    )
    params.update(overrides)
    return AESTETIK(**params)


# ----------------------------- Static contract -----------------------------


def test_clone_round_trips_params():
    model = _make_model()
    clone_ = clone(model)
    assert clone_.get_params() == model.get_params()
    # Cloned estimators must not share state objects (sklearn invariant).
    assert clone_ is not model


def test_set_params_then_clone_preserves_overrides():
    model = _make_model().set_params(n_cluster=11)
    assert clone(model).n_cluster == 11


def test_repr_does_not_explode():
    repr(_make_model())  # repr() must work on un-fitted estimators


def test_get_params_does_not_leak_fitted_attrs(small_adata):
    """Trailing-underscore attrs must never appear in get_params()."""
    logging.getLogger("lightning").setLevel(logging.ERROR)
    model = _make_model().fit(small_adata)
    params = model.get_params()
    leaked = [k for k in params if k.endswith("_")]
    assert not leaked, f"fitted attrs leaked into get_params: {leaked}"


# ----------------------------- ClusterMixin contract ----------------------


@pytest.mark.slow
def test_fit_predict_returns_labels_attr(small_adata):
    """ClusterMixin.fit_predict must return ``self.labels_``."""
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(0)
    model = _make_model()
    labels = model.fit_predict(small_adata)
    np.testing.assert_array_equal(labels, model.labels_)


@pytest.mark.slow
def test_modality_cluster_attrs_present(small_adata):
    """After fit the modality-only cluster baselines are exposed as
    ``transcriptomics_cluster_`` / ``morphology_cluster_`` (added in 0.3
    to replace the old AnnData side effects).
    """
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(0)
    model = _make_model().fit(small_adata)
    assert hasattr(model, "transcriptomics_cluster_")
    assert hasattr(model, "morphology_cluster_")
    assert model.transcriptomics_cluster_.shape == (small_adata.n_obs,)
    assert model.morphology_cluster_.shape == (small_adata.n_obs,)


@pytest.mark.slow
def test_fit_returns_self(small_adata):
    logging.getLogger("lightning").setLevel(logging.ERROR)
    fix_seed(0)
    model = _make_model()
    assert model.fit(small_adata) is model
