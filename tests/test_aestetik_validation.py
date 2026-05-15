"""Tests for the validation surface of the sklearn-style ``AESTETIK``
estimator.

These tests poke the lightweight validation paths (param checks, obsm /
obs presence checks, fit-vs-predict dim guards, sklearn ``check_is_fitted``
behavior). They never run Lightning training so they live in the fast
suite.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from aestetik import AESTETIK


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
    )
    params.update(overrides)
    return AESTETIK(**params)


# ----------------------------- sklearn surface -----------------------------


def test_default_constructor_has_no_required_args():
    """sklearn contract: estimators must be constructible with no args."""
    AESTETIK()


def test_get_params_returns_constructor_args():
    model = _make_model()
    params = model.get_params()
    assert params["morphology_weight"] == 1.5
    assert params["n_cluster"] == 3
    assert params["window_size"] == 3
    # Underscore-suffix attributes (fitted state) must NOT appear in get_params:
    assert "embedding_" not in params
    assert "labels_" not in params


def test_set_params_round_trips():
    model = _make_model()
    model.set_params(n_cluster=5, morphology_weight=2.0)
    assert model.n_cluster == 5
    assert model.morphology_weight == 2.0


def test_predict_before_fit_raises_not_fitted(small_adata):
    model = _make_model()
    with pytest.raises(NotFittedError):
        check_is_fitted(model, ["model_", "trainer_"])
    with pytest.raises(NotFittedError):
        model.transform(small_adata)
    with pytest.raises(NotFittedError):
        model.predict(small_adata)


# ----------------------------- Parameter validation -----------------------


def test_invalid_window_size_raises_on_fit(small_adata):
    """Param validation happens in fit (per sklearn convention), not __init__."""
    model = _make_model(window_size=4)
    with pytest.raises(ValueError, match="window_size should be an odd integer"):
        model.fit(small_adata)


def test_total_weight_below_morphology_weight_raises(small_adata):
    model = _make_model(morphology_weight=5.0, total_weight=2.0)
    with pytest.raises(ValueError, match="total_weight"):
        model.fit(small_adata)


def test_validation_split_must_be_in_unit_interval(small_adata):
    with pytest.raises(ValueError, match="validation_split"):
        _make_model(validation_split=1.5).fit(small_adata)
    with pytest.raises(ValueError, match="validation_split"):
        _make_model(validation_split=-0.1).fit(small_adata)


# ----------------------------- AnnData validation -------------------------


def test_fit_reports_missing_obsm_key(small_adata):
    model = _make_model()
    small_adata.obsm.pop("X_pca_morphology")
    with pytest.raises(KeyError) as exc_info:
        model.fit(small_adata)
    msg = str(exc_info.value)
    assert "X_pca_morphology" in msg
    assert "fit" in msg


def test_fit_reports_missing_obs_column(small_adata):
    model = _make_model()
    del small_adata.obs["x_array"]
    with pytest.raises(KeyError, match="x_array"):
        model.fit(small_adata)


# ----------------------------- Predict-side calibration -------------------


def test_calibrate_predict_inputs_truncates_extra_dims(small_adata):
    model = _make_model()
    # Pretend fit completed:
    model.obsm_transcriptomics_dim_ = 3
    model.num_input_channels_ = 6
    rng = np.random.default_rng(0)
    small_adata.obsm["X_pca_transcriptomics"] = rng.standard_normal((small_adata.n_obs, 8)).astype(np.float32)
    small_adata.obsm["X_pca_morphology"] = rng.standard_normal((small_adata.n_obs, 8)).astype(np.float32)
    model._calibrate_predict_inputs(small_adata)
    assert small_adata.obsm["X_pca_transcriptomics"].shape[1] == 3
    assert small_adata.obsm["X_pca_morphology"].shape[1] == 3  # 6 - 3


def test_validate_anndata_rejects_undersized_features(small_adata):
    model = _make_model()
    model.obsm_transcriptomics_dim_ = 4
    model.num_input_channels_ = 8
    small_adata.obsm["X_pca_transcriptomics"] = np.zeros((small_adata.n_obs, 2), dtype=np.float32)
    small_adata.obsm["X_pca_morphology"] = np.zeros((small_adata.n_obs, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="too small"):
        model._validate_anndata(small_adata, method="predict")


# ----------------------------- Misc ---------------------------------------


def test_version_is_string() -> None:
    import aestetik
    assert isinstance(aestetik.__version__, str)
    # Must be PEP 440-ish (digit-dotted) or the sentinel.
    assert aestetik.__version__ == "0.0.0+unknown" or aestetik.__version__[0].isdigit()
