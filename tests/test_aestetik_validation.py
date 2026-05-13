"""Tests for the validation surface of ``aestetik.AESTETIK``.

Exercises the private helpers (`_check_fitted`, `_validate_obsm_keys`,
`_validate_obs_columns`, `_calibrate_predict_inputs`,
`_create_early_stopping_params`) without needing to spin up Lightning.
"""
from __future__ import annotations

import numpy as np
import pytest

from aestetik import AESTETIK


def _make_model(**overrides) -> AESTETIK:
    params = dict(
        nCluster=3,
        morphology_weight=1.5,
        total_weight=3.0,
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


def test_invalid_window_size_raises() -> None:
    with pytest.raises(ValueError, match="window_size should be an odd integer"):
        _make_model(window_size=4)


def test_predict_before_fit_raises() -> None:
    model = _make_model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model._check_fitted()


def test_validate_obsm_keys_reports_missing(small_adata) -> None:
    model = _make_model()
    # Drop a required obsm key.
    small_adata.obsm.pop("X_pca_morphology")
    with pytest.raises(KeyError) as exc_info:
        model._validate_fit_inputs(
            X=small_adata,
            used_obsm_transcriptomics="X_pca_transcriptomics",
            used_obsm_morphology="X_pca_morphology",
        )
    msg = str(exc_info.value)
    assert "X_pca_morphology" in msg
    assert "fit" in msg


def test_validate_obs_columns_reports_missing(small_adata) -> None:
    model = _make_model()
    del small_adata.obs["x_array"]
    with pytest.raises(KeyError, match="x_array"):
        model._validate_fit_inputs(
            X=small_adata,
            used_obsm_transcriptomics="X_pca_transcriptomics",
            used_obsm_morphology="X_pca_morphology",
        )


def test_calibrate_predict_inputs_trims_extra_dims(small_adata) -> None:
    model = _make_model()
    # Pretend fit has been called.
    model.grid_params["obsm_transcriptomics_dim"] = 3
    model.grid_params["num_input_channels"] = 6
    # Pad the obsm arrays to be larger than required.
    small_adata.obsm["X_pca_transcriptomics"] = np.random.randn(small_adata.n_obs, 8).astype(np.float32)
    small_adata.obsm["X_pca_morphology"] = np.random.randn(small_adata.n_obs, 8).astype(np.float32)
    model._calibrate_predict_inputs(
        small_adata, "X_pca_transcriptomics", "X_pca_morphology"
    )
    assert small_adata.obsm["X_pca_transcriptomics"].shape[1] == 3
    assert small_adata.obsm["X_pca_morphology"].shape[1] == 3  # 6 - 3


def test_validate_predict_rejects_undersized_features(small_adata) -> None:
    model = _make_model()
    model.grid_params["obsm_transcriptomics_dim"] = 4
    model.grid_params["num_input_channels"] = 8
    # Make obsm too small to match the model.
    small_adata.obsm["X_pca_transcriptomics"] = np.zeros((small_adata.n_obs, 2), dtype=np.float32)
    small_adata.obsm["X_pca_morphology"] = np.zeros((small_adata.n_obs, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="too small"):
        model._validate_predict_inputs(
            X=small_adata,
            used_obsm_transcriptomics="X_pca_transcriptomics",
            used_obsm_morphology="X_pca_morphology",
        )


def test_early_stopping_params_strip_forbidden_keys() -> None:
    model = _make_model()
    user_params = {"monitor": "trying_to_override", "patience": 9, "min_delta": 0.01}
    merged = model._create_early_stopping_params(user_params)
    assert merged["monitor"] == "val_loss"
    assert merged["mode"] == "min"
    assert merged["patience"] == 9
    assert merged["min_delta"] == 0.01


def test_early_stopping_params_defaults_when_none() -> None:
    model = _make_model()
    merged = model._create_early_stopping_params(None)
    assert merged == {"monitor": "val_loss", "mode": "min", "patience": 5}


def test_version_is_string() -> None:
    assert isinstance(AESTETIK.version(), str)
