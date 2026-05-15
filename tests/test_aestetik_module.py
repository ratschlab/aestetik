"""Tests for ``aestetik.modules.aestetik_module.AESTETIKModel``.

We construct a real ``AESTETIKDataModule`` so that ``configure_model``
has a populated ``X_st_grid`` to read its input-channel count from.
"""
from __future__ import annotations

import pytest
import torch

from aestetik.data_modules.data_module import AESTETIKDataModule
from aestetik.modules.aestetik_module import AESTETIKModel


def _build_datamodule(adata):
    return AESTETIKDataModule(
        adata=adata,
        validation_split=0.2,
        used_obsm_transcriptomics="X_pca_transcriptomics",
        used_obsm_morphology="X_pca_morphology",
        used_obsm_combined="X_pca",
        used_obs_batch=None,
        dataloader_params={"batch_size": 4, "num_workers": 0},
        clustering_params={
            "nCluster": 3,
            "clustering_method": "kmeans",
            "n_neighbors": 3,
            "refine_cluster": False,
        },
        grid_params={
            "morphology_dim": 3,
            "num_input_channels": None,
            "obsm_transcriptomics_dim": adata.obsm["X_pca_transcriptomics"].shape[1],
        },
        loss_regularization_params={
            "multi_triplet_loss": True,
            "n_repeats": 1,
            "morphology_weight": 1.5,
            "transcriptomics_weight": None,
            "rec_alpha": 1.0,
            "triplet_alpha": 1.0,
            "total_weight": 3.0,
        },
        data_handling_params={"n_jobs": 1, "train_size": None},
    )


def _build_model(datamodule):
    return AESTETIKModel(
        datamodule=datamodule,
        grid_params={
            "morphology_dim": 3,
            "num_input_channels": None,
            "obsm_transcriptomics_dim": datamodule.adata.obsm["X_pca_transcriptomics"].shape[1],
        },
        model_architecture_params={
            "latent_dim": 4,
            "c_hid": 8,
            "kernel_size": 2,
            "p": 0.0,
            "n_ensemble_encoder": 1,
            "n_ensemble_decoder": 1,
        },
        training_params={"rec_alpha": 1.0, "triplet_alpha": 1.0},
        optimizer_params={"lr": 1e-3, "weight_decay": 0.0},
    )


@pytest.fixture
def ready_model(small_adata):
    dm = _build_datamodule(small_adata)
    dm.setup("fit")
    model = _build_model(dm)
    model.setup("fit")
    model.configure_model()
    return model, dm


def test_setup_attaches_losses_and_weights(ready_model):
    model, _ = ready_model
    assert "rec_loss" in model.loss
    assert "triplet_loss" in model.loss
    assert "transcriptomics_weight" in model.weights
    assert "morphology_weight" in model.weights


def test_configure_model_builds_ae_once(ready_model):
    model, _ = ready_model
    assert model.model_built is True
    first = id(model.model)
    # Idempotent
    model.configure_model()
    assert id(model.model) == first


def test_configure_optimizers_returns_adam(ready_model):
    model, _ = ready_model
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["amsgrad"] is True


def test_predict_step_requires_num_repeats(ready_model):
    model, _ = ready_model
    batch_spots = torch.from_numpy(model.datamodule.adata.obsm["X_st_grid"][:2].astype("float32"))
    with pytest.raises(TypeError, match="num_repeats"):
        model.predict_step((batch_spots,), batch_idx=0)


def test_predict_step_returns_mean_embedding(ready_model):
    model, _ = ready_model
    model.predict_params["num_repeats"] = 3
    batch_spots = torch.from_numpy(model.datamodule.adata.obsm["X_st_grid"][:2].astype("float32"))
    z = model.predict_step((batch_spots,), batch_idx=0)
    assert z.shape == (2, 4)
    assert torch.isfinite(z).all()


def test_training_step_returns_finite_scalar(ready_model):
    model, dm = ready_model
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    loss = model.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_finite_scalar(ready_model):
    model, dm = ready_model
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    loss = model.validation_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validate_params_rejects_missing_key(small_adata):
    dm = _build_datamodule(small_adata)
    dm.setup("fit")
    with pytest.raises(ValueError, match="rec_alpha"):
        AESTETIKModel(
            datamodule=dm,
            grid_params={
                "morphology_dim": 3,
                "num_input_channels": None,
                "obsm_transcriptomics_dim": small_adata.obsm["X_pca_transcriptomics"].shape[1],
            },
            model_architecture_params={
                "latent_dim": 4,
                "c_hid": 8,
                "kernel_size": 2,
                "p": 0.0,
                "n_ensemble_encoder": 1,
                "n_ensemble_decoder": 1,
            },
            training_params={"triplet_alpha": 1.0},  # missing rec_alpha
            optimizer_params={"lr": 1e-3, "weight_decay": 0.0},
        )
