"""Tests for ``aestetik.data_modules.data_module.AESTETIKDataModule``.

Covers issue #8: ``train_dataloader`` must use the *split*
``train_dataset`` when ``validation_split > 0``.
"""
import numpy as np
import pytest

from aestetik.data_modules.data_module import AESTETIKDataModule


def _build_datamodule(adata, validation_split: float):
    return AESTETIKDataModule(
        adata=adata,
        validation_split=validation_split,
        used_obsm_transcriptomics="X_pca_transcriptomics",
        used_obsm_morphology="X_pca_morphology",
        used_obsm_combined="X_pca",
        used_obs_batch=None,
        dataloader_params={"batch_size": 2, "num_workers": 0},
        clustering_params={
            "nCluster": 3,
            "clustering_method": "kmeans",
            "n_neighbors": 5,
            "refine_cluster": False,
        },
        grid_params={"morphology_dim": 3,
                     "num_input_channels": None,
                     "obsm_transcriptomics_dim": None},
        loss_regularization_params={
            "multi_triplet_loss": True,
            "n_repeats": 1,
            "morphology_weight": 1.0,
            "transcriptomics_weight": None,
            "rec_alpha": 1.0,
            "triplet_alpha": 1.0,
            "total_weight": 3.0,
        },
        data_handling_params={"n_jobs": 1, "train_size": None},
    )


def test_train_dataloader_respects_validation_split(small_adata):
    """train_dataloader must iterate over only the training portion
    after random_split, not the full dataset (issue #8).
    """
    dm = _build_datamodule(small_adata, validation_split=0.25)
    dm.setup(stage="fit")
    total = len(dm.dataset)
    train_len = len(dm.train_dataset)
    val_len = len(dm.val_dataset)
    assert train_len + val_len == total
    assert val_len > 0
    assert train_len < total

    train_loader = dm.train_dataloader()
    seen = sum(b[0].shape[0] for b in train_loader)
    assert seen == train_len, (
        f"train_dataloader yielded {seen} samples but train_dataset has {train_len} "
        "- regression of issue #8."
    )


def test_train_dataloader_no_validation_split(small_adata):
    """validation_split == 0 -> train_dataset is the full dataset."""
    dm = _build_datamodule(small_adata, validation_split=0.0)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    seen = sum(b[0].shape[0] for b in train_loader)
    assert seen == len(dm.dataset)
    assert dm.val_dataset is None
