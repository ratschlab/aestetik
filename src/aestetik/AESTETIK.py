"""sklearn-style estimator wrapping the AESTETIK multimodal autoencoder."""
from __future__ import annotations

import logging
import os
from typing import List, Literal, Optional, Union

import anndata
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from aestetik.callbacks.callbacks import LossHistoryCallback
from aestetik.data_modules.data_module import AESTETIKDataModule
from aestetik.modules.aestetik_module import AESTETIKModel
from aestetik.utils.utils_clustering import clustering
from aestetik.utils.utils_data import build_grid
from aestetik.utils.utils_grid import fix_seed

logger = logging.getLogger(__name__)


class AESTETIK(TransformerMixin, ClusterMixin, BaseEstimator):
    """sklearn-compatible multimodal autoencoder for spatial transcriptomics.

    The estimator follows the standard scikit-learn ``fit`` / ``transform`` /
    ``predict`` / ``fit_transform`` / ``fit_predict`` surface. Inputs are
    :class:`anndata.AnnData` objects with pre-computed transcriptomics and
    morphology embeddings in ``adata.obsm`` and spatial coordinates
    (``x_array``, ``y_array``) in ``adata.obs``.

    ``fit``, ``transform`` and ``predict`` are read-only on ``X``. Fitted
    attributes (with the customary trailing underscore) expose the trained
    state so callers can attach results to the AnnData themselves if they
    want::

        model = AESTETIK(morphology_weight=1.5, n_cluster=3)
        model.fit(adata)
        adata.obsm["AESTETIK"] = model.embedding_
        adata.obs["AESTETIK_cluster"] = model.labels_

    Parameters
    ----------
    morphology_weight : float, default=1.5
        Weight assigned to the morphology modality in the joint loss.
    n_cluster : int or float, default=7
        If int, target number of clusters for the kmeans / bgm methods.
        If float, used as the resolution parameter for leiden / louvain.
    total_weight : float, default=3.0
        Sum target for transcriptomics + morphology weight after calibration.
    rec_alpha : float, default=1.0
        Reconstruction loss coefficient.
    triplet_alpha : float, default=1.0
        Triplet loss coefficient.
    train_size : float, optional
        If given, the fraction of spots kept after the per-call train/test
        split inside :class:`~aestetik.dataloader.CustomDataset`.
    window_size : int, default=7
        Side length (odd) of the neighborhood grid used as input to the CNN.
    kernel_size : int, default=3
        CNN kernel size.
    latent_dim : int, default=16
        Latent embedding dimensionality.
    c_hid : int, default=64
        Number of CNN hidden channels.
    lr : float, default=1e-3
        Optimizer learning rate.
    p : float, default=0.3
        Dropout probability.
    max_epochs : int, default=100
        Lightning ``max_epochs``.
    multi_triplet_loss : bool, default=True
        Whether to use the multi-class triplet variant.
    n_repeats : int, default=1
        Number of positive / negative repeats per anchor for the triplet loss.
    clustering_method : {"bgm", "kmeans", "louvain", "leiden"}, default="bgm"
        Clustering algorithm applied to the embedding.
    batch_size : int, optional
        DataLoader batch size; defaults to ``min(2**13, len(X))`` at fit time.
    n_ensemble : int, default=3
        Default number of encoder/decoder ensemble members.
    n_ensemble_encoder, n_ensemble_decoder : int, optional
        Per-side ensemble overrides.
    weight_decay : float, default=1e-6
        Adam ``weight_decay``.
    refine_cluster : bool, default=True
        Whether to spatially refine cluster assignments using KNN.
    n_neighbors : int, default=10
        Number of neighbours used by the spatial refinement step.
    n_jobs : int, default=1
        Parallelism for grid construction. ``-1`` uses all CPUs.
    num_workers : int, default=0
        DataLoader worker count.
    used_obsm_transcriptomics : str, default="X_pca_transcriptomics"
        Key for transcriptomics features in ``adata.obsm``.
    used_obsm_morphology : str, default="X_pca_morphology"
        Key for morphology features in ``adata.obsm``.
    used_obsm_combined : str, default="X_pca"
        Key under which the combined modality is stored (used internally
        when building the grid).
    used_obs_batch : str, optional
        Column in ``adata.obs`` containing sample / batch labels.
    validation_split : float, default=0.0
        Fraction of training data held out for Lightning validation.
    early_stopping_patience : int, default=5
        Patience for ``EarlyStopping`` (only used when ``validation_split > 0``).
    early_stopping_min_delta : float, default=0.0
        Minimum delta for ``EarlyStopping``.
    num_repeats_predict : int, default=1000
        Number of stochastic forward passes averaged when computing the
        embedding in ``transform`` / ``predict``.
    random_state : int, default=2023
        Seed used for all RNGs.

    Attributes
    ----------
    model_ : aestetik.modules.aestetik_module.AESTETIKModel
        The trained Lightning module.
    trainer_ : lightning.pytorch.Trainer
        The Lightning trainer used to fit the model.
    losses_ : list of float
        Per-batch training loss values.
    embedding_ : ndarray of shape (n_obs, latent_dim)
        Latent embedding computed on the training data.
    labels_ : ndarray of shape (n_obs,)
        Cluster labels computed on the training data.
    transcriptomics_weight_, morphology_weight_ : float
        Post-calibration modality weights.
    obsm_transcriptomics_dim_ : int
        Width of the transcriptomics obsm seen at fit time.
    num_input_channels_ : int
        Total grid channels (transcriptomics + morphology).
    """

    def __init__(
        self,
        morphology_weight: float = 1.5,
        n_cluster: Union[int, float] = 7,
        total_weight: float = 3.0,
        rec_alpha: float = 1.0,
        triplet_alpha: float = 1.0,
        train_size: Optional[float] = None,
        window_size: int = 7,
        kernel_size: int = 3,
        latent_dim: int = 16,
        c_hid: int = 64,
        lr: float = 1e-3,
        p: float = 0.3,
        max_epochs: int = 100,
        multi_triplet_loss: bool = True,
        n_repeats: int = 1,
        clustering_method: Literal["bgm", "kmeans", "louvain", "leiden"] = "bgm",
        batch_size: Optional[int] = None,
        n_ensemble: int = 3,
        n_ensemble_encoder: Optional[int] = None,
        n_ensemble_decoder: Optional[int] = None,
        weight_decay: float = 1e-6,
        refine_cluster: bool = True,
        n_neighbors: int = 10,
        n_jobs: int = 1,
        num_workers: int = 0,
        used_obsm_transcriptomics: str = "X_pca_transcriptomics",
        used_obsm_morphology: str = "X_pca_morphology",
        used_obsm_combined: str = "X_pca",
        used_obs_batch: Optional[str] = None,
        validation_split: float = 0.0,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.0,
        num_repeats_predict: int = 1000,
        random_state: int = 2023,
    ):
        # sklearn convention: __init__ only stores params verbatim.
        self.morphology_weight = morphology_weight
        self.n_cluster = n_cluster
        self.total_weight = total_weight
        self.rec_alpha = rec_alpha
        self.triplet_alpha = triplet_alpha
        self.train_size = train_size
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.c_hid = c_hid
        self.lr = lr
        self.p = p
        self.max_epochs = max_epochs
        self.multi_triplet_loss = multi_triplet_loss
        self.n_repeats = n_repeats
        self.clustering_method = clustering_method
        self.batch_size = batch_size
        self.n_ensemble = n_ensemble
        self.n_ensemble_encoder = n_ensemble_encoder
        self.n_ensemble_decoder = n_ensemble_decoder
        self.weight_decay = weight_decay
        self.refine_cluster = refine_cluster
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.num_workers = num_workers
        self.used_obsm_transcriptomics = used_obsm_transcriptomics
        self.used_obsm_morphology = used_obsm_morphology
        self.used_obsm_combined = used_obsm_combined
        self.used_obs_batch = used_obs_batch
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.num_repeats_predict = num_repeats_predict
        self.random_state = random_state

    # ----------------------------- sklearn API -----------------------------

    def fit(self, X: anndata.AnnData, y=None) -> "AESTETIK":
        """Train the model on ``X``.

        Returns ``self`` (sklearn convention). The training-set
        embedding and cluster labels are cached on ``self.embedding_``
        and ``self.labels_``; ``X`` itself is not modified.
        """
        self._validate_params()
        self._validate_anndata(X, method="fit")
        fix_seed(self.random_state)

        # Work on a private copy so the public X is never mutated.
        adata = X.copy()

        kernel_size = (
            self.kernel_size
            if self.kernel_size < self.window_size
            else max(1, self.window_size - 1)
        )
        n_enc = self.n_ensemble_encoder or self.n_ensemble
        n_dec = self.n_ensemble_decoder or self.n_ensemble
        batch_size = self.batch_size or min(2 ** 13, adata.n_obs)
        n_jobs = self.n_jobs if self.n_jobs != -1 else (os.cpu_count() or 1)
        obsm_transcriptomics_dim = adata.obsm[self.used_obsm_transcriptomics].shape[1]

        grid_params = {
            "morphology_dim": self.window_size,
            "num_input_channels": None,
            "obsm_transcriptomics_dim": obsm_transcriptomics_dim,
        }
        model_architecture_params = {
            "latent_dim": self.latent_dim,
            "c_hid": self.c_hid,
            "kernel_size": kernel_size,
            "p": self.p,
            "n_ensemble_encoder": n_enc,
            "n_ensemble_decoder": n_dec,
        }
        dataloader_params = {"batch_size": batch_size, "num_workers": self.num_workers}
        clustering_params = {
            "nCluster": self.n_cluster,
            "clustering_method": self.clustering_method,
            "n_neighbors": self.n_neighbors,
            "refine_cluster": self.refine_cluster and self.n_neighbors > 1,
        }
        loss_regularization_params = {
            "multi_triplet_loss": self.multi_triplet_loss,
            "rec_alpha": self.rec_alpha,
            "triplet_alpha": self.triplet_alpha,
            "n_repeats": self.n_repeats,
            "morphology_weight": self.morphology_weight,
            "transcriptomics_weight": None,
            "total_weight": self.total_weight,
        }
        data_handling_params = {"n_jobs": n_jobs, "train_size": self.train_size}

        logger.info("Initializing datamodule ...")
        datamodule = AESTETIKDataModule(
            adata=adata,
            used_obsm_transcriptomics=self.used_obsm_transcriptomics,
            used_obsm_morphology=self.used_obsm_morphology,
            used_obsm_combined=self.used_obsm_combined,
            used_obs_batch=self.used_obs_batch,
            dataloader_params=dataloader_params,
            clustering_params=clustering_params,
            grid_params=grid_params,
            loss_regularization_params=loss_regularization_params,
            data_handling_params=data_handling_params,
            validation_split=self.validation_split,
        )

        training_step_params = {"rec_alpha": self.rec_alpha, "triplet_alpha": self.triplet_alpha}
        optimizer_step_params = {"lr": self.lr, "weight_decay": self.weight_decay}
        lit_model = AESTETIKModel(
            datamodule=datamodule,
            grid_params=grid_params,
            model_architecture_params=model_architecture_params,
            training_params=training_step_params,
            optimizer_params=optimizer_step_params,
        )

        callbacks = self._create_callbacks()
        logger.info("Fit AESTETIKModel ...")
        trainer = Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
        trainer.fit(lit_model, datamodule=datamodule)

        # Store fitted state.
        self.model_ = lit_model
        self.trainer_ = trainer
        self.losses_ = list(callbacks[0].losses)
        self.transcriptomics_weight_ = datamodule.loss_regularization_params["transcriptomics_weight"]
        self.morphology_weight_ = datamodule.loss_regularization_params["morphology_weight"]
        self.obsm_transcriptomics_dim_ = obsm_transcriptomics_dim
        self.num_input_channels_ = datamodule.adata.obsm["X_st_grid"].shape[1]
        self.grid_params_ = {
            "morphology_dim": self.window_size,
            "num_input_channels": self.num_input_channels_,
            "obsm_transcriptomics_dim": self.obsm_transcriptomics_dim_,
        }

        # Cache embedding + labels on the (already-built) training grid.
        self.embedding_ = self._compute_latent_space(datamodule.adata, built_grid=True)
        self.labels_ = self._cluster_embedding(datamodule.adata, self.embedding_)

        return self

    def transform(self, X: anndata.AnnData) -> np.ndarray:
        """Return the latent embedding for ``X`` as a (n_obs, latent_dim) array."""
        check_is_fitted(self, ["model_", "trainer_"])
        self._validate_anndata(X, method="transform")
        adata = X.copy()
        self._calibrate_predict_inputs(adata)
        return self._compute_latent_space(adata, built_grid=False)

    def predict(self, X: anndata.AnnData) -> np.ndarray:
        """Return cluster labels for ``X`` as a (n_obs,) array."""
        check_is_fitted(self, ["model_", "trainer_"])
        self._validate_anndata(X, method="predict")
        adata = X.copy()
        self._calibrate_predict_inputs(adata)
        embedding = self._compute_latent_space(adata, built_grid=False)
        return self._cluster_embedding(adata, embedding)

    @staticmethod
    def version() -> str:
        return "0.3.0"

    # --------------------------- Internal helpers --------------------------

    def _validate_params(self) -> None:
        if self.window_size % 2 == 0:
            raise ValueError("window_size should be an odd integer")
        if self.total_weight < self.morphology_weight:
            raise ValueError(
                f"total_weight ({self.total_weight}) must be >= morphology_weight ({self.morphology_weight})"
            )
        if self.validation_split < 0 or self.validation_split >= 1:
            raise ValueError(f"validation_split must be in [0, 1); got {self.validation_split}")

    def _validate_anndata(self, X: anndata.AnnData, method: str) -> None:
        self._require_obsm(X, [self.used_obsm_transcriptomics, self.used_obsm_morphology], method)
        self._require_obs(X, ["x_array", "y_array"], method)
        if method in ("transform", "predict"):
            t_dim = X.obsm[self.used_obsm_transcriptomics].shape[1]
            m_dim = X.obsm[self.used_obsm_morphology].shape[1]
            m_target = self.num_input_channels_ - self.obsm_transcriptomics_dim_
            if t_dim < self.obsm_transcriptomics_dim_ or m_dim < m_target:
                raise ValueError(
                    "Dimensionality of obsm transcriptomics or morphology features is too small. "
                    f"Transcriptomics dim: {t_dim} (required >= {self.obsm_transcriptomics_dim_}); "
                    f"Morphology dim: {m_dim} (required >= {m_target})."
                )

    @staticmethod
    def _require_obsm(X: anndata.AnnData, keys: List[str], method: str) -> None:
        missing = [k for k in keys if k not in X.obsm]
        if missing:
            raise KeyError(
                f"AESTETIK.{method}: required keys {missing} must be present in X.obsm. "
                f"Available keys: {list(X.obsm.keys())}"
            )

    @staticmethod
    def _require_obs(X: anndata.AnnData, columns: List[str], method: str) -> None:
        missing = [c for c in columns if c not in X.obs]
        if missing:
            raise KeyError(
                f"AESTETIK.{method}: required columns {missing} must be present in X.obs. "
                f"Available columns: {list(X.obs.columns)}"
            )

    def _calibrate_predict_inputs(self, adata: anndata.AnnData) -> None:
        """Truncate obsm to match the dims seen at fit time."""
        m_target = self.num_input_channels_ - self.obsm_transcriptomics_dim_
        if adata.obsm[self.used_obsm_transcriptomics].shape[1] > self.obsm_transcriptomics_dim_:
            logger.info("Truncating %s to fit-time dim %d",
                        self.used_obsm_transcriptomics, self.obsm_transcriptomics_dim_)
            adata.obsm[self.used_obsm_transcriptomics] = adata.obsm[
                self.used_obsm_transcriptomics
            ][:, : self.obsm_transcriptomics_dim_]
        if adata.obsm[self.used_obsm_morphology].shape[1] > m_target:
            logger.info("Truncating %s to fit-time dim %d",
                        self.used_obsm_morphology, m_target)
            adata.obsm[self.used_obsm_morphology] = adata.obsm[
                self.used_obsm_morphology
            ][:, :m_target]

    def _create_callbacks(self) -> list:
        callbacks: list = [LossHistoryCallback()]
        if self.validation_split > 0.0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=self.early_stopping_patience,
                    min_delta=self.early_stopping_min_delta,
                )
            )
            callbacks.append(
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    filename="best-checkpoint",
                )
            )
        return callbacks

    def _compute_latent_space(self, adata: anndata.AnnData, built_grid: bool) -> np.ndarray:
        n_jobs = self.n_jobs if self.n_jobs != -1 else (os.cpu_count() or 1)
        if not built_grid:
            build_grid(
                adata,
                used_obsm_transcriptomics=self.used_obsm_transcriptomics,
                used_obsm_morphology=self.used_obsm_morphology,
                used_obs_batch=self.used_obs_batch,
                window_size=self.window_size,
                n_jobs=n_jobs,
            )

        batch_size = self.batch_size or min(2 ** 13, adata.n_obs)
        all_spots = torch.from_numpy(adata.obsm["X_st_grid"].astype(np.float32))
        dataset = TensorDataset(all_spots)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.model_.predict_params["num_repeats"] = self.num_repeats_predict
        chunks = self.trainer_.predict(self.model_, dataloaders=loader)
        return torch.cat(chunks, dim=0).cpu().numpy()

    def _cluster_embedding(
        self, adata: anndata.AnnData, embedding: np.ndarray
    ) -> np.ndarray:
        """Run the clustering algorithm on ``embedding`` and return labels.

        The helper writes the embedding to ``adata.obsm`` temporarily so
        the existing ``clustering`` utility (which expects a key in
        ``obsm``) can consume it. ``adata`` here is a private copy so
        this mutation is not user-visible.
        """
        emb_key = "_aestetik_emb"
        adata.obsm[emb_key] = embedding
        clustering(
            adata,
            num_cluster=self.n_cluster,
            used_obsm=emb_key,
            method=self.clustering_method,
            refine_cluster=self.refine_cluster and self.n_neighbors > 1,
            n_neighbors=self.n_neighbors,
            used_obs_batch=self.used_obs_batch,
        )
        return adata.obs[f"{emb_key}_cluster"].to_numpy()
