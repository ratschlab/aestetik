import logging
import os
from typing import Dict, List, Literal, Optional, Union

import anndata
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from aestetik.callbacks.callbacks import LossHistoryCallback
from aestetik.data_modules.data_module import AESTETIKDataModule
from aestetik.modules.aestetik_module import AESTETIKModel
from aestetik.utils.utils_clustering import clustering
from aestetik.utils.utils_data import build_grid
from aestetik.utils.utils_grid import fix_seed

logger = logging.getLogger(__name__)


class AESTETIK:

    # ================================================================= #
    #                       Initialization                              #
    # ================================================================= #
    def __init__(
        self,
        nCluster: Union[int, float],
        morphology_weight: float,
        total_weight: float = 3,
        rec_alpha: float = 1,
        triplet_alpha: float = 1,
        train_size: Optional[float] = None,
        window_size: int = 7,
        kernel_size: int = 3,
        latent_dim: int = 16,
        c_hid: int = 64,
        lr: float = 0.001,
        p: float = 0.3,
        max_epochs: int = 100,
        multi_triplet_loss: bool = True,
        n_repeats: int = 1,
        clustering_method: Literal["bgm", "kmeans", "louvain", "leiden"] = "bgm",
        batch_size: Optional[int] = None,
        n_ensemble: int = 3,
        n_ensemble_encoder: Optional[int] = None,
        n_ensemble_decoder: Optional[int] = None,
        random_seed: int = 2023,
        n_neighbors: int = 10,
        weight_decay: float = 1e-6,
        refine_cluster: bool = True,
        n_jobs: int = 1,
        num_workers: int = 0
        ):
        """
        Initialize the model with the given parameters.

        Parameters
        ----------
        nCluster : Union[int, float]
            if int: Number of clusters.
            if float: Resolution parameter in leiden and louvain.
        morphology_weight : float
            Weight for the morphology modality.
        total_weight : float, optional (default=3)
            Total loss weight.
        rec_alpha : float, optional (default=1)
            Alpha value for reconstruction.
        triplet_alpha : float, optional (default=1)
            Alpha value for triplet loss.
        train_size : float, optional
            Size of the training set. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
        window_size : int, optional (default=7)
            Size of the window grid.
        kernel_size : int, optional (default=3)
            Size of the CNN kernel.
        latent_dim : int, optional (default=16)
            Dimensionality of the latent space.
        c_hid : int, optional (default=64)
            Number of channels produced by the convolution
        lr : float, optional (default=0.001)
            Learning rate.
        p : float, optional (default=0.3)
            Dropout probability.
        max_epochs : int, optional (default=100)
            Maximum number of training epochs.
        multi_triplet_loss : bool, optional (default=True)
            Whether to use multi-triplet loss.
        n_repeats : int, optional (default=1)
            Number of repeats per class in multi_triplet_loss.
        clustering_method : Literal["bgm", "kmeans", "louvain", "leiden"], optional (default="bgm")
            Clustering method to use.
        batch_size : int, optional
            Size of the batches.
        n_ensemble : int, optional (default=3)
            Number of ensemble models.
        n_ensemble_encoder : int, optional
            Number of ensemble encoders.
        n_ensemble_decoder : int, optional
            Number of ensemble decoders.
        random_seed : int, optional (default=2023)
            Random seed for reproducibility.
        n_neighbors : int, optional (default=10)
            Number of neighbors used in refining the cluster assignments in spatial space through majority voting.
        weight_decay : float, optional (default=1e-6)
            Weight decay for optimizer.
        refine_cluster : bool, optional (default=True)
            Whether to refine clusters after initial clustering.
        n_jobs : int, optional (default=1)
            Number of parallel jobs to run while building the grid.
        num_workers: int, optional (default=7)
            Number of subprocesses to use for data loading.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size should be an odd integer")

        self.grid_params = {
            "morphology_dim": window_size,
            "num_input_channels": None,
            "obsm_transcriptomics_dim": None
        }

        self.model_architecture_params = {
            "latent_dim": latent_dim,
            "c_hid": c_hid,
            "kernel_size": kernel_size if kernel_size < window_size else max(1, window_size - 1),
            "p": p,
            "n_ensemble_encoder": n_ensemble_encoder if n_ensemble_encoder else n_ensemble,
            "n_ensemble_decoder": n_ensemble_decoder if n_ensemble_decoder else n_ensemble
        }

        self.dataloader_params = {
            "batch_size": batch_size,
            "num_workers": num_workers
        }

        self.training_params = {
            "lr": lr,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
        }

        self.clustering_params = {
            "nCluster": nCluster,
            "clustering_method": clustering_method,
            "n_neighbors": n_neighbors,
            "refine_cluster": refine_cluster and n_neighbors > 1
        }

        self.loss_regularization_params = {
            "multi_triplet_loss": multi_triplet_loss,
            "rec_alpha": rec_alpha,
            "triplet_alpha": triplet_alpha,
            "n_repeats": n_repeats,
            "morphology_weight": morphology_weight,
            "transcriptomics_weight": None,
            "total_weight": total_weight
        }

        self.data_handling_params = {
            "n_jobs": n_jobs if n_jobs != -1 else (os.cpu_count() or 1),
            "train_size": train_size
        }


        self.random_seed = random_seed
        self.lit_aestetik_model: Optional[AESTETIKModel] = None
        self.trainer: Optional[Trainer] = None

        fix_seed(random_seed)

    # ================================================================= #
    #                       Main API Methods                            #
    # ================================================================= #

    def fit(self,
            X: anndata.AnnData,
            used_obsm_transcriptomics: str = "X_pca_transcriptomics",
            used_obsm_morphology: str = "X_pca_morphology",
            used_obsm_combined: str = "X_pca",
            used_obs_batch: Optional[str] = None,
            validation_split: float = 0.0,
            early_stopping_params: Optional[dict] = None
            ) -> None:
        """
        Trains the model on the provided AnnData object.

        Parameters
        ----------
        X : anndata.AnnData
            AnnData object.
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in `obsm`.
        used_obsm_combined : str, optional (default="X_pca")
            Key for combined data in `obsm`.
        used_obs_batch: Optional[str], optional (default=None)
            Key for column in `obs` that contains sample labels.
        validation_split : float, optional (default=0.0)
            Size of the validation set. It should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
        early_stopping_params : dict, optional
            Dictionary with parameters for EarlyStopping callback. Optional keys:
                - 'min_delta': float (default=0.0)
                - 'patience': int (default=3)
        """
        self._validate_fit_inputs(X=X,
                                  used_obsm_transcriptomics=used_obsm_transcriptomics,
                                  used_obsm_morphology=used_obsm_morphology)
        self._set_fit_params(X=X,
                             used_obsm_transcriptomics=used_obsm_transcriptomics)

        logger.info("Initializing datamodule ...")
        datamodule = AESTETIKDataModule(X,
                                        used_obsm_transcriptomics=used_obsm_transcriptomics,
                                        used_obsm_morphology=used_obsm_morphology,
                                        used_obsm_combined=used_obsm_combined,
                                        used_obs_batch=used_obs_batch,
                                        dataloader_params=self.dataloader_params,
                                        clustering_params=self.clustering_params,
                                        grid_params=self.grid_params,
                                        loss_regularization_params=self.loss_regularization_params,
                                        data_handling_params=self.data_handling_params,
                                        validation_split=validation_split)

        self.lit_aestetik_model = self._build_model(datamodule=datamodule)

        callbacks = self._create_callbacks(early_stopping_params=early_stopping_params, validation_split=validation_split)

        logger.info("Fit AESTETIKModel ...")
        self.trainer = Trainer(max_epochs=self.training_params["max_epochs"],
                                callbacks=callbacks,
                                num_sanity_val_steps=0)
        self.trainer.fit(self.lit_aestetik_model, datamodule=datamodule)
        self.losses = callbacks[0].losses

    def predict(self,
                X: anndata.AnnData,
                used_obsm_transcriptomics: str = "X_pca_transcriptomics",
                used_obsm_morphology: str = "X_pca_morphology",
                used_obs_batch: Optional[str] = None,
                save_emb: str = "AESTETIK",
                num_repeats: int = 1000,
                cluster: bool = True) -> None:
        """Compute spot representations for ``X`` and optionally cluster them.

        The fitted embeddings are written **in place** to
        ``X.obsm[save_emb]``; cluster assignments (when ``cluster=True``)
        are written to ``X.obs[f"{save_emb}_cluster"]``. The method
        returns ``None``.

        Parameters
        ----------
        X : anndata.AnnData
            AnnData object. Modified in place.
        num_repeats: int, optional (default=1000)
            Number of stochastic forward passes whose latent codes are
            averaged for the final embedding.
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in ``X.obsm``.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in ``X.obsm``.
        used_obs_batch: Optional[str], optional (default=None)
            Key for column in ``X.obs`` that contains sample labels.
        save_emb : str, optional (default="AESTETIK")
            Key for the embedding column written under ``X.obsm``.
        cluster: bool, optional (default=True)
            Whether to perform clustering on the latent space.
        """
        self._check_fitted()
        self._validate_predict_inputs(X,
                                      used_obsm_transcriptomics=used_obsm_transcriptomics,
                                      used_obsm_morphology=used_obsm_morphology)
        self._set_predict_params(num_repeats=num_repeats)

        all_latent_space = self._compute_latent_space(X,
                                                      used_obsm_transcriptomics=used_obsm_transcriptomics,
                                                      used_obsm_morphology=used_obsm_morphology,
                                                      used_obs_batch=used_obs_batch)
        self._postprocess_predictions(X,
                                      latent_space=all_latent_space,
                                      save_emb=save_emb,
                                      cluster=cluster,
                                      used_obs_batch=used_obs_batch)

    def fit_predict(self,
                    X: anndata.AnnData,
                    used_obsm_transcriptomics: str = "X_pca_transcriptomics",
                    used_obsm_morphology: str = "X_pca_morphology",
                    used_obsm_combined: str = "X_pca",
                    used_obs_batch: Optional[str] = None,
                    validation_split: float = 0.0,
                    save_emb: str = "AESTETIK",
                    num_repeats: int = 1000,
                    cluster: bool = True) -> None:
        """
        Trains the model on the provided AnnData object and then computes spot representations. Then we optionally cluster them into groups.
        
        Parameters
        ----------
        X : anndata.AnnData
            AnnData object.
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in `obsm`.
        used_obsm_combined : str, optional (default="X_pca")
            Key for combined data in `obsm`.
        used_obs_batch: Optional[str], optional (default=None)
            Key for column in `obs` that contains sample labels.
        validation_split : float, optional (default=0.0)
            Size of the validation set. It should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
        save_emb : str, optional (default="AESTETIK")
            Key for saving embeddings.
        num_repeats: int, optional (default=1000)
            Number of repeats for stochastic prediction.
        cluster: bool, optional (default=True)
            Whether to perform clustering on the latent space.
        """
        self.fit(X,
                 used_obsm_transcriptomics=used_obsm_transcriptomics,
                 used_obsm_morphology=used_obsm_morphology,
                 used_obsm_combined=used_obsm_combined,
                 used_obs_batch=used_obs_batch,
                 validation_split=validation_split)

        self._set_predict_params(num_repeats=num_repeats)
        all_latent_space = self._compute_latent_space(X,
                                                      built_grid=True)
        self._postprocess_predictions(X,
                                      latent_space=all_latent_space,
                                      save_emb=save_emb,
                                      cluster=cluster,
                                      used_obs_batch=used_obs_batch)

    # ================================================================= #
    #                      Private Validation Methods                   #
    # ================================================================= #
    def _validate_fit_inputs(self,
                            X: anndata.AnnData,
                            used_obsm_transcriptomics: str,
                            used_obsm_morphology: str) -> None:

        self._validate_obsm_keys(X, [used_obsm_morphology, used_obsm_transcriptomics], "fit")
        self._validate_obs_columns(X, ["x_array", "y_array"], "fit")

    def _validate_predict_inputs(self,
                                 X: anndata.AnnData,
                                 used_obsm_transcriptomics: str,
                                 used_obsm_morphology: str) -> None:
        self._validate_obsm_keys(X, [used_obsm_morphology, used_obsm_transcriptomics], "predict")
        self._validate_obs_columns(X, ["x_array", "y_array"], "predict")

        obsm_transcriptomics_dim = X.obsm[used_obsm_transcriptomics].shape[1]
        obsm_morphology_dim = X.obsm[used_obsm_morphology].shape[1]
        obsm_morphology_dim_target = self.grid_params["num_input_channels"] - self.grid_params["obsm_transcriptomics_dim"]


        if (obsm_transcriptomics_dim < self.grid_params["obsm_transcriptomics_dim"] or
            obsm_morphology_dim < obsm_morphology_dim_target):
            raise ValueError(
                "Dimensionality of obsm transcriptomics or morphology features is too small. "
                f"Transcriptomics dim: {obsm_transcriptomics_dim}, "
                f"Morphology dim: {obsm_morphology_dim}, "
                f"Total: {obsm_transcriptomics_dim + obsm_morphology_dim}, "
                f"Required: transcriptomics >= {self.grid_params['obsm_transcriptomics_dim']}, "
                f"morphology >= {obsm_morphology_dim_target}"
            )
        self._calibrate_predict_inputs(X, used_obsm_transcriptomics, used_obsm_morphology)

    def _check_fitted(self) -> None:
        if self.trainer is None or self.lit_aestetik_model is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")

    def _validate_obsm_keys(self,
                            X: anndata.AnnData,
                            required_keys: List[str],
                            method_name: str) -> None:
        missing = [key for key in required_keys if key not in X.obsm]
        if missing:
            raise KeyError(
                f"AESTETIK.{method_name}: Required keys {missing} must be present in X.obsm. "
                f"Available keys: {list(X.obsm.keys())}"
            )

    def _validate_obs_columns(self,
                              X: anndata.AnnData,
                              required_columns: List[str],
                              method_name: str) -> None:
        missing = [column for column in required_columns if column not in X.obs]
        if missing:
            raise KeyError(
                f"AESTETIK.{method_name}: Required columns {missing} must be present in X.obs. "
                f"Available columns: {list(X.obs.columns)}"
            )

    # ================================================================= #
    #                   Private Data Preparation Methods                #
    # ================================================================= #
    def _set_fit_params(self,
                        X: anndata.AnnData,
                        used_obsm_transcriptomics: str) -> None:
        if self.dataloader_params["batch_size"] is None:
            self.dataloader_params["batch_size"] = min(2 ** 13, len(X))

        self.grid_params["obsm_transcriptomics_dim"] = X.obsm[used_obsm_transcriptomics].shape[1]

    def _set_predict_params(self,
                            num_repeats: int) -> None:
        self.lit_aestetik_model.predict_params["num_repeats"] = num_repeats

    def _calibrate_predict_inputs(self,
                                  X: anndata.AnnData,
                                  used_obsm_transcriptomics: str,
                                  used_obsm_morphology: str) -> None:
        """
        Calibrate the dimensionality of obsm arrays to match grid_params.
        """
        obsm_morphology_dim_target = self.grid_params["num_input_channels"] - self.grid_params["obsm_transcriptomics_dim"]

        if X.obsm[used_obsm_transcriptomics].shape[1] > self.grid_params["obsm_transcriptomics_dim"]:
            logger.info(f"Cut down transcriptomics dimensionality for {used_obsm_transcriptomics}")
            X.obsm[used_obsm_transcriptomics] = X.obsm[used_obsm_transcriptomics][:, :self.grid_params["obsm_transcriptomics_dim"]]

        if X.obsm[used_obsm_morphology].shape[1] > obsm_morphology_dim_target:
            logger.info(f"Cut down morphology dimensionality for {used_obsm_morphology}")
            X.obsm[used_obsm_morphology] = X.obsm[used_obsm_morphology][:, :obsm_morphology_dim_target]


    def _create_predict_dataloader(self,
        X: anndata.AnnData,
        used_obsm_transcriptomics: Optional[str] = None,
        used_obsm_morphology: Optional[str] = None,
        used_obs_batch: Optional[str] = None,
        built_grid: bool = False) -> DataLoader:

        if not built_grid:
            build_grid(X,
                   used_obsm_transcriptomics=used_obsm_transcriptomics,
                   used_obsm_morphology=used_obsm_morphology,
                   used_obs_batch=used_obs_batch,
                   window_size=self.grid_params["morphology_dim"],
                   n_jobs=self.data_handling_params["n_jobs"])

        all_spots = torch.from_numpy(X.obsm["X_st_grid"].astype(np.float32))
        dataset = TensorDataset(all_spots)
        return DataLoader(dataset,
                          shuffle=False,
                          **self.dataloader_params)

    def _create_callbacks(self,
        validation_split: float,
        early_stopping_params: Optional[dict] = None) -> list:

        callbacks = []
        loss_callback = LossHistoryCallback()
        callbacks.append(loss_callback)

        if validation_split > 0.0:
            early_stopping_params = self._create_early_stopping_params(early_stopping_params)
            early_stop_callback = EarlyStopping(**early_stopping_params)
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-checkpoint",
            )
            callbacks.extend([early_stop_callback, checkpoint_callback])
        return callbacks

    def _create_early_stopping_params(self,
        user_params: Optional[Dict] = None) -> Dict:

        default_params = {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 5
        }
        if user_params is None:
            user_params = {}
        for forbidden_key in ["monitor", "mode"]:
            if forbidden_key in user_params:
                user_params.pop(forbidden_key)
                logger.info(f"Removed forbidden key '{forbidden_key}' from early_stopping_params to enforce fixed value.")

        return {**default_params, **user_params}

    # ================================================================= #
    #           Private Prediction and Postprocessing Methods           #
    # ================================================================= #
    def _compute_latent_space(self,
                              X: anndata.AnnData,
                              used_obsm_transcriptomics: Optional[str] = None,
                              used_obsm_morphology: Optional[str] = None,
                              used_obs_batch: Optional[str] = None,
                              built_grid = False) -> np.ndarray:
        predict_dataloader = self._create_predict_dataloader(X,
                                                             used_obsm_transcriptomics=used_obsm_transcriptomics,
                                                             used_obsm_morphology=used_obsm_morphology,
                                                             used_obs_batch=used_obs_batch,
                                                             built_grid=built_grid)
        all_latent_space = self.trainer.predict(self.lit_aestetik_model,
                                                dataloaders=predict_dataloader)
        all_latent_space = torch.cat(all_latent_space, dim=0)
        return all_latent_space

    def _postprocess_predictions(self,
                                 X: anndata.AnnData,
                                 latent_space: np.ndarray,
                                 save_emb:str,
                                 cluster: bool,
                                 used_obs_batch: str) -> None:
        X.obsm[save_emb] = latent_space.cpu().numpy()

        if cluster:
            clustering(X,
            used_obsm=save_emb,
            num_cluster=self.clustering_params["nCluster"],
            method=self.clustering_params["clustering_method"],
            refine_cluster=self.clustering_params["refine_cluster"],
            n_neighbors=self.clustering_params["n_neighbors"],
            used_obs_batch=used_obs_batch)

    # ================================================================= #
    #                       Model Construction                          #
    # ================================================================= #
    def _build_model(self,
                     datamodule: AESTETIKDataModule) -> AESTETIKModel:
        logger.info("Build AESTETIKModel ...")

        training_step_params = {
            "rec_alpha": self.loss_regularization_params["rec_alpha"],
            "triplet_alpha": self.loss_regularization_params["triplet_alpha"]}

        optimizer_step_params = {
            "lr": self.training_params["lr"],
            "weight_decay": self.training_params["weight_decay"]}

        return AESTETIKModel(datamodule=datamodule,
                                grid_params=self.grid_params,
                                model_architecture_params=self.model_architecture_params,
                                training_params=training_step_params,
                                optimizer_params=optimizer_step_params)

    @staticmethod
    def version():
        return "16.06.2025:1"
