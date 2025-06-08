import anndata
import numpy as np
import torch
import logging
import multiprocessing
from lightning.pytorch import Trainer
from torch.utils.data import TensorDataset, DataLoader

from aestetik.data_modules.data_module import AESTETIKDataModule
from aestetik.modules.aestetik_module import AESTETIKModel
from aestetik.utils.utils_clustering import clustering
from aestetik.utils.utils_grid import fix_seed
from aestetik.utils.utils_data import build_grid

from typing import Literal
from typing import Union
from typing import List


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
        train_size: float = None,
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
        batch_size: int = None,
        n_ensemble: int = 3,
        n_ensemble_encoder: int = None,
        n_ensemble_decoder: int = None,
        random_seed: int = 2023,
        n_neighbors: int = 10,
        weight_decay: float = 1e-6,
        refine_cluster: bool = True,
        n_jobs: int = 1,
        num_workers: int = 7
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
            "n_jobs": n_jobs if n_jobs != -1 else int(multiprocessing.cpu_count()),
            "train_size": train_size
        }
        

        self.random_seed = random_seed
        self.lit_aestetik_model = None
        self.trainer = None

        fix_seed(random_seed)

    # ================================================================= #
    #                       Main API Methods                            #
    # ================================================================= #    
    
    def fit(self, 
            X: anndata,
            used_obsm_transcriptomics: str = "X_pca_transcriptomics",
            used_obsm_morphology: str = "X_pca_morphology",
            used_obsm_combined: str = "X_pca",
            ) -> None:
        """
        Train the model on the provided AnnData object.

        Parameters
        ----------
        X : anndata
            anndata object.
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in `obsm`.
        used_obsm_combined : str, optional (default="X_pca")
            Key for combined data in `obsm`.
        """
        self._validate_fit_inputs(X=X,
                                  used_obsm_transcriptomics=used_obsm_transcriptomics,
                                  used_obsm_morphology=used_obsm_morphology)
        self._set_fit_params(X=X,
                             used_obsm_transcriptomics=used_obsm_transcriptomics)

        logging.info("Initializing datamodule ...")
        datamodule = AESTETIKDataModule(X,
                                        used_obsm_transcriptomics=used_obsm_transcriptomics,
                                        used_obsm_morphology=used_obsm_morphology,
                                        used_obsm_combined=used_obsm_combined,
                                        dataloader_params=self.dataloader_params,
                                        clustering_params=self.clustering_params,
                                        grid_params=self.grid_params,
                                        loss_regularization_params=self.loss_regularization_params,
                                        data_handling_params=self.data_handling_params)
        
        self.lit_aestetik_model = self._build_model(datamodule=datamodule)

        logging.info("Fit AESTETIKModel ...")
        self.trainer = Trainer(max_epochs=self.training_params["max_epochs"])
        self.trainer.fit(self.lit_aestetik_model, datamodule=datamodule)

    def predict(self,
                X: anndata,
                used_obsm_transcriptomics: str = "X_pca_transcriptomics",
                used_obsm_morphology: str = "X_pca_morphology",
                save_emb: str = "AESTETIK",
                num_repeats: int = 1000,
                cluster: bool = True) -> None:
        """
        Compute spot representations for all spots in X. Then we optionally cluster them into groups.
        
        Parameters
        ----------
        X : anndata
            anndata object.
        num_repeats: int, optional (default=1000)
            Number of repeats for stochastic prediction.
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in `obsm`.
        save_emb : str, optional (default="AESTETIK")
            Key for saving embeddings.
        cluster: bool, optional (default=True)
            Whether to perform clustering on the latent space.
        """
        self._check_fitted()
        self._validate_predict_inputs(X,
                                      used_obsm_transcriptomics=used_obsm_transcriptomics,
                                      used_obsm_morphology=used_obsm_morphology)
        self._set_predict_params(num_repeats=num_repeats)

        predict_dataloader = self._create_predict_dataloader(X=X,
                                                            used_obsm_transcriptomics=used_obsm_transcriptomics,
                                                            used_obsm_morphology=used_obsm_morphology)
        all_latent_space = self.trainer.predict(self.lit_aestetik_model,
                                                dataloaders=predict_dataloader)
        all_latent_space = torch.cat(all_latent_space, dim=0)
        X.obsm[save_emb] = all_latent_space.cpu().numpy()
        
        if cluster:
            clustering(X,
                       used_obsm=save_emb,
                       num_cluster=self.clustering_params["nCluster"],
                       method=self.clustering_params["clustering_method"],
                       refine_cluster=self.clustering_params["refine_cluster"],
                       n_neighbors=self.clustering_params["n_neighbors"])
    

    # ================================================================= #
    #                       Validation Utilities                        #
    # ================================================================= #     
    def _validate_fit_inputs(self,
                            X: anndata,
                            used_obsm_transcriptomics: str,
                            used_obsm_morphology: str) -> None:
        
        self._validate_obsm_keys(X, [used_obsm_morphology, used_obsm_transcriptomics], "fit")
        self._validate_obs_columns(X, ["x_array", "y_array"], "fit")

    def _validate_predict_inputs(self,
                                 X: anndata,
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
    
    def _check_fitted(self):
        if self.trainer is None or self.lit_aestetik_model is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")

    def _validate_obsm_keys(self, 
                            X: anndata,
                            required_keys: List[str], 
                            method_name: str):
        missing = [key for key in required_keys if key not in X.obsm]
        if missing:
            raise KeyError(
                f"AESTETIK.{method_name}: Required keys {missing} must be present in X.obsm. "
                f"Available keys: {list(X.obsm.keys())}"
            )

    def _validate_obs_columns(self, 
                              X: anndata, 
                              required_columns: List[str], 
                              method_name: str):
        missing = [column for column in required_columns if column not in X.obs]
        if missing:
            raise KeyError(
                f"AESTETIK.{method_name}: Required columns {missing} must be present in X.obs. "
                f"Available columns: {list(X.obs.columns)}"
            )

    # ================================================================= #
    #                       Data Preparation                            #
    # ================================================================= #
    def _set_fit_params(self,
                        X: anndata,
                        used_obsm_transcriptomics: str) -> None:
        if self.dataloader_params["batch_size"] is None:
            self.dataloader_params["batch_size"] = min(2 ** 13, len(X))
        
        self.grid_params["obsm_transcriptomics_dim"] = X.obsm[used_obsm_transcriptomics].shape[1]
    
    def _set_predict_params(self, 
                            num_repeats: int):
        self.lit_aestetik_model.predict_params["num_repeats"] = num_repeats

    def _calibrate_predict_inputs(self,
                                  X: anndata,
                                  used_obsm_transcriptomics: str,
                                  used_obsm_morphology: str) -> None:
        """Calibrate the dimensionality of obsm arrays to match grid_params."""
        obsm_morphology_dim_target = self.grid_params["num_input_channels"] - self.grid_params["obsm_transcriptomics_dim"]

        if X.obsm[used_obsm_transcriptomics].shape[1] > self.grid_params["obsm_transcriptomics_dim"]:
            logging.info(f"Cut down transcriptomics dimensionality for {used_obsm_transcriptomics}")
            X.obsm[used_obsm_transcriptomics] = X.obsm[used_obsm_transcriptomics][:, :self.grid_params["obsm_transcriptomics_dim"]]

        if X.obsm[used_obsm_morphology].shape[1] > obsm_morphology_dim_target:
            logging.info(f"Cut down morphology dimensionality for {used_obsm_morphology}")
            X.obsm[used_obsm_morphology] = X.obsm[used_obsm_morphology][:, :obsm_morphology_dim_target]
    

    def _create_predict_dataloader(self,
        X: anndata,
        used_obsm_transcriptomics: str = "X_pca_transcriptomics",
        used_obsm_morphology: str = "X_pca_morphology") -> DataLoader:
        
        build_grid(X, 
                   used_obsm_transcriptomics=used_obsm_transcriptomics,
                   used_obsm_morphology=used_obsm_morphology,
                   window_size=self.grid_params["morphology_dim"],
                   n_jobs=self.data_handling_params["n_jobs"])
        
        all_spots = torch.from_numpy(X.obsm["X_st_grid"].astype(np.float32))
        dataset = TensorDataset(all_spots)
        return DataLoader(dataset, 
                          shuffle=False,
                          **self.dataloader_params)

    # ================================================================= #
    #                       Model Construction                          #
    # ================================================================= #
    def _build_model(self,
                     datamodule: AESTETIKDataModule) -> AESTETIKModel:
        logging.info("Build AESTETIKModel ...")

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