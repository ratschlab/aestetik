import anndata
import numpy as np
import torch
from lightning.pytorch import Trainer
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing

from aestetik_lightning.data_modules.data_module import AESTETIKDataModule
from aestetik_lightning.modules.aestetik_module import LitAESTETIKModel
from aestetik.utils.utils_clustering import clustering
from aestetik.utils.utils_grid import fix_seed

from typing import Literal
from typing import Union


class LitAESTETIK:
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
        epochs: int = 100,
        multi_triplet_loss: bool = True,
        n_repeats: int = 1,
        clustering_method: Literal["bgm", "kmeans", "louvain", "leiden"] = "bgm",
        batch_size: int = None,
        n_ensemble: int = 3,
        n_ensemble_encoder: int = None,
        n_ensemble_decoder: int = None,
        used_obsm_transcriptomics: str = "X_pca_transcriptomics",
        used_obsm_morphology: str = "X_pca_morphology",
        used_obsm_combined: str = "X_pca",
        save_emb: str = "AESTETIK",
        random_seed: int = 2023,
        n_neighbors: int = 10,
        weight_decay: float = 1e-6,
        spot_diameter_fullres: int = None,
        refine_cluster: bool = True,
        n_jobs: int = 1
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
        epochs : int, optional (default=100)
            Number of training epochs.
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
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in `obsm`.
        used_obsm_combined : str, optional (default="X_pca")
            Key for combined data in `obsm`.
        save_emb : str, optional (default="AESTETIK")
            Key for saving embeddings.
        random_seed : int, optional (default=2023)
            Random seed for reproducibility.
        n_neighbors : int, optional (default=10)
            Number of neighbors used in refining the cluster assignments in spatial space through majority voting.
        weight_decay : float, optional (default=1e-6)
            Weight decay for optimizer.
        spot_diameter_fullres : int, optional
            Diameter of spots in full resolution.
        refine_cluster : bool, optional (default=True)
            Whether to refine clusters after initial clustering.
        n_jobs : int, optional (default=1)
            Number of parallel jobs to run while building the grid.
        """
        self.model_architecture_params = {
            "latent_dim": latent_dim,
            "c_hid": c_hid,
            "kernel_size": kernel_size if kernel_size < window_size else max(1, window_size - 1),
            "p": p,
            "n_ensemble_encoder": n_ensemble_encoder if n_ensemble_encoder else n_ensemble,
            "n_ensemble_decoder": n_ensemble_decoder if n_ensemble_decoder else n_ensemble
        }

        self.training_params = {
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size
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

        self.data_cluster_params = {
            "nCluster": nCluster,
            "train_size": train_size,
            "window_size": window_size,
            "clustering_method": clustering_method,
            "used_obsm_transcriptomics": used_obsm_transcriptomics,
            "used_obsm_morphology": used_obsm_morphology,
            "used_obsm_combined": used_obsm_combined,
            "n_neighbors": n_neighbors,
            "refine_cluster": refine_cluster and n_neighbors > 1,
            "spot_diameter_fullres": spot_diameter_fullres,
            "save_emb": save_emb
        }

        self.misc_params = {
            "n_jobs": n_jobs if n_jobs != -1 else int(multiprocessing.cpu_count()),
            "random_seed": random_seed

        }

        self.lit_aestetik_model = None
        self.trainer = None
        self.adata = None

        fix_seed(random_seed)

    def fit(self, 
            X: anndata) -> None:
        """Train the model on the provided AnnData object.

        Parameters
        ----------
        adata : anndata
            anndata object.
        """
        if (self.data_cluster_params["used_obsm_morphology"] not in X.obsm or 
            self.data_cluster_params["used_obsm_transcriptomics"] not in X.obsm):
            raise KeyError(
                f"LitAESTETIK.fit(self, ): Required keys '{self.data_cluster_params['used_obsm_morphology']}' and '{self.data_cluster_params['used_obsm_transcriptomics']}' must both be present in X.obsm. "
                f"Available keys: {list(X.obsm.keys())}"
            )
    
        self.adata = X
        datamodule = AESTETIKDataModule(adata=self.adata,
                                             nCluster=self.data_cluster_params["nCluster"],
                                             morphology_weight=self.loss_regularization_params["morphology_weight"],
                                             total_weight=self.loss_regularization_params["total_weight"],
                                             train_size=self.data_cluster_params["train_size"],
                                             window_size=self.data_cluster_params["window_size"],
                                             multi_triplet_loss=self.loss_regularization_params["multi_triplet_loss"],
                                             n_repeats=self.loss_regularization_params["n_repeats"],
                                             clustering_method=self.data_cluster_params["clustering_method"],
                                             batch_size=self.training_params["batch_size"],
                                             used_obsm_transcriptomics=self.data_cluster_params["used_obsm_transcriptomics"],
                                             used_obsm_morphology=self.data_cluster_params["used_obsm_morphology"],
                                             used_obsm_combined=self.data_cluster_params["used_obsm_combined"],
                                             n_neighbors=self.data_cluster_params["n_neighbors"],
                                             n_jobs=self.misc_params["n_jobs"])
        self.lit_aestetik_model = LitAESTETIKModel(datamodule=datamodule,
                                                    rec_alpha=self.loss_regularization_params["rec_alpha"],
                                                    triplet_alpha=self.loss_regularization_params["triplet_alpha"],
                                                    lr=self.training_params["lr"],
                                                    epochs=self.training_params["epochs"],
                                                    save_emb=self.data_cluster_params["save_emb"],
                                                    weight_decay=self.training_params["weight_decay"],
                                                    spot_diameter_fullres=self.data_cluster_params["spot_diameter_fullres"],
                                                    refine_cluster=self.data_cluster_params["refine_cluster"],
                                                    **self.model_architecture_params)
        self.trainer = Trainer(max_epochs=self.training_params["epochs"], 
                               min_epochs=self.training_params["epochs"])
        self.trainer.fit(self.lit_aestetik_model, datamodule=datamodule)
        
        self.training_params["batch_size"] = datamodule.training_params["batch_size"]
        self.loss_regularization_params["morphology_weight"] = self.lit_aestetik_model.weights["morphology_weight"]
        self.loss_regularization_params["transcriptomics_weight"] = self.lit_aestetik_model.weights["transcriptomics_weight"]

    def predict(self,
                num_repeats: int = 1000,
                cluster: bool = True) -> None:
        """
        Compute spot representations. Then we optionally cluster them into groups.
        
        Parameters
        ----------
        num_repeats: int, optional (default=1000)
            Number of repeats for stochastic prediction.
        cluster: bool, optional (default=True)
            Whether to perform clustering on the latent space.
        """
        if self.adata is None:
            raise ValueError("AnnData object not loaded. Call fit() first.")

        self.lit_aestetik_model.predict_params["num_repeats"] = num_repeats

        all_spots = torch.from_numpy(self.adata.obsm["X_st_grid"].astype(np.float32))
        dataset = TensorDataset(all_spots)
        predict_dataloader = DataLoader(dataset, 
                                        batch_size=self.training_params["batch_size"],
                                        shuffle=False)
                                        
        all_latent_space = self.trainer.predict(self.lit_aestetik_model,
                                                dataloaders=predict_dataloader)
        all_latent_space = torch.cat(all_latent_space, dim=0)
        self.adata.obsm[self.data_cluster_params["save_emb"]] = all_latent_space.cpu().numpy()

        if cluster:
            clustering(self.adata,
                       num_cluster=self.data_cluster_params["nCluster"],
                       used_obsm=self.data_cluster_params["save_emb"],
                       method=self.data_cluster_params["clustering_method"],
                       refine_cluster=self.data_cluster_params["refine_cluster"],
                       n_neighbors=self.data_cluster_params["n_neighbors"])