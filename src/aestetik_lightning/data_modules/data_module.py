import anndata
import torch 
import lightning as L 

from aestetik.utils.utils_data import prepare_input_for_model
from aestetik.dataloader import CustomDataset

from typing import Union, Literal


class AESTETIKDataModule(L.LightningDataModule):
    def __init__(self,
                 adata: anndata,
                 nCluster: Union[int, float],
                 morphology_weight: float,
                 total_weight: float,
                 train_size: float,
                 window_size: int,
                 multi_triplet_loss: bool,
                 n_repeats: int,
                 clustering_method: Literal["bgm", "kmeans", "louvain", "leiden"],
                 batch_size: int,
                 used_obsm_transcriptomics: str,
                 used_obsm_morphology: str,
                 used_obsm_combined: str,
                 n_neighbors: int,
                 n_jobs: int):
        super().__init__()
        self.adata = adata 

        self.data_cluster_params = {
            "window_size": window_size,
            "used_obsm_transcriptomics": used_obsm_transcriptomics,
            "used_obsm_morphology": used_obsm_morphology,
            "used_obsm_combined": used_obsm_combined,
            "nCluster": nCluster,
            "clustering_method": clustering_method,
            "n_neighbors": n_neighbors
        }
        self.loss_regularization_params = {
            "morphology_weight": morphology_weight,
            "transcriptomics_weight": None,
            "total_weight": total_weight,
            "multi_triplet_loss": multi_triplet_loss,
            "n_repeats": n_repeats
        }
        self.training_params = {
            "batch_size": batch_size if batch_size else min(2 ** 13, len(adata))
        }
        self.misc_params = {
            "train_size": train_size,
            "n_jobs": n_jobs 
        }
    
    def setup(self, stage=None) -> None:
        self.loss_regularization_params["transcriptomics_weight"], self.loss_regularization_params["morphology_weight"] = prepare_input_for_model(adata=self.adata,
                                                                                      total_weight=self.loss_regularization_params["total_weight"],
                                                                                      morphology_weight=self.loss_regularization_params["morphology_weight"],
                                                                                      n_jobs=self.misc_params["n_jobs"],
                                                                                      **self.data_cluster_params)
        self.dataset = CustomDataset(adata=self.adata,
                                     multi_triplet_loss=self.loss_regularization_params["multi_triplet_loss"],
                                     repeats=self.loss_regularization_params["n_repeats"],
                                     train_size=self.misc_params["train_size"],
                                     compute_transcriptomics_list=self.loss_regularization_params["transcriptomics_weight"] > 0,
                                     compute_morphology_list=self.loss_regularization_params["morphology_weight"] > 0)
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.training_params["batch_size"],
                                           shuffle=True)