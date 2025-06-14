import anndata
import torch 
import lightning as L 

from aestetik.utils.utils_data import prepare_input_for_model
from aestetik.dataloader import CustomDataset

from typing import Optional


class AESTETIKDataModule(L.LightningDataModule):
    def __init__(self,
                 adata: anndata,
                 used_obsm_transcriptomics: str,
                 used_obsm_morphology: str,
                 used_obsm_combined: str,
                 dataloader_params: dict,
                 clustering_params: dict,
                 grid_params: dict,
                 loss_regularization_params: dict,
                 data_handling_params: dict,
                 used_obs_batch: Optional[str] = None):
        super().__init__()
        """
        Parameters
        ----------
        adata : anndata
            anndata object.
        used_obsm_transcriptomics : str
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str
            Key for morphology data in `obsm`.
        used_obsm_combined : str
            Key for combined data in `obsm`.
        dataloader_params : dict
            Dictionary with dataloader parameters. Expected keys:
                -'batch_size': int
                    Size of the batches.
                -'num_workers': int
                    Number of subprocesses to use for data loading.
        clustering_params : dict
            Dictionary with clustering parameters. Expected keys:
                -'nCluster': Union[int, float]
                    if int: Number of clusters.
                    if float: Resolution parameter in leiden and louvain.
                -'clustering_method': Literal["bgm", "kmeans", "louvain", "leiden"]
                    Clustering method to use.
                -'n_neighbors': int
                    Number of neighbors used in refining the cluster assignments in spatial space through majority
        grid_params : dict
            Dictionary with grid parameters. Expected keys:
                -'morphology_dim': int
                    Size of the window grid.
        loss_regularization_params : dict
            Dictionary with loss and regularization parameters. Expected keys:
                -'multi_triplet_loss': bool
                    Whether to use multi-triplet loss.
                -'n_repeats': int
                    Number of repeats per class in multi_triplet_loss.
                -'morphology_weight': float
                    Weight for the morphology modality.
                -'transcriptomics_weight': Optional[float]
                    Weight for the transcriptomics modality.
                -'total_weight': float
                    Total loss weight.
        data_handling_params : dict 
            Dictionary with data handling parameters. Expected keys:
                -'n_jobs': int
                    Number of parallel jobs to run while building the grid.
                -'train_size': Optional[float]
                    Size of the training set. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
        used_obs_batch: Optional[str], optional (default=None)
            Key for column in `obs` that differentiates among experiments or batches.

        """
        self.adata = adata 
        self.used_obsm = {
                        "used_obsm_transcriptomics": used_obsm_transcriptomics,
                        "used_obsm_morphology": used_obsm_morphology,
                        "used_obsm_combined": used_obsm_combined}
        self.used_obs = {
                        "used_obs_batch": used_obs_batch}
        self.dataloader_params = dataloader_params
        self.clustering_params = clustering_params 
        self.grid_params = grid_params
        self.loss_regularization_params = loss_regularization_params
        self.data_handling_params = data_handling_params
        
        self._validate_params()
    
    def setup(self, stage=None) -> None:

        self.loss_regularization_params["transcriptomics_weight"], self.loss_regularization_params["morphology_weight"] = prepare_input_for_model(adata=self.adata,
                                                                                                                                                  window_size=self.grid_params["morphology_dim"],
                                                                                                                                                  n_jobs=self.data_handling_params["n_jobs"],
                                                                                                                                                  morphology_weight=self.loss_regularization_params["morphology_weight"],
                                                                                                                                                  total_weight=self.loss_regularization_params["total_weight"],
                                                                                                                                                  n_neighbors=self.clustering_params["n_neighbors"],
                                                                                                                                                  nCluster=self.clustering_params["nCluster"],
                                                                                                                                                  clustering_method=self.clustering_params["clustering_method"],
                                                                                                                                                  **self.used_obsm,
                                                                                                                                                  **self.used_obs
                                                                                                                                                  )
        self.dataset = CustomDataset(self.adata,
                                     multi_triplet_loss=self.loss_regularization_params["multi_triplet_loss"],
                                     repeats=self.loss_regularization_params["n_repeats"],
                                     train_size=self.data_handling_params["train_size"],
                                     compute_transcriptomics_list=(self.loss_regularization_params["transcriptomics_weight"] > 0),
                                     compute_morphology_list=(self.loss_regularization_params["morphology_weight"] > 0))


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.dataset,
                                           **self.dataloader_params,
                                           shuffle=True)
        
    def _validate_params(self) -> None:
        required = {
            'dataloader_params': ['batch_size', 'num_workers'],
            'clustering_params': ['nCluster', 'clustering_method', 'n_neighbors'],
            'grid_params': ['morphology_dim'],
            'loss_regularization_params': ['multi_triplet_loss', 'n_repeats', 'morphology_weight', 'transcriptomics_weight'],
            'data_handling_params': ['n_jobs', 'train_size'],
        }
        for group_name, keys in required.items():
            param_dict = getattr(self, group_name)
            for key in keys:
                if key not in param_dict:
                    raise ValueError(f"Missing required key {key} in '{group_name}' dictionary.")