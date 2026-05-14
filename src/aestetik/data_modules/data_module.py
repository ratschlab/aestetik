from typing import Optional

import anndata
import lightning as L
import torch
from torch.utils.data import random_split

from aestetik.dataloader import CustomDataset
from aestetik.utils.utils_data import prepare_input_for_model


class AESTETIKDataModule(L.LightningDataModule):
    """Internal Lightning DataModule for the AESTETIK pipeline.

    The class is constructed by :class:`aestetik.AESTETIK` and is not
    part of the public API. See ``AESTETIK.fit`` for the parameter
    semantics; the dicts here are the bagged form of the same
    settings.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object. **Mutated in place** by ``setup`` — callers
        should pass a private copy.
    validation_split : float
        Proportion of training data held out for Lightning validation.
    used_obsm_transcriptomics, used_obsm_morphology, used_obsm_combined : str
        Keys under ``adata.obsm`` for the modality / concatenated input.
    used_obs_batch : str, optional
        Column in ``adata.obs`` identifying samples / batches.
    dataloader_params : dict
        ``batch_size`` and ``num_workers``.
    clustering_params : dict
        ``nCluster``, ``clustering_method``, ``n_neighbors``,
        ``refine_cluster``.
    grid_params : dict
        ``morphology_dim`` (= window_size).
    loss_regularization_params : dict
        ``multi_triplet_loss``, ``n_repeats``, ``morphology_weight``,
        ``transcriptomics_weight``, ``total_weight``.
    data_handling_params : dict
        ``n_jobs`` and ``train_size``.
    """

    def __init__(self,
                 adata: anndata,
                 validation_split: float,
                 used_obsm_transcriptomics: str,
                 used_obsm_morphology: str,
                 used_obsm_combined: str,
                 dataloader_params: dict,
                 clustering_params: dict,
                 grid_params: dict,
                 loss_regularization_params: dict,
                 data_handling_params: dict,
                 used_obs_batch: Optional[str] = None,
                ):
        super().__init__()
        self.adata = adata
        self.validation_split = validation_split
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

        if stage == "fit" or stage is None:
            if self.validation_split > 0.0:
                val_len = int(len(self.dataset) * self.validation_split)
                train_len = len(self.dataset) - val_len

                if train_len == 0:
                    raise ValueError(
                        f"Training set length is zero after applying validation_split={self.validation_split}. "
                        "Please decrease validation_split or provide more data."
                    )
                if val_len == 0 and self.validation_split > 0:
                    raise ValueError(
                        f"Validation set length is zero after applying validation_split={self.validation_split}. "
                        "Please increase validation_split or provide more data."
                    )
                self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])
            else:
                self.train_dataset = self.dataset
                self.val_dataset = None


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # Use the split train_dataset so validation_split is honoured
        # (issue #8). When validation_split == 0, setup() assigns
        # self.train_dataset = self.dataset.
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           **self.dataloader_params,
                                           shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # When no validation_split was requested we still have to hand
        # back a DataLoader so Lightning's type checks don't blow up;
        # an empty TensorDataset is the cheapest way to signal "no
        # validation batches".
        if self.val_dataset is None:
            empty = torch.utils.data.TensorDataset(torch.empty(0))
            return torch.utils.data.DataLoader(empty, batch_size=1)
        return torch.utils.data.DataLoader(self.val_dataset, **self.dataloader_params, shuffle=False)

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
