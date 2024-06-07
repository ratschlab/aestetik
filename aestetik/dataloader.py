from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(
            self,
            adata,
            multi_triplet_loss=True,
            repeats=1,
            train_size=None,
            compute_transcriptomics_list=True,
            compute_morphology_list=True):
        super().__init__()
        self.adata = adata.copy()
        self.train_size = train_size
        self.compute_transcriptomics_list = compute_transcriptomics_list
        self.compute_morphology_list = compute_morphology_list

        if self.train_size:
            self._compute_train_idx()

        self.dataset = self.adata.obsm["X_st_grid"].astype(np.float32)

        self.multi_triplet_loss = multi_triplet_loss
        self.label_transcriptomics = self.adata.obs.X_pca_transcriptomics_cluster
        self.label_transcriptomics_unique = self.label_transcriptomics.unique()
        self.label_morphology = self.adata.obs.X_pca_morphology_cluster
        self.label_morphology_unique = self.label_morphology.unique()
        self.repeats = repeats

        # Determine the appropriate compute method based on the provided flags
        if compute_transcriptomics_list and compute_morphology_list:
            self.compute_method = self._compute_transcriptomics_and_morphology
        elif compute_transcriptomics_list:
            self.compute_method = self._compute_transcriptomics_only
        elif compute_morphology_list:
            self.compute_method = self._compute_morphology_only
        else:
            self.compute_method = self._compute_nothing

        # Create a dictionary to store indices for each unique label in label_transcriptomics
        self.label_transcriptomics_dict_idx = {}
        for l in self.label_transcriptomics_unique:
            self.label_transcriptomics_dict_idx[l] = np.random.permutation(
                np.where(self.label_transcriptomics == l)[0]
            )

        # Create a dictionary to store indices for each unique label in label_morphology
        self.label_morphology_dict_idx = {}
        for l in self.label_morphology_unique:
            self.label_morphology_dict_idx[l] = np.random.permutation(
                np.where(self.label_morphology == l)[0]
            )

    def __len__(self):
        return len(self.dataset)

    def _compute_train_idx(self):
        train_idx, _ = train_test_split(self.adata.obs.index, train_size=self.train_size)
        self.adata = self.adata[train_idx, :]

    def _compute_list(
            self,
            anchor_label,
            label_dict_idx,
            label_unique,
            repeats):
        # Randomly select positive indices for the anchor label
        pos_indices = np.random.choice(
            label_dict_idx[anchor_label],
            size=(len(label_unique) - 1) * repeats if self.multi_triplet_loss else repeats,
            replace=True
        )

        # Randomly select negative indices for all other labels except the anchor label
        if self.multi_triplet_loss:
            neg_indices = np.concatenate([
                np.random.choice(label_dict_idx[l], size=repeats, replace=True)
                for l in label_unique if l != anchor_label
            ])
        else:
            random_neg_label = np.random.choice([l for l in label_unique if l != anchor_label])
            neg_indices = np.random.choice(label_dict_idx[random_neg_label], size=repeats, replace=True)

        return pos_indices, neg_indices

    def _compute_transcriptomics_list(self, anchor_label_transcriptomics):
        return self._compute_list(
            anchor_label_transcriptomics,
            self.label_transcriptomics_dict_idx,
            self.label_transcriptomics_unique,
            self.repeats
        )

    def _compute_morphology_list(self, anchor_label_morphology):
        return self._compute_list(
            anchor_label_morphology,
            self.label_morphology_dict_idx,
            self.label_morphology_unique,
            self.repeats
        )

    def _compute_transcriptomics_only(self, anchor_label_transcriptomics, anchor_label_morphology):
        # Compute positive and negative indices for transcriptomics only
        return (
            *self._compute_transcriptomics_list(anchor_label_transcriptomics),
            [],
            []
        )

    def _compute_morphology_only(self, anchor_label_transcriptomics, anchor_label_morphology):
        # Compute positive and negative indices for morphology only
        return (
            [],
            [],
            *self._compute_morphology_list(anchor_label_morphology)
        )

    def _compute_nothing(self, anchor_label_transcriptomics, anchor_label_morphology):
        return (
            [],
            [],
            [],
            []
        )

    def _compute_transcriptomics_and_morphology(
            self,
            anchor_label_transcriptomics,
            anchor_label_morphology):
        # Compute positive and negative indices for both transcriptomics and morphology
        return (
            *self._compute_transcriptomics_list(anchor_label_transcriptomics),
            *self._compute_morphology_list(anchor_label_morphology)
        )

    def __getitem__(self, idx):
        # Retrieve the anchor, anchor label for transcriptomics, and anchor label for morphology
        anchor, anchor_label_transcriptomics, anchor_label_morphology = self.dataset[
            idx], self.label_transcriptomics[idx], self.label_morphology[idx]

        # Compute positive and negative indices based on the compute method
        pos_transcriptomics_indices, neg_transcriptomics_indices, pos_morphology_indices, neg_morphology_indices = self.compute_method(
            anchor_label_transcriptomics, anchor_label_morphology)

        # Retrieve the positive and negative samples for transcriptomics and morphology
        pos_transcriptomics = self.dataset[pos_transcriptomics_indices]
        neg_transcriptomics = self.dataset[neg_transcriptomics_indices]
        pos_morphology = self.dataset[pos_morphology_indices]
        neg_morphology = self.dataset[neg_morphology_indices]

        return anchor, pos_transcriptomics, neg_transcriptomics, pos_morphology, neg_morphology
