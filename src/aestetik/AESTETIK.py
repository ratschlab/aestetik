from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist
from typing_extensions import Literal
from typing import Union, Optional
from tqdm import tqdm
import multiprocessing
from torch import nn
import numpy as np
import logging
import anndata
import torch

from .loss_function import *
from .utils.utils_morphology import *
from .utils.utils_clustering import *
from .utils.utils_vizualization import *
from .utils.utils_grid import *
from .dataloader import *
from .model import *

from aestetik.utils.utils_data import prepare_input_for_model as prepare_input


class AESTETIK:
    def __init__(
        self,
        adata: anndata,
        nCluster: Literal[int, float],
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
        device: Optional[str] = None,
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
        img_path: str = None,
        spot_diameter_fullres: int = None,
        refine_cluster: bool = True,
        n_jobs: int = 1
    ):
        """
        Initialize the model with the given parameters.

        Parameters
        ----------
        adata : anndata
            anndata object.
        nCluster : Literal[int, float]
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
        device : Optional[str], optional
            Device to use (e.g., "cpu" or "cuda").
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
        img_path : str, optional
            Path to the image data.
        spot_diameter_fullres : int, optional
            Diameter of spots in full resolution.
        refine_cluster : bool, optional (default=True)
            Whether to refine clusters after initial clustering.
        n_jobs : int, optional (default=1)
            Number of parallel jobs to run while building the grid.
        """

        # Initialize all the parameters and variables

        self.adata = adata
        self.nCluster = nCluster
        self.morphology_weight = morphology_weight
        self.total_weight = total_weight
        self.window_size = window_size
        self.train_size = train_size
        self.kernel_size = kernel_size if kernel_size < self.window_size else max(1, self.window_size - 1)

        self.latent_dim = latent_dim
        self.c_hid = c_hid
        self.lr = lr
        self.p = p
        self.epochs = epochs

        self.multi_triplet_loss = multi_triplet_loss
        self.rec_alpha = rec_alpha
        self.triplet_alpha = triplet_alpha
        self.n_repeats = n_repeats
        self.clustering_method = clustering_method

        self.batch_size = batch_size if batch_size else min(2 ** 13, len(self.adata))

        self.device = device if device else torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.n_ensemble_encoder = n_ensemble_encoder if n_ensemble_encoder else n_ensemble
        self.n_ensemble_decoder = n_ensemble_decoder if n_ensemble_decoder else n_ensemble
        self.save_emb = save_emb
        self.used_obsm_transcriptomics = used_obsm_transcriptomics
        self.used_obsm_morphology = used_obsm_morphology
        self.obsm_transcriptomics_dim = self.adata.obsm[self.used_obsm_transcriptomics].shape[1]
        self.used_obsm_combined = used_obsm_combined
        self.n_neighbors = n_neighbors
        self.weight_decay = weight_decay
        self.is_model_init = False
        self.img_path = img_path
        self.spot_diameter_fullres = spot_diameter_fullres
        self.refine_cluster = refine_cluster and self.n_neighbors > 1
        self.n_jobs = n_jobs if n_jobs != -1 else int(multiprocessing.cpu_count())

        logging.info(f"The model will be on device: {self.device}...")
        logging.info(f"The batch_size will be: {self.batch_size}.")
        logging.info(f"Number of threads: {self.n_jobs}.")
        logging.info(
            f"""Clustering will be performed using {
                self.clustering_method} and refine_cluster option: {
                self.refine_cluster}...""")

        fix_seed(random_seed)

    def summary(self):
        attr = vars(self)
        for item in attr.items():
            print(*item, sep=": ")

    def prepare_input_for_model(self): 
        """
        Prepare the input for training the model.
        1. Clustering raw input
        2. Grid building
        """
        self.transcriptomics_weight, self.morphology_weight = prepare_input(adata=self.adata,
                                                                                      nCluster=self.nCluster,
                                                                                      used_obsm_transcriptomics=self.used_obsm_transcriptomics,
                                                                                      used_obsm_morphology=self.used_obsm_morphology,
                                                                                      used_obsm_combined=self.used_obsm_combined,
                                                                                      clustering_method=self.clustering_method,
                                                                                      n_neighbors=self.n_neighbors,
                                                                                      window_size=self.window_size,
                                                                                      n_jobs=self.n_jobs,
                                                                                      total_weight=self.total_weight,
                                                                                      morphology_weight=self.morphology_weight)
    
    def _init_data_loader(self):

        self.data_loader = CustomDataset(
            adata=self.adata,
            train_size=self.train_size,
            multi_triplet_loss=self.multi_triplet_loss,
            repeats=self.n_repeats,
            compute_transcriptomics_list=self.transcriptomics_weight > 0,
            compute_morphology_list=self.morphology_weight > 0)

        self.loader = torch.utils.data.DataLoader(dataset=self.data_loader,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)

    def _init_model(self):
        self.model = AE(num_input_channels=self.adata.obsm["X_st_grid"].shape[1],
                        morphology_dim=self.adata.obsm["X_st_grid"].shape[2],
                        c_hid=self.c_hid,
                        latent_dim=self.latent_dim,
                        kernel_size=self.kernel_size,
                        p=self.p,
                        device=self.device,
                        n_ensemble_encoder=self.n_ensemble_encoder,
                        n_ensemble_decoder=self.n_ensemble_decoder)

        self.model.to(self.device)

    def _init_optim_and_loss(self):
        self.triplet_loss = nn.TripletMarginLoss()
        self.rec_loss = nn.L1Loss()

        # Using an Adam Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          amsgrad=True,
                                          weight_decay=self.weight_decay)

    def train(self):
        """
        Train model
        """
        if not self.is_model_init:
            logging.info(f"Init model attributes...")
            self._init_data_loader()
            self._init_model()
            self._init_optim_and_loss()
            self.is_model_init = True

        self.losses = []
        logging.info(f"Start training...")
        for epoch in (pbar := tqdm(range(self.epochs))):

            for (
                anchor,
                pos_transcriptomics_list,
                neg_transcriptomics_list,
                pos_morphology_list,
                    neg_morphology_list) in self.loader:

                anchor = anchor.to(self.device)

                anchor_encode, anchor_decode = self.model(anchor)

                total_loss, rec_loss_transcriptomics, rec_loss_morphology, triplet_loss_transcriptomics, triplet_loss_morphology = compute_loss(self.model,
                                                                                                                                                anchor,
                                                                                                                                                anchor_encode,
                                                                                                                                                anchor_decode,
                                                                                                                                                pos_transcriptomics_list, neg_transcriptomics_list,
                                                                                                                                                pos_morphology_list, neg_morphology_list,
                                                                                                                                                self.transcriptomics_weight,
                                                                                                                                                self.morphology_weight,
                                                                                                                                                self.triplet_loss,
                                                                                                                                                self.triplet_alpha,
                                                                                                                                                self.rec_loss,
                                                                                                                                                self.rec_alpha,
                                                                                                                                                self.obsm_transcriptomics_dim,
                                                                                                                                                self.device)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Storing the losses in a list for plotting
                total_loss = total_loss.cpu().detach().numpy()
                self.losses.append(total_loss)

            pbar.set_postfix_str(
                f"""E:{epoch},L:{
                    total_loss:.2f},RC:{
                    rec_loss_transcriptomics:.2f},RI:{
                    rec_loss_morphology:.2f},TC:{
                    triplet_loss_transcriptomics:.2f},TI:{
                        triplet_loss_morphology:.2f}""")

    def compute_spot_representations(
            self,
            n_repeats: int = 1000,
            cluster: bool = True):
        """
        Computing spot representations by sampling n_times. Then we cluster them into groups.

        Parameters
        ----------
        n_repeats : int, optional (default=1000)
            Number of times to repeat the computation.
        cluster : bool, optional (default=True)
            Whether to perform clustering on the computed representations.
        """
        logging.info(f"Running inference...")

        self.adata.obsm[self.save_emb] = self._compute_latent_space(n_repeats=n_repeats, batch_size=self.batch_size)

        if cluster:
            logging.info(f"Clustering using {self.clustering_method}...")
            clustering(
                self.adata,
                num_cluster=self.nCluster,
                used_obsm=self.save_emb,
                method=self.clustering_method,
                refine_cluster=self.refine_cluster,
                n_neighbors=self.n_neighbors)

    def _compute_latent_space(self, n_repeats, batch_size):
        all_spots = self.adata.obsm["X_st_grid"].astype(np.float32)
        all_latent_space = []
        for batch_spots in batch(all_spots, batch_size=batch_size):
            batch_spots = torch.from_numpy(batch_spots)
            batch_spots = batch_spots.to(self.device)
            batch_latent_space = []
            for i in range(n_repeats):
                out = self.model.encoder(batch_spots)
                out = out.cpu().detach().numpy()
                batch_latent_space.append(out)
            batch_latent_space = np.mean(batch_latent_space, axis=0)
            all_latent_space.extend(batch_latent_space)
        return np.array(all_latent_space)

    def _compute_centroid(self, topN=5):
        logging.info(f"Loading centroid info...")
        self.nc = NearestCentroid()
        self.nc.fit(self.adata.obsm[self.save_emb], self.adata.obs[f"{self.save_emb}_cluster"])

        self.dist_from_centroid = cdist(self.nc.centroids_, self.adata.obsm[self.save_emb])

        self.adata.obs["centroid"] = np.nan

        self.topN_centroid_idx = np.argpartition(self.dist_from_centroid, topN, axis=1)[
            :, :topN].reshape(-1, order="F")
        self.topN_centroid_label = np.tile(self.nc.classes_, topN)

        self.adata.obs.loc[self.adata.obs.index[self.topN_centroid_idx], "centroid"] = self.topN_centroid_label

        for dist_label, label in zip(self.dist_from_centroid, self.nc.classes_):
            self.adata.obs[f"dist_from_{label}"] = abs(((dist_label - dist_label.min()) /
                                                        (dist_label.max() - dist_label.min())) - 1)

    def _compute_centroid_morphology(self):
        logging.info(f"Loading centroid morphology spots...")
        if self.img_path and self.spot_diameter_fullres:
            plot_spots(
                self.img_path,
                self.adata,
                self.topN_centroid_idx,
                self.spot_diameter_fullres,
                f"{self.save_emb}_cluster")
        else:
            print("Morphology path or spot diameter is not specified...")

    def vizualize(self,
                  plot_loss: bool = False,
                  plot_clusters: bool = False,
                  plot_centroid: bool = False,
                  img_alpha: float = 0.6,
                  dot_size: int = 5,
                  ncols: int = 5):
        """
        Visualize different aspects of the model's output.

        Parameters
        ----------
        plot_loss : bool, optional (default=False)
            Whether to plot the training loss over epochs.
        plot_clusters : bool, optional (default=False)
            Whether to plot the clusters.
        plot_centroid : bool, optional (default=False)
            Whether to plot the centroids of the clusters.
        img_alpha : float, optional (default=0.6)
            Alpha blending value for the image (opacity).
        dot_size : int, optional (default=5)
            Size of the dots in the scatter plot.
        ncols : int, optional (default=5)
            Number of columns to use in the subplot grid.
        """

        if plot_loss:
            plot_loss_values(self.losses)
        if plot_clusters:
            plot_spatial_scatter_ari(self.adata,
                                     self.used_obsm_transcriptomics,
                                     self.used_obsm_morphology,
                                     self.save_emb,
                                     img_alpha=img_alpha,
                                     dot_size=dot_size,
                                     ncols=ncols)
        if self.adata.obs[f"{self.save_emb}_cluster"].unique().size > 1 and plot_centroid:
            self._compute_centroid()
            plot_spatial_centroids_and_distance(self.adata,
                                                self.save_emb,
                                                img_alpha=img_alpha,
                                                dot_size=dot_size,
                                                ncols=ncols)
            self._compute_centroid_morphology()

    @staticmethod
    def version():
        return "07.06.2024:8"
