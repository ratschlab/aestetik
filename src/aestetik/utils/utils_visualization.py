import matplotlib.pyplot as plt
import squidpy as sq
import anndata
import numpy as np
import pyvips
import math
import logging
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist

from aestetik.AESTETIK import AESTETIK

from typing import Optional 

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def visualize(model: AESTETIK,
              adata: Optional[anndata.AnnData] = None,
              img_path: Optional[str] = None,
              spot_diameter_fullres: Optional[int] = None,
              used_obsm_transcriptomics: str = "X_pca_transcriptomics",
              used_obsm_morphology: str = "X_pca_morphology",
              save_emb: str = "AESTETIK",
              plot_loss: bool = False,
              plot_clusters: bool = False,
              plot_centroid: bool = False,
              img_alpha: float = 0.6,
              dot_size: int = 5,
              ncols: int = 5) -> None:
        """
        Visualize different aspects of the model's output.

        Parameters
        ----------
        model : AESTETIK
            AESTETIK model.
        adata : Optional[anndata.AnnData], required for plotting clusters or centroids (default=None)
            AnnData object.
        img_path : Optional[str], required for plotting centroids (default=None)
            Path to the image data.
        spot_diameter_fullres : Optional[int], required for plotting centroids (default=None)
            Diameter of spots in full resolution.
        used_obsm_transcriptomics : str, optional (default="X_pca_transcriptomics")
            Key for transcriptomics data in `obsm`.
        used_obsm_morphology : str, optional (default="X_pca_morphology")
            Key for morphology data in `obsm`.
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
        _validate_visualize_inputs(adata=adata,
                                  img_path=img_path,
                                  spot_diameter_fullres=spot_diameter_fullres,
                                  save_emb=save_emb,
                                  plot_loss=plot_loss,
                                  plot_clusters=plot_clusters,
                                  plot_centroid=plot_centroid)
        if plot_loss:
            _plot_loss_values(model.losses)
        if plot_clusters:
            _plot_spatial_scatter_ari(adata,
                                      used_obsm_transcriptomics,
                                      used_obsm_morphology,
                                      save_emb,
                                      img_alpha=img_alpha,
                                      dot_size=dot_size,
                                      ncols=ncols)
        if plot_centroid:
                topN_centroid_idx = _compute_centroid(adata=adata,
                                                      save_emb=save_emb)
                _plot_spatial_centroids_and_distance(adata,
                                                     save_emb,
                                                     img_alpha=img_alpha,
                                                     dot_size=dot_size,
                                                     ncols=ncols)
                _compute_centroid_morphology(img_path=img_path,
                                                  adata=adata,
                                                  topN_centroid_idx=topN_centroid_idx,
                                                  spot_diameter_fullres=spot_diameter_fullres,
                                                  save_emb=save_emb)

# ================================================================= #
#                    Private Plotting Methods                       #
# ================================================================= # 
def _plot_loss_values(losses):
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()


def _plot_spatial_centroids_and_distance(adata,
                                         save_emb,
                                         img_alpha: float = 0.6,
                                         dot_size: int = 5,
                                         ncols: int = 5):

    sq.pl.spatial_scatter(
        adata,
        img_alpha=img_alpha,
        color=[f"{save_emb}_cluster",
               "centroid", *[col for col in adata.obs.columns if "dist_from_" in col]],
        size=dot_size,
        ncols=ncols)


def _plot_spatial_scatter_ari(adata,
                              used_obsm_transcritpomics,
                              used_obsm_morphology,
                              save_emb,
                              img_alpha: float = 0.6,
                              dot_size: int = 5,
                              ncols: int = 5):

    label_list = [
        f"{used_obsm_transcritpomics}_cluster",
        f"{used_obsm_morphology}_cluster",
        f"{save_emb}_cluster"
    ]
    lebel_list = [l for l in label_list if l in adata.obs.columns]

    ari_list = [
        adjusted_rand_score(
            adata.obs.ground_truth,
            adata.obs[label].values) for label in lebel_list]

    title = [
        f"{label}, ARI:{ari:.2f}" for label,
        ari in zip(
            lebel_list,
            ari_list)]

    sq.pl.spatial_scatter(
        adata,
        img_alpha=img_alpha,
        color=["ground_truth", *lebel_list],
        title=["ground_truth", *title],
        size=dot_size,
        ncols=ncols)

def _plot_spots(img_path, adata, indeces_to_plot, spot_diameter_fullres, label=None):

    image = pyvips.Image.new_from_file(img_path)
    tab = adata.obs.iloc[indeces_to_plot]

    n_labels = np.unique(tab[label]).size

    columns = min(10, n_labels)
    rows = math.ceil(len(tab) / columns)

    fig = plt.figure(figsize=(columns * 2, rows * 2))
    for i in range(1, len(tab) + 1):
        row = tab.iloc[i - 1]
        img = _get_spot(image, row.y_pixel, row.x_pixel, spot_diameter_fullres)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.title(f"Centroid {row[label]}")
        plt.axis('off')
    plt.show()

# ================================================================= #
#                     Private Extraction Methods                    #
# ================================================================= # 
def _get_spot(image, x, y, spot_diameter_fullres):
    x = x - int(spot_diameter_fullres // 2)
    y = y - int(spot_diameter_fullres // 2)
    spot = image.crop(x, y, spot_diameter_fullres, spot_diameter_fullres)
    spot_array = np.ndarray(buffer=spot.write_to_memory(),
                            dtype=format_to_dtype[spot.format],
                            shape=[spot.height, spot.width, spot.bands])
    return spot_array

# ================================================================= #
#              Private Centroid Computation Methods                 #
# ================================================================= # 
def _compute_centroid(adata: anndata.AnnData,
                      save_emb: str,
                      topN: int = 5) -> np.ndarray:
    logging.info("Loading centroid info...")
    nc = NearestCentroid()
    nc.fit(adata.obsm[save_emb], adata.obs[f"{save_emb}_cluster"])

    dist_from_centroid = cdist(nc.centroids_, adata.obsm[save_emb])

    adata.obs["centroid"] = np.nan

    topN_centroid_idx = np.argpartition(dist_from_centroid, topN, axis=1)[:, :topN].reshape(-1, order="F")
    topN_centroid_label = np.tile(nc.classes_, topN)

    adata.obs.loc[adata.obs.index[topN_centroid_idx], "centroid"] = topN_centroid_label

    for dist_label, label in zip(dist_from_centroid, nc.classes_):
        adata.obs[f"dist_from_{label}"] = abs(((dist_label - dist_label.min()) /
                                                        (dist_label.max() - dist_label.min())) - 1)
    return topN_centroid_idx

def _compute_centroid_morphology(img_path: str,
                                 adata: anndata.AnnData,
                                 topN_centroid_idx: np.ndarray,
                                 spot_diameter_fullres: int,
                                 save_emb: str) -> None: 
    logging.info("Loading centroid morphology spots...")
    if img_path and spot_diameter_fullres:
        _plot_spots(
                    img_path,
                    adata,
                    topN_centroid_idx,
                    spot_diameter_fullres,
                    f"{save_emb}_cluster")
    else:
        print("Morphology path or spot diameter is not specified...")

# ================================================================= #
#                    Private Validation Methods                     #
# ================================================================= # 
def _validate_visualize_inputs(
    adata: Optional[anndata.AnnData],
    img_path: Optional[str],
    spot_diameter_fullres: Optional[int],
    save_emb: str,
    plot_loss: bool,
    plot_clusters: bool,
    plot_centroid: bool) -> None:
    
    if plot_clusters and adata is None:
        raise ValueError("Cannot plot clusters: 'adata' must be provided (not None)."
                                 "Please specify a valid AnnData object.")
    if plot_centroid:
        if adata is None:
            raise ValueError("Cannot plot centroids: 'adata' must be provided (not None).")
        if adata.obs[f"{save_emb}_cluster"].nunique() <= 1:
            raise ValueError(f"Cannot plot centroids: more than one cluster required in '{save_emb}_cluster'.")
        if spot_diameter_fullres is None or img_path is None:
            raise ValueError("Cannot plot centroids: 'spot_diameter_fullres' and 'img_path' must be provided.")