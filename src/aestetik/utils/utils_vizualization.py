from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import squidpy as sq
import scanpy as sc
import numpy as np
import pyvips
import math

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


def plot_loss_values(losses):
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()


def plot_spatial_centroids_and_distance(adata,
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


def plot_spatial_scatter_ari(adata,
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


def get_spot(image, x, y, spot_diameter_fullres):
    x = x - int(spot_diameter_fullres // 2)
    y = y - int(spot_diameter_fullres // 2)
    spot = image.crop(x, y, spot_diameter_fullres, spot_diameter_fullres)
    spot_array = np.ndarray(buffer=spot.write_to_memory(),
                            dtype=format_to_dtype[spot.format],
                            shape=[spot.height, spot.width, spot.bands])
    return spot_array


def plot_spots(img_path, adata, indeces_to_plot, spot_diameter_fullres, label=None):

    image = pyvips.Image.new_from_file(img_path)
    tab = adata.obs.iloc[indeces_to_plot]

    n_labels = np.unique(tab[label]).size

    columns = min(10, n_labels)
    rows = math.ceil(len(tab) / columns)

    fig = plt.figure(figsize=(columns * 2, rows * 2))
    for i in range(1, len(tab) + 1):
        row = tab.iloc[i - 1]
        img = get_spot(image, row.y_pixel, row.x_pixel, spot_diameter_fullres)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.title(f"Centroid {row[label]}")
        plt.axis('off')
    plt.show()
