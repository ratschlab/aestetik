"""Helpers for extracting per-spot morphology embeddings from a tissue
image. The pyvips dependency is loaded lazily so this module imports
cleanly even when libvips is not available.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from aestetik.utils._pyvips_dtype import FORMAT_TO_DTYPE


def extract_morphology_embeddings(
    img_path,
    model,
    x_pixel,
    y_pixel,
    spot_diameter,
    device,
    preprocess,
    feature_dim,
    n_components=15,
    apply_pca=True,
):
    """Extract morphology embeddings from a tissue image.

    Parameters
    ----------
    img_path : str
        Path to the image file.
    model : torch.nn.Module
        Model used for extracting features.
    x_pixel, y_pixel : Sequence[int]
        Pixel coordinates of each spot's center.
    spot_diameter : int
        Diameter of the spot in the image (pixels).
    device : torch.device or str
        Device on which to run ``model``.
    preprocess : Callable[[np.ndarray], torch.Tensor]
        Preprocessing function applied to each cropped spot before
        being fed to ``model``.
    feature_dim : int
        Output dimensionality of ``model``.
    n_components : int, optional (default=15)
        PCA dimensionality. Only used when ``apply_pca`` is True.
    apply_pca : bool, optional (default=True)
        Whether to reduce the morphology features via PCA.

    Returns
    -------
    embeddings : np.ndarray of shape (n_spots, feature_dim) or
        (n_spots, n_components) when ``apply_pca`` is True.
    """
    import pyvips  # optional dep: install via the [vips] extra

    model.to(device)
    image = pyvips.Image.new_from_file(img_path)

    n_spots = len(x_pixel)
    morphology_representation = np.zeros((n_spots, feature_dim))
    for i, (x, y) in tqdm(enumerate(zip(x_pixel, y_pixel)), total=n_spots):
        x = x - int(spot_diameter // 2)
        y = y - int(spot_diameter // 2)

        spot = image.crop(x, y, spot_diameter, spot_diameter)
        spot = np.ndarray(
            buffer=spot.write_to_memory(),
            dtype=FORMAT_TO_DTYPE[spot.format],
            shape=[spot.height, spot.width, spot.bands],
        )

        input_tensor = preprocess(spot)
        input_batch = input_tensor.unsqueeze(0).to(device)
        output = model(input_batch).detach().cpu().numpy().squeeze()

        morphology_representation[i, :] = output

    if apply_pca:
        pca = PCA(n_components=n_components)
        morphology_representation = pca.fit_transform(morphology_representation)

    return morphology_representation
