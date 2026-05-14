import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import anndata
import lightning as L
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from torch.backends import cudnn

logger = logging.getLogger(__name__)


def fix_seed(seed: int) -> None:
    """Set all random seeds and configurations for reproducibility.

    Side effects
    ------------
    This function flips ``torch.backends.cudnn.deterministic = True`` and
    ``benchmark = False`` *globally* in the calling process and also
    sets ``PYTHONHASHSEED`` / ``CUBLAS_WORKSPACE_CONFIG`` environment
    variables. Other code running in the same process inherits those
    settings.

    Parameters
    ----------
    seed : int
        Value used for the Python, NumPy, PyTorch (CPU + CUDA), and
        Lightning RNGs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    L.seed_everything(seed, workers=True)



def create_st_grid(adata: anndata,
                   used_obsm: str,
                   window_size: int,
                   cpu_count: int,
                   used_obs_batch: Optional[str] = None) -> np.ndarray:
    """
    Creates a grid of features for each spot. Then it stores the result in adata.obsm['st_grid'].
    """
    x_array = adata.obs["x_array"].to_numpy()
    y_array = adata.obs["y_array"].to_numpy()
    embs = adata.obsm[used_obsm]
    embs = MinMaxScaler(feature_range=(0, 1)).fit_transform(embs)

    n_spots, dim_emb = embs.shape

    if used_obs_batch is not None and used_obs_batch in adata.obs.columns:
        batch_labels = adata.obs[used_obs_batch].astype("category").cat.codes.to_numpy()
    else:
        # Single-sample fixtures are the common case; demote to DEBUG
        # so it doesn't dominate test output.
        logger.debug(
            "No batch column specified or found in adata.obs; "
            "treating all spots as a single tissue slice."
        )
        batch_labels = np.zeros(len(x_array), dtype=int)

    trees, batch_to_indices = _build_trees(x_array=x_array,
                                           y_array=y_array,
                                           batch_labels=batch_labels)

    half = window_size // 2
    offsets_flat = _compute_offsets_flat(start=-half, end=-half + window_size)

    x_array.setflags(write=False)
    y_array.setflags(write=False)
    embs.setflags(write=False)
    batch_labels.setflags(write=False)
    offsets_flat.setflags(write=False)

    batch_indices = np.array_split(range(n_spots), cpu_count)
    delayed_create_batch_grid = delayed(_create_batch_grid)

    spot_grid = Parallel(n_jobs=cpu_count,
                       prefer="threads")(
                        delayed_create_batch_grid(spot_indices=spot_indices,
                                                  x_array=x_array,
                                                  y_array=y_array,
                                                  batch_labels=batch_labels,
                                                  embs=embs,
                                                  trees=trees,
                                                  batch_to_indices=batch_to_indices,
                                                  offsets_flat=offsets_flat,
                                                  window_size=window_size) for spot_indices in batch_indices
                       )
    spot_grid = np.concatenate(spot_grid)
    spot_grid = np.moveaxis(spot_grid, 3, 1)
    return spot_grid # shape: (num_spots, dim_emb, window_size, window_size)

def _build_trees(x_array: np.ndarray,
                 y_array: np.ndarray,
                 batch_labels: np.ndarray) -> Tuple[Dict[int, KDTree], Dict[int, np.ndarray]]:
    batch_ids = np.unique(batch_labels)
    trees: Dict[int, KDTree] = {}
    batch_to_indices: Dict[int, np.ndarray] = {}

    for batch_id in batch_ids:
        spot_indices = np.where(batch_labels == batch_id)[0]
        coords = np.column_stack([x_array[spot_indices], y_array[spot_indices]])
        trees[batch_id] = KDTree(coords)
        batch_to_indices[batch_id] = spot_indices

    return trees, batch_to_indices

def _create_batch_grid(spot_indices: np.ndarray,
                       x_array: np.ndarray,
                       y_array: np.ndarray,
                       batch_labels: np.ndarray,
                       embs: np.ndarray,
                       trees: Dict,
                       batch_to_indices: Dict,
                       offsets_flat: np.ndarray,
                       window_size: int) -> List[np.ndarray]:
    batch_grids = []
    for spot_index in spot_indices:
        spot = _create_spot(spot_idx=spot_index,
                            x_array=x_array,
                            y_array=y_array,
                            batch_labels=batch_labels,
                            embs=embs,
                            trees=trees,
                            batch_to_indices=batch_to_indices,
                            offsets_flat=offsets_flat,
                            window_size=window_size)
        batch_grids.append(spot)
    return batch_grids

def _create_spot(spot_idx: int,
                 x_array: np.ndarray,
                 y_array: np.ndarray,
                 batch_labels: np.ndarray,
                 embs:np.ndarray,
                 trees: Dict,
                 batch_to_indices: Dict,
                 offsets_flat: np.ndarray,
                 window_size: int) -> np.ndarray:
    """
    Creates a grid for a single spot.
    """
    x_center, y_center, batch_id = x_array[spot_idx], y_array[spot_idx], batch_labels[spot_idx]
    center = np.array([x_center, y_center])

    grid = np.full((window_size, window_size, embs.shape[1]),
                    fill_value=np.nan,
                    dtype=embs.dtype)
    indices_in_batch = batch_to_indices[batch_id]

    for offset_idx, (dx_offset, dy_offset) in enumerate(offsets_flat):
        position = center + np.array([dx_offset, dy_offset])
        distance, neighbor_idx = trees[batch_id].query(position)
        if distance > 0:
            continue
        grid_row, grid_column = np.unravel_index(offset_idx,
                                           shape=(window_size, window_size))

        grid[grid_row, grid_column] = embs[indices_in_batch[neighbor_idx]]

    # Impute NaNs with the per-embedding-dimension median. grid has shape
    # (window_size, window_size, dim_emb); nan_indices[2] gives the
    # embedding dimension of each NaN entry, while nan_indices[1] would
    # give a column position - silently corrupting data when window_size
    # <= dim_emb and crashing when window_size > dim_emb (issue #9).
    median_spot = np.nanmedian(grid, axis=(0, 1))
    nan_indices = np.where(np.isnan(grid))
    if nan_indices[0].size:
        grid[nan_indices] = np.take(median_spot, nan_indices[2])

    return grid

def _compute_offsets_flat(start:int, end:int) -> np.ndarray:
    offsets = np.arange(start, end)
    dx, dy = np.meshgrid(offsets, offsets)
    offsets_flat = np.stack([dx.ravel(), dy.ravel()], axis=1)
    return offsets_flat # shape: (N,2)
