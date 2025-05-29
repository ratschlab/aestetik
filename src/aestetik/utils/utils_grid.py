from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from torch.backends import cudnn
from tqdm import tqdm
import numpy as np
import pyvips
import random
import torch
import os


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


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def _create_spot(matrix, x, y, window_size):

    start_x = x - window_size // 2
    end_x = start_x + window_size

    start_y = y - window_size // 2
    end_y = start_y + window_size

    spot = matrix[start_x:end_x, start_y:end_y]

    # filna with median across channels
    median_spot = np.nanmedian(spot, axis=(0, 1))
    inds = np.where(np.isnan(spot))
    spot[inds] = np.take(median_spot, inds[1])

    return spot


def _create_batch_grid(matrix, coord_x_y, window_size, idx):

    spot_grid = []

    for x, y in tqdm(coord_x_y, position=idx):

        spot = _create_spot(matrix, x, y, window_size=window_size)
        spot_grid.append(spot)

    return spot_grid


def create_st_grid(adata, used_obsm, window_size, cpu_count):

    minX = adata.obs.x_array.min()
    maxX = adata.obs.x_array.max() + window_size * 2
    minY = adata.obs.y_array.min()
    maxY = adata.obs.y_array.max() + window_size * 2
    x_array = adata.obs.x_array.values + window_size
    y_array = adata.obs.y_array.values + window_size

    emb = adata.obsm[used_obsm]

    # scale data
    emb = MinMaxScaler(feature_range=(0, 1)).fit_transform(emb)

    matrix = np.ones((maxX, maxY, emb.shape[1]), dtype=np.float64) * np.nan
    matrix[x_array, y_array] = emb
    # extract spots

    coord_x_y = np.array_split(adata.obs[["x_array", "y_array"]].values + window_size, cpu_count)

    delayed_create_batch_grid = delayed(_create_batch_grid)

    spot_grid = Parallel(
        cpu_count,
        prefer="threads")(
        delayed_create_batch_grid(
            matrix,
            coord,
            window_size,
            idx) for idx,
        coord in enumerate(coord_x_y))

    spot_grid = np.concatenate(spot_grid)
    spot_grid = np.moveaxis(spot_grid, 3, 1)

    return spot_grid


def batch(x, batch_size):
    spots_list = []
    for spot in x:
        spots_list.append(spot)

        if len(spots_list) > 0 and len(spots_list) % batch_size == 0:
            spots_list = np.array(spots_list)
            yield spots_list
            spots_list = []

    if len(spots_list) > 0:
        spots_list = np.array(spots_list)
        yield spots_list
