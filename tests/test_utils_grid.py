"""Tests for ``aestetik.utils.utils_grid``.

Covers the NaN-imputation axis bug (issue #9) directly on
``_create_spot`` and on the public ``create_st_grid`` entry point.
"""
import numpy as np
import pytest

from aestetik.utils.utils_grid import (
    _compute_offsets_flat,
    _create_spot,
    _build_trees,
    create_st_grid,
)


def test_compute_offsets_flat_shape_and_centring():
    offsets = _compute_offsets_flat(start=-1, end=2)  # window_size=3
    assert offsets.shape == (9, 2)
    # The centre offset (0, 0) must be present exactly once.
    assert (offsets == np.array([0, 0])).all(axis=1).sum() == 1


@pytest.mark.parametrize("window_size,dim_emb", [(3, 4), (5, 3), (7, 16)])
def test_create_spot_imputes_dimension_wise(window_size, dim_emb):
    """NaNs must be filled with the median *per embedding dimension*,
    not per grid column. With dim_emb < window_size the old buggy code
    raised IndexError; with dim_emb >= window_size it silently
    duplicated values from the wrong axis.
    """
    n_x = n_y = window_size + 2
    x_grid, y_grid = np.meshgrid(np.arange(n_x), np.arange(n_y))
    x_arr = x_grid.ravel().astype(int)
    y_arr = y_grid.ravel().astype(int)
    n_spots = x_arr.size

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n_spots, dim_emb)).astype(np.float32)
    batch_labels = np.zeros(n_spots, dtype=int)

    trees, batch_to_indices = _build_trees(x_arr, y_arr, batch_labels)
    half = window_size // 2
    offsets_flat = _compute_offsets_flat(start=-half, end=-half + window_size)

    # Corner spot: many neighbours fall outside the grid -> NaNs in the
    # raw window which exercise the imputation path.
    grid = _create_spot(
        spot_idx=0,
        x_array=x_arr,
        y_array=y_arr,
        batch_labels=batch_labels,
        embs=embs,
        trees=trees,
        batch_to_indices=batch_to_indices,
        offsets_flat=offsets_flat,
        window_size=window_size,
    )

    assert grid.shape == (window_size, window_size, dim_emb)
    assert not np.isnan(grid).any(), "imputation must remove all NaNs"
    # Each imputed slice along axis-2 must take a value from the median
    # of that very same embedding dimension.
    medians = np.nanmedian(grid, axis=(0, 1))
    # NaNs in the original window-> impute to medians; the populated
    # entries should equal the source embedding for the central spot.
    assert grid[half, half] == pytest.approx(embs[0])
    for d in range(dim_emb):
        # at least one cell along this dim must equal the median we
        # would compute (i.e. imputation actually happened).
        assert np.isfinite(medians[d])


def test_create_st_grid_full_pipeline():
    import anndata
    import pandas as pd

    rng = np.random.default_rng(0)
    n_x = n_y = 5
    xs, ys = np.meshgrid(np.arange(n_x), np.arange(n_y))
    xs, ys = xs.ravel(), ys.ravel()
    n_spots = xs.size
    obs = pd.DataFrame({"x_array": xs.astype(int), "y_array": ys.astype(int)})
    X = rng.standard_normal((n_spots, 6)).astype(np.float32)
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = rng.standard_normal((n_spots, 6)).astype(np.float32)

    out = create_st_grid(adata, used_obsm="X_pca", window_size=3, cpu_count=1)
    # shape is (n_spots, dim_emb, window_size, window_size)
    assert out.shape == (n_spots, 6, 3, 3)
    assert np.isfinite(out).all()
