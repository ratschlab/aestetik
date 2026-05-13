"""End-to-end fit_predict smoke test on tiny synthetic data."""
import logging

import pytest

from aestetik import AESTETIK


@pytest.mark.slow
def test_fit_predict_smoke(small_adata):
    """Run the full fit -> predict pipeline on a tiny 6x6 grid for a
    handful of epochs. The test asserts that an embedding column was
    written and has the expected shape; we don't check learning.
    """
    logging.getLogger("lightning").setLevel(logging.ERROR)

    model = AESTETIK(
        nCluster=3,
        morphology_weight=1.5,
        total_weight=3.0,
        window_size=3,
        kernel_size=3,
        latent_dim=4,
        c_hid=8,
        max_epochs=1,
        clustering_method="kmeans",
        n_ensemble=1,
        n_neighbors=3,
        refine_cluster=False,
        n_jobs=1,
        num_workers=0,
    )
    model.fit(small_adata)
    model.predict(small_adata, num_repeats=2)

    assert "AESTETIK" in small_adata.obsm
    assert small_adata.obsm["AESTETIK"].shape == (small_adata.n_obs, 4)
    assert "AESTETIK_cluster" in small_adata.obs.columns
