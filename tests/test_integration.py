"""End-to-end smoke test of the sklearn surface on tiny synthetic data."""
import logging

import numpy as np
import pytest

from aestetik import AESTETIK


@pytest.mark.slow
def test_fit_transform_predict_smoke(small_adata):
    """fit -> transform -> predict on a tiny 6x6 grid, single epoch.

    We don't check learning quality here; we only assert that the
    sklearn surface produces arrays of the right shape and that X is
    not mutated.
    """
    logging.getLogger("lightning").setLevel(logging.ERROR)

    obsm_before = set(small_adata.obsm.keys())
    obs_before = set(small_adata.obs.columns)

    model = AESTETIK(
        n_cluster=3,
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
        num_repeats_predict=2,
    )
    fitted = model.fit(small_adata)
    assert fitted is model  # sklearn convention: fit returns self

    # Fitted-state attributes (trailing underscore).
    assert hasattr(model, "model_")
    assert hasattr(model, "trainer_")
    assert hasattr(model, "embedding_")
    assert hasattr(model, "labels_")
    assert hasattr(model, "losses_")
    assert model.embedding_.shape == (small_adata.n_obs, 4)
    assert model.labels_.shape == (small_adata.n_obs,)

    # transform / predict return arrays, never mutate.
    emb = model.transform(small_adata)
    labels = model.predict(small_adata)
    assert isinstance(emb, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert emb.shape == (small_adata.n_obs, 4)
    assert labels.shape == (small_adata.n_obs,)

    # X is untouched.
    assert set(small_adata.obsm.keys()) == obsm_before
    assert set(small_adata.obs.columns) == obs_before
