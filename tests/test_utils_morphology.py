"""Tests for ``aestetik.utils.utils_morphology.extract_morphology_embeddings``.

The full extractor requires the optional ``pyvips`` package (and a
system libvips). We skip the module if either isn't available so the
test runs on machines without the morphology stack.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from torch import nn


# Import pyvips with a broad exception net: in some conda envs libvips
# fails to load (libjpeg ABI mismatch) and pytest.importorskip
# (ImportError-only) is not enough.
try:
    import pyvips  # noqa: F401
except Exception as _pyvips_exc:  # pragma: no cover
    pytest.skip(
        f"pyvips not usable in this env: {_pyvips_exc!r}",
        allow_module_level=True,
    )

from aestetik.utils.utils_morphology import extract_morphology_embeddings  # noqa: E402


class _IdentityChannelMean(nn.Module):
    """Tiny stand-in for an image backbone: pools spatial dims to a fixed
    feature vector. Avoids downloading Inception V3 in tests.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        # linear projection from RGB-mean (3 channels) to feature_dim
        self.proj = nn.Linear(3, feature_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W). Take per-channel spatial mean -> (B, 3).
        pooled = x.mean(dim=(2, 3))
        return self.proj(pooled)


def _make_test_image(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    image_path = tmp_path / "tile.png"
    pyvips.Image.new_from_array(arr).write_to_file(str(image_path))
    return image_path


def test_extract_morphology_embeddings_with_pca(tmp_path):
    img = _make_test_image(tmp_path)
    feature_dim = 16
    model = _IdentityChannelMean(feature_dim)

    def preprocess(arr):  # arr is (H, W, 3) uint8
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    x_pixel = [16, 32, 48]
    y_pixel = [16, 32, 48]
    out = extract_morphology_embeddings(
        img_path=str(img),
        model=model,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        spot_diameter=16,
        device=torch.device("cpu"),
        preprocess=preprocess,
        feature_dim=feature_dim,
        n_components=4,
        apply_pca=True,
    )
    assert out.shape == (3, 4)


def test_extract_morphology_embeddings_without_pca(tmp_path):
    img = _make_test_image(tmp_path)
    feature_dim = 8
    model = _IdentityChannelMean(feature_dim)

    def preprocess(arr):
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    out = extract_morphology_embeddings(
        img_path=str(img),
        model=model,
        x_pixel=[16, 32],
        y_pixel=[16, 32],
        spot_diameter=16,
        device=torch.device("cpu"),
        preprocess=preprocess,
        feature_dim=feature_dim,
        apply_pca=False,
    )
    assert out.shape == (2, feature_dim)
