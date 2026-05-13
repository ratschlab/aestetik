"""Tests for ``aestetik.metrics.loss_function``."""
import torch
from torch import nn

from aestetik.metrics.loss_function import (
    compute_loss,
    compute_reconstruction_loss,
)
from aestetik.models.model import AE


def test_reconstruction_loss_is_zero_for_identical_inputs():
    a = torch.zeros(4, 6)
    out = compute_reconstruction_loss(a, a.clone(), nn.L1Loss())
    assert out.item() == 0.0


def test_compute_loss_returns_finite_scalar():
    model = AE(
        num_input_channels=6,
        morphology_dim=5,
        c_hid=8,
        latent_dim=4,
        kernel_size=3,
        p=0.0,
        n_ensemble_encoder=1,
        n_ensemble_decoder=1,
    )
    B = 3
    anchor = torch.randn(B, 6, 5, 5)
    anchor_encode = torch.randn(B, 4)
    anchor_decode = torch.randn(B, 6, 5, 5)
    # triplet lists have shape (B, num_samples, C, H, W)
    pos = torch.randn(B, 2, 6, 5, 5)
    neg = torch.randn(B, 2, 6, 5, 5)

    total, rc, ri, tc, ti = compute_loss(
        model=model,
        anchor=anchor,
        anchor_encode=anchor_encode,
        anchor_decode=anchor_decode,
        pos_transcriptomics_list=pos,
        neg_transcriptomics_list=neg,
        pos_morphology_list=pos,
        neg_morphology_list=neg,
        transcriptomics_weight=1.0,
        morphology_weight=1.0,
        triplet_loss=nn.TripletMarginLoss(),
        triplet_alpha=1.0,
        rec_loss=nn.L1Loss(),
        rec_alpha=1.0,
        obsm_transcriptomics_dim=3,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(total)
    assert total.dim() == 0
