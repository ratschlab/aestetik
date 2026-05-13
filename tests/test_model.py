"""Tests for the autoencoder model architecture."""
import torch

from aestetik.models.model import AE


def test_ae_forward_shapes():
    num_input_channels = 8
    window_size = 7
    model = AE(
        num_input_channels=num_input_channels,
        morphology_dim=window_size,
        c_hid=16,
        latent_dim=8,
        kernel_size=3,
        p=0.1,
        n_ensemble_encoder=2,
        n_ensemble_decoder=2,
    )
    x = torch.randn(4, num_input_channels, window_size, window_size)
    encoded, decoded = model(x)
    # encoded: (B, latent_dim)
    assert encoded.shape == (4, 8)
    # decoded: (B, num_input_channels, window_size, window_size)
    assert decoded.shape == (4, num_input_channels, window_size, window_size)


def test_ae_encoder_decoder_separately():
    model = AE(
        num_input_channels=4,
        morphology_dim=5,
        c_hid=8,
        latent_dim=6,
        kernel_size=3,
        p=0.0,
        n_ensemble_encoder=1,
        n_ensemble_decoder=1,
    )
    x = torch.randn(2, 4, 5, 5)
    z = model.encoder(x)
    assert z.shape == (2, 6)
    out = model.decoder(z)
    assert out.shape == (2, 4, 5, 5)
