"""Definition of AE Model"""

from torch import nn
import torch
import math


class Encode(nn.Module):
    def __init__(self, num_input_channels, c_hid, latent_dim, kernel_size, p, conv_out_2):
        super().__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=kernel_size, padding=0, stride=1),
            nn.BatchNorm2d(c_hid),
            nn.Dropout2d(p=p),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=3),
            nn.Flatten(),
            nn.Linear(conv_out_2 * conv_out_2 * c_hid, latent_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x[:, None, :]
        return x


class Decode(nn.Module):
    def __init__(self, num_input_channels, c_hid, latent_dim, kernel_size, p, conv_out_1, conv_out_2):
        super().__init__()

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, conv_out_2 * conv_out_2 * c_hid),
            nn.Unflatten(dim=1, unflattened_size=(-1, conv_out_2, conv_out_2)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(size=(conv_out_1, conv_out_1)),
            nn.Dropout2d(p=p),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=kernel_size, output_padding=0, padding=0, stride=1),
            nn.BatchNorm2d(num_input_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x[:, None, :]
        return x


class AE(nn.Module):
    def __init__(
            self,
            num_input_channels,
            morphology_dim,
            c_hid,
            latent_dim,
            kernel_size,
            p,
            n_ensemble_encoder=2,
            n_ensemble_decoder=1):
        super().__init__()

        conv_out_1 = (morphology_dim - kernel_size) + 1
        conv_out_2 = math.floor((conv_out_1 - kernel_size) / 3 + 1)

        # Encoder ensemble
        encoder_ensemble = [
            Encode(num_input_channels, c_hid, latent_dim, kernel_size, p, conv_out_2)
            for _ in range(n_ensemble_encoder)
        ]
        self.encoder_ensemble = nn.ModuleList(encoder_ensemble)

        # Decoder ensemble
        decoder_ensemble = [
            Decode(num_input_channels, c_hid, latent_dim, kernel_size, p, conv_out_1, conv_out_2)
            for _ in range(n_ensemble_decoder)
        ]
        self.decoder_ensemble = nn.ModuleList(decoder_ensemble)

    def encoder(self, x):
        x_encode = [unit(x) for unit in self.encoder_ensemble]
        x_encode = torch.cat(x_encode, 1)
        x_encode_median, _ = torch.median(x_encode, 1)
        return x_encode_median

    def decoder(self, x):
        x_decode = [unit(x) for unit in self.decoder_ensemble]
        x_decode = torch.cat(x_decode, 1)
        x_decode_median, _ = torch.median(x_decode, 1)
        return x_decode_median

    def forward(self, x):
        x_encode = self.encoder(x)
        x_decode = self.decoder(x_encode)
        return x_encode, x_decode
