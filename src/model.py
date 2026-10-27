"""
Routine autoencoder.

The model learns to rebuild a normal day. If a new day cannot be rebuilt well,
it is different from what the model has learned, so it gets a high anomaly
score.

It has two parts:
  - encoder: compresses the day into 32 numbers. In personalised federated
    learning (FedPer) this part is SHARED between homes.
  - decoder: rebuilds the day from those 32 numbers. In FedPer this part stays
    in each home and is never sent anywhere, so it can learn that home's
    own habits and sensor layout.
"""

import torch
from torch import nn

from .features import N_ROOMS, N_SLOTS

D_IN = N_ROOMS * N_SLOTS


class RoutineAE(nn.Module):
    def __init__(self, hidden=128, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, D_IN),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def masked_mse(recon, x, cmask):
    """Mean squared error over the cells of rooms the home actually has."""
    err = (recon - x) ** 2 * cmask
    return err.sum(dim=1) / cmask.sum(dim=1).clamp(min=1)


def n_params(module):
    return sum(p.numel() for p in module.parameters())
