import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, obs_dim, latent_dim=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.network(x)