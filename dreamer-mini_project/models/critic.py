import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.network(z)