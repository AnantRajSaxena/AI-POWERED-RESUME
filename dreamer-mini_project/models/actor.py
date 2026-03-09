import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, z):
        return self.network(z)