import torch
import torch.nn as nn


class WorldModel(nn.Module):

    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.gru = nn.GRU(latent_dim + action_dim, 128)

        self.fc_state = nn.Linear(128, latent_dim)
        self.fc_reward = nn.Linear(128, 1)

    def forward(self, z, action, hidden):

        x = torch.cat([z, action], dim=-1)

        x = x.unsqueeze(0)

        out, hidden = self.gru(x, hidden)

        h = out.squeeze(0)

        next_z = self.fc_state(h)

        reward = self.fc_reward(h)

        return next_z, reward, hidden