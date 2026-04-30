import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes raw state into a latent vector."""

    def __init__(self, state_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decodes a latent vector back to raw state space."""

    def __init__(self, latent_dim: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, z):
        return self.net(z)


class LatentDynamicsModel(nn.Module):
    """Predicts next latent state given current latent state and action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, action_onehot):
        x = torch.cat([z, action_onehot], dim=-1)
        return self.net(x)


class WorldModel(nn.Module):
    """
    Latent World Model:
      - Encoder maps raw state → latent (z)
      - LatentDynamicsModel predicts z_{t+1} from z_t + action
      - Decoder maps z back to raw state space for reward / planning

    The model supports both:
      - VAE-style training (reparameterisation trick, KL loss)
      - Deterministic mode (just use mu, skip sampling)
    """

    def __init__(
        self,
        state_dim: int = 165,
        action_dim: int = 4,
        latent_dim: int = 32,
        hidden: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(state_dim, latent_dim, hidden)
        self.decoder = Decoder(latent_dim, state_dim, hidden)
        self.dynamics = LatentDynamicsModel(latent_dim, action_dim, hidden)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reparameterise(self, mu, logvar):
        """Sample z ~ N(mu, exp(0.5*logvar)) during training."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, state, deterministic: bool = True):
        """Return latent vector for a raw state tensor."""
        mu, logvar = self.encoder(state)
        if deterministic:
            return mu
        return self.reparameterise(mu, logvar)

    # ------------------------------------------------------------------
    # Forward: predict next raw state (used by MPC planner)
    # ------------------------------------------------------------------

    def forward(self, state, action_onehot, deterministic: bool = True):
        """
        state        : (B, state_dim) float tensor
        action_onehot: (B, action_dim) float tensor
        returns      : (B, state_dim) predicted next state
        """
        mu, logvar = self.encoder(state)
        z = mu if deterministic else self.reparameterise(mu, logvar)
        z_next = self.dynamics(z, action_onehot)
        next_state = self.decoder(z_next)
        return next_state

    # ------------------------------------------------------------------
    # Training forward: returns everything needed to compute losses
    # ------------------------------------------------------------------

    def training_forward(self, state, action_onehot, next_state):
        """
        Returns a dict with:
          pred_next  : reconstructed next state
          z          : sampled latent of current state
          mu, logvar : for KL loss
          z_next     : predicted next latent
          recon_state: reconstruction of current state (for recon loss)
        """
        mu, logvar = self.encoder(state)
        z = self.reparameterise(mu, logvar)

        # Reconstruct current state
        recon_state = self.decoder(z)

        # Predict next latent and decode
        z_next = self.dynamics(z, action_onehot)
        pred_next = self.decoder(z_next)

        return {
            "pred_next": pred_next,
            "recon_state": recon_state,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_next": z_next,
        }
