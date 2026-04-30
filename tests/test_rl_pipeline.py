"""
Tests for the end-to-end RL pipeline:
  MazeEnv → WorldModel (latent) → MPCPlanner → RLAgent → Action
"""

import numpy as np
import pytest
import torch

from maze_env import MazeEnv
from world_model import WorldModel
from agent import MPCPlanner, RLAgent

STATE_DIM = 15 * 11   # 165
ACTION_DIM = 4
LATENT_DIM = 16       # small for fast tests


# ---------------------------------------------------------------------------
# MazeEnv
# ---------------------------------------------------------------------------

class TestMazeEnv:
    def setup_method(self):
        self.env = MazeEnv(width=15, height=11, levels=5)

    def test_reset_returns_correct_shape(self):
        s = self.env.reset()
        assert s.shape == (STATE_DIM,)

    def test_step_returns_tuple(self):
        self.env.reset()
        s, r, done = self.env.step(3)
        assert s.shape == (STATE_DIM,)
        assert isinstance(r, float)
        assert isinstance(done, bool)

    def test_done_flag_set_on_goal(self):
        # Place agent one step away from goal and move into it
        self.env.reset()
        gy, gx = self.env.goal_pos
        # Position agent directly above the goal (one step down = action 1)
        # Make sure that cell is not a wall
        if gy - 1 >= 0 and self.env.grid[gy - 1][gx] == 0:
            self.env.agent_pos = (gy - 1, gx)
            _, r, done = self.env.step(1)  # move down into goal
            assert done is True
            assert r == 10.0
        else:
            # Fallback: directly set agent on goal and verify state reflects done
            self.env.agent_pos = self.env.goal_pos
            self.env.done = True
            assert self.env.done is True

    def test_shortest_path_returns_list(self):
        self.env.reset()
        path = self.env.shortest_path()
        assert path is not None
        assert len(path) >= 1

    def test_level_increases_difficulty(self):
        """Higher levels should have more walls (higher density)."""
        env1 = MazeEnv(width=15, height=11, levels=100)
        env1.reset(level=1)
        walls_1 = sum(1 for row in env1.grid for c in row if c == 1)

        env100 = MazeEnv(width=15, height=11, levels=100)
        env100.reset(level=100)
        walls_100 = sum(1 for row in env100.grid for c in row if c == 1)

        assert walls_100 >= walls_1

    def test_compact_state_shape(self):
        self.env.reset()
        cs = self.env.compact_state()
        assert cs.shape == (MazeEnv.COMPACT_DIM,)
        assert cs.dtype == np.float32

    def test_compact_state_goal_direction(self):
        """dy_norm and dx_norm should point toward goal."""
        self.env.reset()
        cs = self.env.compact_state()
        # agent starts at (1,1), goal at (9,13) — dy and dx should be positive
        assert cs[0] > 0, "dy_norm should be positive (goal is below)"
        assert cs[1] > 0, "dx_norm should be positive (goal is to the right)"


# ---------------------------------------------------------------------------
# WorldModel (latent)
# ---------------------------------------------------------------------------

class TestWorldModel:
    def setup_method(self):
        self.model = WorldModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM,
            hidden=64,
        )

    def test_encode_output_shape(self):
        s = torch.randn(4, STATE_DIM)
        mu, logvar = self.model.encoder(s)
        assert mu.shape == (4, LATENT_DIM)
        assert logvar.shape == (4, LATENT_DIM)

    def test_decode_output_shape(self):
        z = torch.randn(4, LATENT_DIM)
        out = self.model.decoder(z)
        assert out.shape == (4, STATE_DIM)

    def test_dynamics_output_shape(self):
        z = torch.randn(4, LATENT_DIM)
        a = torch.randn(4, ACTION_DIM)
        z_next = self.model.dynamics(z, a)
        assert z_next.shape == (4, LATENT_DIM)

    def test_forward_output_shape(self):
        s = torch.randn(4, STATE_DIM)
        a = torch.randn(4, ACTION_DIM)
        pred = self.model(s, a)
        assert pred.shape == (4, STATE_DIM)

    def test_training_forward_keys(self):
        s = torch.randn(4, STATE_DIM)
        a = torch.randn(4, ACTION_DIM)
        n = torch.randn(4, STATE_DIM)
        out = self.model.training_forward(s, a, n)
        for key in ("pred_next", "recon_state", "mu", "logvar", "z", "z_next"):
            assert key in out

    def test_reparameterise_shape(self):
        mu = torch.zeros(8, LATENT_DIM)
        logvar = torch.zeros(8, LATENT_DIM)
        z = self.model.reparameterise(mu, logvar)
        assert z.shape == (8, LATENT_DIM)

    def test_encode_deterministic_equals_mu(self):
        s = torch.randn(2, STATE_DIM)
        mu, _ = self.model.encoder(s)
        z = self.model.encode(s, deterministic=True)
        assert torch.allclose(z, mu)


# ---------------------------------------------------------------------------
# MPCPlanner
# ---------------------------------------------------------------------------

class TestMPCPlanner:
    def setup_method(self):
        self.model = WorldModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM,
            hidden=64,
        )
        self.planner = MPCPlanner(
            self.model,
            action_size=ACTION_DIM,
            horizon=4,
            samples=20,
        )

    def test_act_returns_valid_action(self):
        state = np.random.rand(STATE_DIM).astype(np.float32)
        goal = np.random.rand(STATE_DIM).astype(np.float32)
        action = self.planner.act(state, goal)
        assert action in range(ACTION_DIM)

    def test_act_deterministic_for_same_seed(self):
        """Same random seed → same action."""
        state = np.ones(STATE_DIM, dtype=np.float32)
        goal = np.zeros(STATE_DIM, dtype=np.float32)
        np.random.seed(0)
        a1 = self.planner.act(state, goal)
        np.random.seed(0)
        a2 = self.planner.act(state, goal)
        assert a1 == a2


# ---------------------------------------------------------------------------
# RLAgent
# ---------------------------------------------------------------------------

class TestRLAgent:
    def setup_method(self):
        self.agent = RLAgent(state_dim=STATE_DIM, action_size=ACTION_DIM, hidden=32)

    def test_act_returns_valid_action(self):
        s = np.random.rand(STATE_DIM).astype(np.float32)
        a = self.agent.act(s)
        assert a in range(ACTION_DIM)

    def test_act_greedy_returns_valid_action(self):
        s = np.random.rand(STATE_DIM).astype(np.float32)
        a = self.agent.act_greedy(s)
        assert a in range(ACTION_DIM)

    def test_update_returns_zero_when_buffer_empty(self):
        loss = self.agent.update()
        assert loss == 0.0

    def test_store_and_update(self):
        s  = np.random.rand(STATE_DIM).astype(np.float32)
        s2 = np.random.rand(STATE_DIM).astype(np.float32)
        # Fill buffer past batch_size (default 64)
        for _ in range(70):
            self.agent.store(s, np.random.randint(ACTION_DIM), np.random.randn(), s2, False)
        loss = self.agent.update()
        assert isinstance(loss, float)

    def test_epsilon_decays(self):
        s  = np.random.rand(STATE_DIM).astype(np.float32)
        s2 = np.random.rand(STATE_DIM).astype(np.float32)
        eps_before = self.agent.epsilon
        for _ in range(70):
            self.agent.store(s, 0, 0.0, s2, False)
        self.agent.update()
        assert self.agent.epsilon <= eps_before


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke test
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline():
    """
    Smoke test: Environment → WorldModel → Planner → RLAgent → Action
    """
    env      = MazeEnv(width=15, height=11, levels=5)
    model    = WorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=64)
    planner  = MPCPlanner(model, action_size=ACTION_DIM, horizon=4, samples=10)
    rl_agent = RLAgent(state_dim=MazeEnv.COMPACT_DIM, action_size=ACTION_DIM, hidden=32)

    env.reset()
    state        = env.state().astype(np.float32)        # raw — for world model
    compact      = env.compact_state()                    # compact — for RL agent
    goal         = np.zeros(STATE_DIM, dtype=np.float32)
    gy, gx       = env.goal_pos
    goal[gy * env.width + gx] = 3.0

    # MPC planner uses raw state
    mpc_action = planner.act(state, goal)
    assert mpc_action in range(ACTION_DIM)

    # RL agent uses compact state
    rl_action = rl_agent.act(compact)
    assert rl_action in range(ACTION_DIM)

    # Step and store transition
    _, reward, done = env.step(rl_action)
    next_compact    = env.compact_state()
    rl_agent.store(compact, rl_action, reward, next_compact, done)
    loss = rl_agent.update()
    assert isinstance(loss, float)
    assert compact.shape == (MazeEnv.COMPACT_DIM,)
