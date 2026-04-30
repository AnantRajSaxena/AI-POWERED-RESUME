import numpy as np
import torch
import torch.nn as nn


class RLAgent:
    """
    DQN-style agent (epsilon-greedy + experience replay) trained on
    REAL environment transitions — not imagination rollouts.

    This avoids the compounding error problem where decoded latent states
    drift away from valid grid states, causing policy collapse.

    The model-based component is used for planning (MPC), while the RL
    agent learns a value function from actual environment experience.
    """

    def __init__(
        self,
        state_dim: int,
        action_size: int = 4,
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps = 0

        # Q-network: state → Q-values for each action
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        ).to(device)

        # Target network for stable training
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        ).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — more stable than MSE

        # Replay buffer
        self._buf_s:  list = []
        self._buf_a:  list = []
        self._buf_r:  list = []
        self._buf_s2: list = []
        self._buf_d:  list = []
        self._buf_size = buffer_size

        # REINFORCE buffers (kept for compatibility, not used in DQN mode)
        self._log_probs: list = []
        self._rewards:   list = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state_np: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return self.act_greedy(state_np)

    def act_greedy(self, state_np: np.ndarray) -> int:
        """Greedy action — highest Q-value."""
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(state)
        return int(torch.argmax(q, dim=-1).item())

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def store(self, s, a, r, s2, done):
        """Store a real transition in the replay buffer."""
        if len(self._buf_s) >= self._buf_size:
            self._buf_s.pop(0);  self._buf_a.pop(0)
            self._buf_r.pop(0);  self._buf_s2.pop(0)
            self._buf_d.pop(0)
        self._buf_s.append(s.astype(np.float32))
        self._buf_a.append(int(a))
        self._buf_r.append(float(r))
        self._buf_s2.append(s2.astype(np.float32))
        self._buf_d.append(float(done))

    def update(self, gamma: float = None) -> float:
        """Sample a mini-batch and do one DQN gradient step."""
        if len(self._buf_s) < self.batch_size:
            return 0.0

        g = gamma if gamma is not None else self.gamma
        idx = np.random.choice(len(self._buf_s), self.batch_size, replace=False)

        s  = torch.tensor(np.stack([self._buf_s[i]  for i in idx]), dtype=torch.float32).to(self.device)
        a  = torch.tensor([self._buf_a[i]  for i in idx], dtype=torch.long).to(self.device)
        r  = torch.tensor([self._buf_r[i]  for i in idx], dtype=torch.float32).to(self.device)
        s2 = torch.tensor(np.stack([self._buf_s2[i] for i in idx]), dtype=torch.float32).to(self.device)
        d  = torch.tensor([self._buf_d[i]  for i in idx], dtype=torch.float32).to(self.device)

        # Current Q-values
        q_vals = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            next_actions = self.policy(s2).argmax(dim=1)
            next_q = self.target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = r + g * next_q * (1.0 - d)

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

        # Sync target network every 100 steps
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    def record_reward(self, reward: float):
        """Compatibility shim — not used in DQN mode."""
        self._rewards.append(reward)

    def clear_buffers(self):
        self._log_probs = []
        self._rewards = []


class MPCPlanner:
    """
    Random-shooting MPC planner operating in latent space.
    Samples action sequences, rolls them out through the latent dynamics
    model, and returns the first action of the best sequence.
    """

    def __init__(self, model, action_size: int = 4, horizon: int = 10,
                 samples: int = 500, device: str = "cpu"):
        self.model = model
        self.action_size = action_size
        self.horizon = horizon
        self.samples = samples
        self.device = device

    def _onehot(self, actions: np.ndarray) -> np.ndarray:
        return np.eye(self.action_size)[actions]

    def act(self, state_np: np.ndarray, goal_np: np.ndarray) -> int:
        S, H = self.samples, self.horizon

        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal  = torch.tensor(goal_np,  dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z_current = self.model.encode(state, deterministic=True)
            z_goal    = self.model.encode(goal,  deterministic=True)

        seqs    = np.random.randint(0, self.action_size, size=(S, H))
        seqs_oh = torch.tensor(self._onehot(seqs), dtype=torch.float32).to(self.device)

        z = z_current.repeat(S, 1)
        total_rewards = torch.zeros(S, device=self.device)

        with torch.no_grad():
            for t in range(H):
                a = seqs_oh[:, t, :]
                z = self.model.dynamics(z, a)
                r = -torch.norm(z - z_goal.expand(S, -1), dim=1)
                total_rewards += r

        best = torch.argmax(total_rewards).item()
        return int(seqs[best, 0])
