import numpy as np
import torch


class MPCPlanner:
    """Random shooting MPC over learned world model."""

    def __init__(self, model, action_size=4, horizon=8, samples=200, device='cpu'):
        self.model = model
        self.action_size = action_size
        self.horizon = horizon
        self.samples = samples
        self.device = device

    def _onehot(self, actions):
        # actions: (S,H) ints
        return np.eye(self.action_size)[actions]

    def act(self, state_np, goal_np):
        # state_np, goal_np: 1D arrays
        S = self.samples
        H = self.horizon
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal = torch.tensor(goal_np, dtype=torch.float32).to(self.device)

        # sample random action sequences
        seqs = np.random.randint(0, self.action_size, size=(S, H))
        seqs_oh = torch.tensor(self._onehot(seqs), dtype=torch.float32).to(self.device)

        states = state.repeat(S, 1)
        total_rewards = torch.zeros(S, device=self.device)

        for t in range(H):
            a = seqs_oh[:, t, :]
            pred = self.model(states, a)
            # simple reward: negative L2 distance to goal
            r = -torch.norm(pred - goal, dim=1)
            total_rewards += r
            states = pred

        best = torch.argmax(total_rewards).item()
        return int(seqs[best, 0])
