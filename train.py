"""
Model-Based RL Training Pipeline
==================================
End-to-end:  Environment → World Model (latent) → MPC Planner → RL Agent → Action

Phase 1 — World Model pre-training
  Uses full 165-dim raw state.  Trains Encoder + LatentDynamics + Decoder.

Phase 2 — DQN RL Agent
  Uses 31-dim COMPACT state (relative goal position + local wall patch).
  Trains on REAL environment transitions with experience replay.
  Compact state makes learning tractable in ~200 episodes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from world_model import WorldModel
from agent import MPCPlanner, RLAgent
from maze_env import MazeEnv

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
RAW_DIM     = 15 * 11          # 165 — full grid for world model
COMPACT_DIM = MazeEnv.COMPACT_DIM  # 31  — compact state for RL agent
ACTION_DIM  = 4
LATENT_DIM  = 32
HIDDEN      = 128
DEVICE      = "cpu"


# -----------------------------------------------------------------------
# Phase 1: World Model pre-training  (uses raw 165-dim state)
# -----------------------------------------------------------------------

def collect_random_data(env: MazeEnv, episodes: int = 300, max_steps: int = 50):
    data = []
    for _ in range(episodes):
        s = env.reset().astype(np.float32)
        for _ in range(max_steps):
            a = np.random.randint(0, ACTION_DIM)
            s2, _, done = env.step(a)
            data.append((s.copy(), a, s2.astype(np.float32)))
            s = s2.astype(np.float32)
            if done:
                break
    return data


def train_world_model(epochs: int = 20, kl_weight: float = 0.1) -> WorldModel:
    print("[trainer] Phase 1 — World Model pre-training")
    env  = MazeEnv(width=15, height=11, levels=100)
    data = collect_random_data(env, episodes=300, max_steps=50)
    print(f"[trainer] collected {len(data):,} transitions")

    states  = np.stack([d[0] for d in data])
    actions = np.array([d[1] for d in data], dtype=np.int64)
    nexts   = np.stack([d[2] for d in data])
    act_oh  = np.eye(ACTION_DIM)[actions].astype(np.float32)

    ds     = TensorDataset(torch.from_numpy(states),
                           torch.from_numpy(act_oh),
                           torch.from_numpy(nexts))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = WorldModel(state_dim=RAW_DIM, action_dim=ACTION_DIM,
                       latent_dim=LATENT_DIM, hidden=HIDDEN)
    model.to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total = 0.0
        for b_s, b_a, b_n in loader:
            b_s, b_a, b_n = b_s.to(DEVICE), b_a.to(DEVICE), b_n.to(DEVICE)
            out   = model.training_forward(b_s, b_a, b_n)
            recon = loss_fn(out["recon_state"], b_s)
            dyn   = loss_fn(out["pred_next"],   b_n)
            mu, lv = out["mu"], out["logvar"]
            kl    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss  = recon + dyn + kl_weight * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * b_s.size(0)
        print(f"[trainer] WM epoch {epoch+1}/{epochs}  loss={total/len(ds):.5f}")

    torch.save(model.state_dict(), "world_model.pth")
    print("[trainer] world model saved → world_model.pth")
    return model


# -----------------------------------------------------------------------
# Phase 2: DQN RL Agent  (uses compact 31-dim state)
# -----------------------------------------------------------------------

def train_rl_agent(
    rl_agent: RLAgent,
    env: MazeEnv,
    episodes: int = 400,
    max_steps: int = 200,
) -> RLAgent:
    """
    Train DQN on real environment using compact state representation.
    Compact state (31-dim) is much easier to learn from than raw 165-dim grid.
    """
    print(f"[trainer] Phase 2 — DQN RL Agent  (compact {COMPACT_DIM}-dim state)")
    print(f"[trainer] episodes={episodes}  max_steps={max_steps}  "
          f"ε_start={rl_agent.epsilon:.2f}  ε_end={rl_agent.epsilon_end:.2f}")

    episode_rewards = []

    for ep in range(episodes):
        env.reset()
        s       = env.compact_state()
        gy, gx  = env.goal_pos
        total_r = 0.0

        for _ in range(max_steps):
            a = rl_agent.act(s)
            _, r, done = env.step(a)
            s2 = env.compact_state()

            # Shaped reward: step penalty + distance shaping
            ay, ax = env.agent_pos
            dist   = abs(ay - gy) + abs(ax - gx)
            shaped = r - 0.005 * dist

            rl_agent.store(s, a, shaped, s2, done)
            rl_agent.update()
            total_r += r
            s = s2
            if done:
                break

        episode_rewards.append(total_r)

        if (ep + 1) % max(1, episodes // 10) == 0:
            mean_r = float(np.mean(episode_rewards[-30:]))
            solved = sum(1 for x in episode_rewards[-30:] if x > 0)
            print(f"[trainer] RL ep {ep+1:>4}/{episodes}  "
                  f"mean_r={mean_r:>7.3f}  solved={solved}/30  "
                  f"ε={rl_agent.epsilon:.3f}  buf={len(rl_agent._buf_s)}")

    # Greedy evaluation
    print("[trainer] Evaluating DQN agent (greedy, 30 episodes)…")
    solved_eval = 0
    for _ in range(30):
        env.reset()
        s = env.compact_state()
        for _ in range(200):
            a = rl_agent.act_greedy(s)
            _, r, done = env.step(a)
            s = env.compact_state()
            if done:
                solved_eval += 1
                break
    print(f"[trainer] Eval — solved: {solved_eval}/30")

    torch.save(rl_agent.policy.state_dict(), "rl_agent.pth")
    print("[trainer] RL agent saved → rl_agent.pth")
    return rl_agent


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def run_training(epochs: int = 20, rl_episodes: int = 400):
    print(f"[trainer] ═══ Starting end-to-end training "
          f"(wm_epochs={epochs}, rl_episodes={rl_episodes}) ═══")

    # Phase 1 — World Model (raw state)
    model = train_world_model(epochs=epochs)
    model.eval()

    # Phase 2 — DQN (compact state)
    env      = MazeEnv(width=15, height=11, levels=100)
    rl_agent = RLAgent(
        state_dim=COMPACT_DIM,
        action_size=ACTION_DIM,
        hidden=128,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        buffer_size=20000,
        batch_size=64,
        device=DEVICE,
    )
    train_rl_agent(rl_agent, env, episodes=rl_episodes, max_steps=200)

    print("[trainer] ═══ Training complete ═══")


if __name__ == "__main__":
    run_training(epochs=20, rl_episodes=400)
