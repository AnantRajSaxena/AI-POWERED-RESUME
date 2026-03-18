import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from world_model import WorldModel
from maze_env import MazeEnv


def run_training(epochs=10):
    print(f"[trainer] start training for {epochs} epochs")
    env = MazeEnv()

    # collect random dataset
    data = []
    for ep in range(200):
        s = env.reset().astype(np.float32)
        for t in range(30):
            a = np.random.randint(0, 4)
            s2, r, done = env.step(a)
            data.append((s.copy(), a, s2.copy()))
            if done:
                break

    states = np.stack([d[0] for d in data]).astype(np.float32)
    actions = np.array([d[1] for d in data], dtype=np.int64)
    nexts = np.stack([d[2] for d in data]).astype(np.float32)

    # prepare tensors
    act_oh = np.eye(4)[actions].astype(np.float32)

    ds = TensorDataset(torch.from_numpy(states), torch.from_numpy(act_oh), torch.from_numpy(nexts))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = WorldModel(state_dim=9, action_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total = 0.0
        for b_s, b_a, b_n in loader:
            pred = model(b_s, b_a)
            loss = loss_fn(pred, b_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * b_s.size(0)
        print(f"[trainer] epoch {epoch+1}/{epochs} loss={total/len(ds):.4f}")

    # Save model for later use
    torch.save(model.state_dict(), 'world_model.pth')
    print('[trainer] training complete, model saved to world_model.pth')


if __name__ == '__main__':
    run_training(epochs=10)
