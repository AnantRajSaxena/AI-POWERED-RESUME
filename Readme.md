# NeuroMaze — Reinforcement Learning with Learned World Models

End-to-end RL system: **Environment → World Model → MPC Planner → RL Agent → Action**

## What it does

- Procedurally generated mazes (100 levels, increasing difficulty)
- Latent world model (VAE-style encoder + dynamics + decoder) trained on environment transitions
- MPC planner does imagination-based planning in latent space
- DQN RL agent trained on real environment with compact state representation
- Flask web UI — no Streamlit/Gradio
- Full CI/CD pipeline (GitHub Actions: lint → test → Docker build)

## Run locally

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# or: .venv/bin/pip install -r requirements.txt  # Linux/Mac

python app.py
```

Open http://127.0.0.1:5000

## Train the AI

Click **⚡ Train World Model** in the sidebar. The training dashboard shows:
- Phase 1: World model loss curve (should drop from ~0.09 → ~0.04)
- Phase 2: DQN reward curve + solved count increasing over 400 episodes

## Docker

```bash
docker build -t neuromaze .
docker run -p 5000:5000 neuromaze
```

## Deploy to Render

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Set environment variable `FLASK_SECRET` to a random string

Or use the `render.yaml` blueprint for one-click deploy.

## Project structure

```
app.py          — Flask web server + game routes + training endpoint
world_model.py  — Latent world model (Encoder + LatentDynamics + Decoder)
agent.py        — DQN RL Agent + MPC Planner
train.py        — End-to-end training pipeline
maze_env.py     — MazeEnv (100 levels, raw + compact state)
templates/      — HTML UI (login, register, game dashboard)
tests/          — 33 pytest tests
.github/        — CI/CD pipeline (lint → test → Docker)
render.yaml     — Render deployment config
Dockerfile      — Production container
```

## CI/CD

GitHub Actions runs on every push to `main`:
1. **Lint** — flake8
2. **Test** — pytest on Python 3.10, 3.11, 3.12
3. **Docker** — build image + smoke test
