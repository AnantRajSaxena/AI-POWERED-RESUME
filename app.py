import json
import os
import threading

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash

from agent import MPCPlanner, RLAgent
from maze_env import MazeEnv
from train import run_training
from world_model import WorldModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-for-production")

os.makedirs(app.instance_path, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(app.instance_path, "app.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# On Render, /app/instance is the persistent disk mount point
# Model files are stored there so they survive deploys
INSTANCE_DIR = os.environ.get("INSTANCE_PATH", app.instance_path)
os.makedirs(INSTANCE_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(INSTANCE_DIR, "world_model.pth")
RL_AGENT_PATH = os.path.join(INSTANCE_DIR, "rl_agent.pth")

db = SQLAlchemy(app)
limiter = Limiter(app=app, key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

STATE_DIM   = 15 * 11          # 165 — raw grid for world model
COMPACT_DIM = 31               # compact state for RL agent
ACTION_DIM  = 4

ENV = MazeEnv(width=15, height=11, levels=100)

# ---------------------------------------------------------------------------
# Load world model + planner + RL agent (optional — graceful fallback)
# ---------------------------------------------------------------------------

MODEL: WorldModel | None = None
PLANNER: MPCPlanner | None = None
RL_AGENT: RLAgent | None = None

try:
    MODEL = WorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden=128)
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    MODEL.eval()
    PLANNER = MPCPlanner(MODEL, action_size=ACTION_DIM, horizon=8, samples=200)
    print("[app] loaded world_model.pth — planner ready")
except Exception:
    MODEL = None
    PLANNER = None

try:
    _agent = RLAgent(state_dim=COMPACT_DIM, action_size=ACTION_DIM, hidden=128)
    _agent.policy.load_state_dict(torch.load(RL_AGENT_PATH, map_location="cpu"))
    _agent.policy.eval()
    RL_AGENT = _agent
    print("[app] loaded rl_agent.pth — RL agent ready")
except Exception:
    RL_AGENT = None

# ---------------------------------------------------------------------------
# Training state — live log visible to the UI
# ---------------------------------------------------------------------------

TRAINING_LOG: list = []          # list of log-line dicts
TRAINING_ACTIVE: bool = False
TRAINING_PHASE: str = ""         # "world_model" | "rl_agent" | "done" | ""
TRAINING_EPOCH: int = 0
TRAINING_TOTAL_EPOCHS: int = 0
TRAINING_RL_EP: int = 0
TRAINING_RL_TOTAL: int = 0
WM_LOSSES: list = []             # [(epoch, loss), ...]
RL_REWARDS: list = []            # [(episode, reward), ...]


def _log(msg: str, kind: str = "info"):
    """Append a structured log entry visible via /training_status."""
    import time
    TRAINING_LOG.append({"t": round(time.time(), 2), "msg": msg, "kind": kind})
    print(msg)


# ---------------------------------------------------------------------------
# Progress state (per-session, persisted to DB or fallback file)
# ---------------------------------------------------------------------------

PROGRESS_FILE = os.path.join(INSTANCE_DIR, "progress.json")
UNLOCKED_LEVEL = 1
TOTAL_SCORE = 0
LEVEL_HELP_COUNTS: dict = {}
LEVEL_SCORES: dict = {}


# ---------------------------------------------------------------------------
# Database model
# ---------------------------------------------------------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    unlocked_level = db.Column(db.Integer, default=1)
    total_score = db.Column(db.Integer, default=0)
    password_hash = db.Column(db.String(128), nullable=True)
    help_counts = db.Column(db.Text, default="{}")
    scores = db.Column(db.Text, default="{}")

    def to_dict(self):
        return {
            "username": self.username,
            "unlocked_level": int(self.unlocked_level),
            "total_score": int(self.total_score),
            "help_counts": json.loads(self.help_counts or "{}"),
            "scores": json.loads(self.scores or "{}"),
        }


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_progress():
    global UNLOCKED_LEVEL, TOTAL_SCORE, LEVEL_HELP_COUNTS, LEVEL_SCORES
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                p = json.load(f)
            UNLOCKED_LEVEL = p.get("unlocked_level", 1)
            TOTAL_SCORE = p.get("total_score", 0)
            LEVEL_HELP_COUNTS = {int(k): int(v) for k, v in p.get("help_counts", {}).items()}
            LEVEL_SCORES = {int(k): int(v) for k, v in p.get("scores", {}).items()}
        except Exception:
            UNLOCKED_LEVEL, TOTAL_SCORE, LEVEL_HELP_COUNTS, LEVEL_SCORES = 1, 0, {}, {}

    # Ensure DB tables exist and run safe migration
    with app.app_context():
        db.create_all()
        try:
            with db.engine.connect() as con:
                result = con.execute(text("PRAGMA table_info('user')"))
                cols = [row[1] for row in result.fetchall()]
                if "password_hash" not in cols:
                    con.execute(text("ALTER TABLE user ADD COLUMN password_hash VARCHAR(128)"))
                    con.commit()
        except Exception:
            pass


def save_progress():
    """Persist progress to the logged-in user's DB row, or to the fallback file."""
    global UNLOCKED_LEVEL, TOTAL_SCORE, LEVEL_HELP_COUNTS, LEVEL_SCORES

    # Only attempt session access inside an active request context
    username = None
    try:
        username = session.get("user")
    except RuntimeError:
        pass  # called outside request context (e.g. startup)

    if username:
        try:
            u = User.query.filter_by(username=username).first()
            if u:
                u.unlocked_level = int(UNLOCKED_LEVEL)
                u.total_score = int(TOTAL_SCORE)
                u.help_counts = json.dumps(LEVEL_HELP_COUNTS)
                u.scores = json.dumps(LEVEL_SCORES)
                db.session.commit()
                return
        except Exception:
            pass

    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(
                {
                    "unlocked_level": UNLOCKED_LEVEL,
                    "total_score": TOTAL_SCORE,
                    "help_counts": LEVEL_HELP_COUNTS,
                    "scores": LEVEL_SCORES,
                },
                f,
            )
    except Exception:
        pass


load_progress()


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@limiter.limit("8 per minute")
@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "") or "")
    if not username:
        return jsonify({"error": "missing_username"}), 400
    try:
        u = User.query.filter_by(username=username).first()
        if not u:
            u = User(username=username, unlocked_level=1, total_score=0, help_counts="{}", scores="{}")
            if password:
                u.password_hash = generate_password_hash(password)
            db.session.add(u)
            db.session.commit()
        else:
            if u.password_hash:
                if not password or not check_password_hash(u.password_hash, password):
                    return jsonify({"error": "invalid_credentials"}), 400

        session["user"] = username

        global UNLOCKED_LEVEL, TOTAL_SCORE, LEVEL_HELP_COUNTS, LEVEL_SCORES
        UNLOCKED_LEVEL = int(u.unlocked_level)
        TOTAL_SCORE = int(u.total_score)
        try:
            LEVEL_HELP_COUNTS = {int(k): int(v) for k, v in (json.loads(u.help_counts) if u.help_counts else {}).items()}
        except Exception:
            LEVEL_HELP_COUNTS = {}
        try:
            LEVEL_SCORES = {int(k): int(v) for k, v in (json.loads(u.scores) if u.scores else {}).items()}
        except Exception:
            LEVEL_SCORES = {}
        try:
            ENV.reset(level=UNLOCKED_LEVEL)
        except Exception:
            pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "ok", "user": username, "unlocked_level": UNLOCKED_LEVEL})


@limiter.limit("6 per minute")
@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "") or "")
    if not username or len(username) < 2 or len(username) > 32 or " " in username:
        return jsonify({"error": "invalid_username"}), 400
    if password and len(password) < 6:
        return jsonify({"error": "password_too_short"}), 400
    try:
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "user_exists"}), 400
        u = User(username=username, unlocked_level=1, total_score=0, help_counts="{}", scores="{}")
        if password:
            u.password_hash = generate_password_hash(password)
        db.session.add(u)
        db.session.commit()
        return jsonify({"status": "ok", "user": username})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logout")
def logout():
    session.pop("user", None)
    load_progress()
    try:
        ENV.reset(level=UNLOCKED_LEVEL)
    except Exception:
        pass
    return jsonify({"status": "ok"})


@app.route("/user")
def user_info():
    user = session.get("user")
    if not user:
        return jsonify({"user": None})
    return jsonify({"user": user, "unlocked_level": UNLOCKED_LEVEL, "total_score": TOTAL_SCORE})


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if not session.get("user"):
        return render_template("login.html")
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Game routes
# ---------------------------------------------------------------------------

@app.route("/state")
def state():
    st = ENV.state()
    return jsonify(
        {
            "state": st.tolist(),
            "done": ENV.done,
            "width": ENV.width,
            "height": ENV.height,
            "level": ENV.level,
            "unlocked_level": UNLOCKED_LEVEL,
            "total_score": TOTAL_SCORE,
            "help_count": LEVEL_HELP_COUNTS.get(ENV.level, 0),
        }
    )


@app.route("/step", methods=["POST"])
def step():
    data = request.json or {}
    action = int(data.get("action", 0))
    obs, reward, done = ENV.step(action)

    global UNLOCKED_LEVEL, TOTAL_SCORE
    # Always include width/height so the frontend can always re-render the grid
    resp = {
        "state": obs.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "width": ENV.width,
        "height": ENV.height,
        "level": ENV.level,
        "total_score": TOTAL_SCORE,
        "help_count": LEVEL_HELP_COUNTS.get(ENV.level, 0),
    }

    if done:
        lvl = ENV.level
        helps = LEVEL_HELP_COUNTS.get(lvl, 0)
        penalty = (2 ** (helps + 1)) - 2 if helps > 0 else 0
        score = max(0, 10 - penalty)
        LEVEL_SCORES[lvl] = score
        TOTAL_SCORE += score
        if UNLOCKED_LEVEL < ENV.levels:
            UNLOCKED_LEVEL = max(UNLOCKED_LEVEL, lvl + 1)
        save_progress()
        resp.update(
            {
                "level_completed": True,
                "level": lvl,
                "level_score": score,
                "total_score": TOTAL_SCORE,
                "unlocked_level": UNLOCKED_LEVEL,
            }
        )

    return jsonify(resp)


@app.route("/help")
def help_step():
    """
    Human hint request — increments hint counter (affects score).
    Priority: MPC planner → RL agent → BFS fallback.
    """
    lvl = ENV.level
    LEVEL_HELP_COUNTS[lvl] = LEVEL_HELP_COUNTS.get(lvl, 0) + 1
    return _get_action(include_path=True)


@app.route("/ai_action")
def ai_action():
    """
    AI Auto-Play action request — does NOT increment hint counter.
    Same priority as /help but silent (no score penalty).
    Always returns the BFS path so auto-follow can use it.
    """
    return _get_action(include_path=True)


def _get_action(include_path: bool = True):
    """Shared logic for hint and AI action — returns best next action."""
    raw_st     = ENV.state().astype("float32")        # for MPC planner
    compact_st = ENV.compact_state()                   # for RL agent

    # 1. MPC planner (latent imagination — uses raw state)
    if PLANNER is not None:
        try:
            goal = np.zeros(STATE_DIM, dtype="float32")
            gy, gx = ENV.goal_pos
            goal[gy * ENV.width + gx] = 3.0
            act  = PLANNER.act(raw_st, goal)
            path = ENV.shortest_path() if include_path else None
            return jsonify({"action": int(act), "source": "mpc_planner", "path": path})
        except Exception:
            pass

    # 2. RL agent (DQN greedy — uses compact state)
    if RL_AGENT is not None:
        try:
            act  = RL_AGENT.act_greedy(compact_st)
            path = ENV.shortest_path() if include_path else None
            return jsonify({"action": int(act), "source": "rl_agent", "path": path})
        except Exception:
            pass

    # 3. BFS fallback
    path = ENV.shortest_path()
    if not path or len(path) < 2:
        return jsonify({"action": None, "source": "bfs", "note": "no path found"})
    cur, nxt = path[0], path[1]
    dy, dx = nxt[0] - cur[0], nxt[1] - cur[1]
    a = 0 if dy == -1 else (1 if dy == 1 else (2 if dx == -1 else 3))
    return jsonify({"action": int(a), "source": "bfs",
                    "path": path if include_path else None})


@app.route("/level", methods=["POST"])
def set_level():
    data = request.json or {}
    lvl = int(data.get("level", 1))
    if lvl > UNLOCKED_LEVEL:
        return jsonify({"error": "level_locked", "unlocked_level": UNLOCKED_LEVEL}), 400
    obs = ENV.reset(level=lvl)
    return jsonify({"state": obs.tolist(), "level": ENV.level, "unlocked_level": UNLOCKED_LEVEL})


@app.route("/reset")
def reset():
    obs = ENV.reset()
    # Clear hint count for this level so AI auto-play doesn't pollute the score
    LEVEL_HELP_COUNTS.pop(ENV.level, None)
    return jsonify({"state": obs.tolist(), "done": ENV.done, "width": ENV.width, "height": ENV.height, "level": ENV.level})


@app.route("/train", methods=["POST"])
def train():
    global TRAINING_ACTIVE
    if TRAINING_ACTIVE:
        return jsonify({"status": "already_running"}), 409

    params = request.json or {}
    epochs = int(params.get("epochs", 10))
    rl_episodes = int(params.get("rl_episodes", 30))

    def _train():
        global MODEL, PLANNER, RL_AGENT
        global TRAINING_ACTIVE, TRAINING_PHASE, TRAINING_EPOCH, TRAINING_TOTAL_EPOCHS
        global TRAINING_RL_EP, TRAINING_RL_TOTAL, WM_LOSSES, RL_REWARDS

        TRAINING_ACTIVE = True
        TRAINING_LOG.clear()
        WM_LOSSES.clear()
        RL_REWARDS.clear()
        TRAINING_EPOCH = 0
        TRAINING_RL_EP = 0
        TRAINING_TOTAL_EPOCHS = epochs
        TRAINING_RL_TOTAL = rl_episodes

        try:
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from world_model import WorldModel as WM
            from agent import MPCPlanner as MPC, RLAgent as RLA
            from maze_env import MazeEnv as ME

            _STATE_DIM = 15 * 11
            _ACTION_DIM = 4
            _LATENT_DIM = 32
            _HIDDEN = 128

            # ── Phase 1: World Model ──────────────────────────────────────
            TRAINING_PHASE = "world_model"
            _log("━━━ Phase 1 — World Model Pre-training ━━━", "phase")
            _log(f"Collecting random transitions from environment…", "info")

            env = ME(width=15, height=11, levels=100)
            data = []
            for _ in range(300):
                s = env.reset().astype(np.float32)
                for _ in range(50):
                    a = np.random.randint(0, _ACTION_DIM)
                    s2, _, done = env.step(a)
                    data.append((s.copy(), a, s2.astype(np.float32)))
                    s = s2.astype(np.float32)
                    if done:
                        break

            _log(f"Collected {len(data):,} transitions from {300} episodes", "success")

            states  = np.stack([d[0] for d in data])
            actions = np.array([d[1] for d in data], dtype=np.int64)
            nexts   = np.stack([d[2] for d in data])
            act_oh  = np.eye(_ACTION_DIM)[actions].astype(np.float32)

            ds = TensorDataset(
                torch.from_numpy(states),
                torch.from_numpy(act_oh),
                torch.from_numpy(nexts),
            )
            loader = DataLoader(ds, batch_size=64, shuffle=True)

            model = WM(state_dim=_STATE_DIM, action_dim=_ACTION_DIM,
                       latent_dim=_LATENT_DIM, hidden=_HIDDEN)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            _log(f"Training world model for {epochs} epochs…", "info")

            for epoch in range(epochs):
                TRAINING_EPOCH = epoch + 1
                total = 0.0
                for b_s, b_a, b_n in loader:
                    out = model.training_forward(b_s, b_a, b_n)
                    recon = loss_fn(out["recon_state"], b_s)
                    dyn   = loss_fn(out["pred_next"], b_n)
                    mu, lv = out["mu"], out["logvar"]
                    kl    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    loss  = recon + dyn + 0.1 * kl
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total += float(loss.item()) * b_s.size(0)

                avg = total / len(ds)
                WM_LOSSES.append({"epoch": epoch + 1, "loss": round(avg, 5)})
                pct = int((epoch + 1) / epochs * 100)
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                _log(f"Epoch {epoch+1:>3}/{epochs}  [{bar}] {pct:>3}%  loss={avg:.5f}", "epoch")

            torch.save(model.state_dict(), MODEL_PATH)
            _log("World model saved → world_model.pth", "success")

            # ── Phase 2: DQN RL Agent — compact 31-dim state ────────────
            TRAINING_PHASE = "rl_agent"
            _COMPACT_DIM = 31  # MazeEnv.COMPACT_DIM
            _log("━━━ Phase 2 — DQN RL Agent (compact state) ━━━", "phase")
            _log(f"Training DQN for {rl_episodes} episodes  "
                 f"(compact {_COMPACT_DIM}-dim state, real environment)…", "info")
            _log("Compact state: relative goal pos + local 5×5 wall patch", "info")

            model.eval()
            rl_agent = RLA(
                state_dim=_COMPACT_DIM, action_size=_ACTION_DIM,
                hidden=128, lr=1e-3,
                epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.992,
                buffer_size=20000, batch_size=64,
            )

            episode_rewards = []
            for ep in range(rl_episodes):
                TRAINING_RL_EP = ep + 1
                env.reset()
                s      = env.compact_state()
                gy, gx = env.goal_pos
                total_r = 0.0

                for _ in range(200):
                    a = rl_agent.act(s)
                    _, r, done = env.step(a)
                    s2 = env.compact_state()

                    # Distance shaping
                    ay2, ax2 = env.agent_pos
                    dist     = abs(ay2 - gy) + abs(ax2 - gx)
                    shaped_r = r - 0.005 * dist

                    rl_agent.store(s, a, shaped_r, s2, done)
                    rl_agent.update()
                    total_r += r
                    s = s2
                    if done:
                        break

                episode_rewards.append(total_r)
                RL_REWARDS.append({"ep": ep + 1, "reward": round(total_r, 3)})

                if (ep + 1) % max(1, rl_episodes // 10) == 0:
                    mean_r = float(np.mean(episode_rewards[-30:]))
                    pct    = int((ep + 1) / rl_episodes * 100)
                    bar    = "█" * (pct // 10) + "░" * (10 - pct // 10)
                    solved = sum(1 for x in episode_rewards[-30:] if x > 0)
                    _log(
                        f"RL ep {ep+1:>4}/{rl_episodes}  [{bar}] {pct:>3}%  "
                        f"mean_r={mean_r:>7.3f}  solved={solved}/30  "
                        f"ε={rl_agent.epsilon:.3f}",
                        "epoch"
                    )

            # Evaluation (greedy)
            _log("Evaluating DQN agent (greedy, 30 episodes)…", "info")
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
            _log(f"Eval — solved: {solved_eval}/30 mazes", "success")

            torch.save(rl_agent.policy.state_dict(), RL_AGENT_PATH)
            _log("RL agent saved → rl_agent.pth", "success")

            # ── Reload into app ───────────────────────────────────────────
            m = WM(state_dim=_STATE_DIM, action_dim=_ACTION_DIM,
                   latent_dim=_LATENT_DIM, hidden=_HIDDEN)
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            m.eval()
            MODEL   = m
            PLANNER = MPC(MODEL, action_size=_ACTION_DIM, horizon=10, samples=500)

            ag = RLA(state_dim=_COMPACT_DIM, action_size=_ACTION_DIM, hidden=128)
            ag.policy.load_state_dict(torch.load(RL_AGENT_PATH, map_location="cpu"))
            ag.policy.eval()
            RL_AGENT = ag

            TRAINING_PHASE = "done"
            _log("━━━ Training complete — models hot-reloaded ━━━", "phase")

        except Exception as e:
            _log(f"Training error: {e}", "error")
            TRAINING_PHASE = "error"
        finally:
            TRAINING_ACTIVE = False

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()
    return jsonify({"status": "training_started", "epochs": epochs, "rl_episodes": rl_episodes})


@app.route("/training_status")
def training_status():
    """Polled by the UI every second to show live training progress."""
    return jsonify({
        "active":        TRAINING_ACTIVE,
        "phase":         TRAINING_PHASE,
        "epoch":         TRAINING_EPOCH,
        "total_epochs":  TRAINING_TOTAL_EPOCHS,
        "rl_ep":         TRAINING_RL_EP,
        "rl_total":      TRAINING_RL_TOTAL,
        "wm_losses":     WM_LOSSES[-50:],   # last 50 points
        "rl_rewards":    RL_REWARDS[-50:],
        "log":           TRAINING_LOG[-60:], # last 60 lines
        "model_ready":   MODEL is not None,
        "agent_ready":   RL_AGENT is not None,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
