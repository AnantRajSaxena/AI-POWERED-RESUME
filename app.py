from flask import Flask, render_template, jsonify, request, session
import threading
from train import run_training
from maze_env import MazeEnv
from world_model import WorldModel
from agent import MPCPlanner
import torch
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# Secret key for sessions (override in env if needed)
app.secret_key = 'change-me-for-production'

# Database (SQLite) for user accounts and progress
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Global environment for the web UI: use MazeEnv with 100 levels
ENV = MazeEnv(width=15, height=11, levels=100)

# Try to load a learned world model if present (optional)
MODEL = None
PLANNER = None
try:
    state_dim = ENV.width * ENV.height + 0  # placeholder if model expects different input
    MODEL = WorldModel(state_dim=ENV.width*ENV.height, action_dim=4)
    MODEL.load_state_dict(torch.load('world_model.pth', map_location='cpu'))
    MODEL.eval()
    PLANNER = MPCPlanner(MODEL, action_size=4)
    print('[app] loaded world_model.pth, planner ready')
except Exception:
    MODEL = None
    PLANNER = None

# Progress + scoring state (persisted)
PROGRESS_FILE = 'progress.json'
UNLOCKED_LEVEL = 1
TOTAL_SCORE = 0
LEVEL_HELP_COUNTS = {}  # level -> helps used
LEVEL_SCORES = {}

# (user accounts persisted in the SQLite database via SQLAlchemy)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    unlocked_level = db.Column(db.Integer, default=1)
    total_score = db.Column(db.Integer, default=0)
    help_counts = db.Column(db.Text, default='{}')
    scores = db.Column(db.Text, default='{}')

    def to_dict(self):
        return {
            'username': self.username,
            'unlocked_level': int(self.unlocked_level),
            'total_score': int(self.total_score),
            'help_counts': json.loads(self.help_counts or '{}'),
            'scores': json.loads(self.scores or '{}')
        }

import json, os


def load_progress():
    global UNLOCKED_LEVEL, TOTAL_SCORE, LEVEL_HELP_COUNTS, LEVEL_SCORES
    # load global progress file as a fallback
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                p = json.load(f)
            UNLOCKED_LEVEL = p.get('unlocked_level', 1)
            TOTAL_SCORE = p.get('total_score', 0)
            LEVEL_HELP_COUNTS = {int(k): int(v) for k, v in p.get('help_counts', {}).items()}
            LEVEL_SCORES = {int(k): int(v) for k, v in p.get('scores', {}).items()}
        except Exception:
            UNLOCKED_LEVEL = 1
            TOTAL_SCORE = 0
            LEVEL_HELP_COUNTS = {}
            LEVEL_SCORES = {}

    # ensure database tables exist
    try:
        db.create_all()
    except Exception:
        pass


def save_progress():
    p = {
        'unlocked_level': UNLOCKED_LEVEL,
        'total_score': TOTAL_SCORE,
        'help_counts': LEVEL_HELP_COUNTS,
        'scores': LEVEL_SCORES,
    }
    # If a user is logged in, save to their DB row; otherwise write global progress file
    user = session.get('user') if request else None
    if user:
        try:
            u = User.query.filter_by(username=user).first()
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
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(p, f)
    except Exception:
        pass


load_progress()


@app.route('/login', methods=['POST'])
def login():
    data = request.json or {}
    username = str(data.get('username', '')).strip()
    if not username:
        return jsonify({'error': 'missing_username'}), 400
    try:
        # lookup or create user in DB
        u = User.query.filter_by(username=username).first()
        if not u:
            u = User(username=username, unlocked_level=1, total_score=0, help_counts='{}', scores='{}')
            db.session.add(u)
            db.session.commit()

        session['user'] = username

        # load user progress into globals
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
        return jsonify({'error': str(e)}), 500
    return jsonify({'status': 'ok', 'user': username, 'unlocked_level': UNLOCKED_LEVEL})


@app.route('/register', methods=['POST'])
def register():
    data = request.json or {}
    username = str(data.get('username', '')).strip()
    if not username:
        return jsonify({'error': 'missing_username'}), 400
    try:
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'user_exists'}), 400
        u = User(username=username, unlocked_level=1, total_score=0, help_counts='{}', scores='{}')
        db.session.add(u)
        db.session.commit()
        return jsonify({'status': 'ok', 'user': username})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logout')
def logout():
    session.pop('user', None)
    # revert to global progress
    load_progress()
    try:
        ENV.reset(level=UNLOCKED_LEVEL)
    except Exception:
        pass
    return jsonify({'status': 'ok'})


@app.route('/user')
def user_info():
    user = session.get('user')
    if not user:
        return jsonify({'user': None})
    return jsonify({'user': user, 'unlocked_level': UNLOCKED_LEVEL, 'total_score': TOTAL_SCORE})


@app.route('/')
def index():
    # require login first
    if not session.get('user'):
        return render_template('login.html')
    return render_template('index.html')


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')


@app.route('/state')
def state():
    st = ENV.state()
    return jsonify({'state': st.tolist(), 'done': ENV.done, 'width': ENV.width, 'height': ENV.height, 'level': ENV.level,
                    'unlocked_level': UNLOCKED_LEVEL, 'total_score': TOTAL_SCORE,
                    'help_count': LEVEL_HELP_COUNTS.get(ENV.level, 0)})


@app.route('/step', methods=['POST'])
def step():
    data = request.json or {}
    action = int(data.get('action', 0))
    obs, reward, done = ENV.step(action)
    global UNLOCKED_LEVEL, TOTAL_SCORE
    resp = {'state': obs.tolist(), 'reward': float(reward), 'done': bool(done)}

    # if level completed, award points and unlock next
    if done:
        lvl = ENV.level
        helps = LEVEL_HELP_COUNTS.get(lvl, 0)
        # compute penalty: sum_{i=1..helps} 2^i = 2^{helps+1}-2
        penalty = 0
        if helps > 0:
            penalty = (2 ** (helps + 1)) - 2
        base = 10
        score = max(0, base - penalty)
        LEVEL_SCORES[lvl] = score
        TOTAL_SCORE += score
        # unlock next level if available
        if UNLOCKED_LEVEL < ENV.levels:
            UNLOCKED_LEVEL = max(UNLOCKED_LEVEL, lvl + 1)
        resp.update({'level_completed': True, 'level': lvl, 'level_score': score, 'total_score': TOTAL_SCORE, 'unlocked_level': UNLOCKED_LEVEL})

    return jsonify(resp)


@app.route('/help')
def help_step():
    """Return a single next-step suggestion (0=up,1=down,2=left,3=right).
    Uses learned model+planner if available, otherwise BFS shortest path.
    """
    # Count that user asked for help for current level
    lvl = ENV.level
    LEVEL_HELP_COUNTS[lvl] = LEVEL_HELP_COUNTS.get(lvl, 0) + 1

    # If planner exists, try to use it (planner expects numeric state vector)
    if PLANNER is not None:
        try:
            st = ENV.state().astype('float32')
            goal = np.zeros_like(st, dtype='float32')
            gy, gx = ENV.goal_pos
            goal_idx = gy * ENV.width + gx
            goal[goal_idx] = 1.0
            act = PLANNER.act(st, goal)
            path = ENV.shortest_path()
            return jsonify({'action': int(act), 'source': 'planner', 'path': path})
        except Exception:
            pass

    # fallback: BFS shortest path
    path = ENV.shortest_path()
    if not path or len(path) < 2:
        return jsonify({'action': None, 'source': 'bfs', 'note': 'no path found'})
    cur = path[0]
    nxt = path[1]
    dy = nxt[0] - cur[0]
    dx = nxt[1] - cur[1]
    if dy == -1:
        a = 0
    elif dy == 1:
        a = 1
    elif dx == -1:
        a = 2
    else:
        a = 3
    return jsonify({'action': int(a), 'source': 'bfs', 'path': path})


@app.route('/level', methods=['POST'])
def set_level():
    data = request.json or {}
    lvl = int(data.get('level', 1))
    # enforce unlocking rule
    if lvl > UNLOCKED_LEVEL:
        return jsonify({'error': 'level_locked', 'unlocked_level': UNLOCKED_LEVEL}), 400
    obs = ENV.reset(level=lvl)
    return jsonify({'state': obs.tolist(), 'level': ENV.level, 'unlocked_level': UNLOCKED_LEVEL})


@app.route('/reset')
def reset():
    obs = ENV.reset()
    return jsonify({'state': obs.tolist(), 'done': ENV.done})


@app.route('/train', methods=['POST'])
def train():
    params = request.json or {}
    epochs = int(params.get('epochs', 10))
    thread = threading.Thread(target=run_training, args=(epochs,))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'training_started', 'epochs': epochs})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
