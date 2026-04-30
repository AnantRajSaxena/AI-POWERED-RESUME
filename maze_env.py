import numpy as np
from collections import deque


class MazeEnv:
    """
    Grid-maze environment.
    - Levels: 1..N (increasing difficulty)
    - Grid: fixed size (width x height)
    - Raw state: flattened grid with 0=floor,1=wall,2=agent,3=goal  (for world model)
    - Compact state: [dy_norm, dx_norm, can_up, can_down, can_left, can_right,
                      local 5x5 wall patch (25 values)]  → 31 dims  (for RL agent)
    - Actions: 0=up, 1=down, 2=left, 3=right
    """

    # Compact state dimension used by RL agent and DQN
    COMPACT_DIM = 31

    def __init__(self, width=15, height=11, levels=100):
        self.width  = width
        self.height = height
        self.levels = max(1, levels)
        self.level  = 1
        self.grid   = None
        self.agent_pos = (0, 0)
        self.goal_pos  = (height - 1, width - 1)
        self.done   = False
        self.seeds  = [42 + i for i in range(self.levels)]
        self.reset(1)

    # ------------------------------------------------------------------
    # Level generation
    # ------------------------------------------------------------------

    def generate_level(self, level):
        import random
        level = max(1, min(self.levels, int(level)))
        rnd   = random.Random(self.seeds[level - 1])
        w, h  = self.width, self.height

        grid = [[0] * w for _ in range(h)]

        # Outer walls
        for x in range(w):
            grid[0][x] = 1
            grid[h - 1][x] = 1
        for y in range(h):
            grid[y][0] = 1
            grid[y][w - 1] = 1

        # Internal walls — density increases with level
        density = 0.06 + (level - 1) / (self.levels - 1) * 0.25
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if rnd.random() < density:
                    grid[y][x] = 1

        # Ensure start and goal are free
        grid[1][1]         = 0
        grid[h - 2][w - 2] = 0

        # Carve a guaranteed path
        x, y = 1, 1
        gx, gy = w - 2, h - 2
        while x != gx or y != gy:
            grid[y][x] = 0
            if x < gx and rnd.random() < 0.6:
                x += 1
            elif y < gy and rnd.random() < 0.6:
                y += 1
            else:
                if x < gx:
                    x += 1
                elif y < gy:
                    y += 1
        grid[gy][gx] = 0

        return grid

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, level=None):
        if level is None:
            level = self.level
        else:
            self.level = max(1, min(self.levels, int(level)))

        self.grid      = self.generate_level(self.level)
        self.agent_pos = (1, 1)
        self.goal_pos  = (self.height - 2, self.width - 2)
        self.done      = False
        return self.state()

    def step(self, action):
        if self.done:
            return self.state(), 0.0, True

        ay, ax = self.agent_pos
        ny, nx = ay, ax
        if   action == 0: ny -= 1
        elif action == 1: ny += 1
        elif action == 2: nx -= 1
        elif action == 3: nx += 1

        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny][nx] == 0:
            self.agent_pos = (ny, nx)

        reward = -0.1
        if self.agent_pos == self.goal_pos:
            reward    = 10.0
            self.done = True

        return self.state(), reward, self.done

    # ------------------------------------------------------------------
    # State representations
    # ------------------------------------------------------------------

    def state(self):
        """Full 165-dim raw grid state — used by world model training."""
        g = [row[:] for row in self.grid]
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos
        g[ay][ax] = 2
        g[gy][gx] = 3
        flat = []
        for row in g:
            flat.extend(row)
        return np.array(flat, dtype=np.int64)

    def compact_state(self):
        """
        31-dim compact state for the RL agent:
          [0]   dy_norm  — normalised row distance to goal  (-1..1)
          [1]   dx_norm  — normalised col distance to goal  (-1..1)
          [2]   can_up   — 1 if cell above is free
          [3]   can_down
          [4]   can_left
          [5]   can_right
          [6..30] 5×5 local wall patch centred on agent (25 values, 0=free/1=wall)
        """
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos

        dy_norm = (gy - ay) / max(self.height, 1)
        dx_norm = (gx - ax) / max(self.width,  1)

        def free(y, x):
            return 1.0 if (0 <= y < self.height and 0 <= x < self.width
                           and self.grid[y][x] == 0) else 0.0

        can_up    = free(ay - 1, ax)
        can_down  = free(ay + 1, ax)
        can_left  = free(ay,     ax - 1)
        can_right = free(ay,     ax + 1)

        # 5×5 local patch
        patch = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = ay + dy, ax + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    patch.append(float(self.grid[ny][nx]))
                else:
                    patch.append(1.0)  # out-of-bounds = wall

        vec = [dy_norm, dx_norm, can_up, can_down, can_left, can_right] + patch
        return np.array(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Rendering / pathfinding
    # ------------------------------------------------------------------

    def render(self):
        rows = []
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos
        for y in range(self.height):
            row = ''
            for x in range(self.width):
                if   (y, x) == (ay, ax): row += 'A'
                elif (y, x) == (gy, gx): row += 'G'
                elif self.grid[y][x] == 1: row += '#'
                else: row += '.'
            rows.append(row)
        return '\n'.join(rows)

    def shortest_path(self):
        h, w   = self.height, self.width
        start  = self.agent_pos
        goal   = self.goal_pos
        q      = deque([start])
        prev   = {start: None}
        dirs   = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for d in dirs:
                ny, nx = cur[0] + d[0], cur[1] + d[1]
                if 0 <= ny < h and 0 <= nx < w and self.grid[ny][nx] == 0:
                    nb = (ny, nx)
                    if nb not in prev:
                        prev[nb] = cur
                        q.append(nb)

        if goal not in prev:
            return None

        path, cur = [], goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path
