import numpy as np


class PuzzleEnv:
    """Simple 3x3 sliding puzzle (8-puzzle).
    State: flattened length-9 array with 0 as blank and 1..8 tiles.
    Actions: 0=up,1=down,2=left,3=right
    """

    def __init__(self):
        self.goal = np.array([1,2,3,4,5,6,7,8,0], dtype=np.int64)
        self.reset()

    def reset(self):
        # start from a solvable shuffle
        while True:
            arr = self.goal.copy()
            np.random.shuffle(arr)
            if self._is_solvable(arr.tolist()):
                break
        self.state = arr
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0.0, True

        idx = int(np.where(self.state == 0)[0])
        r, c = divmod(idx, 3)
        move = None
        if action == 0 and r > 0:  # up
            move = (r-1, c)
        elif action == 1 and r < 2:  # down
            move = (r+1, c)
        elif action == 2 and c > 0:  # left
            move = (r, c-1)
        elif action == 3 and c < 2:  # right
            move = (r, c+1)

        if move is not None:
            mi = move[0]*3 + move[1]
            s = self.state.copy()
            s[idx], s[mi] = s[mi], s[idx]
            self.state = s

        reward = -1.0
        if np.array_equal(self.state, self.goal):
            reward = 10.0
            self.done = True

        return self.state, reward, self.done

    def render(self):
        s = self.state.reshape(3,3)
        return '\n'.join(' '.join(str(x) for x in row) for row in s)

    def _is_solvable(self, arr):
        # Count inversions for solvability (3x3)
        a = [x for x in arr if x != 0]
        inv = 0
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                if a[i] > a[j]:
                    inv += 1
        return inv % 2 == 0


class MazeEnv:
    """Grid-maze environment.
    - Levels: 1..100 (increasing difficulty)
    - Grid: fixed size (15x11)
    - State: flattened grid with 0=floor,1=wall,2=agent,3=goal
    - Actions: 0=up,1=down,2=left,3=right
    """

    def __init__(self, width=15, height=11, levels=100):
        self.width = width
        self.height = height
        self.levels = max(1, levels)
        self.level = 1
        self.grid = None
        self.agent_pos = (0, 0)
        self.goal_pos = (height-1, width-1)
        self.done = False
        # Pre-generate level seeds for reproducibility
        self.seeds = [42 + i for i in range(self.levels)]
        self.reset(1)

    def generate_level(self, level):
        import random

        level = max(1, min(self.levels, int(level)))
        seed = self.seeds[level-1]
        rnd = random.Random(seed)
        w, h = self.width, self.height

        # Start with empty grid
        grid = [[0 for _ in range(w)] for _ in range(h)]

        # Add outer walls
        for x in range(w):
            grid[0][x] = 1
            grid[h-1][x] = 1
        for y in range(h):
            grid[y][0] = 1
            grid[y][w-1] = 1

        # Carve some internal walls based on level difficulty
        density = 0.06 + (level-1)/(self.levels-1) * 0.25  # from 6% to ~31%
        for y in range(1, h-1):
            for x in range(1, w-1):
                if rnd.random() < density:
                    grid[y][x] = 1

        # Ensure start and goal are free
        grid[1][1] = 0
        grid[h-2][w-2] = 0

        # Try to ensure solvability by carving a simple path from start to goal
        sx, sy = 1, 1
        gx, gy = w-2, h-2
        x, y = sx, sy
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

    def reset(self, level=None):
        if level is None:
            level = self.level
        else:
            self.level = max(1, min(self.levels, int(level)))

        self.grid = self.generate_level(self.level)
        self.agent_pos = (1, 1)
        self.goal_pos = (self.height-2, self.width-2)
        self.done = False
        return self.state()

    def state(self):
        """Return flattened state array: 0 floor,1 wall,2 agent,3 goal"""
        g = [row[:] for row in self.grid]
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos
        g[ay][ax] = 2
        g[gy][gx] = 3
        flat = []
        for row in g:
            flat.extend(row)
        return np.array(flat, dtype=np.int64)

    def step(self, action):
        if self.done:
            return self.state(), 0.0, True

        ay, ax = self.agent_pos
        ny, nx = ay, ax
        if action == 0:
            ny -= 1
        elif action == 1:
            ny += 1
        elif action == 2:
            nx -= 1
        elif action == 3:
            nx += 1

        # check bounds and walls
        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny][nx] == 0:
            self.agent_pos = (ny, nx)

        reward = -0.1
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            self.done = True

        return self.state(), reward, self.done

    def render(self):
        rows = []
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos
        for y in range(self.height):
            row = ''
            for x in range(self.width):
                if (y, x) == (ay, ax):
                    row += 'A'
                elif (y, x) == (gy, gx):
                    row += 'G'
                elif self.grid[y][x] == 1:
                    row += '#'
                else:
                    row += '.'
            rows.append(row)
        return '\n'.join(rows)

    # simple helper for BFS pathfinding
    def shortest_path(self):
        from collections import deque
        h, w = self.height, self.width
        start = self.agent_pos
        goal = self.goal_pos
        q = deque([start])
        prev = {start: None}
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for d in dirs:
                ny = cur[0] + d[0]
                nx = cur[1] + d[1]
                if 0 <= ny < h and 0 <= nx < w and self.grid[ny][nx] == 0:
                    nb = (ny, nx)
                    if nb not in prev:
                        prev[nb] = cur
                        q.append(nb)

        if goal not in prev:
            return None

        # reconstruct
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

