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
