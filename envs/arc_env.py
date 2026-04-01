"""
ARCEnvironment — loads real ARC task grids from Re-ARC or ARC-AGI-1 on every reset(),
applies transformation-based actions, and computes dense grid similarity rewards.
"""
import numpy as np
import random
import json
import pathlib
from envs.base_env import BaseEnvironment


# ── Transformation primitives ─────────────────────────────────────────────────
def _rotate90(g):  return np.rot90(g, k=1)
def _rotate180(g): return np.rot90(g, k=2)
def _rotate270(g): return np.rot90(g, k=3)
def _flip_h(g):    return np.flip(g, axis=1)
def _flip_v(g):    return np.flip(g, axis=0)
def _shift_r(g):   return np.roll(g, 1,  axis=1)
def _shift_l(g):   return np.roll(g, -1, axis=1)
def _shift_u(g):   return np.roll(g, -1, axis=0)
def _shift_d(g):   return np.roll(g, 1,  axis=0)
def _noop(g):      return g.copy()

ACTIONS = [_noop, _rotate90, _rotate180, _rotate270,
           _flip_h, _flip_v, _shift_r, _shift_l, _shift_u, _shift_d]


class ARCEnvironment(BaseEnvironment):
    """
    ARC environment that:
    1. Loads real input/output grid pairs from Re-ARC JSON files (or falls back to random).
    2. On reset(), samples a new task pair and resets state to the input grid.
    3. On step(action), applies a transformation primitive and computes progress.
    """
    MAX_GRID = 30

    def __init__(self, config: dict):
        super().__init__(**{k: v for k, v in config.items()
                            if k in ("device",)})
        self.action_space_size = config.get("action_dim", len(ACTIONS))
        assert self.action_space_size <= len(ACTIONS), \
            f"action_dim must be ≤ {len(ACTIONS)} (number of grid primitives)"

        self.max_grid = self.MAX_GRID
        self._pairs: list[tuple] = []          # list of (input_arr, output_arr) pairs
        self._load_pairs(config)

        # Will be set on reset()
        self.input_grid:  np.ndarray = np.zeros((self.max_grid, self.max_grid))
        self.target_grid: np.ndarray = np.zeros((self.max_grid, self.max_grid))
        self.state:       np.ndarray = np.zeros((self.max_grid, self.max_grid))
        self.step_count   = 0
        self.max_steps    = config.get("max_steps", 50)

    # ── Data loading ─────────────────────────────────────────────────────────
    def _pad(self, grid_list) -> np.ndarray:
        arr  = np.array(grid_list, dtype=np.float32)
        out  = np.zeros((self.max_grid, self.max_grid), dtype=np.float32)
        h, w = arr.shape
        out[:min(h, self.max_grid), :min(w, self.max_grid)] = \
            arr[:min(h, self.max_grid), :min(w, self.max_grid)]
        return out

    def _load_pairs(self, config: dict):
        data_path = pathlib.Path(
            config.get("rearc_path",
            config.get("arc_path", "data/re-arc"))
        )
        if data_path.is_dir():
            # Recursively search for JSON files (e.g. in re_arc/tasks/)
            task_files = sorted(data_path.rglob("*.json"))
            for fpath in task_files:
                try:
                    examples = json.loads(fpath.read_text())
                    for ex in examples[:100]:
                        inp = self._pad(ex["input"])
                        out = self._pad(ex["output"])
                        self._pairs.append((inp, out))
                except Exception:
                    pass
        if not self._pairs:
            # Fallback: random grids
            rng = np.random.default_rng(0)
            for _ in range(500):
                inp = rng.integers(0, 10, (self.max_grid, self.max_grid)).astype(np.float32)
                out = rng.integers(0, 10, (self.max_grid, self.max_grid)).astype(np.float32)
                self._pairs.append((inp, out))

    # ── Env API ──────────────────────────────────────────────────────────────
    def reset(self) -> dict:
        """Sample a new input/output pair and reset to the input grid."""
        inp, tgt       = random.choice(self._pairs)
        self.input_grid  = inp.copy()
        self.target_grid = tgt.copy()
        self.state       = inp.copy()
        self.step_count  = 0
        return {"state": self.state.copy()}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        # Decode action index
        if np.ndim(action) >= 1 and len(action) > 1:
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action) % self.action_space_size

        action_idx    = action_idx % self.action_space_size
        transform_fn  = ACTIONS[action_idx]
        self.state    = transform_fn(self.state).astype(np.float32)
        self.step_count += 1

        reward = self.compute_reward(self.state, self.target_grid)
        done   = self.is_goal(self.state, self.target_grid) or \
                 self.step_count >= self.max_steps

        return {"state": self.state.copy()}, reward, done, {"step": self.step_count}

    def render(self):
        print(f"Step {self.step_count}")
        print("State:\n",  self.state[:6, :6])
        print("Target:\n", self.target_grid[:6, :6])

    def sample_action(self) -> np.ndarray:
        action = np.zeros(self.action_space_size, dtype=np.float32)
        action[np.random.randint(0, self.action_space_size)] = 1.0
        return action

    def compute_reward(self, grid: np.ndarray, target: np.ndarray) -> float:
        """Dense fractional cell-match reward in [0, 1]."""
        if grid.shape != target.shape:
            return 0.0
        match_ratio  = float(np.sum(grid == target) / grid.size)
        # Bonus for exact match
        exact_bonus  = 1.0 if np.array_equal(grid, target) else 0.0
        return match_ratio + exact_bonus

    def is_goal(self, grid: np.ndarray, target: np.ndarray) -> bool:
        if grid.shape != target.shape:
            return False
        return bool(np.array_equal(grid, target))
