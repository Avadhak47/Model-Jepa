import os, json, random, pathlib
import torch
import numpy as np

class ReARCDataset:
    """
    Loader for the Re-ARC dataset (github.com/michaelhodel/re-arc).

    Re-ARC procedurally regenerates 100+ grid pairs per ARC task using the
    original DSL generators, giving 40,000+ training pairs vs ~1,400 in ARC-AGI-1.

    Expected directory layout (after `git clone` or download):
        <data_path>/
            0a938d79.json
            0b148d64.json
            ...   (one JSON per task, named by ARC task ID)

    Each JSON file is a list of {"input": [[...]], "output": [[...]]} dicts.

    Falls back to random mock data if path is not found, so the notebook
    always runs without crashing.
    """

    MAX_GRID = 30

    def __init__(self, data_path: str, max_pairs_per_task: int = 100):
        self.data_path = pathlib.Path(data_path)
        self.max_pairs = max_pairs_per_task
        self.pairs: list[tuple] = []   # list of (input_grid, output_grid) numpy arrays

        if self.data_path.is_dir():
            self._load(self.data_path)
        else:
            print(f"Warning: ReARC path '{data_path}' not found. "
                  "Using random mock data. Clone re-arc and point ARC_DATA_PATH there.")
            self._use_mock()

    def _pad(self, grid: list) -> np.ndarray:
        """Pad a 2D list grid to MAX_GRID×MAX_GRID."""
        arr = np.array(grid, dtype=np.float32)
        h, w = arr.shape
        out  = np.zeros((self.MAX_GRID, self.MAX_GRID), dtype=np.float32)
        out[:min(h, self.MAX_GRID), :min(w, self.MAX_GRID)] = \
            arr[:min(h, self.MAX_GRID), :min(w, self.MAX_GRID)]
        return out

    def _load(self, path: pathlib.Path):
        # Recursively search for JSON files in case they are in re_arc/tasks/
        task_files = sorted(path.rglob("*.json"))
        if not task_files:
            print(f"Warning: no .json files found in '{path}'. Using mock data.")
            self._use_mock()
            return

        for fpath in task_files:
            try:
                examples = json.loads(fpath.read_text())
                for ex in examples[: self.max_pairs]:
                    inp  = self._pad(ex["input"])
                    out  = self._pad(ex["output"])
                    self.pairs.append((inp, out))
            except Exception:
                pass   # skip malformed files silently

        print(f"ReARCDataset: loaded {len(self.pairs):,} grid pairs "
              f"from {len(task_files)} tasks in '{path}'")

    def _use_mock(self, n: int = 2000):
        rng = np.random.default_rng(42)
        for _ in range(n):
            inp = rng.integers(0, 10, (self.MAX_GRID, self.MAX_GRID)).astype(np.float32)
            out = rng.integers(0, 10, (self.MAX_GRID, self.MAX_GRID)).astype(np.float32)
            self.pairs.append((inp, out))

    def __len__(self):
        return len(self.pairs)

    def sample(self, batch_size: int) -> dict:
        """Return a TensorDict batch compatible with all NS-ARC modules."""
        batch_size = min(batch_size, len(self.pairs))
        chosen = random.sample(self.pairs, batch_size)

        states  = np.stack([p[0] for p in chosen])   # [B, H, W]
        targets = np.stack([p[1] for p in chosen])

        # Shape: [B, 1, 30, 30]  (1 channel so CNNEncoder / TransformerEncoder work directly)
        s_t = torch.tensor(states,  dtype=torch.float32).unsqueeze(1)
        t_t = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        return {
            "state":          s_t,
            "target_state":   t_t,
            # Aliases expected by World Model loss and ReplayBuffer
            "target_latent":  t_t.view(batch_size, -1)[:, :128] if t_t.numel() // batch_size >= 128
                              else t_t.view(batch_size, -1).repeat(1, 128)[:, :128],
            "target_reward":  torch.zeros(batch_size),
        }
