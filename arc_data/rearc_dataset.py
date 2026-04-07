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

    def __init__(self, data_path: str, max_pairs_per_task: int = 100, validation_split: float = 0.1):
        self.data_path = pathlib.Path(data_path)
        self.max_pairs = max_pairs_per_task
        self.pairs: list[tuple] = []   # list of (input_grid, output_grid) numpy arrays

        if self.data_path.is_dir():
            self._load(self.data_path)
        else:
            print(f"Warning: ReARC path '{data_path}' not found. "
                  "Using random mock data. Clone re-arc and point ARC_DATA_PATH there.")
            self._use_mock()
            
        # Shuffle and split
        random.seed(42) # Deterministic split
        random.shuffle(self.pairs)
        val_size = int(len(self.pairs) * validation_split)
        self.val_pairs = self.pairs[:val_size]
        self.train_pairs = self.pairs[val_size:]
        print(f"Dataset Split | Train: {len(self.train_pairs)} pairs | Val: {len(self.val_pairs)} pairs")

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
        """Generates procedural geometric blocks rather than TV static so Slot Attention can actually learn objects."""
        rng = np.random.default_rng(42)
        for _ in range(n):
            inp = np.zeros((self.MAX_GRID, self.MAX_GRID), dtype=np.float32)
            out = np.zeros((self.MAX_GRID, self.MAX_GRID), dtype=np.float32)
            
            # Add 2-6 distinct semantic objects
            for _ in range(rng.integers(2, 7)):
                color = rng.integers(1, 10)
                obj_type = rng.integers(0, 3)
                r, c = rng.integers(0, self.MAX_GRID-1, size=2)
                
                if obj_type == 0: # 1x1 Dot (stress-tests focal loss)
                    inp[r, c] = color
                    out[r, c] = color
                elif obj_type == 1: # Horizontal Line
                    w = rng.integers(3, 10)
                    inp[r:min(r+w, self.MAX_GRID), c] = color
                    out[r:min(r+w, self.MAX_GRID), c] = color
                elif obj_type == 2: # Solid Rectangle
                    w, h = rng.integers(3, 8, size=2)
                    inp[r:min(r+w, self.MAX_GRID), c:min(c+h, self.MAX_GRID)] = color
                    out[r:min(r+w, self.MAX_GRID), c:min(c+h, self.MAX_GRID)] = color

            self.pairs.append((inp, out))

    def __len__(self):
        return len(self.pairs)

    def sample(self, batch_size: int, split: str = 'train') -> dict:
        """Return a TensorDict batch compatible with all NS-ARC modules."""
        pool = self.val_pairs if split == 'val' else self.train_pairs
        if not pool: pool = self.pairs # Fallback if split fails bounds
        
        batch_size = min(batch_size, len(pool))
        chosen = random.sample(pool, batch_size)

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
