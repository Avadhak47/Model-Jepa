import json
import os
import torch
import numpy as np

class ARCDataset:
    """Loader for the true ARC JSON dataset benchmark."""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tasks = []
        self.max_grid_size = 30
        
        if os.path.exists(data_path) and os.path.isdir(data_path):
            count = 0
            for file in os.listdir(data_path):
                if file.endswith('.json'):
                    with open(os.path.join(data_path, file), 'r') as f:
                        task = json.load(f)
                        self.tasks.append(task)
                        count += 1
            print(f"Loaded {count} ARC JSON tasks from {data_path}")
        else:
            print(f"Warning: ARC path '{data_path}' not found. Generating minimal mock JSON structures.")
            self.tasks.append({"train": [{"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}]})
        
    def pad_grid(self, grid: list) -> np.ndarray:
        arr = np.array(grid)
        h, w = arr.shape
        # Ensure we don't exceed max grid size
        h, w = min(h, self.max_grid_size), min(w, self.max_grid_size)
        padded = np.zeros((self.max_grid_size, self.max_grid_size))
        padded[:h, :w] = arr[:h, :w]
        return padded
        
    def sample(self, batch_size: int) -> dict:
        """Parses actual JSON tasks into padded batched TensorDicts."""
        states = []
        targets = []
        
        for _ in range(batch_size):
            task = self.tasks[np.random.randint(0, len(self.tasks))]
            # Sample a train pair
            pair = task["train"][np.random.randint(0, len(task["train"]))]
            inp_pad = self.pad_grid(pair["input"])
            out_pad = self.pad_grid(pair["output"])
            
            states.append(inp_pad)
            targets.append(out_pad)
            
        states_tensor  = torch.from_numpy(np.stack(states)).float().unsqueeze(1)   # [B, 1, H, W]
        targets_tensor = torch.from_numpy(np.stack(targets)).float().unsqueeze(1)
        
        return {"state": states_tensor, "target_latent": targets_tensor, "target_reward": torch.zeros(batch_size, 1)}
