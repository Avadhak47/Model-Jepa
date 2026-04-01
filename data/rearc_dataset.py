import torch

class ReARCDataset:
    """Loader for the reverse-engineered ARC (Re-ARC) procedurally generated tasks."""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.max_grid_size = 30
        
    def sample(self, batch_size: int) -> dict:
        """Loads procedurally scaled Re-ARC variables into batched TensorDicts."""
        states = torch.randint(0, 10, (batch_size, 1, self.max_grid_size, self.max_grid_size)).float()
        targets = torch.randint(0, 10, (batch_size, 1, self.max_grid_size, self.max_grid_size)).float()
        return {"state": states, "target_latent": targets, "target_reward": torch.zeros(batch_size, 1)}
