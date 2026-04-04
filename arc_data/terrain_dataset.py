import torch

class TerrainDataset:
    """Generator for synthetic N-dimensional procedural terrains."""
    def __init__(self, config: dict):
        self.config = config
        self.dim = config.get("latent_dim", 128)

    def __len__(self):
        return 10000 # Generator-style datasets have arbitrary large length for evaluation batches

    def sample(self, batch_size: int) -> dict:
        """Creates procedural mathematical noises to simulate infinite state terrains."""
        noise_states = torch.randn(batch_size, self.dim)
        shifted_targets = noise_states + torch.randn_like(noise_states) * 0.1
        return {"state": noise_states, "target_latent": shifted_targets, "target_reward": torch.ones(batch_size, 1)}
