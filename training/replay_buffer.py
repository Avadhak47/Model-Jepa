import random
import collections
import torch

class ReplayBuffer:
    """High-throughput O(1) append/pop circular buffer with typed TensorDict output."""
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.storage = collections.deque(maxlen=capacity)  # O(1) append+pop

    def add(self, transition: tuple):
        """Store (state_tensor, action_tensor, reward_float, next_state_tensor, done_bool)."""
        self.storage.append(transition)

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size: int) -> dict:
        """Sample a batch of transitions as TensorDicts moved to target device."""
        actual_size = min(batch_size, len(self.storage))
        if actual_size == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")
        batch = random.sample(list(self.storage), actual_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "state":      torch.stack(states).to(self.device),
            "action":     torch.stack(actions).to(self.device),
            "reward":     torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device),
            "next_state": torch.stack(next_states).to(self.device),
            "done":       torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device),
            "latent":     torch.stack(states).to(self.device),          # alias for module compat
            "target_latent": torch.stack(next_states).to(self.device),  # alias
            "target_reward": torch.tensor(rewards, dtype=torch.float32).to(self.device),
        }
