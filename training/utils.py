import torch

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    """Computes Generalized Advantage Estimation (GAE)."""
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * (1 - dones[t]) * last_gae_lam
        
    returns = advantages + values
    return advantages, returns
