import torch

def mean_aggregator(stacked_tensors):
    """Aggregates a list/stack of tensors by averaging."""
    return torch.mean(stacked_tensors, dim=0)

def max_aggregator(stacked_tensors):
    """Aggregates a list/stack of tensors by max pooling."""
    return torch.max(stacked_tensors, dim=0)[0]

def attention_aggregator(stacked_tensors, weights):
    """Aggregates a stack of tensors [N, B, D] using attention weights [B, N]."""
    weights = weights.unsqueeze(-1)  # [B, N, 1]
    stacked_tensors = stacked_tensors.transpose(0, 1)  # [B, N, D]
    aggregated = torch.sum(stacked_tensors * weights, dim=1)  # [B, D]
    return aggregated
