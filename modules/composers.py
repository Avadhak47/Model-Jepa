import torch
import torch.nn.functional as F
from core.base_module import BaseModule

class SequentialComposer(BaseModule):
    """Chains modules sequentially, merging their outputs into the running dict."""
    def __init__(self, modules: list, config: dict):
        super().__init__(config)
        self.pipes = modules

    def forward(self, inputs: dict) -> dict:
        x = inputs
        for m in self.pipes:
            out = m(x)
            x = {**x, **out}
        return x

class ParallelComposer(BaseModule):
    """Runs modules in parallel then combines their `latent` outputs via learned softmax weighting."""
    def __init__(self, modules: list, config: dict):
        super().__init__(config)
        self.pipes = modules
        n = len(modules)
        # Learned importance weights over parallel branches
        self.branch_logits = torch.nn.Parameter(torch.zeros(n))

    def forward(self, inputs: dict) -> dict:
        branch_outputs = [m(inputs) for m in self.pipes]
        weights = F.softmax(self.branch_logits, dim=0)  # [N]

        # Merge `latent` tensors with weighted sum (skip branches w/o latent key)
        latents = [o["latent"] for o in branch_outputs if "latent" in o]
        if latents:
            stacked = torch.stack(latents, dim=0)          # [N, B, D]
            merged = (stacked * weights[:len(latents)].view(-1, 1, 1)).sum(0)  # [B, D]
        else:
            merged = None

        combined: dict = {}
        for o in branch_outputs:
            combined.update(o)
        if merged is not None:
            combined["latent"] = merged
        combined["parallel_outputs"] = branch_outputs
        return combined

