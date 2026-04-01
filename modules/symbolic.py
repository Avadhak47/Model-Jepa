import torch
import torch.nn as nn
from modules.interfaces import BaseSymbolicModule

class ProgramGenerator(BaseSymbolicModule):
    """Translates continuous latents into discrete symbolic codes (e.g. DSL primitives)."""
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        vocab_size = config.get("vocab_size", 50)
        self.net = nn.Linear(latent_dim, vocab_size)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        logits = self.net(z)
        symbols = torch.argmax(logits, dim=-1)
        return {"symbol_logits": logits, "symbols": symbols}

class ConstraintMask(BaseSymbolicModule):
    """Applies rigorous logical constraints by masking out illegal actions."""
    def __init__(self, config: dict):
        super().__init__(config)

    def forward(self, inputs: dict) -> dict:
        action_logits = inputs["action_logits"].to(self.device)
        illegal_mask = inputs["illegal_action_mask"].to(self.device)
        
        masked_logits = action_logits.masked_fill(illegal_mask.bool(), float('-inf'))
        return {"action_logits": masked_logits}
