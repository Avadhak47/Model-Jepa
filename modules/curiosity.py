import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.interfaces import BaseCuriosity

class PredErrCuriosity(BaseCuriosity):
    """Intrinsic reward based on forward dynamics prediction error."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.scale = config.get("curiosity_scale", 1.0)
        
    def forward(self, inputs: dict) -> dict:
        """Requires true_next_latent and pred_next_latent."""
        true_z = inputs["target_latent"]
        pred_z = inputs["next_latent"]
        
        err = F.mse_loss(pred_z, true_z, reduction='none').mean(dim=-1, keepdim=True)
        intrinsic_reward = err * self.scale
        return {"intrinsic_reward": intrinsic_reward}

class RNDCuriosity(BaseCuriosity):
    """Random Network Distillation for epistemic novelty."""
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        self.scale = config.get("curiosity_scale", 1.0)
        
        self.target = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        for param in self.target.parameters():
            param.requires_grad = False
            
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        
        with torch.no_grad():
            target_feat = self.target(z)
        pred_feat = self.predictor(z)
        
        err = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=-1, keepdim=True)
        return {"intrinsic_reward": err * self.scale, "rnd_loss": err.mean()}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        return {"loss": outputs["rnd_loss"]}

class EnsembleDisagreementCuriosity(BaseCuriosity):
    """Intrinsic bonus based on variance of an ensemble of world models."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.scale = config.get("curiosity_scale", 1.0)

    def forward(self, inputs: dict) -> dict:
        ensemble_preds = inputs["ensemble_next_latents"].to(self.device) 
        variance = ensemble_preds.var(dim=0).mean(dim=-1, keepdim=True)
        return {"intrinsic_reward": variance * self.scale}
