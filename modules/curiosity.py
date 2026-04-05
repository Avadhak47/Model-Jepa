import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.interfaces import BaseCuriosity


class PredErrCuriosity(BaseCuriosity):
    """
    Forward-model prediction error curiosity.
    Computes ||WM_pred_z - true_next_z||_2 as intrinsic reward.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.scale = config.get("curiosity_scale", 1.0)

    def forward(self, inputs: dict) -> dict:
        true_z = inputs["target_latent"].to(self.device)
        pred_z = inputs["next_latent"].to(self.device)
        err = F.mse_loss(pred_z, true_z, reduction='none').mean(dim=-1, keepdim=True)
        return {
            "intrinsic_reward": err * self.scale,
            "pred_err": err.mean(),
        }

    def loss(self, inputs: dict, outputs: dict) -> dict:
        return {"loss": outputs["pred_err"]}


class RNDCuriosity(BaseCuriosity):
    """
    Random Network Distillation (Burda et al. 2019).
    Trains a predictor network to match a fixed random target encoder.
    Prediction error = epistemic novelty signal.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        hidden     = 256
        self.scale = config.get("curiosity_scale", 1.0)

        # Fixed random target — outputs are never trained
        self.target = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Learned predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)

        with torch.no_grad():
            t_feat = self.target(z)           # [B, H] fixed

        p_feat = self.predictor(z)            # [B, H] trained

        err             = F.mse_loss(p_feat, t_feat, reduction='none').mean(dim=-1)  # [B]
        intrinsic_reward = (err * self.scale).unsqueeze(-1)                           # [B, 1]

        return {
            "intrinsic_reward": intrinsic_reward,
            "rnd_loss":         err.mean(),        # scalar for backward
        }

    def loss(self, inputs: dict, outputs: dict) -> dict:
        return {"loss": outputs["rnd_loss"]}


class EnsembleDisagreementCuriosity(BaseCuriosity):
    """
    Epistemic uncertainty via ensemble disagreement:
    Maintains K small predictor heads; intrinsic reward = variance across their predictions.
    All K heads are trained to predict the fixed RND target — disagreement = novelty.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim  = config.get("latent_dim", 128)
        self.K      = config.get("ensemble_size", 5)
        self.scale  = config.get("curiosity_scale", 1.0)
        hidden      = 256

        # Fixed random target
        self.target = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # K independent predictor heads
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(self.K)
        ])
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)   # [B, D]

        with torch.no_grad():
            t_feat = self.target(z)             # [B, H]

        preds = torch.stack([h(z) for h in self.predictors], dim=0)   # [K, B, H]

        # Variance across ensemble heads → disagreement
        variance        = preds.var(dim=0).mean(dim=-1, keepdim=True)  # [B, 1]
        intrinsic_reward = variance * self.scale

        # Per-head MSE loss for training predictors
        t_expanded = t_feat.unsqueeze(0).expand_as(preds)  # [K, B, H]
        pred_losses = F.mse_loss(preds, t_expanded, reduction='none').mean(dim=-1)  # [K, B]
        ensemble_loss = pred_losses.mean()

        return {
            "intrinsic_reward": intrinsic_reward,
            "ensemble_variance": variance.mean(),
            "ensemble_loss":     ensemble_loss,
        }

    def loss(self, inputs: dict, outputs: dict) -> dict:
        return {"loss": outputs["ensemble_loss"]}

class SlotRNDCuriosity(BaseCuriosity):
    """
    Object-Centric Random Network Distillation.
    Computes novelty per-slot and averages it, encouraging exploration of novel abstract objects.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        hidden     = 256
        self.scale = config.get("curiosity_scale", 1.0)
        
        # Fixed random target evaluated on individual slots
        self.target = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        for p in self.target.parameters():
            p.requires_grad = False
            
        # Learned predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.to(self.device)
        
    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device) # [B, num_slots, latent_dim]
        B, S, D = z.shape
        
        # Flatten to [B*S, latent_dim]
        z_flat = z.view(-1, D)
        
        with torch.no_grad():
            t_feat = self.target(z_flat)
            
        p_feat = self.predictor(z_flat)
        
        # Error per object: [B*S]
        err = F.mse_loss(p_feat, t_feat, reduction='none').mean(dim=-1)
        
        # Average object-novelties for final image novelty
        err = err.view(B, S).mean(dim=1) # [B]
        
        intrinsic_reward = (err * self.scale).unsqueeze(-1) # [B, 1]
        
        return {
            "intrinsic_reward": intrinsic_reward,
            "rnd_loss": err.mean()
        }
        
    def loss(self, inputs: dict, outputs: dict) -> dict:
        return {"loss": outputs["rnd_loss"]}
