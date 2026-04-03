import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.interfaces import BaseWorldModel

class MLPDynamicsModel(BaseWorldModel):
    """Simple continuous dynamics model."""
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        action_dim = config.get("action_dim", 32)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        self.reward_head = nn.Linear(256, 1)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        a = inputs["action"].to(self.device)
        x = torch.cat([z, a], dim=-1)
        
        hidden = x
        for layer in self.net[:-1]:
            hidden = layer(hidden)
            
        next_z = self.net[-1](hidden)
        reward = self.reward_head(hidden)
        return {"next_latent": next_z, "predicted_reward": reward}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        target_z = inputs["target_latent"].to(self.device)
        target_r = inputs["target_reward"].to(self.device)
        z_loss = F.mse_loss(outputs["next_latent"], target_z)
        r_loss = F.mse_loss(outputs["predicted_reward"].squeeze(-1), target_r)
        return {"loss": z_loss + r_loss, "z_loss": z_loss.detach(), "r_loss": r_loss.detach()}

class GaussianDynamicsModel(BaseWorldModel):
    """Probabilistic dynamics predicting mean and log-variance."""
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        action_dim = config.get("action_dim", 32)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        )
        self.mean_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        self.reward_head = nn.Linear(256, 1)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        a = inputs["action"].to(self.device)
        x = torch.cat([z, a], dim=-1)
        
        hidden = self.net(x)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)
        reward = self.reward_head(hidden)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        next_z = mean + eps * std
        
        return {"next_latent": next_z, "mean": mean, "logvar": logvar, "predicted_reward": reward}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        target_z = inputs["target_latent"].to(self.device)
        mean = outputs["mean"]
        logvar = outputs["logvar"]
        
        nll_loss = 0.5 * torch.mean(logvar + (target_z - mean)**2 / torch.exp(logvar))
        return {"loss": nll_loss, "nll_loss": nll_loss.detach()}

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class AdaLN(nn.Module):
    """Adaptive Layer Normalization for Action Injection."""
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 2),
            nn.SiLU()
        )
    def forward(self, x, action):
        gamma, beta = self.mlp(action).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

class AttentionResidual(nn.Module):
    """Attention over prior block outputs to prevent magnitude scaling collapse in deep models."""
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, current_x, history_x):
        history_stack = torch.stack(history_x, dim=2) # [B, T, L, D]
        q = self.query(current_x).unsqueeze(2) # [B, T, 1, D]
        k = self.key(history_stack) # [B, T, L, D]
        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1]**0.5), dim=-1) # [B, T, 1, L]
        attended_history = torch.matmul(attn_weights, history_stack).squeeze(2) # [B, T, D]
        return attended_history + current_x

class TransformerWorldModel(BaseWorldModel):
    """
    Action-Conditioned Autoregressive Transformer with AttnRes.
    Includes SIGReg metrics for stability computation.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = config.get("latent_dim", 128)
        self.action_dim = config.get("action_dim", 32)
        self.embed_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 4)
        self.nhead = config.get("nhead", 8)
        self.use_attn_res = config.get("use_attn_res", False)
        
        self.state_embed = nn.Linear(self.latent_dim, self.embed_dim)
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.embed_dim, self.nhead, batch_first=True) for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim)
            ) for _ in range(self.num_layers)
        ])
        self.adaln_attn = nn.ModuleList([AdaLN(self.embed_dim, self.action_dim) for _ in range(self.num_layers)])
        self.adaln_ffn = nn.ModuleList([AdaLN(self.embed_dim, self.action_dim) for _ in range(self.num_layers)])
        
        if self.use_attn_res:
            self.attn_res_layers = nn.ModuleList([AttentionResidual(self.embed_dim) for _ in range(self.num_layers)])
            self.ffn_res_layers = nn.ModuleList([AttentionResidual(self.embed_dim) for _ in range(self.num_layers)])
            
        self.next_state_head = nn.Linear(self.embed_dim, self.latent_dim)
        self.reward_head = nn.Linear(self.embed_dim, 1)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z_seq = inputs["latent"].to(self.device)
        a_seq = inputs["action"].to(self.device)
        
        if z_seq.dim() == 2:
            z_seq = z_seq.unsqueeze(1)
            a_seq = a_seq.unsqueeze(1)
            
        B, T, _ = z_seq.shape
        x = self.state_embed(z_seq)
        
        # Ensure action is [B, action_dim] — AdaLN injects per-sequence-step
        a_flat = a_seq[:, 0, :] if a_seq.dim() == 3 else a_seq  # take first action or use directly
        
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(self.device)
        self.attention_entropies = []
        
        history_x = [x]
        
        for i in range(self.num_layers):
            x_norm = self.adaln_attn[i](x, a_flat)
            attn_out, attn_weights = self.attn_layers[i](x_norm, x_norm, x_norm, attn_mask=mask, average_attn_weights=True)
            
            x = self.attn_res_layers[i](attn_out, history_x) if self.use_attn_res else x + attn_out
            if self.use_attn_res: history_x.append(x)
            
            ent = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1).mean()
            self.attention_entropies.append(ent)
            
            x_norm = self.adaln_ffn[i](x, a_flat)
            ffn_out = self.ffn_layers[i](x_norm)
            
            x = self.ffn_res_layers[i](ffn_out, history_x) if self.use_attn_res else x + ffn_out
            if self.use_attn_res: history_x.append(x)
            
        next_z = self.next_state_head(x)
        reward = self.reward_head(x)
        
        return {
            "next_latent": next_z, 
            "predicted_reward": reward,
            "attention_entropy": sum(self.attention_entropies) / len(self.attention_entropies)
        }
        
    def loss(self, inputs: dict, outputs: dict) -> dict:
        target_z = inputs["target_latent"].to(self.device)
        target_r = inputs["target_reward"].to(self.device)
        
        # Ensure shapes match to prevent broadcasting warnings
        if pred_z.dim() > target_z.dim():
            pred_z = pred_z.squeeze(1)
        if pred_r.dim() > target_r.dim():
            pred_r = pred_r.squeeze(-1)

        z_loss = F.mse_loss(pred_z, target_z)
        r_loss = F.mse_loss(pred_r, target_r)
        
        std_z = torch.sqrt(pred_z.var(dim=0) + 1e-4).mean()
        sigreg_loss = F.relu(1.0 - std_z)
        
        total_loss = z_loss + r_loss + 0.1 * sigreg_loss
        
        return {
            "loss": total_loss, 
            "z_loss": z_loss.detach(),
            "r_loss": r_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "attention_entropy": outputs["attention_entropy"].detach()
        }

class TransformerWorldModel32(TransformerWorldModel):
    """32-layer Deep AttnRes Transformer"""
    def __init__(self, config: dict):
        config_override = config.copy()
        config_override["num_layers"] = 32
        config_override["use_attn_res"] = True
        super().__init__(config_override)

class TransformerWorldModel64(TransformerWorldModel):
    """64-layer Deep AttnRes Transformer"""
    def __init__(self, config: dict):
        config_override = config.copy()
        config_override["num_layers"] = 64
        config_override["use_attn_res"] = True
        super().__init__(config_override)

class TransformerWorldModel128(TransformerWorldModel):
    """128-layer Ultra-Deep AttnRes Transformer"""
    def __init__(self, config: dict):
        config_override = config.copy()
        config_override["num_layers"] = 128
        config_override["use_attn_res"] = True
        super().__init__(config_override)
