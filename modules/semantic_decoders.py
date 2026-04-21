import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.register_buffer("grid", self._build_grid(resolution))

    @staticmethod
    def _build_grid(resolution):
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1)
        grid = torch.cat([grid, 1.0 - grid], dim=-1)
        return grid.unsqueeze(0)

    def forward(self, x):
        pos = self.dense(self.grid).permute(0, 3, 1, 2)
        return x + pos

def vicreg_loss(z, std_coeff=25.0, cov_coeff=1.0):
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4) # enforce variance
    std_loss = torch.mean(F.relu(1.0 - std))
    cov = (z.T @ z) / (z.shape[0] - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum() / z.shape[1]
    return std_coeff * std_loss + cov_coeff * cov_loss

def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class SemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.embed_dim = config['hidden_dim']
        self.vocab_size = config['vocab_size']
        self.grid_size = config['grid_size']
        
        self.decoder_pos = SoftPositionEmbed(self.embed_dim, (self.grid_size, self.grid_size))
        
        self.spatial_broadcast = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        
        self.color_mlp = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, self.vocab_size, 3, padding=1)
        )
        self.alpha_mlp = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        self.to(self.device)

    def forward(self, inputs):
        z = inputs["latent"] 
        B, num_slots, D = z.shape
        H, W = self.grid_size, self.grid_size
        
        z_tiled = z.view(B * num_slots, D, 1, 1).expand(-1, -1, H, W)
        z_tiled = self.decoder_pos(z_tiled)
        
        decoded_features = self.spatial_broadcast(z_tiled)

        colors = self.color_mlp(decoded_features).view(B, num_slots, self.vocab_size, H, W)
        alphas = self.alpha_mlp(decoded_features).view(B, num_slots, 1, H, W)
        
        alphas_normalized = F.softmax(alphas, dim=1) 
        reconstruction = torch.sum(alphas_normalized * colors, dim=1)
        
        return {"reconstruction": reconstruction, "alphas": alphas_normalized}

    def loss(self, inputs, outputs, w_fg=50.0):
        target = inputs["state"].to(self.device).long().squeeze(1)
        if target.dim() == 2: target = target.unsqueeze(0)
        recon_logits = outputs["reconstruction"] 
        
        # 1. Pixel Focal Loss (With Foreground Monopoly Erasure Weighting)
        ce_loss = F.cross_entropy(recon_logits, target, reduction='none')
        
        # Apply 50x weighting to non-zero (non-background) pixels!
        mask_fg = (target > 0).float()
        weight_matrix = 1.0 + (mask_fg * (w_fg - 1.0))
        
        pt = torch.exp(-ce_loss)
        recon_loss = (((1 - pt) ** 2.0) * ce_loss * weight_matrix).mean()

        # 2. VICReg Regularization for Slots
        z = inputs["latent"] 
        B, S, D = z.shape
        z_flat = z.view(B * S, D)
        vic_loss = vicreg_loss(z_flat, std_coeff=10.0, cov_coeff=1.0) 
        
        total_loss = recon_loss + (0.5 * vic_loss)
        return {"loss": total_loss, "recon_loss": recon_loss, "vic_loss": vic_loss}
