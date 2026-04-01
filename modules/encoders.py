import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.interfaces import BaseEncoder

class MLPEncoder(BaseEncoder):
    """Simple continuous state encoder."""
    def __init__(self, config: dict):
        super().__init__(config)
        in_dim = config.get("input_dim", 64)
        latent_dim = config.get("latent_dim", 128)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        x = inputs["state"].to(self.device).float()
        x = x.view(x.shape[0], -1)  # flatten any spatial dims
        return {"latent": self.net(x)}

class CNNEncoder(BaseEncoder):
    """Extracts spatial priors from visual grid inputs."""
    def __init__(self, config: dict):
        super().__init__(config)
        in_channels = config.get("in_channels", 1)  # ARC grids are single-channel
        latent_dim = config.get("latent_dim", 128)
        # For small grids like ARC
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.projector = nn.Linear(128, latent_dim)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        img = inputs["state"].float().to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        z = self.projector(self.conv(img))
        return {"latent": z}

class TransformerEncoder(BaseEncoder):
    """Vision Transformer (ViT) backbone with CLS token projection designed to prevent representation collapse."""
    def __init__(self, config: dict):
        super().__init__(config)
        in_channels = config.get("in_channels", 3)
        patch_size = config.get("patch_size", 14)
        embed_dim = config.get("hidden_dim", 192)
        latent_dim = config.get("latent_dim", 128)
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Tiny ViT config: 12 layers, 3 heads
        # num_heads must divide embed_dim; derive the largest valid one from config
        raw_nhead = config.get("nhead", 4)
        nhead = max(1, raw_nhead) if embed_dim % raw_nhead == 0 else 1
        # Fall back to 1 if nothing works, then try common divisors
        for h in [raw_nhead, 4, 2, 1]:
            if embed_dim % h == 0:
                nhead = h
                break
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # Crucial anti-collapse projection (LeWM Section 3.1)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim)
        )
        self.to(self.device)
        
    def forward(self, inputs: dict) -> dict:
        img = inputs["state"].float().to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        x = self.patch_embed(img) # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, embed_dim]
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x)
        cls_out = x[:, 0] # Extract [CLS]
        z = self.projector(cls_out)
        return {"latent": z}

class Decoder(BaseEncoder):
    """Inverts the latent space back to pixel space for autencoding metrics."""
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        out_dim = config.get("input_dim", 64) # flattened
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        recon = self.net(z)
        return {"reconstruction": recon}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        target = inputs["state"].to(self.device)
        target = target.view(target.shape[0], -1) 
        recon = outputs["reconstruction"]
        return {"loss": F.mse_loss(recon, target)}
