import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.extract_arc_objects import PrimitiveDataset
from modules.basis_vq import BasisVQ

# ─────────────────────────── Helpers ────────────────────────────────────────

def focal_loss(logits, target, mask, gamma=2.0):
    """
    Focal Loss for ARC: Penalizes the model more for missing foreground pixels.
    logits: [B, 10, 15, 15], target: [B, 15, 15], mask: [B, 1, 15, 15]
    """
    ce_loss = F.cross_entropy(logits, target, reduction='none')
    pt = torch.exp(-ce_loss) # probability of the correct class
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce_loss * mask.squeeze(1)).sum() / (mask.sum() + 1e-8)
    return loss

def apply_2d_rope(x, d_model):
    """2D Rotary Positional Embedding for a 15x15 grid."""
    b, n, d = x.shape
    h = int(np.sqrt(n))
    device = x.device
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(h, device=device), indexing='ij')
    y_coords = y_coords.flatten().float()
    x_coords = x_coords.flatten().float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 4, device=device).float() / d))
    sin_y = torch.sin(y_coords.unsqueeze(1) * inv_freq)
    cos_y = torch.cos(y_coords.unsqueeze(1) * inv_freq)
    sin_x = torch.sin(x_coords.unsqueeze(1) * inv_freq)
    cos_x = torch.cos(x_coords.unsqueeze(1) * inv_freq)
    x_rot = x.clone()
    half = d // 2
    x_rot[:, :, :half:2]   = x[:, :, :half:2]   * cos_y - x[:, :, 1:half:2]   * sin_y
    x_rot[:, :, 1:half:2]  = x[:, :, :half:2]   * sin_y + x[:, :, 1:half:2]   * cos_y
    x_rot[:, :, half::2]   = x[:, :, half::2]   * cos_x - x[:, :, half+1::2]  * sin_x
    x_rot[:, :, half+1::2] = x[:, :, half::2]   * sin_x + x[:, :, half+1::2]  * cos_x
    return x_rot

# ─────────────────────────── Encoder ────────────────────────────────────────

class BasisEncoder(nn.Module):
    """
    Phoenix v6:
    - Returns raw slot features [B, K, d_model] (NOT logits over basis)
    - Color AND shape information are both preserved in slot_features
    - BasisVQ does the nearest-neighbour lookup separately
    """
    def __init__(self, k_slots=5, d_model=256, n_basis=1024):
        super().__init__()
        self.k_slots = k_slots
        self.n_basis = n_basis
        self.patch_proj = nn.Linear(10, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8,
                                       dim_feedforward=1024, batch_first=True,
                                       dropout=0.1)
            for _ in range(8)
        ])
        self.slot_queries = nn.Parameter(torch.randn(1, k_slots, d_model))
        self.slot_attn    = nn.MultiheadAttention(d_model, 8, batch_first=True)
        # No basis_proj here — we output raw slot features

    def forward(self, x_onehot):
        b, c, h, w = x_onehot.shape
        x = x_onehot.permute(0, 2, 3, 1).reshape(b, h * w, 10)
        x = self.patch_proj(x)
        x = apply_2d_rope(x, x.shape[-1])
        curr = x
        for block in self.blocks:
            curr = block(curr)
        queries = self.slot_queries.expand(b, -1, -1)
        slots, _ = self.slot_attn(queries, curr, curr)
        return slots   # [B, K, d_model]  — contains color + shape info

# ─────────────────────────── Decoder ────────────────────────────────────────

class AlgebraicDecoder(nn.Module):
    """
    Phoenix v6 Broadcast Decoder:
    Forces slot competition by making each slot reconstruct its own 
    local patch + alpha mask. Final image = sum(slot_color * slot_alpha).
    """
    def __init__(self, n_slots=5, basis_dim=2025):
        super().__init__()
        self.n_slots   = n_slots
        self.basis_dim = basis_dim

        # Per-slot processing (Shared across slots)
        self.slot_net = nn.Sequential(
            nn.Linear(basis_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 15 * 15 * 11), # 10 colors + 1 alpha mask
        )

    def forward(self, q_st):
        # q_st: [B, K, basis_dim]
        b, k, _ = q_st.shape
        
        # Process each slot independently
        # [B, K, 15*15*11] -> [B, K, 11, 15, 15]
        decoded = self.slot_net(q_st).view(b, k, 11, 15, 15)
        
        slot_colors = decoded[:, :, :10, :, :]  # [B, K, 10, 15, 15]
        slot_alphas = decoded[:, :, 10:, :, :]  # [B, K, 1, 15, 15]
        
        # Softmax masks over slots so they "partition" the image
        masks = F.softmax(slot_alphas, dim=1)   # [B, K, 1, 15, 15]
        
        # Weighted sum of colors
        combined_logits = (slot_colors * masks).sum(dim=1) # [B, 10, 15, 15]
        
        # For shape loss, we use the aggregate mask intensity
        shape_logits = masks.sum(dim=1) # [B, 1, 15, 15] (should be ~1.0 everywhere)
        
        return combined_logits, shape_logits

# ─────────────────────────── Training ────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'k_slots':           5,
        'd_model':           256,
        'n_basis':           1024,
        'batch_size':        128,
        'epochs':            500,       # v6 converges fast (no quantization noise)
        'lr':                5e-4,
        'lambda_diversity':  2.0,       # Increased to fight collapse
        'lambda_shape':      1.0,       
        'dead_revive_every': 20,        # Reset more frequently
        'library_path':      'arc_data/primitive_library.pt'
    }

    dataset    = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'],
                            shuffle=True, drop_last=True)

    encoder = BasisEncoder(k_slots=cfg['k_slots'],
                           d_model=cfg['d_model'],
                           n_basis=cfg['n_basis']).to(device)
    vq      = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt',
                      d_model=cfg['d_model'],
                      n_basis=cfg['n_basis']).to(device)
    
    decoder = AlgebraicDecoder(n_slots=cfg['k_slots'], 
                               basis_dim=vq.basis_dim).to(device)

    # Train encoder + VQ projection + decoder
    optimizer = optim.AdamW(
        list(encoder.parameters()) +
        list(vq.slot_proj.parameters()) +   # the projection into NMF space
        list(decoder.parameters()),
        lr=cfg['lr'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-5)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train(); vq.train(); decoder.train()

        # Dead atom revival
        if epoch % cfg['dead_revive_every'] == 0 and epoch > 50:
            n_dead = vq.revive_dead_atoms()
            if n_dead > 0:
                tqdm.write(f"  [Revival] {n_dead} dead atoms reset at epoch {epoch}")

        vq.usage_counts.zero_()

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)   # [B, 1, 15, 15]
            mask  = batch['valid_mask'].to(device)

            x_onehot = F.one_hot(state.squeeze(1).long(),
                                  num_classes=10).permute(0, 3, 1, 2).float()

            optimizer.zero_grad()

            # ── Forward ──
            slot_features              = encoder(x_onehot)              # [B, K, d_model]
            q_st, indices, vq_loss, entropy = vq(slot_features)         # [B, K, basis_dim]
            color_logits, shape_logits = decoder(q_st)                  # [B, 10/1, 15, 15]

            # ── Losses ──
            target     = state.squeeze(1).long()
            color_loss = focal_loss(color_logits, target, mask, gamma=2.0)

            shape_target = (state > 0).float()
            shape_loss   = F.binary_cross_entropy_with_logits(shape_logits, shape_target)

            loss = (color_loss
                    + vq_loss                               # commitment loss
                    + cfg['lambda_shape'] * shape_loss
                    - cfg['lambda_diversity'] * entropy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()

            # Pixel accuracy metric
            with torch.no_grad():
                preds = color_logits.argmax(dim=1)
                px_acc = (preds == target).float().mean().item() * 100

            progress.set_postfix({
                'Col':    f"{color_loss.item():.3f}",
                'VQ':     f"{vq_loss.item():.3f}",
                'Acc':    f"{px_acc:.1f}%",
                'Act':    vq.usage_counts.gt(0).sum().item(),
            })

        scheduler.step()

        if epoch % 50 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'vq_proj': vq.slot_proj.state_dict(),
                'decoder': decoder.state_dict(),
                'cfg':     cfg,
                'epoch':   epoch,
            }, f"checkpoints/phoenix_v6_e{epoch}.pth")
            tqdm.write(f"  [Saved] Checkpoint at epoch {epoch}")

if __name__ == "__main__":
    train()
