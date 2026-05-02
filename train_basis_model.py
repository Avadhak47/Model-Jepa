import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from tools.extract_arc_objects import PrimitiveDataset
from modules.basis_vq import BasisVQ

# ─────────────────────────── Helpers ───────────────────────────

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

# ─────────────────────────── Encoder ───────────────────────────

class BasisEncoder(nn.Module):
    def __init__(self, k_slots=5, d_model=256, n_basis=1024):
        super().__init__()
        self.k_slots = k_slots
        self.n_basis = n_basis
        self.patch_proj = nn.Linear(10, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8,
                                       dim_feedforward=1024, batch_first=True, dropout=0.1)
            for _ in range(8)
        ])
        self.slot_queries = nn.Parameter(torch.randn(1, k_slots, d_model))
        self.slot_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.basis_proj = nn.Linear(d_model, n_basis)

    def forward(self, x_onehot):
        b, c, h, w = x_onehot.shape
        x = x_onehot.permute(0, 2, 3, 1).reshape(b, h*w, 10)
        x = self.patch_proj(x)
        x = apply_2d_rope(x, x.shape[-1])
        curr = x
        for block in self.blocks:
            curr = block(curr)
        queries = self.slot_queries.expand(b, -1, -1)
        slots, _ = self.slot_attn(queries, curr, curr)
        return self.basis_proj(slots)   # [B, K, 1024]

# ─────────────────────────── Decoder ───────────────────────────

class AlgebraicDecoder(nn.Module):
    """
    Phoenix v5: Decodes COLOR and SHAPE from separate NMF axes.
    - Color comes from color_manifold (sum of K color-only basis slices)
    - Shape mask from pos_manifold   (sum of K position-only basis slices)
    A lightweight learnable head projects the summed color manifold to logits.
    This avoids the "summed NMF noise" problem.
    """
    def __init__(self, n_slots=5):
        super().__init__()
        self.n_slots = n_slots
        # color_manifold per slot: [B, K, 2250] -> sum -> [B, 2250]
        # Small learnable head to fix the "sum = blur" problem
        self.color_head = nn.Sequential(
            nn.Linear(2250, 512),
            nn.ReLU(),
            nn.Linear(512, 15 * 15 * 10)   # output: pixel-wise color logits
        )
        self.shape_head = nn.Sequential(
            nn.Linear(450, 15 * 15 * 1)    # output: pixel-wise binary mask
        )

    def forward(self, color_manifold, pos_manifold):
        # color_manifold: [B, K, 2250],  pos_manifold: [B, K, 450]
        c_sum = color_manifold.mean(dim=1)   # [B, 2250]  (mean > sum to avoid scale explosion)
        p_sum = pos_manifold.mean(dim=1)     # [B, 450]

        color_logits = self.color_head(c_sum).view(-1, 10, 15, 15)   # [B, 10, 15, 15]
        shape_logits = self.shape_head(p_sum).view(-1, 1, 15, 15)    # [B, 1, 15, 15]
        return color_logits, shape_logits

# ─────────────────────────── Training ───────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'k_slots':          5,
        'd_model':          256,
        'n_basis':          1024,
        'batch_size':       128,
        'epochs':           1000,
        'lr':               2e-4,
        # Gumbel temperature: start high (explore), anneal to near-0 (snap)
        'tau_start':        2.0,
        'tau_end':          0.1,
        'warmup_epochs':    100,   # Hold tau_start for first 100 epochs
        'lambda_diversity':  0.5,  # Entropy bonus (maximise codebook usage)
        'lambda_shape':      5.0,  # Shape loss weight
        'dead_revive_every': 50,   # Epochs between dead-atom revival passes
        'library_path':     'arc_data/primitive_library.pt'
    }

    dataset   = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'],
                            shuffle=True, drop_last=True)

    encoder = BasisEncoder(k_slots=cfg['k_slots'],
                           d_model=cfg['d_model'],
                           n_basis=cfg['n_basis']).to(device)
    vq      = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt').to(device)
    decoder = AlgebraicDecoder(n_slots=cfg['k_slots']).to(device)

    # Train encoder + the small decoder heads (basis is frozen in BasisVQ)
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg['lr'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-5)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train(); vq.train(); decoder.train()

        # ── Temperature schedule (Gumbel tau) ──
        if epoch <= cfg['warmup_epochs']:
            curr_tau = cfg['tau_start']
        else:
            t = (epoch - cfg['warmup_epochs']) / (cfg['epochs'] - cfg['warmup_epochs'])
            curr_tau = cfg['tau_start'] * (cfg['tau_end'] / cfg['tau_start']) ** t
        vq.temperature = curr_tau

        # ── Dead atom revival ──
        if epoch % cfg['dead_revive_every'] == 0 and epoch > cfg['warmup_epochs']:
            n_dead = vq.revive_dead_atoms(None)
            if n_dead > 0:
                tqdm.write(f"  [Revival] {n_dead} dead atoms reset at epoch {epoch}")

        vq.usage_counts.zero_()
        epoch_col, epoch_shp = 0.0, 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)    # [B, 1, 15, 15]
            mask  = batch['valid_mask'].to(device)

            x_onehot = F.one_hot(state.squeeze(1).long(),
                                  num_classes=10).permute(0, 3, 1, 2).float()

            optimizer.zero_grad()

            logits                          = encoder(x_onehot)          # [B, K, 1024]
            c_mani, p_mani, indices, entropy = vq(logits)                 # forward
            color_logits, shape_logits      = decoder(c_mani, p_mani)    # [B,10,15,15]

            # ── Losses ──
            target     = state.squeeze(1).long()
            color_loss = F.cross_entropy(color_logits, target, reduction='none')
            color_loss = (color_loss * mask).sum() / (mask.sum() + 1e-8)

            shape_target = (state > 0).float()
            shape_loss   = F.binary_cross_entropy_with_logits(shape_logits, shape_target)

            # Maximise entropy of codebook usage (prevents collapse)
            loss = (color_loss
                    + cfg['lambda_shape']    * shape_loss
                    - cfg['lambda_diversity'] * entropy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            epoch_col += color_loss.item()
            epoch_shp += shape_loss.item()

            progress.set_postfix({
                'Col':  f"{color_loss.item():.3f}",
                'Shp':  f"{shape_loss.item():.3f}",
                'Act':  vq.usage_counts.gt(0).sum().item(),
                'τ':    f"{curr_tau:.3f}",
            })

        scheduler.step()

        if epoch % 50 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'cfg':     cfg,
                'epoch':   epoch,
            }, f"checkpoints/phoenix_v5_e{epoch}.pth")
            tqdm.write(f"  [Saved] Checkpoint at epoch {epoch}")

if __name__ == "__main__":
    train()
