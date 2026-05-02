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

def apply_2d_rope(x, d_model):
    """2D Rotary Positional Embedding for a 15x15 grid."""
    b, n, d = x.shape
    h = int(np.sqrt(n))
    device = x.device
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(h, device=device), indexing='ij')
    y_coords = y_coords.flatten().float()
    x_coords = x_coords.flatten().float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 4, device=device).float() / d))
    sin_y = torch.sin(y_coords.unsqueeze(1) * inv_freq)
    cos_y = torch.cos(y_coords.unsqueeze(1) * inv_freq)
    sin_x = torch.sin(x_coords.unsqueeze(1) * inv_freq)
    cos_x = torch.cos(x_coords.unsqueeze(1) * inv_freq)
    x_rot = x.clone()
    half = d // 2
    x_rot[:, :, :half:2] = x[:, :, :half:2] * cos_y - x[:, :, 1:half:2] * sin_y
    x_rot[:, :, 1:half:2] = x[:, :, :half:2] * sin_y + x[:, :, 1:half:2] * cos_y
    x_rot[:, :, half::2] = x[:, :, half::2] * cos_x - x[:, :, half+1::2] * sin_x
    x_rot[:, :, half+1::2] = x[:, :, half::2] * sin_x + x[:, :, half+1::2] * cos_x
    return x_rot

class BasisEncoder(nn.Module):
    def __init__(self, k_slots=3, d_model=256, n_basis=1024):
        super().__init__()
        self.k_slots = k_slots
        self.n_basis = n_basis
        self.patch_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, batch_first=True, dropout=0.1)
            for _ in range(6)
        ])
        self.slot_queries = nn.Parameter(torch.randn(1, k_slots, d_model))
        self.slot_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.basis_proj = nn.Linear(d_model, n_basis)

    def forward(self, x_mask):
        b, c, h, w = x_mask.shape
        x = x_mask.view(b, h*w, 1)
        x = self.patch_proj(x)
        x = apply_2d_rope(x, x.shape[-1])
        curr = x
        for block in self.blocks:
            curr = block(curr)
        queries = self.slot_queries.expand(b, -1, -1)
        slots, _ = self.slot_attn(queries, curr, curr)
        basis_logits = self.basis_proj(slots)
        return basis_logits

class AlgebraicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, manifold_slots):
        combined = manifold_slots.sum(dim=1)
        grid_3d = combined.view(-1, 15, 15, 12)
        color_logits = grid_3d[:, :, :, :10]
        mask_logits = grid_3d[:, :, :, 10:12].sum(dim=-1, keepdim=True)
        return color_logits.permute(0, 3, 1, 2), mask_logits.permute(0, 3, 1, 2)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'k_slots': 3,
        'd_model': 256,
        'n_basis': 1024,
        'batch_size': 128,
        'epochs': 1000,
        'lr_init': 1e-4,
        'gain_start': 1.0,
        'gain_end': 50.0,
        'warmup_epochs': 250,
        'lambda_diversity': 0.3,
        'lambda_shape': 5.0,
        'library_path': 'arc_data/primitive_library.pt'
    }
    
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    
    encoder = BasisEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model'], n_basis=cfg['n_basis']).to(device)
    vq = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt').to(device)
    decoder = AlgebraicDecoder().to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=cfg['lr_init'], weight_decay=1e-4)
    
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        vq.train()
        
        if epoch <= cfg['warmup_epochs']:
            curr_gain = cfg['gain_start']
        else:
            progress_ratio = (epoch - cfg['warmup_epochs']) / (cfg['epochs'] - cfg['warmup_epochs'])
            curr_gain = cfg['gain_start'] + progress_ratio * (cfg['gain_end'] - cfg['gain_start'])
            
        vq.snapping_gain = curr_gain
        vq.usage_counts.zero_()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)
            mask = batch['valid_mask'].to(device)
            x_mask = (state > 0).float()
            
            optimizer.zero_grad()
            basis_logits = encoder(x_mask)
            q_manifold, indices = vq(basis_logits)
            color_logits, mask_logits = decoder(q_manifold)
            
            target = state.squeeze(1).long()
            color_loss = F.cross_entropy(color_logits, target, reduction='none')
            color_loss = (color_loss * mask).sum() / (mask.sum() + 1e-8)
            
            shape_target = x_mask
            shape_loss = F.binary_cross_entropy_with_logits(mask_logits, shape_target)
            
            probs = F.softmax(basis_logits.view(-1, cfg['n_basis']), dim=-1)
            avg_probs = probs.mean(dim=0)
            diversity_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
            
            loss = color_loss + cfg['lambda_shape'] * shape_loss - cfg['lambda_diversity'] * diversity_loss
            loss.backward()
            optimizer.step()
            
            progress.set_postfix({
                'Col': f"{color_loss.item():.2f}",
                'Shp': f"{shape_loss.item():.3f}",
                'Act': vq.usage_counts.gt(0).sum().item(),
                'G': f"{curr_gain:.1f}"
            })
            
        if epoch % 50 == 0:
            torch.save({'encoder': encoder.state_dict(), 'cfg': cfg}, f"checkpoints/phoenix_v4_e{epoch}.pth")

if __name__ == "__main__":
    train()
