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
    """Transformer Encoder that maps grids to NMF Basis IDs."""
    def __init__(self, k_slots=3, d_model=256, n_basis=1024):
        super().__init__()
        self.k_slots = k_slots
        self.n_basis = n_basis
        self.patch_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, batch_first=True)
            for _ in range(8)
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
        
        # Predict logits over the 1024 Basis Atoms
        basis_logits = self.basis_proj(slots) # [B, K, 1024]
        return basis_logits

class AlgebraicDecoder(nn.Module):
    """Zero-parameter Linear Assembly of NMF basis atoms."""
    def __init__(self):
        super().__init__()

    def forward(self, manifold_slots):
        # manifold_slots: [B, K, 2700] (from BasisVQ)
        # Sum the selected Lego parts
        combined = manifold_slots.sum(dim=1) # [B, 2700]
        grid_3d = combined.view(-1, 15, 15, 12)
        color_logits = grid_3d[:, :, :, :10] # First 10 channels are colors
        return color_logits.permute(0, 3, 1, 2)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'k_slots': 3,
        'd_model': 256,
        'n_basis': 1024,
        'batch_size': 64,
        'epochs': 200,
        'lr': 3e-4,
        'library_path': 'arc_data/primitive_library.pt'
    }
    
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    
    encoder = BasisEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model'], n_basis=cfg['n_basis']).to(device)
    vq = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt').to(device)
    decoder = AlgebraicDecoder().to(device)
    
    # We only train the Encoder! The VQ Basis and Decoder are frozen/fixed.
    optimizer = optim.Adam(encoder.parameters(), lr=cfg['lr'])
    
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        total_loss = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)
            mask = batch['valid_mask'].to(device)
            x_mask = (state > 0).float()
            
            optimizer.zero_grad()
            
            # 1. Forward
            basis_logits = encoder(x_mask)
            q_manifold, indices = vq(basis_logits)
            logits = decoder(q_manifold)
            
            # 2. Loss (Reconstruction only)
            target = state.squeeze(1).long()
            loss = F.cross_entropy(logits, target, reduction='none')
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress.set_postfix({'Recon': f"{loss.item():.4f}", 'Active': vq.usage_counts.gt(0).sum().item()})
            
        if epoch % 20 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'cfg': cfg
            }, f"checkpoints/phoenix_basis_e{epoch}.pth")

if __name__ == "__main__":
    train()
