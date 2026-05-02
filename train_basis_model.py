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

# ... [apply_2d_rope, BasisEncoder, AlgebraicDecoder remain same] ...

def apply_2d_rope(x, d_model):
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
            for _ in range(6) # Slightly shallower for faster iteration
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
        return color_logits.permute(0, 3, 1, 2)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'k_slots': 3,
        'd_model': 256,
        'n_basis': 1024,
        'batch_size': 64,
        'epochs': 300,
        'lr_init': 1e-3,
        'lr_final': 1e-4,
        'gain_start': 1.0,
        'gain_end': 30.0,
        'lambda_diversity': 0.05,
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
        
        # 1. ANNEALING SCHEDULE
        # Increase gain (snapping) and decrease LR over time
        progress_ratio = min(1.0, (epoch - 1) / (cfg['epochs'] * 0.7))
        curr_gain = cfg['gain_start'] + progress_ratio * (cfg['gain_end'] - cfg['gain_start'])
        curr_lr = cfg['lr_init'] - progress_ratio * (cfg['lr_init'] - cfg['lr_final'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
            
        vq.snapping_gain = curr_gain
        vq.usage_counts.zero_() # Reset for each epoch audit
        
        total_recon = 0
        total_div = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)
            mask = batch['valid_mask'].to(device)
            x_mask = (state > 0).float()
            
            optimizer.zero_grad()
            
            # 2. FORWARD
            basis_logits = encoder(x_mask)
            q_manifold, indices = vq(basis_logits)
            logits = decoder(q_manifold)
            
            # 3. LOSSES
            # Reconstruction (Cross-Entropy)
            target = state.squeeze(1).long()
            recon_loss = F.cross_entropy(logits, target, reduction='none')
            recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-8)
            
            # Diversity Loss: Maximize entropy of basis usage in the batch
            # avg_probs: [1024]
            probs = F.softmax(basis_logits.view(-1, cfg['n_basis']), dim=-1)
            avg_probs = probs.mean(dim=0)
            diversity_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
            # Negative entropy because we want to MAXIMIZE it
            
            loss = recon_loss - cfg['lambda_diversity'] * diversity_loss
            loss.backward()
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_div += diversity_loss.item()
            
            progress.set_postfix({
                'Rec': f"{recon_loss.item():.3f}",
                'Div': f"{diversity_loss.item():.2f}",
                'Act': vq.usage_counts.gt(0).sum().item(),
                'G': f"{curr_gain:.1f}"
            })
            
        if epoch % 20 == 0:
            torch.save({'encoder': encoder.state_dict(), 'cfg': cfg}, f"checkpoints/phoenix_basis_v2_e{epoch}.pth")

if __name__ == "__main__":
    train()
