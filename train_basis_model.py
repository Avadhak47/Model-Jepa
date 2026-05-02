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

# ... [apply_2d_rope remains same] ...

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
        # Added: Return mask prediction separately for shape-first learning
        mask_logits = grid_3d[:, :, :, 10:12].sum(dim=-1, keepdim=True) # Structural energy
        return color_logits.permute(0, 3, 1, 2), mask_logits.permute(0, 3, 1, 2)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'k_slots': 3,
        'd_model': 256,
        'n_basis': 1024,
        'batch_size': 128, # Increased batch size for diversity
        'epochs': 1000,
        'lr_init': 1e-4,
        'gain_start': 1.0,
        'gain_end': 50.0,
        'warmup_epochs': 250, # EXTENDED CHILDHOOD: Hold gain at 1.0
        'lambda_diversity': 0.3, # Harder penalty on collapse
        'lambda_shape': 5.0, # SHAPE-FIRST: Heavy weight on getting the mask right
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
        
        # 1. ANNEALING (Now with Warmup Delay)
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
            
            # 2. FORWARD
            basis_logits = encoder(x_mask)
            q_manifold, indices = vq(basis_logits)
            color_logits, mask_logits = decoder(q_manifold)
            
            # 3. LOSSES
            # Reconstruction (Color)
            target = state.squeeze(1).long()
            color_loss = F.cross_entropy(color_logits, target, reduction='none')
            color_loss = (color_loss * mask).sum() / (mask.sum() + 1e-8)
            
            # SHAPE LOSS (Binary Cross Entropy on the mask)
            shape_target = x_mask
            shape_loss = F.binary_cross_entropy_with_logits(mask_logits, shape_target)
            
            # DIVERSITY
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
            
        if epoch % 20 == 0:
            torch.save({'encoder': encoder.state_dict(), 'cfg': cfg}, f"checkpoints/phoenix_v3_e{epoch}.pth")

if __name__ == "__main__":
    train()
