import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from tools.extract_arc_objects import PrimitiveDataset
from modules.vq import NMFInformedVQ

def apply_2d_rope(x, d_model):
    b, n, d = x.shape
    h = int(np.sqrt(n))
    device = x.device
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(h, device=device), indexing='ij')
    y_coords, x_coords = y_coords.flatten().float(), x_coords.flatten().float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 4, device=device).float() / d))
    sin_y, cos_y = torch.sin(y_coords.unsqueeze(1) * inv_freq), torch.cos(y_coords.unsqueeze(1) * inv_freq)
    sin_x, cos_x = torch.sin(x_coords.unsqueeze(1) * inv_freq), torch.cos(x_coords.unsqueeze(1) * inv_freq)
    x_rot = x.clone()
    half = d // 2
    x_rot[:, :, :half:2] = x[:, :, :half:2] * cos_y - x[:, :, 1:half:2] * sin_y
    x_rot[:, :, 1:half:2] = x[:, :, :half:2] * sin_y + x[:, :, 1:half:2] * cos_y
    x_rot[:, :, half::2] = x[:, :, half::2] * cos_x - x[:, :, half+1::2] * sin_x
    x_rot[:, :, half+1::2] = x[:, :, half::2] * sin_x + x[:, :, half+1::2] * cos_x
    return x_rot

class AlgebraicTransformerEncoder(nn.Module):
    def __init__(self, k_slots=3, d_model=128, n_basis=200):
        super().__init__()
        self.k_slots = k_slots
        self.d_model = d_model
        self.patch_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True)
            for _ in range(6)
        ])
        self.slot_queries = nn.Parameter(torch.randn(1, k_slots, d_model))
        self.slot_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.coeff_proj = nn.Linear(d_model, n_basis)
        self.pose_proj = nn.Linear(d_model, 16)

    def forward(self, x_mask):
        b, c, h, w = x_mask.shape
        x = x_mask.reshape(b, h*w, 1)
        x = self.patch_proj(x)
        x = apply_2d_rope(x, self.d_model)
        curr = x
        for block in self.blocks:
            curr = block(curr)
        queries = self.slot_queries.expand(b, -1, -1)
        slots, _ = self.slot_attn(queries, curr, curr)
        coeffs = F.softplus(self.coeff_proj(slots))
        poses = self.pose_proj(slots)
        return coeffs, poses

class AlgebraicBasisDecoder(nn.Module):
    def __init__(self, basis_path, k_slots=3):
        super().__init__()
        data = torch.load(basis_path, weights_only=False)
        self.register_buffer('basis', data['basis']) 
        self.n_basis, self.dim_flat = self.basis.shape
        self.k_slots = k_slots

    def forward(self, coeffs, poses):
        reconstructions = torch.matmul(coeffs, self.basis)
        grid_sum = reconstructions.sum(dim=1)
        grid_3d = grid_sum.view(-1, 15, 15, 12)
        logits = grid_3d[:, :, :, :10].permute(0, 3, 1, 2)
        return logits

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    cfg = {
        'k_slots': 3,
        'd_model': 128,
        'n_basis': 200,
        'num_codes': 1024,
        'batch_size': 64,
        'epochs': 300,
        'lr': 3e-4,
        'lambda_cycle': 1.0,
        'basis_path': 'arc_data/arc_basis_nmf_200.pt',
        'library_path': 'arc_data/primitive_library.pt'
    }
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    encoder = AlgebraicTransformerEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model'], n_basis=cfg['n_basis']).to(device)
    vq = NMFInformedVQ(num_codes=cfg['num_codes'], basis_dim=cfg['n_basis']).to(device)
    decoder = AlgebraicBasisDecoder(cfg['basis_path'], k_slots=cfg['k_slots']).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(vq.parameters()), lr=cfg['lr'])
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        vq.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)
            mask = batch['valid_mask'].to(device)
            x_mask = (state > 0).float()
            optimizer.zero_grad()
            coeffs, poses = encoder(x_mask)
            z_q, vq_loss, _ = vq(coeffs)
            logits = decoder(z_q, poses)
            target = state.squeeze(1).long()
            recon_loss = F.cross_entropy(logits, target, reduction='none')
            recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-8)
            recon_mask = (logits.argmax(1).unsqueeze(1) > 0).float()
            rec_coeffs, _ = encoder(recon_mask.detach())
            cycle_loss = F.mse_loss(rec_coeffs, coeffs.detach()) * cfg['lambda_cycle']
            loss = recon_loss + vq_loss + cycle_loss
            loss.backward()
            optimizer.step()
            progress.set_postfix({'Rec': f"{recon_loss.item():.4f}", 'Cyc': f"{cycle_loss.item():.4f}"})
        if epoch % 20 == 0:
            torch.save({'encoder': encoder.state_dict(), 'vq': vq.state_dict(), 'cfg': cfg}, f"checkpoints/nmf_algebraic_e{epoch}.pth")

if __name__ == "__main__":
    train()
