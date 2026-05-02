import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from tools.extract_arc_objects import PrimitiveDataset
from modules.vq import EvolutionaryClusterVQ

def apply_2d_rope(x, d_model):
    """Simple 2D Rotary Positional Embedding for a 15x15 grid."""
    b, n, d = x.shape
    h = int(np.sqrt(n))
    device = x.device
    
    # Create grid
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(h, device=device), indexing='ij')
    y_coords = y_coords.flatten().float()
    x_coords = x_coords.flatten().float()
    
    # RoPE frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 4, device=device).float() / d))
    
    # Compute angles
    sin_y = torch.sin(y_coords.unsqueeze(1) * inv_freq)
    cos_y = torch.cos(y_coords.unsqueeze(1) * inv_freq)
    sin_x = torch.sin(x_coords.unsqueeze(1) * inv_freq)
    cos_x = torch.cos(x_coords.unsqueeze(1) * inv_freq)
    
    # Apply rotations (simplified)
    # We rotate different chunks of the embedding with X and Y angles
    x_rot = x.clone()
    half = d // 2
    # Rotate first half with Y
    x_rot[:, :, :half:2] = x[:, :, :half:2] * cos_y - x[:, :, 1:half:2] * sin_y
    x_rot[:, :, 1:half:2] = x[:, :, :half:2] * sin_y + x[:, :, 1:half:2] * cos_y
    # Rotate second half with X
    x_rot[:, :, half::2] = x[:, :, half::2] * cos_x - x[:, :, half+1::2] * sin_x
    x_rot[:, :, half+1::2] = x[:, :, half::2] * sin_x + x[:, :, half+1::2] * cos_x
    
    return x_rot

class TransformerEncoder(nn.Module):
    """Transformer Encoder with RoPE for structural object encoding."""
    def __init__(self, k_slots=3, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.k_slots = k_slots
        self.d_model = d_model
        
        self.patch_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Learned queries to extract K slots
        self.slot_queries = nn.Parameter(torch.randn(1, k_slots, d_model))
        self.slot_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.pose_proj = nn.Linear(d_model, 16) # pose_dim=16

    def forward(self, x_mask):
        # x_mask: [B, 1, 15, 15]
        b, c, h, w = x_mask.shape
        x = x_mask.view(b, h*w, 1)
        x = self.patch_proj(x)
        
        # Apply RoPE
        x = apply_2d_rope(x, self.d_model)
        
        # Residual Multi-Scale Fusion
        features = []
        curr = x
        for i, block in enumerate(self.blocks):
            curr = block(curr)
            if i in [2, 5]: # Capture mid and late features
                features.append(curr)
        
        # Fuse spatial features
        spatial_feat = torch.stack(features).mean(0)
        
        # Extract Slots
        queries = self.slot_queries.expand(b, -1, -1)
        slots, _ = self.slot_attn(queries, spatial_feat, spatial_feat)
        
        # Pose is extracted from the slots
        z_pose = self.pose_proj(slots)
        
        return slots, z_pose

class TransformerDecoder(nn.Module):
    """Transformer Decoder with Cross-Attention for grid reconstruction."""
    def __init__(self, k_slots=3, d_model=128, pose_dim=16):
        super().__init__()
        self.d_model = d_model
        self.k_slots = k_slots
        
        self.input_proj = nn.Linear(d_model + pose_dim, d_model)
        self.target_grid = nn.Parameter(torch.randn(1, 225, d_model)) # 15*15 target tokens
        
        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True)
            for _ in range(4)
        ])
        
        self.head = nn.Linear(d_model, 10) # 10 colors

    def forward(self, z_q, z_pose):
        b = z_q.shape[0]
        # Combine quantized slots and poses
        slots = torch.cat([z_q, z_pose], dim=-1)
        slots = self.input_proj(slots) # [B, K, D]
        
        # Cross-attend target grid to slots
        targets = self.target_grid.expand(b, -1, -1)
        # Apply RoPE to targets so it knows WHERE to decode
        targets = apply_2d_rope(targets, self.d_model)
        
        out, _ = self.cross_attn(targets, slots, slots)
        
        for block in self.blocks:
            out = block(out)
            
        logits = self.head(out) # [B, 225, 10]
        return logits.view(b, 10, 15, 15)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg = {
        'k_slots': 3,
        'd_model': 128,
        'pose_dim': 16,
        'num_codes': 1024,
        'batch_size': 64,
        'epochs': 300,
        'lr': 5e-4,
        'lambda_cycle': 1.0,
        'snapping_gain': 25.0,
        'library_path': 'arc_data/primitive_library.pt',
        'checkpoint_dir': 'checkpoints/'
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    
    encoder = TransformerEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model']).to(device)
    vq = EvolutionaryClusterVQ(num_codes=cfg['num_codes'], embedding_dim=cfg['d_model'], snapping_gain=cfg['snapping_gain']).to(device)
    decoder = TransformerDecoder(k_slots=cfg['k_slots'], d_model=cfg['d_model'], pose_dim=cfg['pose_dim']).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(vq.parameters()) + list(decoder.parameters()), lr=cfg['lr'])
    
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        vq.train()
        decoder.train()
        vq.reset_epoch()
        
        total_recon = 0
        total_vq = 0
        total_cycle = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device)
            mask = batch['valid_mask'].to(device)
            x_mask = (state > 0).float()
            
            optimizer.zero_grad()
            
            # 1. Forward Pass
            slots, z_pose = encoder(x_mask)
            
            # (Color Factorization would be here, simplifying for the structural focus)
            # For strict factorization, we only pass shape to VQ
            half = cfg['d_model'] // 2
            z_shape = slots[:, :, :half]
            # (In a full version, we'd cat color here, but we focus on the Shape/RoPE logic)
            
            z_q, vq_loss, s_idx, _ = vq(slots) # Passing full slots to VQ
            
            logits = decoder(z_q, z_pose)
            
            # Recon Loss
            target = state.squeeze(1).long()
            recon_loss = F.cross_entropy(logits, target, reduction='none')
            recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-8)
            
            # 2. Latent Cycle Consistency
            # Re-encode the reconstruction to see if we get the same latents
            recon_mask = (logits.argmax(1).unsqueeze(1) > 0).float()
            rec_slots, _ = encoder(recon_mask.detach())
            cycle_loss = F.mse_loss(rec_slots, slots.detach()) * cfg['lambda_cycle']
            
            loss = recon_loss + vq_loss + cycle_loss
            loss.backward()
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_cycle += cycle_loss.item()
            
            progress.set_postfix({
                'Rec': f"{recon_loss.item():.3f}",
                'Cyc': f"{cycle_loss.item():.3f}",
                'Codes': vq.active_shape_codes.item()
            })
            
        if epoch % 20 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'vq': vq.state_dict(),
                'decoder': decoder.state_dict(),
                'cfg': cfg
            }, f"checkpoints/object_codebook_v2_e{epoch}.pth")

if __name__ == "__main__":
    train()
