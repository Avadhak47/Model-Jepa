import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.extract_arc_objects import PrimitiveDataset
from modules.vq import StructureAwareDynamicVQ

class ObjectEncoder(nn.Module):
    """Encodes a 15x15 primitive object into a flat latent vector."""
    def __init__(self, in_channels=10, hidden_dim=256, latent_dim=128, pose_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.pose_dim = pose_dim
        
        # Simple ConvNet for 15x15 -> 1x1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=1, padding=0), # 1x1
            nn.ReLU()
        )
        self.proj = nn.Linear(hidden_dim, latent_dim + pose_dim)
        
    def forward(self, x):
        # x: [B, 1, 15, 15]
        x_onehot = F.one_hot(x.long().squeeze(1), num_classes=10) # [B, 15, 15, 10]
        x_onehot = x_onehot.permute(0, 3, 1, 2).float() # [B, 10, 15, 15]
        
        feat = self.net(x_onehot).view(x.shape[0], -1) # [B, hidden_dim]
        latent = self.proj(feat) # [B, latent_dim + pose_dim]
        
        z_vq = latent[:, :self.latent_dim]
        z_pose = latent[:, self.latent_dim:]
        
        # 50% Information Dropout on z_pose during training to prevent Codebook Bypass
        if self.training:
            dropout_mask = (torch.rand(z_pose.shape[0], 1, device=z_pose.device) > 0.5).float()
            z_pose = z_pose * dropout_mask
            
        return {
            'latent_vq': z_vq.unsqueeze(1), # [B, 1, latent_dim]
            'latent_pose': z_pose.unsqueeze(1) # [B, 1, pose_dim]
        }

class ObjectDecoder(nn.Module):
    """Decodes a flat latent vector back into a 15x15 primitive object."""
    def __init__(self, hidden_dim=256, latent_dim=128, pose_dim=16):
        super().__init__()
        in_dim = latent_dim + pose_dim
        self.proj = nn.Linear(in_dim, hidden_dim)
        
        # More robust spatial decoding:
        self.spatial = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4), # 4x4
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 8x8
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(15, 15)), # 15x15
            nn.Conv2d(hidden_dim // 2, 10, kernel_size=3, padding=1)
        )
        
    def forward(self, latent):
        # latent: [B, 1, in_dim]
        feat = self.proj(latent.squeeze(1)).unsqueeze(-1).unsqueeze(-1) # [B, hidden_dim, 1, 1]
        logits = self.spatial(feat) # [B, 10, 15, 15]
        return logits

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ── Configurations ──
    cfg = {
        'latent_dim': 128,
        'pose_dim': 16,  # Reduced capacity!
        'num_shape_codes': 1024,
        'num_color_codes': 16,
        'commitment_cost': 0.25,
        'batch_size': 128,
        'epochs': 100,
        'lr': 1e-3,
        'lambda_affine': 0.5,
        'lambda_attractive': 0.1,
        'novelty_threshold': 2.0,
        'repulsion_weight': 0.1,
        'library_path': 'arc_data/primitive_library.pt',
        'checkpoint_dir': 'checkpoints/'
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    
    # ── Data Loading ──
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} objects from {cfg['library_path']}")
    
    # ── Models ──
    encoder = ObjectEncoder(latent_dim=cfg['latent_dim'], pose_dim=cfg['pose_dim']).to(device)
    vq = StructureAwareDynamicVQ(
        max_shape_codes=cfg['num_shape_codes'],
        max_color_codes=cfg['num_color_codes'],
        embedding_dim=cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
        novelty_threshold=cfg['novelty_threshold'],
        repulsion_weight=cfg['repulsion_weight']
    ).to(device)
    decoder = ObjectDecoder(latent_dim=cfg['latent_dim'], pose_dim=cfg['pose_dim']).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(vq.parameters()) + list(decoder.parameters()), lr=cfg['lr'])
    
    # ── Training Loop ──
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        vq.train()
        decoder.train()
        
        total_recon_loss = 0
        total_vq_loss = 0
        total_rep_loss = 0
        total_attr_loss = 0
        total_equiv_loss = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device) # [B, 1, 15, 15]
            pixel_mask = batch['valid_mask'].to(device) # [B, 15, 15]
            B = state.shape[0]
            
            optimizer.zero_grad()
            
            # 1. Forward Pass
            enc_out = encoder(state)
            z_vq = enc_out['latent_vq'] # [B, 1, D]
            z_pose = enc_out['latent_pose']
            
            # Dynamic Vector Quantization
            z_q, vq_loss, rep_loss, s_idx, c_idx = vq(z_vq)
            
            # Reconstruction
            latent_combined = torch.cat([z_q, z_pose], dim=-1)
            logits = decoder(latent_combined) # [B, 10, 15, 15]
            
            target = state.squeeze(1).long()
            recon_loss = F.cross_entropy(logits, target, reduction='none') # [B, 15, 15]
            recon_loss = (recon_loss * pixel_mask).sum() / (pixel_mask.sum() + 1e-8)
            
            # 2. Structural Contrastive Loss (Attractive Force)
            masks_flat = pixel_mask.view(B, -1).float() # [B, 225]
            intersection = torch.mm(masks_flat, masks_flat.t()) # [B, B]
            area = masks_flat.sum(dim=1) # [B]
            union = area.unsqueeze(0) + area.unsqueeze(1) - intersection
            iou = intersection / (union + 1e-8) # [B, B]
            
            z_shape = z_vq.squeeze(1)[:, :cfg['latent_dim']//2]
            dist_matrix = torch.cdist(z_shape, z_shape, p=2) # [B, B]
            diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
            attr_loss = (iou[diag_mask] * dist_matrix[diag_mask]).mean() * cfg['lambda_attractive']
            
            # 3. D4 Equivariance (Rotation/Flip Invariance)
            state_rot = torch.rot90(state, k=1, dims=[2, 3])
            state_flip = torch.flip(state, dims=[3])
            
            z_vq_rot = encoder(state_rot)['latent_vq']
            z_vq_flip = encoder(state_flip)['latent_vq']
            
            s_dim = cfg['latent_dim'] // 2
            shape_orig = z_vq[:, :, :s_dim]
            equiv_loss = F.mse_loss(z_vq_rot[:, :, :s_dim], shape_orig) + F.mse_loss(z_vq_flip[:, :, :s_dim], shape_orig)
            equiv_loss = equiv_loss * cfg['lambda_affine']
            
            # Total Loss
            loss = recon_loss + vq_loss + rep_loss + attr_loss + equiv_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_rep_loss += rep_loss.item()
            total_attr_loss += attr_loss.item()
            total_equiv_loss += equiv_loss.item()
            
            progress.set_postfix({
                'Recon': f"{recon_loss.item():.3f}",
                'VQ': f"{vq_loss.item():.3f}",
                'Rep': f"{rep_loss.item():.3f}",
                'Attr': f"{attr_loss.item():.3f}"
            })
            
        print(f"Epoch {epoch} | Recon: {total_recon_loss/len(dataloader):.4f} | VQ: {total_vq_loss/len(dataloader):.4f} | Attr: {total_attr_loss/len(dataloader):.4f} | Active Shapes: {vq.active_shape_codes.item()}")
        
        if epoch % 10 == 0:
            ckpt_path = os.path.join(cfg['checkpoint_dir'], f'object_codebook_e{epoch}.pth')
            torch.save({
                'encoder': encoder.state_dict(),
                'vq': vq.state_dict(),
                'decoder': decoder.state_dict(),
                'cfg': cfg
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
