import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.extract_arc_objects import PrimitiveDataset
from modules.vq import FactorizedVectorQuantizer

class ObjectEncoder(nn.Module):
    """Encodes a 15x15 primitive object into a flat latent vector."""
    def __init__(self, in_channels=10, hidden_dim=256, latent_dim=256, pose_dim=64):
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
        # One-hot encode the ARC colors (10 classes)
        x_onehot = F.one_hot(x.long().squeeze(1), num_classes=10) # [B, 15, 15, 10]
        x_onehot = x_onehot.permute(0, 3, 1, 2).float() # [B, 10, 15, 15]
        
        feat = self.net(x_onehot).view(x.shape[0], -1) # [B, hidden_dim]
        latent = self.proj(feat) # [B, latent_dim + pose_dim]
        
        return {
            'latent_vq': latent[:, :self.latent_dim].unsqueeze(1), # [B, 1, latent_dim]
            'latent_pose': latent[:, self.latent_dim:].unsqueeze(1) # [B, 1, pose_dim]
        }

class ObjectDecoder(nn.Module):
    """Decodes a flat latent vector back into a 15x15 primitive object."""
    def __init__(self, hidden_dim=256, latent_dim=256, pose_dim=64):
        super().__init__()
        in_dim = latent_dim + pose_dim
        self.proj = nn.Linear(in_dim, hidden_dim)
        
        # Transposed ConvNet for 1x1 -> 15x15
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=1, padding=0), # 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=0), # 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14 -> output_padding=1 makes it 14? Wait.
            # 7 -> (7-1)*2 - 2 + 3 + 1 = 12 + 1 + 1 = 14. Let's use Upsample + Conv instead.
        )
        
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


def entropy(counts):
    """Calculates Shannon Entropy for a count distribution."""
    p = counts / (counts.sum() + 1e-8)
    p = p[p > 0]
    return -(p * torch.log(p)).sum()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ── Configurations ──
    cfg = {
        'latent_dim': 128,
        'pose_dim': 64,
        'num_shape_codes': 1024,
        'num_color_codes': 16,
        'commitment_cost': 0.25,
        'batch_size': 128,
        'epochs': 100,
        'lr': 1e-3,
        'lambda_entropy': 0.5,
        'lambda_affine': 0.5,
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
    vq = FactorizedVectorQuantizer(
        num_shape_codes=cfg['num_shape_codes'],
        num_color_codes=cfg['num_color_codes'],
        embedding_dim=cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost']
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
        total_entropy_loss = 0
        total_equiv_loss = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device) # [B, 1, 15, 15]
            valid_mask = batch['valid_mask'].to(device).view(-1, 1) # [B, 1] - since it's a single object representation
            
            optimizer.zero_grad()
            
            # 1. Forward Pass
            enc_out = encoder(state)
            z_vq = enc_out['latent_vq'] # [B, 1, D]
            z_pose = enc_out['latent_pose']
            
            # The VQ layer expects [B, N, C]. We have N=1. 
            # We bypass the spatial valid mask logic internally by passing None since we want to quantize the global descriptor.
            z_q, vq_loss, p_s, p_c, z_flat, s_idx, c_idx = vq(z_vq, valid_mask=None)
            
            # Reconstruction
            latent_combined = torch.cat([z_q, z_pose], dim=-1)
            logits = decoder(latent_combined) # [B, 10, 15, 15]
            
            # Only calculate recon loss on the valid pixels of the padded object
            target = state.squeeze(1).long()
            recon_loss = F.cross_entropy(logits, target, reduction='none') # [B, 15, 15]
            # Average over valid mask
            pixel_mask = batch['valid_mask'].to(device) # [B, 15, 15]
            recon_loss = (recon_loss * pixel_mask).sum() / (pixel_mask.sum() + 1e-8)
            
            # 2. Shannon Entropy Loss (Codebook Usage)
            h_s = entropy(vq.shape_usage)
            h_c = entropy(vq.color_usage)
            entropy_loss = - (h_s + h_c) * cfg['lambda_entropy']
            
            # 3. D4 Equivariance (Rotation/Flip Invariance)
            # Rotate by 90 degrees and flip
            state_rot = torch.rot90(state, k=1, dims=[2, 3])
            state_flip = torch.flip(state, dims=[3])
            
            z_vq_rot = encoder(state_rot)['latent_vq']
            z_vq_flip = encoder(state_flip)['latent_vq']
            
            # The shape encoding (first half) should be invariant
            s_dim = cfg['latent_dim'] // 2
            shape_orig = z_vq[:, :, :s_dim]
            equiv_loss = F.mse_loss(z_vq_rot[:, :, :s_dim], shape_orig) + F.mse_loss(z_vq_flip[:, :, :s_dim], shape_orig)
            equiv_loss = equiv_loss * cfg['lambda_affine']
            
            # Total Loss
            loss = recon_loss + vq_loss + entropy_loss + equiv_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_equiv_loss += equiv_loss.item()
            
            progress.set_postfix({
                'Recon': f"{recon_loss.item():.3f}",
                'VQ': f"{vq_loss.item():.3f}",
                'Equiv': f"{equiv_loss.item():.3f}"
            })
            
        print(f"Epoch {epoch} | Recon: {total_recon_loss/len(dataloader):.4f} | VQ: {total_vq_loss/len(dataloader):.4f} | Equiv: {total_equiv_loss/len(dataloader):.4f} | Codes Used: {(vq.shape_usage>0).sum().item()}")
        
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
