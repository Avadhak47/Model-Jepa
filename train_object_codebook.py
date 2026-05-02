import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.extract_arc_objects import PrimitiveDataset
from modules.vq import SemanticEvolutionaryVQ

class MultiSlotEncoder(nn.Module):
    """Encodes a 15x15 primitive into K distinct slots, strictly splitting shape and color."""
    def __init__(self, k_slots=3, latent_dim=128, pose_dim=16):
        super().__init__()
        self.k_slots = k_slots
        self.half_dim = latent_dim // 2
        self.pose_dim = pose_dim
        
        # Shape branch: completely blind to color
        self.shape_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0), # 1x1
            nn.ReLU()
        )
        self.shape_proj = nn.Linear(256, k_slots * (self.half_dim + pose_dim))
        
        # Color branch: permutation and rotation invariant
        self.color_net = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global distribution, rotation invariant
        )
        self.color_proj = nn.Linear(64, k_slots * self.half_dim)
        
    def forward(self, state):
        # state: [B, 1, 15, 15]
        x_onehot = F.one_hot(state.long().squeeze(1), num_classes=10).permute(0, 3, 1, 2).float()
        x_mask = (state > 0).float() # [B, 1, 15, 15] Binary foreground mask
        
        feat_shape = self.shape_net(x_mask).view(-1, 256)
        slots_shape_raw = self.shape_proj(feat_shape).view(-1, self.k_slots, self.half_dim + self.pose_dim)
        
        slots_shape = slots_shape_raw[:, :, :self.half_dim]
        slots_pose = slots_shape_raw[:, :, self.half_dim:]
        
        feat_color = self.color_net(x_onehot).view(-1, 64)
        slots_color = self.color_proj(feat_color).view(-1, self.k_slots, self.half_dim)
        
        z_vq = torch.cat([slots_shape, slots_color], dim=-1) # [B, K, latent_dim]
        
        # Information Dropout on continuous pose
        if self.training:
            dropout_mask = (torch.rand(slots_pose.shape[0], 1, 1, device=slots_pose.device) > 0.5).float()
            slots_pose = slots_pose * dropout_mask
            
        return {'latent_vq': z_vq, 'latent_pose': slots_pose}

class DeepMultiSlotDecoder(nn.Module):
    """Deep decoder fusing K slots to reconstruct dense grids."""
    def __init__(self, k_slots=3, latent_dim=128, pose_dim=16):
        super().__init__()
        in_dim = (latent_dim + pose_dim) * k_slots
        self.proj = nn.Linear(in_dim, 512)
        
        self.spatial = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4), # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 8x8
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=(15, 15)), # 15x15
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 10, kernel_size=3, padding=1)
        )
        
    def forward(self, z_vq, z_pose):
        combined = torch.cat([z_vq, z_pose], dim=-1).view(z_vq.shape[0], -1) # [B, K * D]
        feat = self.proj(combined).unsqueeze(-1).unsqueeze(-1) # [B, 512, 1, 1]
        logits = self.spatial(feat) # [B, 10, 15, 15]
        return logits

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg = {
        'k_slots': 3,
        'latent_dim': 128,
        'pose_dim': 16,
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
        'anchor_alpha': 0.3, # How strongly the anchor map pulls novel items
        'library_path': 'arc_data/primitive_library.pt',
        'checkpoint_dir': 'checkpoints/'
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} objects")
    
    encoder = MultiSlotEncoder(k_slots=cfg['k_slots'], latent_dim=cfg['latent_dim'], pose_dim=cfg['pose_dim']).to(device)
    vq = SemanticEvolutionaryVQ(
        max_shape_codes=cfg['num_shape_codes'],
        max_color_codes=cfg['num_color_codes'],
        embedding_dim=cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
        novelty_threshold=cfg['novelty_threshold'],
        repulsion_weight=cfg['repulsion_weight'],
        anchor_alpha=cfg['anchor_alpha']
    ).to(device)
    decoder = DeepMultiSlotDecoder(k_slots=cfg['k_slots'], latent_dim=cfg['latent_dim'], pose_dim=cfg['pose_dim']).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(vq.parameters()) + list(decoder.parameters()), lr=cfg['lr'])
    
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        vq.train()
        decoder.train()
        
        # Epoch-Wise Reset (Evolutionary Anchor mapping)
        vq.reset_epoch()
        
        total_recon_loss = 0
        total_vq_loss = 0
        total_rep_loss = 0
        total_equiv_loss = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for batch in progress:
            state = batch['state'].to(device) # [B, 1, 15, 15]
            pixel_mask = batch['valid_mask'].to(device) # [B, 15, 15]
            
            optimizer.zero_grad()
            
            # 1. Forward Pass
            enc_out = encoder(state)
            z_vq = enc_out['latent_vq'] # [B, K, D]
            z_pose = enc_out['latent_pose']
            
            z_q, vq_loss, rep_loss, _, _ = vq(z_vq)
            
            logits = decoder(z_q, z_pose) # [B, 10, 15, 15]
            
            target = state.squeeze(1).long()
            recon_loss = F.cross_entropy(logits, target, reduction='none')
            recon_loss = (recon_loss * pixel_mask).sum() / (pixel_mask.sum() + 1e-8)
            
            # 2. D4 Equivariance (Rotation/Flip Invariance)
            state_rot = torch.rot90(state, k=1, dims=[2, 3])
            state_flip = torch.flip(state, dims=[3])
            
            z_vq_rot = encoder(state_rot)['latent_vq']
            z_vq_flip = encoder(state_flip)['latent_vq']
            
            s_dim = cfg['latent_dim'] // 2
            shape_orig = z_vq[:, :, :s_dim]
            equiv_loss = F.mse_loss(z_vq_rot[:, :, :s_dim], shape_orig) + F.mse_loss(z_vq_flip[:, :, :s_dim], shape_orig)
            equiv_loss = equiv_loss * cfg['lambda_affine']
            
            loss = recon_loss + vq_loss + rep_loss + equiv_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_rep_loss += rep_loss.item()
            total_equiv_loss += equiv_loss.item()
            
            progress.set_postfix({
                'Recon': f"{recon_loss.item():.3f}",
                'VQ': f"{vq_loss.item():.3f}",
                'Rep': f"{rep_loss.item():.3f}",
                'Codes': vq.active_shape_codes.item()
            })
            
        print(f"Epoch {epoch} | Recon: {total_recon_loss/len(dataloader):.4f} | VQ: {total_vq_loss/len(dataloader):.4f} | Active Shapes: {vq.active_shape_codes.item()}")
        
        if epoch % 10 == 0:
            ckpt_path = os.path.join(cfg['checkpoint_dir'], f'object_codebook_e{epoch}.pth')
            torch.save({
                'encoder': encoder.state_dict(),
                'vq': vq.state_dict(),
                'decoder': decoder.state_dict(),
                'cfg': cfg
            }, ckpt_path)

if __name__ == "__main__":
    train()
