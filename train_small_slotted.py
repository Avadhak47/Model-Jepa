#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for Linux servers
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from pathlib import Path

# Local repo imports
sys.path.insert(0, os.path.abspath('.'))
try:
    from arc_data.rearc_dataset import ReARCDataset
    from analysis.plot_utils import plot_reconstruction_dashboard, plot_slot_masks
except ImportError:
    print("❌ Critical Import Error: Ensure you are running this from the root of the Model-Jepa repository.")
    sys.exit(1)

# ========================================================
# 1. CORE MODEL DEFINITIONS
# ========================================================

class HarmonicSlotEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.embed_dim = config['hidden_dim']
        self.num_slots = config['num_slots']
        self.slot_iters = config['slot_iters']
        self.temperature = config['slot_temperature']
        
        self.patch_embed = nn.Conv2d(config['in_channels'], self.embed_dim, kernel_size=config['patch_size'], stride=config['patch_size'])
        self.patch_pos_embed = nn.Parameter(torch.randn(1, 1024, self.embed_dim))
        
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        self.harmonic_priors = nn.Parameter(self._build_harmonic_priors(self.num_slots, self.embed_dim), requires_grad=False)
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.slot_logsigma = nn.Parameter(torch.ones(1, 1, self.embed_dim) * -2.0)
        
        self.norm_inputs = nn.LayerNorm(self.embed_dim)
        self.norm_slots  = nn.LayerNorm(self.embed_dim)
        self.norm_mlp    = nn.LayerNorm(self.embed_dim)
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.gru = nn.GRUCell(self.embed_dim, self.embed_dim)
        
        self.slot_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        self.to(self.device)

    def _build_harmonic_priors(self, num_slots, dim):
        priors = torch.zeros(1, num_slots, dim)
        position = torch.arange(0, num_slots, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        priors[0, :, 0::2] = torch.sin(position * div_term)
        priors[0, :, 1::2] = torch.cos(position * div_term)
        return priors

    def forward(self, inputs):
        img = inputs["state"].float().to(self.device)
        B = img.shape[0]
        x = self.patch_embed(img)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        N = x.shape[1]
        x = x + self.patch_pos_embed[:, :N, :]
        x = self.patch_mlp(x)
        x = self.norm_inputs(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        slots = self.harmonic_priors.expand(B, -1, -1) + self.slot_mu
        if self.training:
            slots = slots + torch.randn_like(slots) * self.slot_logsigma.exp()
        for _ in range(self.slot_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.q_proj(slots_norm)
            attn_logits = torch.bmm(k, q.transpose(1, 2)) * (self.embed_dim ** -0.5)
            attn = F.softmax(attn_logits / self.temperature, dim=-1)
            attn_norm = attn + 1e-8
            attn_norm = attn_norm / attn_norm.sum(dim=1, keepdim=True)
            updates = torch.bmm(attn_norm.transpose(1, 2), v)
            slots = self.gru(updates.reshape(-1, self.embed_dim), slots_prev.reshape(-1, self.embed_dim)).reshape(B, self.num_slots, self.embed_dim)
            slots = slots + self.slot_mlp(self.norm_mlp(slots))
        masks = attn.transpose(1, 2).view(B, self.num_slots, H, W).detach()
        return {"latent": slots, "masks": masks, "slot_logsigma": self.slot_logsigma}

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.register_buffer("grid", self._build_grid(resolution))

    @staticmethod
    def _build_grid(resolution):
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1)
        grid = torch.cat([grid, 1.0 - grid], dim=-1)
        return grid.unsqueeze(0)

    def forward(self, x):
        pos = self.dense(self.grid).permute(0, 3, 1, 2)
        return x + pos

class HarmonicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.embed_dim = config['hidden_dim']
        self.vocab_size = config['vocab_size']
        self.grid_size = config['grid_size']
        self.decoder_pos = SoftPositionEmbed(self.embed_dim, (self.grid_size, self.grid_size))
        self.spatial_broadcast = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.color_mlp = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, self.vocab_size, 3, padding=1))
        self.alpha_mlp = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 1, 3, padding=1))
        self.to(self.device)

    def forward(self, inputs):
        z = inputs["latent"]
        B, num_slots, D = z.shape
        H, W = self.grid_size, self.grid_size
        z_tiled = z.view(B * num_slots, D, 1, 1).expand(-1, -1, H, W)
        z_tiled = self.decoder_pos(z_tiled)
        decoded_features = self.spatial_broadcast(z_tiled)
        colors = self.color_mlp(decoded_features).view(B, num_slots, self.vocab_size, H, W)
        alphas = self.alpha_mlp(decoded_features).view(B, num_slots, 1, H, W)
        alphas_normalized = F.softmax(alphas, dim=1)
        reconstruction = torch.sum(alphas_normalized * colors, dim=1)
        return {"reconstruction": reconstruction, "alphas": alphas_normalized}

    def loss(self, inputs, outputs):
        target = inputs["state"].to(self.device).long().squeeze()
        if target.dim() == 2: target = target.unsqueeze(0)
        recon_logits = outputs["reconstruction"] 
        ce_loss = F.cross_entropy(recon_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        recon_loss = (((1 - pt) ** 2.0) * ce_loss).mean()
        z = inputs["latent"] 
        z_norm = F.normalize(z, dim=-1)
        sim = torch.bmm(z_norm, z_norm.transpose(1, 2))
        S = sim.size(1)
        eye = torch.eye(S, device=self.device).unsqueeze(0).bool()
        sim_off = sim.masked_fill(eye, 0.0)
        ortho_loss = F.relu(sim_off.abs() - 0.3).mean()
        total_loss = recon_loss + (0.5 * ortho_loss)
        return {"loss": total_loss, "recon_loss": recon_loss, "ortho_loss": ortho_loss}

# ========================================================
# 2. CLI ARGUMENT PARSER
# ========================================================

def get_args():
    parser = argparse.ArgumentParser(description="NS-ARC Slotted Training Utility (Small)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_slots", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="arc_data/re-arc")
    parser.add_argument("--wandb_project", type=str, default="NS-ARC-Scaling")
    parser.add_argument("--run_name", type=str, default="Harmonic-Slot-Small-CLI")
    parser.add_argument("--save_dir", type=str, default="checkpoints/small_slot")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in save_dir")
    return parser.parse_args()

# ========================================================
# 3. MAIN TRAINING LOOP
# ========================================================

def main():
    args = get_args()
    
    # Device setup
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    config = {
        'device': device,
        'in_channels': 1,
        'input_dim': 64,
        'patch_size': 2,
        'hidden_dim': 128,
        'latent_dim': 128,
        'num_slots': args.num_slots,
        'slot_iters': 5,
        'slot_temperature': 0.1,
        'vocab_size': 10,
        'grid_size': 30,
        'focal_gamma': 2.0,
    }
    
    # Initialize models
    encoder = HarmonicSlotEncoder(config)
    decoder = HarmonicDecoder(config)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 1
    
    # Resume logic
    if args.resume:
        latest_enc = os.path.join(args.save_dir, "encoder_latest.pt")
        latest_dec = os.path.join(args.save_dir, "decoder_latest.pt")
        if os.path.exists(latest_enc) and os.path.exists(latest_dec):
            print(f"🔄 Resuming from latent checkpoints in {args.save_dir}")
            encoder.load_state_dict(torch.load(latest_enc, map_location=device))
            decoder.load_state_dict(torch.load(latest_dec, map_location=device))
            # Optional: parse epoch from filename if saved with epoch suffix
    
    # Dataset
    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"⚠️ Warning: Dataset path {data_path} not found. Ensure you are in the correct directory.")
    dataset = ReARCDataset(data_path=data_path)
    
    # W&B
    wb_run = wandb.init(project=args.wandb_project, name=args.run_name, config={**config, **vars(args)})
    
    print(f"🚀 Training starting on {device} for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        decoder.train()
        epoch_losses, epoch_recon, epoch_ortho = [], [], []
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        for _ in pbar:
            batch = dataset.sample(args.batch_size)
            state_batch = batch['state'].to(device)
            
            z_dict = encoder({"state": state_batch})
            out_dict = decoder({"latent": z_dict["latent"], "state": state_batch})
            loss_dict = decoder.loss({"state": state_batch, "latent": z_dict["latent"]}, out_dict)
            
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss_dict["loss"].item())
            epoch_recon.append(loss_dict["recon_loss"].item())
            epoch_ortho.append(loss_dict["ortho_loss"].item())
            pbar.set_postfix(loss=np.mean(epoch_losses))
        
        avg_loss = np.mean(epoch_losses)
        avg_recon = np.mean(epoch_recon)
        avg_ortho = np.mean(epoch_ortho)
        
        # Validation & Logging
        if epoch % 5 == 0 or epoch == 1:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                val_batch = dataset.sample(4)
                val_state = val_batch['state'].to(device)
                z_dict_val = encoder({"state": val_state})
                out_dict_val = decoder({"latent": z_dict_val["latent"], "state": val_state})
                
                # Visuals
                os.makedirs("evaluation_reports/plots", exist_ok=True)
                viz_path = f"evaluation_reports/plots/recon_epoch_{epoch}.png"
                mask_path = f"evaluation_reports/plots/masks_epoch_{epoch}.png"
                
                z_flat = z_dict_val["latent"].view(4 * config['num_slots'], -1)
                recon_grid = out_dict_val["reconstruction"].argmax(dim=1).float()
                
                plot_reconstruction_dashboard(val_state[0, 0].cpu(), recon_grid[0].cpu(), z_flat.cpu(), epoch, viz_path)
                plot_slot_masks(out_dict_val["alphas"], epoch, mask_path)
                
                # Save models
                torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder_latest.pt"))
                torch.save(decoder.state_dict(), os.path.join(args.save_dir, "decoder_latest.pt"))
                
                # W&B Artifacts
                enc_art = wandb.Artifact(f"{args.run_name}_encoder", type="model")
                enc_art.add_file(os.path.join(args.save_dir, "encoder_latest.pt"))
                wb_run.log_artifact(enc_art)
                
                dec_art = wandb.Artifact(f"{args.run_name}_decoder", type="model")
                dec_art.add_file(os.path.join(args.save_dir, "decoder_latest.pt"))
                wb_run.log_artifact(dec_art)
                
                wb_run.log({
                    "Train/Total_Loss": avg_loss,
                    "Train/Recon_Loss": avg_recon,
                    "Train/Ortho_Loss": avg_ortho,
                    "Visuals/Reconstruction": wandb.Image(viz_path),
                    "Visuals/Slot_Masks": wandb.Image(mask_path)
                }, step=epoch)
                
                print(f" ✅ [Epoch {epoch:03d}] Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Ortho: {avg_ortho:.4f}")
        else:
            wb_run.log({
                "Train/Total_Loss": avg_loss, 
                "Train/Recon_Loss": avg_recon, 
                "Train/Ortho_Loss": avg_ortho
            }, step=epoch)

    wb_run.finish()
    print("✨ Training Complete.")

if __name__ == "__main__":
    main()
