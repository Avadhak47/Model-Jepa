"""
Phase 0 Standalone Training Script — NS-ARC Factorized Codebook Pretraining
============================================================================
Run this directly on the server instead of Jupyter to avoid kernel crashes:

    python train_phase0.py

Features:
- NaN guard: skips corrupt batches instead of crashing the whole run
- LR warmup after surgery: prevents gradient explosion from sudden code teleportation
- Soft 25% quantile resurrection: replaces only weakest codes, not all dead ones
- Padding-masked perplexity and surgery pool: reports true codebook utilization
- Graceful SIGINT/SIGTERM: saves checkpoint before exiting on Ctrl+C or server kill
- Auto-resume from latest_phase0_checkpoint.pth
- WandB logging with offline fallback
"""

import sys
import os
import signal
import json
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arc_data.rearc_dataset import ReARCDataset
from modules.encoders import PatchTransformerEncoder
from modules.decoders import PatchDecoder
from modules.vq import FactorizedVectorQuantizer

# ── WandB (graceful offline fallback) ──────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not found — metrics will only be logged locally to metrics_log.jsonl")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CFG = {
    'device':           'cuda' if torch.cuda.is_available() else 'cpu',
    'in_channels':      10,       # One-Hot encoded ARC color channels
    'patch_size':       2,
    'hidden_dim':       128,
    'latent_dim':       128,
    'vocab_size':       10,
    'grid_size':        30,
    'focal_gamma':      2.0,
    # Training
    'epochs':           300,
    'steps_per_epoch':  100,
    'batch_size':       128,
    'lr':               1e-3,
    'lr_post_surgery':  1e-4,     # Reduced LR for 2 epochs after surgery (prevents spike)
    'lr_warmup_epochs': 2,        # How many epochs to hold post-surgery LR
    'grad_clip':        1.0,
    # Codebook
    'num_shape_codes':  256,
    'num_color_codes':  16,
    'commitment_cost':  0.25,
    # Surgery schedule
    'surgery_interval': 25,       # Every N epochs
    'surgery_quantile': 0.25,     # Replace bottom 25% by usage (not all dead)
    # Checkpointing
    'checkpoint_path':  'latest_phase0_checkpoint.pth',
    'save_interval':    10,
    # WandB
    'wandb_project':    'NS-ARC-Scaling',
    'wandb_run_name':   'Phase0-Stable-v3',
    # Data
    'data_path':        'arc_data/re-arc',
    # Early stop thresholds
    'stop_perp_shape':  240.0,
    'stop_perp_color':  15.0,
    'stop_recon':       0.05,
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════
class Phase0Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PatchTransformerEncoder(cfg)
        self.vq      = FactorizedVectorQuantizer(
            num_shape_codes=cfg['num_shape_codes'],
            num_color_codes=cfg['num_color_codes'],
            embedding_dim=cfg['latent_dim'],
            commitment_cost=cfg['commitment_cost'],
        )
        self.decoder = PatchDecoder(cfg)
        self.to(cfg['device'])

    def forward(self, inputs, valid_mask=None):
        z             = self.encoder(inputs)['latent']
        z_q, vq_loss, p_shape, p_color = self.vq(z, valid_mask=valid_mask)
        out           = self.decoder({'latent': z_q})
        return out, vq_loss, p_shape, p_color

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def compute_valid_patch_mask(state: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Returns bool [B, N] — True for patches that contain at least one real (non-zero) pixel.
    Used to exclude zero-padding from perplexity calculation and surgery pool.
    """
    s = state.squeeze(1).long()          # [B, H, W]
    B, H, W = s.shape
    Ph, Pw = H // patch_size, W // patch_size
    patches = s.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches_flat = patches.reshape(B, Ph, Pw, patch_size * patch_size)
    valid = patches_flat.any(dim=-1)     # [B, Ph, Pw]
    return valid.reshape(B, Ph * Pw)     # [B, N]


def set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g['lr'] = lr


def log_metrics(wb_run, metrics: dict, step: int):
    """Log to WandB if available, always append to local JSONL."""
    if WANDB_AVAILABLE and wb_run is not None:
        wb_run.log(metrics, step=step)
    with open('metrics_log.jsonl', 'a') as f:
        f.write(json.dumps({'step': step, **metrics}) + '\n')


# ═══════════════════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════
_STOP_REQUESTED = False

def _signal_handler(sig, frame):
    global _STOP_REQUESTED
    print(f"\n🛑 Signal {sig} received — will save checkpoint and exit after this epoch.")
    _STOP_REQUESTED = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def train(cfg: dict):
    global _STOP_REQUESTED

    device = cfg['device']
    print(f"🖥️  Device: {device}")
    print(f"🔧 Config: epochs={cfg['epochs']}, batch={cfg['batch_size']}, "
          f"surgery every {cfg['surgery_interval']} epochs\n")

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = ReARCDataset(data_path=cfg['data_path'])

    # ── Model & Optimiser ───────────────────────────────────────────────────
    model = Phase0Autoencoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])

    start_epoch       = 1
    post_surgery_cd   = 0   # Countdown for post-surgery LR warmup

    # ── Resume checkpoint ───────────────────────────────────────────────────
    if os.path.exists(cfg['checkpoint_path']):
        print(f"📥 Resuming from {cfg['checkpoint_path']}...")
        ckpt = torch.load(cfg['checkpoint_path'], map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model'], strict=False)
            opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch'] + 1
            print(f"✅ Resumed from epoch {ckpt['epoch']}.")
        except Exception as e:
            print(f"⚠️  Checkpoint incompatible, starting fresh.\n   Reason: {e}")

    # ── WandB Init ──────────────────────────────────────────────────────────
    wb_run = None
    if WANDB_AVAILABLE:
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        try:
            wb_run = wandb.init(
                project=cfg['wandb_project'],
                name=cfg['wandb_run_name'],
                config=cfg,
                resume='allow',
            )
        except Exception as e:
            print(f"⚠️  WandB init failed ({e}), proceeding offline.")

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()

        # Apply post-surgery LR warmup
        if post_surgery_cd > 0:
            set_lr(opt, cfg['lr_post_surgery'])
            post_surgery_cd -= 1
            if post_surgery_cd == 0:
                set_lr(opt, cfg['lr'])  # Restore normal LR
                print(f"   [Epoch {epoch}] LR restored to {cfg['lr']:.2e}")
        
        epoch_losses, vq_losses, p_shapes, p_colors = [], [], [], []
        nan_batches = 0

        for step in tqdm(range(cfg['steps_per_epoch']), desc=f"Epoch {epoch:03d}", leave=False):
            batch  = dataset.sample(cfg['batch_size'])
            state  = batch['state'].to(device)

            valid_mask = compute_valid_patch_mask(state, cfg['patch_size']).to(device)

            out, vq_loss, p_shape, p_color = model({'state': state}, valid_mask=valid_mask)
            loss_dict  = model.decoder.loss({'state': state}, out)
            total_loss = loss_dict['loss'] + vq_loss

            # ── NaN Guard — skip corrupt batches ──────────────────────────
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_batches += 1
                opt.zero_grad()
                if nan_batches >= 5:
                    print(f"\n❌ NaN/Inf persists for 5 consecutive steps at epoch {epoch}. "
                          "Saving checkpoint and exiting.")
                    _STOP_REQUESTED = True
                    break
                continue
            else:
                nan_batches = 0  # Reset consecutive counter on healthy batch

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()

            epoch_losses.append(loss_dict['loss'].item())
            vq_losses.append(vq_loss.item())
            p_shapes.append(p_shape.item())
            p_colors.append(p_color.item())

        # Skip logging if no valid steps completed
        if not epoch_losses:
            print(f"⚠️  Epoch {epoch} produced no valid steps — skipping metrics.")
        else:
            avg_recon    = float(np.mean(epoch_losses))
            avg_vq       = float(np.mean(vq_losses))
            avg_p_shape  = float(np.mean(p_shapes))
            avg_p_color  = float(np.mean(p_colors))
            current_lr   = opt.param_groups[0]['lr']

            log_metrics(wb_run, {
                'P0/Recon_Loss':            avg_recon,
                'P0/VQ_Loss':               avg_vq,
                'P0/Perplexity_Shape_256':  avg_p_shape,
                'P0/Perplexity_Color_16':   avg_p_color,
                'P0/LR':                    current_lr,
            }, step=epoch)

            if epoch % cfg['save_interval'] == 0 or _STOP_REQUESTED:
                print(f"Epoch {epoch:03d} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} "
                      f"| Shape: {avg_p_shape:.1f}/256 | Color: {avg_p_color:.1f}/16 "
                      f"| LR: {current_lr:.2e}")

            # ── Codebook Resurrection (every surgery_interval epochs) ──────
            if epoch % cfg['surgery_interval'] == 0:
                print(f"⚡ [Epoch {epoch}] Codebook Resurrection (bottom {int(cfg['surgery_quantile']*100)}% by usage)...")
                with torch.no_grad():
                    z_raw = model.encoder({'state': state})['latent']
                    n_s, n_c = model.vq.resurrect_dead_codes(
                        z_raw,
                        valid_mask=valid_mask,
                        aggression_quantile=cfg['surgery_quantile'],
                    )
                print(f"   Resurrected {n_s} shape codes, {n_c} color codes.")
                # LR warmup: hold reduced LR for warmup_epochs after surgery
                set_lr(opt, cfg['lr_post_surgery'])
                post_surgery_cd = cfg['lr_warmup_epochs']

            # ── Checkpoint Save ────────────────────────────────────────────
            if epoch % cfg['save_interval'] == 0 or _STOP_REQUESTED:
                torch.save({
                    'model': model.state_dict(),
                    'opt':   opt.state_dict(),
                    'epoch': epoch,
                    'cfg':   cfg,
                }, cfg['checkpoint_path'])

            # ── Early Stop ─────────────────────────────────────────────────
            if (avg_p_shape > cfg['stop_perp_shape']
                    and avg_p_color > cfg['stop_perp_color']
                    and avg_recon   < cfg['stop_recon']):
                print(f"\n✅ Codebook fully baked at epoch {epoch}! "
                      f"Shape={avg_p_shape:.1f} Color={avg_p_color:.1f} Recon={avg_recon:.4f}")
                break

        # ── Graceful Exit ──────────────────────────────────────────────────
        if _STOP_REQUESTED:
            print(f"\n💾 Emergency checkpoint saved at epoch {epoch}. Exiting cleanly.")
            break

    if wb_run:
        wb_run.finish()

    print("\n🏁 Phase 0 Training Complete.")
    print(f"   Final checkpoint: {cfg['checkpoint_path']}")
    return model.vq


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    frozen_vq = train(CFG)
    # Save the frozen VQ separately for Phase 1 FPS injection
    torch.save(frozen_vq.state_dict(), 'frozen_vq_codebook.pth')
    print("💾 Frozen VQ codebook saved to frozen_vq_codebook.pth")
