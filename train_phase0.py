"""
Full NS-ARC Training Pipeline — Phase 0 + Phase 1
====================================================
Runs the complete pipeline end-to-end on the server without Jupyter:

    python train_phase0.py

All outputs (checkpoints, metrics, plots) are saved to:
    runs/<run_name>_<timestamp>/
    └── phase0/
    │   ├── latest_checkpoint.pth
    │   ├── frozen_vq_codebook.pth
    │   └── metrics_log.jsonl
    └── phase1/
        ├── latest_slot_checkpoint.pth
        ├── metrics_log.jsonl
        └── plots/
            ├── recon_epoch_*.png
            └── masks_epoch_*.png

This directory is .gitignored so nothing leaks into version control.

Stability features:
- NaN guard: skips corrupt batches, exits after 5 consecutive NaN
- LR warmup after surgery: prevents VQ spike gradient explosion
- Soft 25% quantile resurrection: replaces weakest codes
- Padding-masked perplexity and surgery pool
- Graceful SIGINT/SIGTERM: saves checkpoint on Ctrl+C or server kill
- resume_from: pass checkpoint path in CFG to resume a run
- WandB logging with local JSONL fallback
"""

import sys
import os

# Reduce CUDA memory fragmentation (helps when two models share the GPU sequentially)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import signal
import json
import datetime
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arc_data.rearc_dataset import ReARCDataset
from arc_data.arc_dataset import ARCDataset
from modules.encoders import PatchTransformerEncoder, DeepTransformerEncoder
from modules.decoders import PatchDecoder, TransformerDecoder
from modules.vq import FactorizedVectorQuantizer
from modules.semantic_encoders import SemanticSlotEncoder
from modules.semantic_decoders import SemanticDecoder
from analysis.evaluator import run_validation_epoch
from analysis.plot_utils import plot_reconstruction_dashboard, plot_slot_masks

# ── WandB (graceful offline fallback) ──────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not found — metrics logged locally to metrics_log.jsonl only")

# ═══════════════════════════════════════════════════════════════════════════
# SHARED CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CFG = {
    # Run identity
    'run_name':             'FactorizedFPS-v3',
    'wandb_project':        'NS-ARC-Scaling',

    # Architecture
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'in_channels':          10,       # One-Hot encoded ARC colors
    'patch_size':           2,
    'hidden_dim':           128,
    'latent_dim':           128,
    'vocab_size':           10,
    'grid_size':            30,
    'focal_gamma':          2.0,
    'num_slots':            10,
    'slot_iters':           3,
    'slot_temperature':     0.1,

    # Phase 0 training
    'p0_epochs':            399,   # Resume adds 99 more epochs on top of the 300 already done
    'p0_steps':             100,
    'p0_batch':             128,
    'p0_lr':                1e-3,
    'p0_lr_post_surgery':   1e-4,
    'p0_lr_warmup_epochs':  2,
    'p0_grad_clip':         1.0,
    'p0_save_interval':     10,

    # Phase 0 codebook
    'num_shape_codes':      256,
    'num_color_codes':      16,
    'commitment_cost':      0.25,
    'surgery_interval':     25,
    'surgery_quantile':     0.25,

    # Phase 0 early stop
    'p0_stop_perp_shape':   240.0,
    'p0_stop_perp_color':   15.0,
    'p0_stop_recon':        0.05,

    # Phase 1 training
    'p1_epochs':            1000,
    'p1_steps':             200,
    'p1_batch':             32,     # Reduced from 128 — SemanticDecoder spatial-broadcast is VRAM-heavy
    'p1_lr':                1e-4,
    'p1_grad_clip':         1.0,
    'p1_save_interval':     10,
    'p1_val_interval':      10,     # Run validation every N epochs

    # Data
    'data_path':            'arc_data/re-arc',
    'eval_data_path':       'arc_data/re-arc/arc_original/evaluation',
    'val_batch_size':       8,

    # Resume — point at the exact checkpoint files to continue from
    # Phase 0 is DONE (399 epochs). Point at its final checkpoint so the loop
    # detects start_epoch > p0_epochs and skips straight to Phase 1.
    'p0_resume_from': 'runs/FactorizedFPS-v3_2026-04-23_14-40-34/phase0/latest_checkpoint.pth',
    # Phase 1 is fresh
    'p1_resume_from': None,
}

# ═══════════════════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════
_STOP = False

def _handle_signal(sig, frame):
    global _STOP
    print(f"\n🛑 Signal {sig} received — saving checkpoint at end of this epoch.")
    _STOP = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def make_run_dir(cfg: dict) -> tuple[str, str, str]:
    """Creates the full run directory tree and returns (root, p0_dir, p1_dir)."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = os.path.join('runs', f"{cfg['run_name']}_{ts}")
    p0_dir = os.path.join(root, 'phase0')
    p1_dir = os.path.join(root, 'phase1')
    os.makedirs(p0_dir, exist_ok=True)
    os.makedirs(p1_dir, exist_ok=True)
    os.makedirs(os.path.join(p1_dir, 'plots'), exist_ok=True)
    with open(os.path.join(root, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"📂 Run directory: {root}")
    return root, p0_dir, p1_dir


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g['lr'] = lr


def log_metrics(wb_run, metrics: dict, step: int, log_path: str):
    if WANDB_AVAILABLE and wb_run is not None:
        wb_run.log(metrics, step=step)
    with open(log_path, 'a') as f:
        f.write(json.dumps({'step': step, **metrics}) + '\n')


def compute_valid_patch_mask(state: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Bool [B, N] — True for patches containing at least one real (non-zero) pixel."""
    s = state.squeeze(1).long()
    B, H, W = s.shape
    Ph, Pw = H // patch_size, W // patch_size
    patches = s.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return patches.reshape(B, Ph, Pw, patch_size * patch_size).any(dim=-1).reshape(B, Ph * Pw)


def init_wandb(cfg, run_dir, phase_tag, wb_run=None):
    """Initialise or reuse a WandB run. Returns wb_run (may be None on failure)."""
    if not WANDB_AVAILABLE:
        return None
    if wb_run is not None:
        return wb_run          # reuse across phases
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception:
            pass
    try:
        return wandb.init(
            project=cfg['wandb_project'],
            name=f"{cfg['run_name']}-{phase_tag}",
            config=cfg,
            resume='allow',
            dir=run_dir,
        )
    except Exception as e:
        print(f"⚠️  WandB init failed ({e}), proceeding offline.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════
class Phase0Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PatchTransformerEncoder(cfg)
        self.vq = FactorizedVectorQuantizer(
            num_shape_codes=cfg['num_shape_codes'],
            num_color_codes=cfg['num_color_codes'],
            embedding_dim=cfg['latent_dim'],
            commitment_cost=cfg['commitment_cost'],
        )
        self.decoder = PatchDecoder(cfg)
        self.to(cfg['device'])

    def forward(self, inputs, valid_mask=None):
        z = self.encoder(inputs)['latent']
        z_q, vq_loss, p_shape, p_color = self.vq(z, valid_mask=valid_mask)
        out = self.decoder({'latent': z_q})
        return out, vq_loss, p_shape, p_color


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0 — FACTORIZED CODEBOOK PRETRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_phase_0(cfg, p0_dir, wb_run, dataset):
    global _STOP
    device = cfg['device']

    print("\n" + "═"*60)
    print("🚀  PHASE 0: Factorized Codebook Pretraining (SLATE Mode)")
    print("═"*60)

    ckpt_path    = os.path.join(p0_dir, 'latest_checkpoint.pth')
    vq_path      = os.path.join(p0_dir, 'frozen_vq_codebook.pth')
    metrics_path = os.path.join(p0_dir, 'metrics_log.jsonl')

    model = Phase0Autoencoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['p0_lr'])

    start_epoch   = 1
    post_surg_cd  = 0

    # ── Resume ──────────────────────────────────────────────────────────────
    resume = cfg.get('p0_resume_from') or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume and os.path.exists(resume):
        print(f"📥 P0 resuming from {resume}...")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model'], strict=False)
            opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch'] + 1
            print(f"   ✅ Resumed from epoch {ckpt['epoch']}.")
        except Exception as e:
            print(f"   ⚠️  Incompatible checkpoint, starting fresh. ({e})")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg['p0_epochs'] + 1):
        model.train()

        if post_surg_cd > 0:
            set_lr(opt, cfg['p0_lr_post_surgery'])
            post_surg_cd -= 1
            if post_surg_cd == 0:
                set_lr(opt, cfg['p0_lr'])

        ep_loss, ep_vq, ep_ps, ep_pc = [], [], [], []
        nan_streak = 0

        for _ in tqdm(range(cfg['p0_steps']), desc=f"P0 E{epoch:03d}", leave=False):
            batch = dataset.sample(cfg['p0_batch'])
            state = batch['state'].to(device)
            vmask = compute_valid_patch_mask(state, cfg['patch_size']).to(device)

            out, vq_loss, p_shape, p_color = model({'state': state}, valid_mask=vmask)
            loss_dict  = model.decoder.loss({'state': state}, out)
            total_loss = loss_dict['loss'] + vq_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_streak += 1
                opt.zero_grad()
                if nan_streak >= 5:
                    print(f"\n❌ P0: NaN persists for 5 steps at epoch {epoch} — saving & stopping.")
                    _STOP = True
                    break
                continue
            nan_streak = 0

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['p0_grad_clip'])
            opt.step()

            ep_loss.append(loss_dict['loss'].item())
            ep_vq.append(vq_loss.item())
            ep_ps.append(p_shape.item())
            ep_pc.append(p_color.item())

        if not ep_loss:
            if _STOP: break
            continue

        avg_recon   = float(np.mean(ep_loss))
        avg_vq      = float(np.mean(ep_vq))
        avg_p_shape = float(np.mean(ep_ps))
        avg_p_color = float(np.mean(ep_pc))
        cur_lr      = opt.param_groups[0]['lr']

        log_metrics(wb_run, {
            'P0/Recon_Loss':           avg_recon,
            'P0/VQ_Loss':              avg_vq,
            'P0/Perplexity_Shape_256': avg_p_shape,
            'P0/Perplexity_Color_16':  avg_p_color,
            'P0/LR':                   cur_lr,
        }, step=epoch, log_path=metrics_path)

        if epoch % cfg['p0_save_interval'] == 0 or _STOP:
            print(f"P0 E{epoch:03d} | Recon:{avg_recon:.4f} VQ:{avg_vq:.4f} "
                  f"Shape:{avg_p_shape:.1f}/256 Color:{avg_p_color:.1f}/16 LR:{cur_lr:.1e}")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch, 'cfg': cfg}, ckpt_path)

        # Codebook resurrection
        if epoch % cfg['surgery_interval'] == 0:
            q_pct = int(cfg['surgery_quantile'] * 100)
            print(f"⚡ P0 E{epoch}: Resurrection (bottom {q_pct}% by usage)...")
            with torch.no_grad():
                z_raw = model.encoder({'state': state})['latent']
                n_s, n_c = model.vq.resurrect_dead_codes(
                    z_raw, valid_mask=vmask,
                    aggression_quantile=cfg['surgery_quantile'])
            print(f"   Resurrected {n_s} shape codes, {n_c} color codes.")
            set_lr(opt, cfg['p0_lr_post_surgery'])
            post_surg_cd = cfg['p0_lr_warmup_epochs']

        # Early stop
        if (avg_p_shape > cfg['p0_stop_perp_shape']
                and avg_p_color > cfg['p0_stop_perp_color']
                and avg_recon < cfg['p0_stop_recon']):
            print(f"\n✅ P0 Codebook fully baked at epoch {epoch}!")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch, 'cfg': cfg}, ckpt_path)
            break

        if _STOP:
            print(f"\n💾 P0 emergency checkpoint saved (epoch {epoch}).")
            break

    # Save frozen VQ for Phase 1 FPS injection
    # Detach the VQ from the full model before returning so the encoder and decoder
    # weights become unreferenced and can be garbage-collected.
    frozen_vq = model.vq.eval()
    del model          # Free encoder + decoder weights from GPU now, not at GC whim
    torch.save(frozen_vq.state_dict(), vq_path)
    print(f"\n💾 Frozen VQ codebook → {vq_path}")
    return frozen_vq, vq_path, epoch   # ← return final epoch for WandB offset


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — SLOTTED JEPA + BASELINE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def train_phase_1(cfg, p1_dir, wb_run, frozen_vq, train_dataset, eval_dataset, step_offset: int = 0):
    """Phase 1 Slotted JEPA training. step_offset = last Phase 0 epoch so WandB steps are monotonic."""
    global _STOP
    device = cfg['device']

    print("\n" + "═"*60)
    print("🧠  PHASE 1: Slotted JEPA + Baseline Comparison (1000 Epochs)")
    print("═"*60)

    ckpt_path       = os.path.join(p1_dir, 'latest_slot_checkpoint.pth')
    base_ckpt_path  = os.path.join(p1_dir, 'latest_base_checkpoint.pth')
    metrics_path    = os.path.join(p1_dir, 'metrics_log.jsonl')
    plots_dir       = os.path.join(p1_dir, 'plots')

    # ── FPS Initialization: extract semantic slot priors from frozen VQ ─────
    print("🔍 Extracting 10 FPS Semantic Slots from frozen codebook...")
    with torch.no_grad():
        semantic_inits = frozen_vq.get_farthest_point_samples(target_slots=cfg['num_slots'])

    # ── Models ──────────────────────────────────────────────────────────────
    slot_enc  = SemanticSlotEncoder(cfg)
    slot_dec  = SemanticDecoder(cfg)
    slot_enc.inject_semantic_priors(semantic_inits)

    base_enc  = DeepTransformerEncoder(cfg)
    base_dec  = TransformerDecoder(cfg)

    opt_slot  = torch.optim.AdamW(
        list(slot_enc.parameters()) + list(slot_dec.parameters()), lr=cfg['p1_lr'])
    opt_base  = torch.optim.AdamW(
        list(base_enc.parameters()) + list(base_dec.parameters()), lr=cfg['p1_lr'])

    start_epoch = 1

    # ── Resume ──────────────────────────────────────────────────────────────
    resume = cfg.get('p1_resume_from') or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume and os.path.exists(resume):
        print(f"📥 P1 resuming from {resume}...")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        try:
            slot_enc.load_state_dict(ckpt['slot_enc'], strict=False)
            slot_dec.load_state_dict(ckpt['slot_dec'], strict=False)
            opt_slot.load_state_dict(ckpt['opt_slot'])
            start_epoch = ckpt['epoch'] + 1
            print(f"   ✅ Slotted model resumed from epoch {ckpt['epoch']}.")
        except Exception as e:
            print(f"   ⚠️  Slot checkpoint incompatible, starting fresh. ({e})")

    if os.path.exists(base_ckpt_path):
        try:
            b_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
            base_enc.load_state_dict(b_ckpt['base_enc'], strict=False)
            base_dec.load_state_dict(b_ckpt['base_dec'], strict=False)
            opt_base.load_state_dict(b_ckpt['opt_base'])
            print(f"   ✅ Baseline model resumed from epoch {b_ckpt['epoch']}.")
        except Exception:
            pass

    # ── Training loop ────────────────────────────────────────────────────────
    history = {'slot_recon': [], 'base_recon': [], 'slot_val_acc': [], 'base_val_acc': []}

    for epoch in range(start_epoch, cfg['p1_epochs'] + 1):
        slot_enc.train(); slot_dec.train()
        base_enc.train(); base_dec.train()

        ep_stot, ep_srecon, ep_svic, ep_brecon = [], [], [], []
        nan_streak = 0

        for _ in tqdm(range(cfg['p1_steps']), desc=f"P1 E{epoch:03d}", leave=False):
            batch  = train_dataset.sample(cfg['p1_batch'])
            states = batch['state'].to(device)

            # ── Slotted model step ────────────────────────────────────────
            z_slot = slot_enc({'state': states})
            out_slot = slot_dec({'latent': z_slot['latent']})
            loss_slot = slot_dec.loss({'state': states, 'latent': z_slot['latent']}, out_slot)

            s_loss = loss_slot['loss']
            if not (torch.isnan(s_loss) or torch.isinf(s_loss)):
                opt_slot.zero_grad()
                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(slot_enc.parameters()) + list(slot_dec.parameters()),
                    cfg['p1_grad_clip'])
                opt_slot.step()
                ep_stot.append(s_loss.item())
                ep_srecon.append(loss_slot.get('recon_loss', s_loss).item())
                ep_svic.append(loss_slot.get('vic_loss', torch.tensor(0.0)).item())

            # ── Baseline model step ───────────────────────────────────────
            z_base = base_enc({'state': states})
            out_base = base_dec({'latent': z_base['latent']})
            loss_base = base_dec.loss({'state': states}, out_base)

            b_loss = loss_base['loss']
            if not (torch.isnan(b_loss) or torch.isinf(b_loss)):
                opt_base.zero_grad()
                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(base_enc.parameters()) + list(base_dec.parameters()),
                    cfg['p1_grad_clip'])
                opt_base.step()
                ep_brecon.append(b_loss.item())

        if not ep_stot:
            if _STOP: break
            continue

        avg_stot  = float(np.mean(ep_stot))
        avg_srecon = float(np.mean(ep_srecon))
        avg_svic  = float(np.mean(ep_svic))
        avg_brecon = float(np.mean(ep_brecon)) if ep_brecon else 0.0

        history['slot_recon'].append(avg_srecon)
        history['base_recon'].append(avg_brecon)

        # Step offset by 100 to keep Phase 0 and Phase 1 in the same WandB timeline
        log_metrics(wb_run, {
            'P1_Slot/Total':    avg_stot,
            'P1_Slot/Recon':    avg_srecon,
            'P1_Slot/VICReg':   avg_svic,
            'P1_Base/Recon':    avg_brecon,
        }, step=step_offset + epoch, log_path=metrics_path)

        # ── Validation + Visualization Block ─────────────────────────────
        if epoch % cfg['p1_val_interval'] == 0 or epoch == 1:
            slot_enc.eval(); slot_dec.eval()
            base_enc.eval(); base_dec.eval()

            mods_slot = {'encoder': slot_enc, 'decoder': slot_dec, 'config': cfg}
            mods_base = {'encoder': base_enc, 'decoder': base_dec, 'config': cfg}

            s_val_loss, s_val_acc, s_val_perf = run_validation_epoch(
                mods_slot, eval_dataset, phase='ae',
                batch_size=cfg['val_batch_size'], device=device)
            b_val_loss, b_val_acc, b_val_perf = run_validation_epoch(
                mods_base, eval_dataset, phase='ae',
                batch_size=cfg['val_batch_size'], device=device)

            history['slot_val_acc'].append(s_val_acc)
            history['base_val_acc'].append(b_val_acc)

            print(f"\nP1 E{epoch:03d} | Slot Recon:{avg_srecon:.4f} Base Recon:{avg_brecon:.4f}")
            print(f"  → [Slot Val]  Pixel.Acc:{s_val_acc*100:.2f}% | Perfect:{s_val_perf*100:.2f}%")
            print(f"  → [Base Val]  Pixel.Acc:{b_val_acc*100:.2f}% | Perfect:{b_val_perf*100:.2f}%")

            log_metrics(wb_run, {
                'Val_Slot/Pixel_Acc':  s_val_acc * 100,
                'Val_Slot/Perfect':    s_val_perf * 100,
                'Val_Base/Pixel_Acc':  b_val_acc * 100,
                'Val_Base/Perfect':    b_val_perf * 100,
                'Val_Slot/Loss':       s_val_loss,
                'Val_Base/Loss':       b_val_loss,
            }, step=step_offset + epoch, log_path=metrics_path)

            # Visualization: slot reconstruction + masks
            with torch.no_grad():
                val_batch  = eval_dataset.sample(4)
                val_states = val_batch['state'].to(device)
                z_val = slot_enc({'state': val_states})
                out_val = slot_dec({'latent': z_val['latent']})

                recon_logits = out_val.get('reconstruction', out_val.get('reconstructed_logits'))
                recon_grid = recon_logits.argmax(dim=1).float()

                z_flat = z_val['latent'].view(4 * cfg['num_slots'], -1)

                viz_path  = os.path.join(plots_dir, f'recon_epoch_{epoch}.png')
                mask_path = os.path.join(plots_dir, f'masks_epoch_{epoch}.png')

                plot_reconstruction_dashboard(
                    val_states[0, 0].cpu(), recon_grid[0].cpu(),
                    z_flat.cpu(), epoch, viz_path)

                if 'masks' in z_val:
                    masks = z_val['masks'].unsqueeze(2)  # [B, S, 1, H, W]
                    plot_slot_masks(masks.cpu(), epoch, mask_path)
                    if WANDB_AVAILABLE and wb_run is not None:
                        wb_run.log({'Visuals/Slot_Masks': wandb.Image(mask_path)},
                                   step=step_offset + epoch)

                if WANDB_AVAILABLE and wb_run is not None:
                    wb_run.log({'Visuals/Slot_Reconstruction': wandb.Image(viz_path)},
                               step=step_offset + epoch)

            slot_enc.train(); slot_dec.train()
            base_enc.train(); base_dec.train()

        # ── Checkpoint save ───────────────────────────────────────────────
        if epoch % cfg['p1_save_interval'] == 0 or _STOP:
            torch.save({
                'slot_enc': slot_enc.state_dict(),
                'slot_dec': slot_dec.state_dict(),
                'opt_slot': opt_slot.state_dict(),
                'epoch':    epoch,
                'history':  history,
            }, ckpt_path)
            torch.save({
                'base_enc': base_enc.state_dict(),
                'base_dec': base_dec.state_dict(),
                'opt_base': opt_base.state_dict(),
                'epoch':    epoch,
            }, base_ckpt_path)

            # WandB artifact every 100 epochs
            if WANDB_AVAILABLE and wb_run is not None and epoch % 100 == 0:
                art = wandb.Artifact(f'slotted_model_ep{step_offset + epoch}', type='model')
                art.add_file(ckpt_path)
                wb_run.log_artifact(art)

        if _STOP:
            print(f"\n💾 P1 emergency checkpoint saved (epoch {epoch}).")
            break

    print("\n🏁 Phase 1 Training Complete.")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   Metrics:    {metrics_path}")
    return slot_enc, slot_dec


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def check_gpu_memory(min_free_gb: float = 4.0):
    """
    Warn if GPU free memory is below threshold.
    Prints nvidia-smi process table so you know what to kill.
    """
    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb  = free_bytes  / 1024**3
    total_gb = total_bytes / 1024**3
    used_gb  = total_gb - free_gb
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total: {total_gb:.1f} GB | Used: {used_gb:.1f} GB | Free: {free_gb:.1f} GB")
    if free_gb < min_free_gb:
        print(f"\n⚠️  WARNING: Only {free_gb:.1f} GB free — need at least {min_free_gb:.1f} GB.")
        print("   A previous process may still be holding GPU memory.")
        print("   Run the following to identify and kill it:")
        print("     nvidia-smi           # find the PID column")
        print("     kill -9 <PID>        # kill the zombie process")
        print("   Then re-run this script.\n")
        # Still continue — expandable_segments may handle it, or user ignored warning intentionally


def main(cfg: dict):
    device = cfg['device']
    check_gpu_memory(min_free_gb=4.0)
    print(f"🖥️  Device: {device}")


    root_dir, p0_dir, p1_dir = make_run_dir(cfg)

    train_dataset = ReARCDataset(data_path=cfg['data_path'])
    try:
        eval_dataset = ARCDataset(data_path=cfg['eval_data_path'])
    except Exception:
        print("⚠️  Eval dataset not found — using train dataset split for validation.")
        eval_dataset = train_dataset

    # ── WandB: single run for both phases ───────────────────────────────────
    wb_run = init_wandb(cfg, root_dir, 'Phase0+1')

    # ── Phase 0 ─────────────────────────────────────────────────────────────
    frozen_vq, _, p0_last_epoch = train_phase_0(cfg, p0_dir, wb_run, train_dataset)

    if _STOP:
        print("\n🛑 Stop requested during Phase 0 — skipping Phase 1.")
        if wb_run:
            wb_run.finish()
        return

    # ── Free GPU before Phase 1 ──────────────────────────────────────────────
    # Phase0Autoencoder (encoder+vq+decoder) is still alive on GPU.
    # Explicitly delete it and clear the cache so Phase 1 has room.
    # frozen_vq itself is tiny (just the embedding tables).
    import gc
    frozen_vq = frozen_vq.cpu()   # Keep the tables, move them off GPU
    gc.collect()
    torch.cuda.empty_cache()
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"\n🧹 GPU cleared after Phase 0. Free VRAM: {free_mb:.0f} MiB")
    frozen_vq = frozen_vq.to(device)  # Move back for FPS sampling

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    train_phase_1(cfg, p1_dir, wb_run, frozen_vq, train_dataset, eval_dataset,
                  step_offset=p0_last_epoch)  # Phase 1 steps are monotonic after Phase 0

    if wb_run:
        wb_run.finish()
    print("\n✅ Full Pipeline Complete.")


if __name__ == '__main__':
    main(CFG)
