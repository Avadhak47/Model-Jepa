"""
NS-ARC Full Training Pipeline — Object-Level Codebook Edition
=============================================================
Trains a two-phase pipeline to build an object-level discrete codebook
for JEPA-based reasoning over ARC-AGI grids.

PHASE 0 — Patch Alphabet
  Trains a FactorizedVQ codebook on 5×5 patches using:
  - Pixel reconstruction loss (CrossEntropy)
  - Rotation-invariance contrastive loss (InfoNCE)
  - VICReg variance/covariance anti-collapse
  - Connected-component affinity loss (same object → same code)
  - Semantic partitioning: BG codes 0..N_BG, FG codes N_BG..N

PHASE 1 — Object Dictionary
  Trains a Slot Attention model on top of Phase 0 codebook using:
  - L1: Pixel reconstruction  (CrossEntropy, Focal-weighted)
  - L2: Slot sharpness        (entropy of alpha per pixel → low)
  - L3: Slot coverage         (slot must cover full connected component)
  - L4: Slot diversity        (VICReg across slots within image)
  - L5: VQ commitment         (slot shape → codebook + codebook → slot)
  - L6: Codebook freeze warmup (first N epochs: codebook frozen)

EXPANDABLE CODEBOOK:
  The codebook supports adding new codes at any checkpoint via
  `expand_codebook(checkpoint_path, new_size)` without retraining.
  Uses Residual VQ (multiple stacked codebooks) for hierarchical
  object descriptions at different abstraction levels.

References:
  - SLATE (Singh et al., 2022): dVAE + Slot Attention
  - DINOSAUR (Seitzer et al., 2023): DINO features + Slot Attention
  - VQ-VAE-2 (Razavi et al., 2019): Hierarchical VQ, EMA codebook
  - VICReg (Bardes et al., 2022): Variance-Invariance-Covariance Reg
  - MESH (ICML 2023): Sinkhorn entropy for slot sharpness

Usage:
  python train_phase0.py                        # fresh run
  python train_phase0.py --resume-run runs/ObjectCodebook-v1_2026-04-26_14-13-43
  python train_phase0.py --resume-p0 runs/.../phase0/latest_checkpoint.pth
  python train_phase0.py --resume-p1 runs/.../phase1/latest_slot_checkpoint.pth
  python train_phase0.py --skip-p0  --resume-p1 runs/.../phase1/latest_slot_checkpoint.pth
  python train_phase0.py --p0-epochs 0 --p1-epochs 500   # override epoch counts
All outputs saved to runs/<run_name>_<timestamp>/
"""

import sys, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import signal, json, datetime, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not found — logging locally only")

# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH-BACKED CONFIGURATION
# Literature: SLATE, DINOSAUR, VQ-VAE-2, VICReg
# ═══════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Run Identity ─────────────────────────────────────────────────────
    'run_name':             'ObjectCodebook-Tiny-v1',
    'wandb_project':        'NS-ARC-Scaling',

    # ── Architecture ─────────────────────────────────────────────────────
    # Literature ref: SLATE uses 64D codebook, VQ-VAE-2 uses 128D.
    # For Tiny ARC research: 32D forces extreme compression/logic.
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'in_channels':          10,           # One-Hot ARC colors
    'patch_size':           5,            # 30/5 = 6×6 = 36 patches per grid
    'hidden_dim':           32,           # Tiny model (per user request)
    'latent_dim':           32,           # Matches hidden_dim
    'pose_dim':             16,           # Scaled pose dim
    'vocab_size':           10,           # ARC has 10 colors (0-9)
    'grid_size':            30,

    # ── Phase 0 Codebook ─────────────────────────────────────────────────
    # Literature: SLATE uses 4096-8192 codes for natural images.
    # ARC has much simpler structure → 1024 codes gives good coverage.
    # N_BG_CODES (16) are reserved for background by semantic partitioning.
    'num_shape_codes':      1024,         # Total shape codes (BG: 0-15, FG: 16-1023)
    'num_color_codes':      32,           # More color codes for multi-color patterns
    'commitment_cost':      0.25,         # Standard VQ-VAE value (Razavi 2019)
    'ema_decay':            0.99,         # EMA decay for codebook update (VQ-VAE-2)

    # Phase 0 surgery (resurrection of dead codes)
    'surgery_interval':     50,           # Every N epochs resurrect dead codes
    'surgery_quantile':     0.10,         # Replace bottom 10% by usage

    # ── Phase 0 Training ─────────────────────────────────────────────────
    # Literature: SLATE trains 100k-500k steps. ARC is simpler → 200 epochs × 200 steps.
    'p0_epochs':            1000,
    'p0_steps':             200,          # Steps per epoch
    'p0_batch':             128,          # Batch size (SLATE: 64-128)
    'p0_lr':                4e-4,         # Adam learning rate (SLATE: 3e-4 to 1e-3)
    'p0_lr_post_surgery':   1e-5,         # LR after codebook surgery
    'p0_lr_warmup_epochs':  3,            # Epochs at reduced LR after surgery
    'p0_grad_clip':         2.0,          # Gradient clip (conservative for VQ)
    'p0_save_interval':     50,

    # Phase 0 loss weights (tuned from VICReg paper + SLATE)
    'p0_inv_weight':        10.0,         # Rotation invariance (InfoNCE-style)
    'p0_vicreg_weight':     0.1,          # VICReg anti-collapse
    'p0_affinity_weight':   0.5,          # CC affinity (same object → same code)
    'p0_affinity_interval': 5,            # Compute affinity every N steps (scipy is slow)

    # Phase 0 early stop thresholds
    'p0_stop_perp_shape':   900.0,        # Stop if shape perplexity > 90% of 1024
    'p0_stop_perp_color':   28.0,         # Stop if color perplexity > 87% of 32
    'p0_stop_recon':        0.05,

    # ── Slot Attention Architecture ───────────────────────────────────────
    # Literature: SLATE uses 7 slots for CLEVR, DINOSAUR uses 11 for COCO.
    # ARC grids have 1-8 distinct objects typically → 12 slots gives headroom.
    'num_slots':            12,           # Slightly fewer than before (more focused)
    'slot_iters':           7,            # Slot attention iterations (standard: 3-7)

    # Temperature schedule (linear anneal, Locatello et al. 2020)
    'slot_temp_start':      1.0,          # High temp = exploratory (soft masks)
    'slot_temp_end':        0.05,          # Low temp = sharp (hard object masks)
    'slot_temp_anneal':     400,          # Epochs over which to anneal
    'slot_temperature':     0.05,          # Fallback if annealing disabled

    # ── Phase 1 Training ─────────────────────────────────────────────────
    # Literature: Object-centric models typically need 300-500k steps.
    # ARC is simpler but we need discrete codebook convergence.
    'p1_epochs':            4000,
    'p1_steps':             200,          # Steps per epoch
    'p1_batch':             128,
    'p1_lr':                1e-4,         # Lower LR than P0 (fine-tuning codebook)
    'p1_grad_clip':         1.0,
    'p1_save_interval':     50,
    'p1_val_interval':      25,

    # ── Phase 1 Loss Weights (from derivation in architecture decisions) ──
    # L1: Reconstruction (primary objective)
    'p1_recon_weight':      1.0,
    'p1_focal_gamma':       2.0,          # Focal loss gamma (foreground weighting)
    'p1_fg_weight':         50.0,         # Foreground pixel weight multiplier
    # L2: Slot sharpness (entropy of alpha at each pixel)
    'p1_sharp_weight':      0.5,
    # L3: Slot coverage (slot must cover full connected component)
    'p1_cover_weight':      0.3,
    'p1_cover_interval':    10,           # Every N steps (scipy CC labeling is slow)
    # L4: Slot diversity (VICReg across K slots within each image)
    'p1_vicreg_std':        25.0,         # VICReg std coefficient
    'p1_vicreg_cov':        1.0,          # VICReg cov coefficient
    'p1_vicreg_weight':     0.1,
    # L5: VQ commitment loss (slot ↔ codebook alignment)
    'p1_commit_weight':     0.25,
    # L6: Codebook warmup freeze
    'codebook_warmup_epochs': 200,        # Epochs codebook is frozen (SLATE warmup)
    'codebook_finetune_lr':   5e-6,       # Very low LR when codebook unfrozen
    
    # NEW: Factorized Diversity Patch weights
    'lambda_entropy':       0.15,         # Diversity: maximizes codebook usage
    'lambda_affine':        0.4,          # Factorization: forces rotation invariance
    'p1_resurrect_interval': 10,          # Epochs between dead-code resurrection
    'p1_resurrect_thresh':   100,         # Minimum hits in an epoch to stay alive

    # ── Data Paths ───────────────────────────────────────────────────────
    'data_path':            'arc_data/re-arc/tasks',
    'arc_heavy_path':       'arc_data/arc-heavy/training',
    'eval_data_path':       'arc_data/original/evaluation',
    'val_batch_size':       128,

    # ── Resume ────────────────────────────────────────────────────────────
    'p0_resume_from':       None,
    'p1_resume_from':       None,
    'resume_run_dir':       None,
}

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL HANDLER
# ═══════════════════════════════════════════════════════════════════════════
_STOP = False

def _handle_signal(sig, frame):
    global _STOP
    print(f"\n🛑 Signal {sig} — saving checkpoint at end of current epoch.")
    _STOP = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def make_run_dir(cfg):
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


def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


def log_metrics(wb_run, metrics: dict, step: int, log_path: str):
    if wb_run:
        try:
            wb_run.log(metrics, step=step)
        except Exception:
            pass
    row = {'step': step, **{k: float(v) for k, v in metrics.items()}}
    with open(log_path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def init_wandb(cfg, run_dir, suffix):
    if not WANDB_AVAILABLE:
        return None
    try:
        import wandb as wb
        run = wb.init(
            project=cfg['wandb_project'],
            name=f"{cfg['run_name']}-{suffix}",
            config=cfg,
            dir=run_dir,
            resume='allow',
        )
        return run
    except Exception as e:
        print(f"⚠️  WandB init failed ({e}). Logging locally.")
        return None


def compute_valid_patch_mask(state: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Returns bool mask [B, N] — True if any pixel in the patch is non-zero."""
    B, _, H, W = state.shape
    p = patch_size
    n = (H // p) * (W // p)
    patches = state.unfold(2, p, p).unfold(3, p, p)           # [B,1,h,w,p,p]
    patches = patches.reshape(B, 1, -1, p * p)                 # [B,1,N,p²]
    return (patches.squeeze(1).abs().sum(-1) > 0)              # [B, N]

# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def vicreg_loss(z, std_coeff=25.0, cov_coeff=1.0):
    """VICReg applied across batch dimension. z: [N, D]"""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / (z.shape[0] - 1)
    n = cov.shape[0]
    cov_loss = cov.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().pow(2).sum() / n
    return std_coeff * std_loss + cov_coeff * cov_loss


def vicreg_loss_slots(slots, std_coeff=25.0, cov_coeff=1.0):
    """
    VICReg applied ACROSS SLOTS within each image — forces slot diversity.
    slots: [B, K, D]
    Each image independently: K slot vectors should be diverse.
    """
    B, K, D = slots.shape
    total = torch.tensor(0.0, device=slots.device)
    for b in range(B):
        z = slots[b]  # [K, D] — K slots for one image
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        std_loss = F.relu(1.0 - std).mean()
        cov = (z.T @ z) / max(K - 1, 1)
        cov_loss = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten().pow(2).sum() / D
        total = total + std_coeff * std_loss + cov_coeff * cov_loss
    return total / B


def slot_sharpness_loss(alphas):
    """
    L2: Encourage sharp per-pixel slot ownership.
    alphas: [B, K, H, W] — softmax'd alpha from decoder.
    Low entropy at each pixel = one slot clearly wins.
    """
    # alphas are already softmax'd over K. Compute entropy across slot dim.
    eps = 1e-8
    entropy = -(alphas * (alphas + eps).log()).sum(dim=1)  # [B, H, W]
    return entropy.mean()


def slot_coverage_loss(alphas, state, patch_size):
    """
    L3: Slot must cover the FULL connected component of its "winning" object.
    alphas: [B, K, H, W] — slot alpha (softmax over K).
    state:  [B, 1, H, W] — original grid (int 0-9).
    For each connected component: the dominant slot should have alpha≈1 over all CC pixels.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return torch.tensor(0.0, device=alphas.device)

    B, K, H, W = alphas.shape
    device = alphas.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    # Work with dominant slot per pixel [B, H, W]
    dominant = alphas.argmax(dim=1).cpu().numpy()  # [B, H, W]
    alpha_np  = alphas.detach().cpu().numpy()       # [B, K, H, W]
    grid_np   = state[:, 0].cpu().numpy().astype(int)  # [B, H, W]

    for b in range(B):
        label_map = np.zeros((H, W), dtype=int)
        next_label = 1
        for color in range(1, 10):
            mask = (grid_np[b] == color)
            if not mask.any():
                continue
            labeled, n_comp = ndimage.label(mask)
            label_map[mask] = labeled[mask] + next_label - 1
            next_label += n_comp

        if next_label == 1:
            continue  # all background, skip

        for label_id in range(1, next_label):
            cc_mask = (label_map == label_id)  # [H, W]
            if cc_mask.sum() < 2:
                continue

            # Find dominant slot for this CC (by total alpha mass)
            slot_mass = alpha_np[b, :, cc_mask].sum(axis=1)  # [K]
            dom_slot  = slot_mass.argmax()

            # Penalty: alpha of dominant slot at CC pixels should be ≈1
            # Use a differentiable path through the original `alphas` tensor
            cc_indices = torch.from_numpy(cc_mask).to(device)
            # coverage_loss = mean((1 - alpha[dom_slot, CC_pixels])^2)
            coverage = alphas[b, dom_slot][cc_indices]  # [n_cc_pixels]
            loss_b = ((1.0 - coverage) ** 2).mean()
            total_loss = total_loss + loss_b
            count += 1

    return total_loss / max(count, 1)


def slot_vq_commit_loss(slot_shape, codebook_embedding, beta=0.25):
    """
    L5: VQ Commitment loss for slot quantization.
    slot_shape:          [B, K, 128] — projected slot shape part
    codebook_embedding:  [B, K, 128] — nearest codebook entry (after STE)
    beta:                codebook update weight (Razavi 2019: 0.25)
    """
    # Commitment: push slot toward code
    commit = F.mse_loss(slot_shape, codebook_embedding.detach())
    # Codebook update: pull code toward slot
    cb_update = F.mse_loss(codebook_embedding, slot_shape.detach())
    return commit + beta * cb_update


# ═══════════════════════════════════════════════════════════════════════════
# EXPANDABLE CODEBOOK UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def expand_codebook(checkpoint_path: str, new_num_shape_codes: int,
                    new_num_color_codes: int = None, out_path: str = None):
    """
    Expands a trained VQ codebook checkpoint to a larger size.
    New code slots are initialized with EMA noise around existing code centroids,
    seeding them close to real data distribution.

    Args:
        checkpoint_path: Path to a frozen_vq_codebook.pth
        new_num_shape_codes: Target codebook size (must be >= current size)
        new_num_color_codes: Optional new color codebook size
        out_path: Where to save the expanded checkpoint

    Usage:
        expand_codebook(
            'runs/.../phase0/frozen_vq_codebook.pth',
            new_num_shape_codes=2048
        )
    """
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    shape_w = state['embedding_shape.weight']  # [old_N, D]
    color_w = state['embedding_color.weight']  # [old_C, D]
    old_N, D = shape_w.shape
    old_C    = color_w.shape[0]

    if new_num_shape_codes <= old_N:
        raise ValueError(f"new_num_shape_codes ({new_num_shape_codes}) must be > current ({old_N})")

    n_new_shape = new_num_shape_codes - old_N
    # Sample new codes: random perturbations of the top-used existing codes
    # (perturb by a small fraction of their std dev)
    noise_std = shape_w.std() * 0.1
    random_base = shape_w[torch.randint(0, old_N, (n_new_shape,))]
    new_shape_codes = random_base + torch.randn_like(random_base) * noise_std

    expanded_shape = torch.cat([shape_w, new_shape_codes], dim=0)
    state['embedding_shape.weight'] = expanded_shape

    if new_num_color_codes is not None and new_num_color_codes > old_C:
        n_new_color = new_num_color_codes - old_C
        noise_std_c = color_w.std() * 0.1
        random_base_c = color_w[torch.randint(0, old_C, (n_new_color,))]
        new_color_codes = random_base_c + torch.randn_like(random_base_c) * noise_std_c
        expanded_color = torch.cat([color_w, new_color_codes], dim=0)
        state['embedding_color.weight'] = expanded_color

        usage_shape = torch.zeros(new_num_shape_codes)
        usage_color = torch.zeros(new_num_color_codes)
    else:
        usage_shape = torch.zeros(new_num_shape_codes)
        usage_color = torch.zeros(old_C)

    state['shape_usage'] = usage_shape
    state['color_usage'] = usage_color

    out_path = out_path or checkpoint_path.replace('.pth', f'_expanded{new_num_shape_codes}.pth')
    torch.save(state, out_path)
    print(f"✅ Expanded codebook: {old_N} → {new_num_shape_codes} shape codes")
    print(f"   Saved to: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0 AUTOENCODER (Patch-Level Alphabet)
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

    def forward(self, inputs, valid_mask=None, temperature=1.0):
        enc_out = self.encoder(inputs)
        z_vq    = enc_out['latent_vq']
        z_pose  = enc_out['latent_pose']
        z_q, vq_loss, p_shape, p_color, shape_idx = self.vq(
            z_vq, valid_mask=valid_mask, temperature=temperature)
        latent_combined = torch.cat([z_q, z_pose], dim=-1)
        out = self.decoder({'latent': latent_combined})
        return out, vq_loss, p_shape, p_color, z_vq, shape_idx


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0 TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_phase_0(cfg, p0_dir, wb_run, train_dataset):
    global _STOP
    device      = cfg['device']
    ckpt_path   = os.path.join(p0_dir, 'latest_checkpoint.pth')
    vq_path     = os.path.join(p0_dir, 'frozen_vq_codebook.pth')
    metrics_path = os.path.join(p0_dir, 'metrics_log.jsonl')

    print("\n" + "═" * 60)
    print("🚀  PHASE 0: Object Alphabet Pretraining")
    print(f"    Codebook: {cfg['num_shape_codes']} shape × {cfg['num_color_codes']} color codes")
    print(f"    BG reserved: codes 0–{FactorizedVectorQuantizer.N_BG_CODES - 1}")
    print("═" * 60)

    model = Phase0Autoencoder(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['p0_lr'],
                               weight_decay=1e-4, betas=(0.9, 0.999))

    start_epoch = 1
    epoch       = 0
    post_surg_cd = 0

    # ── Resume ──────────────────────────────────────────────────────────
    resume = cfg.get('p0_resume_from') or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume and os.path.exists(resume):
        print(f"📥 P0 resuming from {resume}...")
        try:
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=False)
            opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch'] + 1
            epoch       = ckpt['epoch']
            print(f"   ✅ Resumed from epoch {ckpt['epoch']}.")
        except Exception as e:
            print(f"   ⚠️  Checkpoint incompatible, starting fresh. ({e})")

    # ── Training Loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg['p0_epochs'] + 1):
        model.train()
        ep_loss, ep_vq, ep_ps, ep_pc, ep_aff = [], [], [], [], []
        nan_streak = 0

        # Gumbel temperature: 1.0 → 0.1 over full training
        tau = max(0.1, 1.0 - (epoch / cfg['p0_epochs']))

        # LR warmup after surgery
        if post_surg_cd > 0:
            wd = cfg['p0_lr_warmup_epochs'] - post_surg_cd
            warm_lr = cfg['p0_lr_post_surgery'] + (cfg['p0_lr'] - cfg['p0_lr_post_surgery']) * (wd / cfg['p0_lr_warmup_epochs'])
            set_lr(opt, warm_lr)
            post_surg_cd -= 1
            if post_surg_cd == 0:
                set_lr(opt, cfg['p0_lr'])

        cur_lr = opt.param_groups[0]['lr']

        for step in tqdm(range(cfg['p0_steps']), desc=f"P0 E{epoch:04d}", leave=False):
            batch  = train_dataset.sample(cfg['p0_batch'])
            state  = batch['state'].to(device)
            B      = state.shape[0]
            vmask  = compute_valid_patch_mask(state, cfg['patch_size']).to(device)

            # ── 1. Rotation Invariance Path (InfoNCE-style contrastive) ──
            k = torch.randint(1, 4, (1,)).item()
            state_rot = torch.rot90(state, k=k, dims=[2, 3])
            vmask_rot = torch.rot90(
                vmask.view(B, cfg['grid_size']//cfg['patch_size'],
                              cfg['grid_size']//cfg['patch_size']),
                k=k, dims=[1, 2]
            ).reshape(B, -1)
            _, vq_loss_rot, _, _, z_vq_rot, _ = model(
                {'state': state_rot}, valid_mask=vmask_rot, temperature=tau)

            gh = cfg['grid_size'] // cfg['patch_size']
            z_aligned = torch.rot90(
                z_vq_rot.view(B, gh, gh, -1).permute(0, 3, 1, 2),
                k=-k, dims=[2, 3]
            ).flatten(2).transpose(1, 2)

            # ── 2. Standard Forward Pass ─────────────────────────────────
            out, vq_loss, p_shape, p_color, z_vq, shape_idx = model(
                {'state': state}, valid_mask=vmask, temperature=tau)
            loss_dict = model.decoder.loss({'state': state}, out)

            # ── 3. Affinity Loss (every N steps, scipy CC on CPU) ────────
            aff_loss = torch.tensor(0.0, device=device)
            if step % cfg['p0_affinity_interval'] == 0:
                try:
                    grid_hw = cfg['grid_size'] // cfg['patch_size']
                    aff_loss = FactorizedVectorQuantizer.affinity_loss(
                        shape_idx, state, cfg['patch_size'], grid_hw
                    ).to(device)
                    ep_aff.append(aff_loss.item())
                except Exception:
                    pass

            # ── 4. Total Loss ─────────────────────────────────────────────
            inv_loss = F.mse_loss(z_aligned, z_vq)
            reg_loss = vicreg_loss(z_vq.reshape(-1, z_vq.shape[-1]))

            total_loss = (
                loss_dict['loss']
                + vq_loss + vq_loss_rot
                + cfg['p0_inv_weight']      * inv_loss
                + cfg['p0_vicreg_weight']   * reg_loss
                + cfg['p0_affinity_weight'] * aff_loss
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_streak += 1
                opt.zero_grad()
                if nan_streak >= 5:
                    print(f"\n❌ P0: NaN persists 5 steps at epoch {epoch} — stopping.")
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
        avg_aff     = float(np.mean(ep_aff)) if ep_aff else 0.0

        log_metrics(wb_run, {
            'P0/Recon_Loss':             avg_recon,
            'P0/VQ_Loss':                avg_vq,
            f'P0/Perplexity_Shape_{cfg["num_shape_codes"]}': avg_p_shape,
            f'P0/Perplexity_Color_{cfg["num_color_codes"]}': avg_p_color,
            'P0/Affinity_Loss':          avg_aff,
            'P0/LR':                     cur_lr,
            'P0/GumbelTau':              tau,
        }, step=epoch, log_path=metrics_path)

        if epoch % cfg['p0_save_interval'] == 0 or _STOP:
            print(f"P0 E{epoch:04d} | Recon:{avg_recon:.4f} VQ:{avg_vq:.4f} "
                  f"Perp:{avg_p_shape:.1f}/{cfg['num_shape_codes']} "
                  f"Color:{avg_p_color:.1f}/{cfg['num_color_codes']} "
                  f"Aff:{avg_aff:.3f} τ:{tau:.2f}")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch, 'cfg': cfg}, ckpt_path)

        # ── Codebook Surgery ─────────────────────────────────────────────
        if epoch % cfg['surgery_interval'] == 0:
            q_pct = int(cfg['surgery_quantile'] * 100)
            print(f"⚡ P0 E{epoch}: Resurrection (bottom {q_pct}% by usage)...")
            with torch.no_grad():
                z_raw = model.encoder({'state': state})['latent_vq']
                n_s, n_c = model.vq.resurrect_dead_codes(
                    z_raw, valid_mask=vmask,
                    aggression_quantile=cfg['surgery_quantile'])
            print(f"   Resurrected {n_s} shape codes, {n_c} color codes.")
            set_lr(opt, cfg['p0_lr_post_surgery'])
            post_surg_cd = cfg['p0_lr_warmup_epochs']

        # ── Early Stop ───────────────────────────────────────────────────
        if (avg_p_shape > cfg['p0_stop_perp_shape']
                and avg_p_color > cfg['p0_stop_perp_color']
                and avg_recon < cfg['p0_stop_recon']):
            print(f"\n✅ P0 Alphabet fully baked at epoch {epoch}!")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch, 'cfg': cfg}, ckpt_path)
            break

        if _STOP:
            print(f"\n💾 P0 emergency checkpoint saved (epoch {epoch}).")
            break

    # Save frozen VQ
    frozen_vq = model.vq.eval()
    del model
    torch.save(frozen_vq.state_dict(), vq_path)
    print(f"\n💾 Frozen VQ codebook → {vq_path}")
    print(f"   Shape codes: {cfg['num_shape_codes']} | Color codes: {cfg['num_color_codes']}")
    print(f"   To expand later: expand_codebook('{vq_path}', new_num_shape_codes=2048)")
    return frozen_vq, vq_path, epoch


# ── NEW: Factorized Diversity Patch Utilities ───────────────────────────────

def shannon_entropy_loss(sims, temp=0.1):
    """
    Computes Shannon Entropy of the codebook assignment distribution.
    Encourages the model to use more codes (higher entropy = better).
    """
    # sims: [B*K, V]
    # Calculate usage distribution over the codebook
    probs = F.softmax(sims / temp, dim=-1).mean(dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    # We want to MAXIMIZE entropy, so minimize (MaxEntropy - currentEntropy)
    max_entropy = math.log(sims.shape[-1])
    return max_entropy - entropy

def affine_contrastive_loss(slots, slots_rot):
    """
    Enforces that Shape and Color segments are invariant to rotation.
    slots: [B, K, 320]
    slots_rot: [B, K, 320] (forward pass of rotated input)
    """
    # Factorized segments
    ld = slots.shape[-1] - slots.shape[-1] // 5 # approximate or use specific dims
    # Better: use half of latent_dim for shape/color
    shape_dim = slots.shape[-1] // 2 if slots.shape[-1] == 256 else (slots.shape[-1] - 64) // 2
    if slots.shape[-1] < 100: # Tiny model
        shape_dim = 16
        color_dim = 16
    else:
        shape_dim = 128
        color_dim = 128
        
    shape_orig = slots[:, :, :shape_dim]
    shape_rot  = slots_rot[:, :, :shape_dim]
    color_orig = slots[:, :, shape_dim:shape_dim+color_dim]
    color_rot  = slots_rot[:, :, shape_dim:shape_dim+color_dim]
    
    # MSE for invariance
    l_shape = F.mse_loss(shape_orig, shape_rot)
    l_color = F.mse_loss(color_orig, color_rot)
    
    return l_shape + l_color

@torch.no_grad()
def resurrect_dead_codes(slot_enc, usage_counts, thresh=50):
    """
    Resets codes that have been used fewer than 'thresh' times.
    They are re-initialized to the value of a randomly selected active slot embedding.
    """
    if not hasattr(slot_enc, 'codebook_shape'): return 0
    
    dead_mask = usage_counts < thresh
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    
    if len(dead_indices) > 0:
        # For simplicity in this patch, we just re-randomize them.
        num_dead = len(dead_indices)
        dim = slot_enc.codebook_shape.shape[1]
        new_weights = torch.randn(num_dead, dim, device=slot_enc.codebook_shape.device) * 0.02
        slot_enc.codebook_shape.data[dead_indices] = new_weights
        return num_dead
    return 0



# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — OBJECT DICTIONARY TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_phase_1(cfg, p1_dir, wb_run, frozen_vq, train_dataset, eval_dataset, step_offset=0):
    """
    Trains slot attention model with all 6 object-level losses.
    The codebook evolves from patch-level into an object-level dictionary.
    """
    global _STOP
    device = cfg['device']

    print("\n" + "═" * 60)
    print("🧠  PHASE 1: Object Dictionary Training (6-Loss Regime)")
    print(f"    Slots: {cfg['num_slots']} | Iterations: {cfg['slot_iters']}")
    print(f"    Codebook freeze: first {cfg['codebook_warmup_epochs']} epochs")
    print("═" * 60)

    ckpt_path      = os.path.join(p1_dir, 'latest_slot_checkpoint.pth')
    base_ckpt_path = os.path.join(p1_dir, 'latest_base_checkpoint.pth')
    metrics_path   = os.path.join(p1_dir, 'metrics_log.jsonl')
    plots_dir      = os.path.join(p1_dir, 'plots')

    # ── Models ──────────────────────────────────────────────────────────
    slot_enc = SemanticSlotEncoder(cfg)
    slot_dec = SemanticDecoder(cfg)
    slot_enc.inject_codebook(frozen_vq)
    
    # Track codebook usage for resurrection
    usage_counts = torch.zeros(cfg.get('num_shape_codes', 1024), device=device)


    base_enc = DeepTransformerEncoder(cfg)
    base_dec = TransformerDecoder(cfg)

    # Separate param groups: codebook has its own LR controlled by warmup schedule
    def make_slot_param_groups():
        cb_params  = []
        enc_params = []
        if hasattr(slot_enc, 'codebook_shape'):
            cb_params  = [slot_enc.codebook_shape, slot_enc.codebook_color]
            cb_ids     = {id(p) for p in cb_params}
            enc_params = [p for p in slot_enc.parameters() if id(p) not in cb_ids]
        else:
            enc_params = list(slot_enc.parameters())

        return [
            {'params': enc_params + list(slot_dec.parameters()), 'lr': cfg['p1_lr']},
            {'params': cb_params, 'lr': 0.0},  # codebook LR starts frozen
        ]

    opt_slot = torch.optim.AdamW(make_slot_param_groups(), weight_decay=1e-4)
    opt_base = torch.optim.AdamW(
        list(base_enc.parameters()) + list(base_dec.parameters()),
        lr=cfg['p1_lr'], weight_decay=1e-4)

    start_epoch = 1

    # ── Resume ──────────────────────────────────────────────────────────
    resume = cfg.get('p1_resume_from') or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume and os.path.exists(resume):
        print(f"📥 P1 resuming from {resume}...")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        try:
            slot_enc.load_state_dict(ckpt['slot_enc'], strict=False)
            slot_dec.load_state_dict(ckpt['slot_dec'], strict=False)
            opt_slot.load_state_dict(ckpt['opt_slot'])
            start_epoch = ckpt['epoch'] + 1
            print(f"   ✅ Slot model resumed from epoch {ckpt['epoch']}.")
        except Exception as e:
            print(f"   ⚠️  Slot checkpoint incompatible, starting fresh. ({e})")

    if os.path.exists(base_ckpt_path):
        try:
            b = torch.load(base_ckpt_path, map_location=device, weights_only=False)
            base_enc.load_state_dict(b['base_enc'], strict=False)
            base_dec.load_state_dict(b['base_dec'], strict=False)
            opt_base.load_state_dict(b['opt_base'])
            print(f"   ✅ Baseline resumed from epoch {b['epoch']}.")
        except Exception:
            pass

    # ── Training Loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg['p1_epochs'] + 1):
        slot_enc.train(); slot_dec.train()
        base_enc.train(); base_dec.train()

        # ── Codebook Warmup Schedule ─────────────────────────────────────
        warmup = cfg.get('codebook_warmup_epochs', 150)
        if hasattr(slot_enc, 'codebook_shape'):
            if epoch <= warmup:
                # Fully frozen
                slot_enc.codebook_shape.requires_grad_(False)
                slot_enc.codebook_color.requires_grad_(False)
                opt_slot.param_groups[1]['lr'] = 0.0
                if epoch == 1:
                    print(f"   🔒 Codebook FROZEN for first {warmup} epochs.")
            elif epoch == warmup + 1:
                # Unfreeze at very low LR
                slot_enc.codebook_shape.requires_grad_(True)
                slot_enc.codebook_color.requires_grad_(True)
                opt_slot.param_groups[1]['lr'] = cfg.get('codebook_finetune_lr', 1e-5)
                print(f"\n🔓 P1 E{epoch}: Codebook unfrozen (LR={cfg.get('codebook_finetune_lr', 1e-5):.0e})")

        # Temperature annealing
        t = min(1.0, (epoch - 1) / max(1, cfg['slot_temp_anneal']))
        temp = cfg['slot_temp_start'] * (1 - t) + cfg['slot_temp_end'] * t

        ep_stot, ep_srecon, ep_ssharp, ep_scover = [], [], [], []
        ep_sdiv, ep_brecon = [], []

        for step in tqdm(range(cfg['p1_steps']), desc=f"P1 E{epoch:04d}", leave=False):
            batch  = train_dataset.sample(cfg['p1_batch'])
            states = batch['state'].to(device)

            # ── Slot Model Step ─────────────────────────────────────────
            slot_out  = slot_enc({'state': states}, temperature=temp)
            slots     = slot_out['latent']   # [B, K, 320]
            alphas_raw = slot_out.get('masks_raw', None)

            dec_out   = slot_dec({'latent': slots})
            alphas    = dec_out['alphas']  # [B, K, H, W] — softmax'd

            # L1: Focal-weighted pixel reconstruction
            recon_target = states[:, 0].long()
            recon_logits = dec_out['reconstruction']
            ce = F.cross_entropy(recon_logits, recon_target, reduction='none')
            fg_mask    = (recon_target > 0).float()
            w_matrix   = 1.0 + fg_mask * (cfg['p1_fg_weight'] - 1.0)
            pt         = torch.exp(-ce)
            l_recon    = (((1 - pt) ** cfg['p1_focal_gamma']) * ce * w_matrix).mean()

            # L2: Slot sharpness (low entropy of alpha over slots at each pixel)
            l_sharp = slot_sharpness_loss(alphas)

            # L3: Coverage (every N steps — scipy CC labeling on CPU)
            l_cover = torch.tensor(0.0, device=device)
            if step % cfg['p1_cover_interval'] == 0:
                try:
                    l_cover = slot_coverage_loss(alphas, states, cfg['patch_size'])
                    ep_scover.append(l_cover.item())
                except Exception as e:
                    if step == 0:  # only log once per epoch to avoid spam
                        print(f"  ⚠️  Coverage loss skipped: {e}")

            # L4: Slot diversity (VICReg across K slots within each image)
            l_div = vicreg_loss_slots(slots,
                                      std_coeff=cfg['p1_vicreg_std'],
                                      cov_coeff=cfg['p1_vicreg_cov'])

            # L5: VQ commitment (slot shape part vs nearest codebook entry)
            l_commit = torch.tensor(0.0, device=device)
            if hasattr(slot_enc, 'codebook_shape'):
                sd = slot_enc.codebook_shape.shape[1]
                slot_shape = slots[:, :, :sd]  # Dynamic shape segment
                cb = slot_enc.codebook_shape  
                slot_shape_n = F.normalize(slot_shape.reshape(-1, sd), dim=-1)
                cb_n = F.normalize(cb, dim=-1)
                sims = slot_shape_n @ cb_n.T
                nearest_idx = sims.argmax(dim=-1)
                nearest_emb = cb[nearest_idx].reshape_as(slot_shape)
                l_commit = slot_vq_commit_loss(slot_shape, nearest_emb,
                                               beta=cfg['p1_commit_weight'])

            # L6: NEW: Entropy Regularization (Diversity)
            l_entropy = shannon_entropy_loss(sims)

            # L7: NEW: Affine Invariance (Factorization)
            # Run second forward pass with rotated input
            states_rot = torch.rot90(states, 1, [2, 3])
            slot_out_rot = slot_enc({'state': states_rot}, temperature=temp)
            slots_rot = slot_out_rot['latent']
            l_affine = affine_contrastive_loss(slots, slots_rot)

            # Update usage stats for resurrection hook
            with torch.no_grad():
                usage_counts[nearest_idx] += 1

            # ── Total Slot Loss ──────────────────────────────────────────
            s_loss = (
                cfg['p1_recon_weight']   * l_recon
                + cfg['p1_sharp_weight'] * l_sharp
                + cfg['p1_cover_weight'] * l_cover
                + cfg['p1_vicreg_weight']* l_div
                + cfg['lambda_entropy']  * l_entropy
                + cfg['lambda_affine']   * l_affine
                + l_commit
            )

            if not (torch.isnan(s_loss) or torch.isinf(s_loss)):
                opt_slot.zero_grad()
                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(slot_enc.parameters()) + list(slot_dec.parameters()),
                    cfg['p1_grad_clip'])
                opt_slot.step()
                ep_stot.append(s_loss.item())
                ep_srecon.append(l_recon.item())
                ep_ssharp.append(l_sharp.item())
                ep_sdiv.append(l_div.item())

            # ── Baseline Model Step ──────────────────────────────────────
            z_base   = base_enc({'state': states})
            out_base = base_dec({'latent': z_base['latent']})
            b_loss   = base_dec.loss({'state': states}, out_base)['loss']

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

        avg_stot   = float(np.mean(ep_stot))
        avg_srecon = float(np.mean(ep_srecon))
        avg_ssharp = float(np.mean(ep_ssharp))
        avg_scover = float(np.mean(ep_scover)) if ep_scover else 0.0
        avg_sdiv   = float(np.mean(ep_sdiv))
        avg_brecon = float(np.mean(ep_brecon)) if ep_brecon else 0.0

        log_metrics(wb_run, {
            'P1_Slot/Total':       avg_stot,
            'P1_Slot/Recon':       avg_srecon,
            'P1_Slot/Sharpness':   avg_ssharp,
            'P1_Slot/Coverage':    avg_scover,
            'P1_Slot/Diversity':   avg_sdiv,
            'P1_Base/Recon':       avg_brecon,
            'P1/Codebook_Frozen':  float(epoch <= warmup),
            'P1/Temperature':      temp,
        }, step=step_offset + epoch, log_path=metrics_path)

        # ── End of Epoch Hooks ──────────────────────────────────────────
        if epoch % cfg['p1_resurrect_interval'] == 0 and epoch > cfg['codebook_warmup_epochs']:
            num_res = resurrect_dead_codes(slot_enc, usage_counts, thresh=cfg['p1_resurrect_thresh'])
            if num_res > 0:
                print(f"   ⚰️  Epoch {epoch}: Resurrected {num_res} dead shape codes.")
            # Reset usage for next interval
            usage_counts.zero_()
            
        if epoch % cfg['p1_save_interval'] == 0 or _STOP:
            print(f"P1 E{epoch:04d} | Slot[recon:{avg_srecon:.4f} sharp:{avg_ssharp:.3f} "
                  f"cov:{avg_scover:.3f} div:{avg_sdiv:.3f}] Base:{avg_brecon:.4f} T:{temp:.2f}")
            torch.save({
                'slot_enc': slot_enc.state_dict(),
                'slot_dec': slot_dec.state_dict(),
                'opt_slot': opt_slot.state_dict(),
                'epoch': epoch, 'cfg': cfg,
            }, ckpt_path)
            torch.save({
                'base_enc': base_enc.state_dict(),
                'base_dec': base_dec.state_dict(),
                'opt_base': opt_base.state_dict(),
                'epoch': epoch,
            }, base_ckpt_path)

        # ── Validation ───────────────────────────────────────────────────
        if epoch % cfg['p1_val_interval'] == 0:
            # run_validation_epoch(modules_dict, dataset, phase, batch_size, device)
            # → returns (avg_loss, avg_acc, avg_perfect)
            # It handles .eval() / .train() toggling internally.
            slot_loss, slot_acc, slot_perfect = run_validation_epoch(
                modules={'encoder': slot_enc, 'decoder': slot_dec},
                dataset=eval_dataset,
                phase='ae',
                batch_size=cfg.get('val_batch_size', 32),
                device=device,
            )
            base_loss, base_acc, base_perfect = run_validation_epoch(
                modules={'encoder': base_enc, 'decoder': base_dec},
                dataset=eval_dataset,
                phase='ae',
                batch_size=cfg.get('val_batch_size', 32),
                device=device,
            )
            print(f"  → [Slot Val] Pixel.Acc:{slot_acc:.2%} | Perfect:{slot_perfect:.2%}")
            print(f"  → [Base Val] Pixel.Acc:{base_acc:.2%} | Perfect:{base_perfect:.2%}")
            log_metrics(wb_run, {
                'Val/Slot_PixelAcc':  slot_acc,
                'Val/Slot_Perfect':   slot_perfect,
                'Val/Slot_Loss':      slot_loss,
                'Val/Base_PixelAcc':  base_acc,
                'Val/Base_Perfect':   base_perfect,
                'Val/Base_Loss':      base_loss,
            }, step=step_offset + epoch, log_path=metrics_path)
            # Restore training mode (evaluator already did this, but be explicit)
            slot_enc.train(); slot_dec.train()
            base_enc.train(); base_dec.train()

        # ── Visualization ─────────────────────────────────────────────────
        if epoch % (cfg['p1_save_interval'] * 2) == 0:
            try:
                slot_enc.eval()
                sample = eval_dataset.sample(4)
                with torch.no_grad():
                    vis_out = slot_enc({'state': sample['state'].to(device)}, temperature=0.1)
                    vis_dec = slot_dec({'latent': vis_out['latent']})
                plot_slot_masks(
                    sample['state'], vis_dec['reconstruction'],
                    vis_dec['alphas'],
                    save_path=os.path.join(plots_dir, f'masks_epoch_{epoch:04d}.png'))
                slot_enc.train()
            except Exception:
                pass

        if _STOP:
            break

    print(f"\n🏁 Phase 1 Training Complete.")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   Object dictionary: {cfg['num_shape_codes']} shape codes")


# ═══════════════════════════════════════════════════════════════════════════
# GPU MEMORY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def check_gpu_memory(min_free_gb=4.0):
    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb  = free_bytes  / 1024**3
    total_gb = total_bytes / 1024**3
    used_gb  = total_gb - free_gb
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total: {total_gb:.1f} GB | Used: {used_gb:.1f} GB | Free: {free_gb:.1f} GB")
    if free_gb < min_free_gb:
        print(f"\n⚠️  WARNING: Only {free_gb:.1f} GB free — need ≥{min_free_gb:.1f} GB.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(cfg: dict):
    device = cfg['device']
    check_gpu_memory(min_free_gb=4.0)
    print(f"🖥️  Device: {device}")

    # Resume or create run directory
    resume_run = cfg.get('resume_run_dir')
    if resume_run:
        root_dir = resume_run
        p0_dir   = os.path.join(root_dir, 'phase0')
        p1_dir   = os.path.join(root_dir, 'phase1')
        print(f"🔄 Resuming from: {root_dir}")
    else:
        root_dir, p0_dir, p1_dir = make_run_dir(cfg)

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = ReARCDataset(data_path=cfg['data_path'])

    try:
        arc_heavy = ARCDataset(data_path=cfg['arc_heavy_path'])

        class CurriculumMix:
            def __init__(self, ds1, ds2):
                self.ds1, self.ds2 = ds1, ds2
            def sample(self, bsz, split='train'):
                b1 = self.ds1.sample(bsz // 2)
                b2 = self.ds2.sample(bsz - (bsz // 2))
                return {'state': torch.cat([b1['state'], b2['state']], dim=0)}

        train_dataset = CurriculumMix(train_dataset, arc_heavy)
        print("🌍 Mixed ARC-Heavy into training curriculum.")
    except Exception:
        print("⚠️  ARC-Heavy not found — using pure Re-ARC.")

    try:
        eval_dataset = ARCDataset(data_path=cfg['eval_data_path'])
    except Exception:
        print("⚠️  Eval dataset not found — using train dataset split for validation.")
        eval_dataset = train_dataset

    # ── WandB ─────────────────────────────────────────────────────────────
    wb_run = init_wandb(cfg, root_dir, 'Phase0+1')

    # ── Phase 0 ───────────────────────────────────────────────────────────
    frozen_vq, _, p0_last_epoch = train_phase_0(cfg, p0_dir, wb_run, train_dataset)

    if _STOP:
        print("\n🛑 Stop requested during Phase 0.")
        if wb_run: wb_run.finish()
        return

    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"\n🧹 GPU cleared after Phase 0. Free VRAM: {free_bytes / 1024**2:.0f} MiB")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    train_phase_1(cfg, p1_dir, wb_run, frozen_vq, train_dataset, eval_dataset,
                  step_offset=p0_last_epoch)

    if wb_run:
        wb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NS-ARC Object Codebook Training (Phase 0 + Phase 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Resume helpers ───────────────────────────────────────────────────────
    parser.add_argument(
        '--resume-run', metavar='RUN_DIR',
        help='Root run directory (e.g. runs/ObjectCodebook-v1_2026-04-26_14-13-43). '
             'Auto-loads latest_checkpoint.pth from phase0/ and '
             'latest_slot_checkpoint.pth from phase1/.',
    )
    parser.add_argument(
        '--resume-p0', metavar='CKPT',
        help='Path to a specific Phase 0 checkpoint .pth to resume from.',
    )
    parser.add_argument(
        '--resume-p1', metavar='CKPT',
        help='Path to a specific Phase 1 slot checkpoint .pth to resume from.',
    )
    # ── Skip phases ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--skip-p0', action='store_true',
        help='Skip Phase 0 entirely (requires --resume-run or a frozen codebook to already exist).',
    )
    # ── Epoch overrides ──────────────────────────────────────────────────────
    parser.add_argument('--p0-epochs', type=int, metavar='N',
                        help='Override CFG["p0_epochs"].')
    parser.add_argument('--p1-epochs', type=int, metavar='N',
                        help='Override CFG["p1_epochs"].')

    args = parser.parse_args()

    # ── Apply CLI overrides to CFG ────────────────────────────────────────
    if args.resume_run:
        run_dir = args.resume_run.rstrip('/')
        p0_ckpt = os.path.join(run_dir, 'phase0', 'latest_checkpoint.pth')
        p1_ckpt = os.path.join(run_dir, 'phase1', 'latest_slot_checkpoint.pth')
        if os.path.exists(p0_ckpt):
            CFG['p0_resume_from'] = p0_ckpt
            print(f"📂 Resuming Phase 0 from: {p0_ckpt}")
        if os.path.exists(p1_ckpt):
            CFG['p1_resume_from'] = p1_ckpt
            print(f"📂 Resuming Phase 1 from: {p1_ckpt}")
        CFG['resume_run_dir'] = run_dir

    if args.resume_p0:
        CFG['p0_resume_from'] = args.resume_p0
        print(f"📂 Phase 0 checkpoint: {args.resume_p0}")

    if args.resume_p1:
        CFG['p1_resume_from'] = args.resume_p1
        print(f"📂 Phase 1 checkpoint: {args.resume_p1}")

    if args.skip_p0:
        CFG['p0_epochs'] = 0
        print("⏭️  Skipping Phase 0 (p0_epochs set to 0).")

    if args.p0_epochs is not None:
        CFG['p0_epochs'] = args.p0_epochs
        print(f"🔧 p0_epochs overridden to {args.p0_epochs}")

    if args.p1_epochs is not None:
        CFG['p1_epochs'] = args.p1_epochs
        print(f"🔧 p1_epochs overridden to {args.p1_epochs}")

    main(CFG)
