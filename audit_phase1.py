"""
Phase 1 Slot Model Audit
========================
Deep inspection of the SemanticSlotEncoder + SlotDecoder after Phase 1 training.
All architecture dims auto-detected from the Phase 1 checkpoint.

Panels generated:
  1. Slot Mask Gallery      — attention mask per slot overlaid on input grids
  2. Slot Reconstruction    — input | slot recon | error mask | baseline recon
  3. Slot Specialisation    — what ARC colours / edge types each slot dominates
  4. Prior Delta Analysis   — how far each slot drifted from its FPS anchor
  5. Slot Utilisation       — do all 16 slots "fire" or are some dead?
  6. Slot Collapse Probe    — cosine similarity matrix between slot representations
  7. Accuracy Comparison    — slot model vs baseline histogram comparison

Usage:
    python audit_phase1.py
        --slot_ckpt   runs/.../phase1/latest_checkpoint.pth
        --base_ckpt   runs/.../phase1/base_checkpoint.pth
        --p0_ckpt     runs/.../phase0/latest_checkpoint.pth  (needed for VQ + arch dims)
        --out         evaluation_reports/phase1_audit
        --n_grids     150
"""

import argparse, os, sys, json, warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

# ── ARC colour palette ──────────────────────────────────────────────────────
ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
               '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
ARC_CMAP   = ListedColormap(ARC_COLORS)
ARC_NAMES  = ['Black','Blue','Red','Green','Yellow',
               'Grey','Magenta','Orange','Cyan','Maroon']

# ═══════════════════════════════════════════════════════════════════════════════
# ARCH DETECTION (shared with audit_model.py)
# ═══════════════════════════════════════════════════════════════════════════════
def detect_arch_from_p0(ckpt_path: str) -> dict:
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    arch  = {'patch_size':2,'hidden_dim':128,'latent_dim':128,
              'num_shape_codes':256,'num_color_codes':16,
              'vocab_size':10,'grid_size':30,'in_channels':10,
              'focal_gamma':2.0,'commitment_cost':0.25,
              'device':'cuda' if torch.cuda.is_available() else 'cpu'}
    for key, val in state.items():
        if 'encoder.patch_embed.weight' in key and val.dim() == 4:
            arch['hidden_dim']  = val.shape[0]
            arch['patch_size']  = val.shape[2]
        if 'encoder.projector' in key and '3.weight' in key and val.dim() == 2:
            arch['latent_dim']  = val.shape[0]
        if 'vq.embedding_shape.weight' in key and val.dim() == 2:
            arch['num_shape_codes'] = val.shape[0]
        if 'vq.embedding_color.weight' in key and val.dim() == 2:
            arch['num_color_codes'] = val.shape[0]
    return arch


def detect_slot_params(slot_ckpt_path: str) -> dict:
    """Read num_slots and slot embed_dim from the slot checkpoint."""
    if not os.path.exists(slot_ckpt_path):
        raise FileNotFoundError(
            f"Slot checkpoint not found: {slot_ckpt_path}\n"
            f"  Phase 1 saves checkpoints every p1_save_interval epochs.\n"
            f"  Check the phase1/ directory with: ls <run_dir>/phase1/"
        )
    ckpt  = torch.load(slot_ckpt_path, map_location='cpu', weights_only=False)
    # Phase 1 saves keys 'slot_enc' and 'slot_dec'
    state = ckpt.get('slot_enc', ckpt)
    params = {'num_slots': 10, 'slot_embed_dim': 128,
              'slot_iters': 3, 'slot_temperature': 0.1,
              'vocab_size': 10, 'grid_size': 30, 'patch_size': 2}
    # semantic_priors: [1, num_slots, embed_dim]
    if 'semantic_priors' in state:
        sh = state['semantic_priors'].shape
        params['num_slots']      = sh[1]
        params['slot_embed_dim'] = sh[2]
    # patch_embed: [embed_dim, 12, patch_size, patch_size]
    if 'patch_embed.weight' in state:
        sh = state['patch_embed.weight'].shape
        params['slot_embed_dim'] = sh[0]
        params['patch_size']     = sh[2]
    print(f"  Slot params: num_slots={params['num_slots']}, "
          f"embed_dim={params['slot_embed_dim']}, patch_size={params['patch_size']}")
    return params


def find_phase1_checkpoints(run_dir: str):
    """
    Given a run directory, auto-discover the Phase 1 checkpoint files.
    Returns (slot_ckpt, base_ckpt, p0_ckpt) absolute paths.
    """
    p1_dir = os.path.join(run_dir, 'phase1')
    p0_dir = os.path.join(run_dir, 'phase0')

    # Phase 1 saves as: latest_slot_checkpoint.pth / latest_base_checkpoint.pth
    # (set in train_phase0.py line 391-392)
    slot_ckpt = os.path.join(p1_dir, 'latest_slot_checkpoint.pth')
    base_ckpt = os.path.join(p1_dir, 'latest_base_checkpoint.pth')
    p0_ckpt   = os.path.join(p0_dir, 'latest_checkpoint.pth')

    print(f"  Auto-discovered from run dir: {run_dir}")
    for label, path in [('slot_ckpt', slot_ckpt),
                         ('base_ckpt', base_ckpt),
                         ('p0_ckpt',   p0_ckpt)]:
        status = '✅' if os.path.exists(path) else '❌ NOT FOUND'
        print(f"    {label}: {os.path.basename(path)}  {status}")

    if not os.path.exists(slot_ckpt):
        # List what IS in the phase1 directory to help debug
        if os.path.exists(p1_dir):
            contents = os.listdir(p1_dir)
            print(f"\n  phase1/ contains: {contents or '(empty — Phase 1 has not saved yet)'} ")
        else:
            print(f"\n  phase1/ directory does not exist yet — Phase 1 may not have started.")
        raise FileNotFoundError(
            f"Phase 1 slot checkpoint not found.\n"
            f"  Either Phase 1 hasn't reached p1_save_interval yet, or use\n"
            f"  --slot_ckpt / --base_ckpt to point at a specific file.\n"
            f"  Expected: {slot_ckpt}"
        )
    return slot_ckpt, base_ckpt, p0_ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_phase0_vq(p0_ckpt: str, cfg: dict):
    from modules.vq import FactorizedVectorQuantizer
    vq = FactorizedVectorQuantizer(
        num_shape_codes=cfg['num_shape_codes'],
        num_color_codes=cfg['num_color_codes'],
        embedding_dim  =cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
    ).to(cfg['device'])
    state = torch.load(p0_ckpt, map_location=cfg['device'], weights_only=False)
    s     = state.get('model', state)
    vq_s  = {k.replace('vq.','',1): v for k,v in s.items() if k.startswith('vq.')}
    vq.load_state_dict(vq_s, strict=False)
    vq.eval()
    return vq


def load_slot_model(slot_ckpt: str, cfg: dict, slot_params: dict):
    from modules.semantic_encoders import SemanticSlotEncoder
    from modules.decoders          import SpatialBroadcastDecoder

    # Build a merged config for the slot encoder
    slot_cfg = {**cfg,
                'num_slots':       slot_params['num_slots'],
                'slot_iters':      slot_params.get('slot_iters', 3),
                'slot_temperature': slot_params.get('slot_temperature', 0.1),
                'hidden_dim':      slot_params['slot_embed_dim'],
                'latent_dim':      slot_params['slot_embed_dim'],
                'patch_size':      slot_params['patch_size']}

    enc = SemanticSlotEncoder(slot_cfg).to(cfg['device'])
    dec = SpatialBroadcastDecoder(slot_cfg).to(cfg['device'])

    ckpt     = torch.load(slot_ckpt, map_location=cfg['device'], weights_only=False)
    enc_state = ckpt.get('slot_enc', ckpt)
    dec_state = ckpt.get('slot_dec', {})

    enc.load_state_dict(enc_state, strict=False)
    if dec_state:
        dec.load_state_dict(dec_state, strict=False)
    enc.eval(); dec.eval()
    return enc, dec, slot_cfg


def load_baseline(base_ckpt: str, cfg: dict):
    from modules.encoders import DeepTransformerEncoder
    from modules.decoders import PatchDecoder

    enc = DeepTransformerEncoder(cfg).to(cfg['device'])
    dec = PatchDecoder(cfg).to(cfg['device'])

    ckpt = torch.load(base_ckpt, map_location=cfg['device'], weights_only=False)
    enc_s = ckpt.get('base_enc', {})
    dec_s = ckpt.get('base_dec', {})
    if enc_s: enc.load_state_dict(enc_s, strict=False)
    if dec_s: dec.load_state_dict(dec_s, strict=False)
    enc.eval(); dec.eval()
    return enc, dec


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_slot_inference(slot_enc, slot_dec, base_enc, base_dec,
                       dataset, cfg, slot_cfg, n_grids):
    device  = cfg['device']
    results = []
    slot_mask_store  = []  # [B, num_slots, Ph, Pw]
    slot_latent_store = [] # [B, num_slots, D]
    num_slots = slot_cfg['num_slots']

    batch_size = min(16, n_grids)
    n_iters    = (n_grids + batch_size - 1) // batch_size

    for _ in tqdm(range(n_iters), desc="Slot inference"):
        batch  = dataset.sample(batch_size)
        states = batch['state'].to(device)   # [B, 1, 30, 30]
        B      = states.shape[0]

        # ── Slot forward ──────────────────────────────────────────────────
        # Use final temperature (sharp attention) for eval
        z_slot = slot_enc({'state': states}, temperature=slot_cfg.get('slot_temp_end', 0.1))
        slots  = z_slot['latent']    # [B, num_slots, D]
        masks  = z_slot['masks']     # [B, num_slots, Ph, Pw]

        out_slot = slot_dec({'latent': slots})
        slot_recon_key = next((k for k in ('reconstruction','reconstructed_logits')
                               if k in out_slot), None)
        if slot_recon_key is None:
            # Fallback: try to get any output
            slot_recon_key = list(out_slot.keys())[0]
        slot_logits = out_slot[slot_recon_key]   # [B, 10, 30, 30]
        slot_recon  = slot_logits.argmax(1)       # [B, 30, 30]

        # ── Baseline forward ───────────────────────────────────────────────
        z_base   = base_enc({'state': states})['latent']
        out_base = base_dec({'latent': z_base})
        base_logits = out_base.get('reconstructed_logits', out_base.get('reconstruction'))
        base_recon  = base_logits.argmax(1)

        target = states.squeeze(1).long()   # [B, 30, 30]

        slot_mask_store.append(masks.cpu())
        slot_latent_store.append(slots.cpu())

        for b in range(B):
            if len(results) >= n_grids:
                break
            tgt  = target[b].cpu().numpy()
            sr   = slot_recon[b].cpu().numpy()
            br   = base_recon[b].cpu().numpy()
            results.append({
                'input':      tgt,
                'slot_recon': sr,
                'base_recon': br,
                'slot_acc':   float((tgt == sr).mean()),
                'base_acc':   float((tgt == br).mean()),
                'masks':      masks[b].cpu().numpy(),    # (num_slots, Ph, Pw)
                'slots':      slots[b].cpu().numpy(),    # (num_slots, D)
            })

    return results, slot_mask_store, slot_latent_store


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_slot_mask_gallery(results, slot_cfg, cfg, out_dir, n_show=8):
    """Panel 1: For each grid show input + one subplot per slot (attention mask)."""
    print("🎭 Slot mask gallery...")
    num_slots  = slot_cfg['num_slots']
    patch_size = slot_cfg['patch_size']
    Ph         = cfg['grid_size'] // patch_size

    n_show = min(n_show, len(results))
    # Each row: input + num_slots masks
    ncols = num_slots + 1
    nrows = n_show

    cmap_mask = plt.cm.hot

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.4, nrows * 1.6))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    slot_colors = plt.cm.tab20(np.linspace(0, 1, num_slots))

    for row, r in enumerate(results[:n_show]):
        # Input grid
        axes[row, 0].imshow(r['input'], cmap=ARC_CMAP, vmin=0, vmax=9,
                            interpolation='nearest')
        acc_s = r['slot_acc'] * 100
        axes[row, 0].set_title(f'Input\nSlot {acc_s:.0f}%', fontsize=7)
        axes[row, 0].axis('off')

        # One column per slot — upsample mask to full grid size
        for s in range(num_slots):
            ax  = axes[row, s + 1]
            msk = r['masks'][s]   # (Ph, Pw)
            # Upsample to 30×30 for overlay readability
            msk_up = np.kron(msk, np.ones((patch_size, patch_size)))[:cfg['grid_size'], :cfg['grid_size']]
            ax.imshow(r['input'], cmap=ARC_CMAP, vmin=0, vmax=9,
                      interpolation='nearest', alpha=0.35)
            ax.imshow(msk_up, cmap=cmap_mask, alpha=0.65, interpolation='nearest')
            if row == 0:
                ax.set_title(f'S{s}', fontsize=7,
                             color=to_hex(slot_colors[s]))
            ax.axis('off')

    fig.suptitle(f'Slot Attention Masks  ({num_slots} slots, patch_size={patch_size})',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, '1_slot_masks.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def to_hex(rgba):
    """Convert matplotlib RGBA to hex string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255),
                                         int(rgba[1]*255),
                                         int(rgba[2]*255))


def plot_reconstruction_compare(results, out_dir, n_show=12, cols=3):
    """Panel 2: Input | Slot Recon | Error | Baseline for n_show grids."""
    print("🎨 Reconstruction comparison...")
    n_show = min(n_show, len(results))
    rows   = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows * 4, cols, figsize=(cols * 4, rows * 5))
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_show):
        r        = results[idx]
        rb       = (idx // cols) * 4
        col      = idx % cols

        axes[rb   , col].imshow(r['input'],      cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
        axes[rb   , col].set_title('Input',       fontsize=7)
        axes[rb+1 , col].imshow(r['slot_recon'], cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
        axes[rb+1 , col].set_title(f'Slot  {r["slot_acc"]*100:.1f}%', fontsize=7)
        err = (r['input'] != r['slot_recon']).astype(float)
        axes[rb+2 , col].imshow(err, cmap='Reds',  vmin=0, vmax=1, interpolation='nearest')
        axes[rb+2 , col].set_title('Slot Errors', fontsize=7)
        axes[rb+3 , col].imshow(r['base_recon'], cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
        axes[rb+3 , col].set_title(f'Base  {r["base_acc"]*100:.1f}%', fontsize=7)

        for rb2 in range(4):
            axes[rb+rb2, col].axis('off')

    for idx in range(n_show, rows * cols):
        for rb2 in range(4):
            axes[(idx//cols)*4+rb2, idx%cols].axis('off')

    slot_mean = np.mean([r['slot_acc'] for r in results]) * 100
    base_mean = np.mean([r['base_acc'] for r in results]) * 100
    fig.suptitle(f'Slot vs Baseline Reconstruction\n'
                 f'Slot mean acc: {slot_mean:.1f}%   Baseline mean acc: {base_mean:.1f}%',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, '2_reconstruction_compare.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_slot_specialisation(results, slot_cfg, cfg, out_dir):
    """Panel 3: For each slot, what is the dominant ARC colour in its attended patches?"""
    print("🔬 Slot specialisation analysis...")
    num_slots  = slot_cfg['num_slots']
    patch_size = slot_cfg['patch_size']
    Ph = cfg['grid_size'] // patch_size

    # For each slot, collect the input colours at its highest-attention patches
    # mask: (num_slots, Ph, Pw)
    slot_colour_hist = np.zeros((num_slots, 10), dtype=np.float64)
    slot_mean_attn   = np.zeros(num_slots)

    for r in results:
        inp  = r['input']   # (30, 30)
        for s in range(num_slots):
            msk = r['masks'][s]   # (Ph, Pw)
            slot_mean_attn[s] += msk.mean()
            # Upsample mask and sample colours under it
            msk_up = np.kron(msk, np.ones((patch_size, patch_size)))[:cfg['grid_size'], :cfg['grid_size']]
            # Weight colour occurrences by attention weight
            for c in range(10):
                slot_colour_hist[s, c] += (msk_up * (inp == c)).sum()

    # Normalise
    total = slot_colour_hist.sum(1, keepdims=True)
    total = np.maximum(total, 1)
    slot_colour_hist_norm = slot_colour_hist / total
    slot_mean_attn /= len(results)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Stacked bar: colour mix per slot
    bottoms = np.zeros(num_slots)
    for c in range(10):
        heights = slot_colour_hist_norm[:, c]
        ax1.bar(range(num_slots), heights, bottom=bottoms,
                color=ARC_COLORS[c], label=ARC_NAMES[c], width=0.8)
        bottoms += heights

    ax1.set_xticks(range(num_slots))
    ax1.set_xticklabels([f'S{i}' for i in range(num_slots)])
    ax1.set_ylabel('Fraction of attended pixels')
    ax1.set_title('Slot Colour Specialisation — what does each slot attend to?', fontsize=12)
    ax1.legend(loc='upper right', ncol=5, fontsize=8)

    # Bar: mean attention weight per slot
    colours_bar = plt.cm.viridis(slot_mean_attn / slot_mean_attn.max())
    ax2.bar(range(num_slots), slot_mean_attn * 100, color=colours_bar, width=0.8)
    ax2.axhline(100 / num_slots, color='red', lw=1.5, ls='--',
                label=f'Uniform ({100/num_slots:.1f}%)')
    ax2.set_xticks(range(num_slots))
    ax2.set_xticklabels([f'S{i}' for i in range(num_slots)])
    ax2.set_ylabel('Mean attention weight (%)')
    ax2.set_title('Slot Utilisation — "firing" rate (below uniform = dead/weak slot)', fontsize=12)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, '3_slot_specialisation.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_prior_delta(slot_enc, slot_cfg, out_dir):
    """Panel 4: magnitude and direction of prior_delta for each slot."""
    print("📐 Prior delta analysis...")
    if not hasattr(slot_enc, 'prior_delta') or slot_enc.prior_delta is None:
        print("  ℹ️  No prior_delta found (model trained without it) — skipping.")
        return

    priors = slot_enc.semantic_priors.detach().cpu()  # [1, S, D]
    delta  = slot_enc.prior_delta.detach().cpu()       # [1, S, D]
    num_slots = delta.shape[1]

    delta_norm = delta.norm(dim=-1).squeeze(0).numpy()   # [S]
    prior_norm = priors.norm(dim=-1).squeeze(0).numpy()

    # Cosine similarity between delta and prior (is it moving toward or away?)
    cos_sim = F.cosine_similarity(delta.squeeze(0), priors.squeeze(0), dim=-1).numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    ax1.bar(range(num_slots), delta_norm, color='#e74c3c', width=0.8)
    ax1.set_title('||prior_delta||  — magnitude of slot position adjustment\n'
                  '(zero = slot stayed at FPS anchor; large = slot moved significantly)',
                  fontsize=11)
    ax1.set_xticks(range(num_slots)); ax1.set_xticklabels([f'S{i}' for i in range(num_slots)])
    ax1.set_ylabel('L2 norm')

    ax2.bar(range(num_slots), prior_norm, color='#3498db', width=0.8)
    ax2.set_title('||semantic_prior||  — FPS anchor magnitude (reference scale)', fontsize=11)
    ax2.set_xticks(range(num_slots)); ax2.set_xticklabels([f'S{i}' for i in range(num_slots)])
    ax2.set_ylabel('L2 norm')

    colors_cos = ['#2ecc71' if c > 0 else '#e74c3c' for c in cos_sim]
    ax3.bar(range(num_slots), cos_sim, color=colors_cos, width=0.8)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Cosine similarity(delta, prior)  — green=aligned, red=opposed', fontsize=11)
    ax3.set_xticks(range(num_slots)); ax3.set_xticklabels([f'S{i}' for i in range(num_slots)])
    ax3.set_ylabel('cos sim'); ax3.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    path = os.path.join(out_dir, '4_prior_delta.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_slot_collapse(results, out_dir):
    """Panel 5: cosine similarity matrix between mean slot representations."""
    print("🌀 Slot collapse probe...")
    # Stack all slot latents and compute mean per slot
    all_slots = np.stack([r['slots'] for r in results], axis=0)  # (N, S, D)
    mean_slots = all_slots.mean(0)  # (S, D)
    num_slots  = mean_slots.shape[0]

    # Normalise
    norms = np.linalg.norm(mean_slots, axis=1, keepdims=True) + 1e-8
    normed = mean_slots / norms
    sim_matrix = normed @ normed.T   # (S, S)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(num_slots):
        for j in range(num_slots):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(sim_matrix[i,j]) > 0.6 else 'black')
    ax.set_xticks(range(num_slots)); ax.set_xticklabels([f'S{i}' for i in range(num_slots)], fontsize=8)
    ax.set_yticks(range(num_slots)); ax.set_yticklabels([f'S{i}' for i in range(num_slots)], fontsize=8)
    ax.set_title('Slot Collapse Probe — Cosine Similarity Between Mean Slot Representations\n'
                 'Diagonal=1.0 expected. Off-diagonal > 0.9 → collapsed/duplicate slots.',
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, '5_slot_collapse.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_accuracy_comparison(results, out_dir):
    """Panel 6: accuracy histogram for slot model vs baseline."""
    print("📊 Accuracy comparison histogram...")
    slot_accs = [r['slot_acc'] * 100 for r in results]
    base_accs = [r['base_acc'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(0, 100, 31)
    ax.hist(base_accs, bins=bins, alpha=0.6, color='#3498db', label=f'Baseline  μ={np.mean(base_accs):.1f}%')
    ax.hist(slot_accs, bins=bins, alpha=0.6, color='#e74c3c', label=f'Slot model  μ={np.mean(slot_accs):.1f}%')
    ax.axvline(np.mean(slot_accs), color='#c0392b', lw=2.5)
    ax.axvline(np.mean(base_accs), color='#2980b9', lw=2.5)
    ax.set_xlabel('Per-Grid Pixel Accuracy (%)')
    ax.set_ylabel('Number of grids')
    ax.set_title('Slot Model vs Baseline — Reconstruction Accuracy Distribution', fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, '6_accuracy_comparison.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def save_phase1_summary(results, slot_enc, out_dir):
    slot_accs = [r['slot_acc'] for r in results]
    base_accs = [r['base_acc'] for r in results]

    delta_norms = None
    if hasattr(slot_enc, 'prior_delta') and slot_enc.prior_delta is not None:
        delta_norms = slot_enc.prior_delta.detach().cpu().norm(dim=-1).squeeze(0).tolist()

    summary = {
        'n_grids':              len(results),
        'slot_mean_acc':        float(np.mean(slot_accs)),
        'slot_median_acc':      float(np.median(slot_accs)),
        'base_mean_acc':        float(np.mean(base_accs)),
        'base_median_acc':      float(np.median(base_accs)),
        'slot_beats_base':      int(sum(s > b for s, b in zip(slot_accs, base_accs))),
        'prior_delta_norms':    delta_norms,
    }
    path = os.path.join(out_dir, 'phase1_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"  PHASE 1 AUDIT SUMMARY")
    print(f"{'─'*55}")
    print(f"  Grids evaluated   : {len(results)}")
    print(f"  Slot mean acc     : {np.mean(slot_accs)*100:.2f}%")
    print(f"  Baseline mean acc : {np.mean(base_accs)*100:.2f}%")
    print(f"  Slot beats base   : {summary['slot_beats_base']}/{len(results)}")
    if delta_norms:
        print(f"  Prior delta norms : {[f'{d:.3f}' for d in delta_norms]}")
    print(f"{'═'*55}")
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 slot model audit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\"""
Examples:
  # Pass a run directory — checkpoints are auto-discovered:
  python audit_phase1.py --run_dir runs/FactorizedFPS-v4_2026-04-24_04-57-55

  # Or pass explicit checkpoint paths:
  python audit_phase1.py \\
    --slot_ckpt runs/.../phase1/latest_slot_checkpoint.pth \\
    --base_ckpt runs/.../phase1/latest_base_checkpoint.pth \\
    --p0_ckpt   runs/.../phase0/latest_checkpoint.pth
"""
    )
    # Option A: pass a run directory
    parser.add_argument('--run_dir',   type=str, default=None,
                        help='Run directory — checkpoints are auto-discovered inside it')
    # Option B: pass individual checkpoint paths (overrides --run_dir)
    parser.add_argument('--slot_ckpt', type=str, default=None,
                        help='Phase 1 slot checkpoint  (latest_slot_checkpoint.pth)')
    parser.add_argument('--base_ckpt', type=str, default=None,
                        help='Phase 1 baseline checkpoint (latest_base_checkpoint.pth)')
    parser.add_argument('--p0_ckpt',   type=str, default=None,
                        help='Phase 0 checkpoint (latest_checkpoint.pth)')
    parser.add_argument('--out',       type=str,
                        default='evaluation_reports/phase1_audit')
    parser.add_argument('--n_grids',   type=int, default=150)
    parser.add_argument('--n_show',    type=int, default=8)
    parser.add_argument('--data_path', type=str, default='arc_data/re-arc')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Resolve checkpoint paths ──────────────────────────────────────────────
    if args.slot_ckpt and args.base_ckpt and args.p0_ckpt:
        slot_ckpt = args.slot_ckpt
        base_ckpt = args.base_ckpt
        p0_ckpt   = args.p0_ckpt
    elif args.run_dir:
        slot_ckpt, base_ckpt, p0_ckpt = find_phase1_checkpoints(args.run_dir)
    else:
        # Try to find the most recent run automatically
        runs_dir = 'runs'
        if os.path.exists(runs_dir):
            all_runs = sorted([
                os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d))
            ])
            for run in reversed(all_runs):
                cand = os.path.join(run, 'phase1', 'latest_slot_checkpoint.pth')
                if os.path.exists(cand):
                    print(f"  Auto-using most recent run with Phase 1 checkpoint: {run}")
                    slot_ckpt, base_ckpt, p0_ckpt = find_phase1_checkpoints(run)
                    break
            else:
                parser.error(
                    "No Phase 1 checkpoint found in any run.\n"
                    "Pass --run_dir <path> or --slot_ckpt / --base_ckpt / --p0_ckpt.")
        else:
            parser.error("Pass --run_dir <path> or --slot_ckpt / --base_ckpt / --p0_ckpt.")

    print(f"{'═'*55}")
    print(f"  Phase 1 Slot Audit")
    print(f"  Slot ckpt  : {args.slot_ckpt}")
    print(f"  Base ckpt  : {args.base_ckpt}")
    print(f"  P0 ckpt    : {args.p0_ckpt}")
    print(f"{'═'*55}\n")

    # ── Detect architecture ──────────────────────────────────────────────────
    cfg         = detect_arch_from_p0(args.p0_ckpt)
    cfg['device'] = device
    slot_params = detect_slot_params(args.slot_ckpt)

    # ── Load models ──────────────────────────────────────────────────────────
    print("⚙️  Loading models...")
    slot_enc, slot_dec, slot_cfg = load_slot_model(args.slot_ckpt, cfg, slot_params)
    base_enc, base_dec           = load_baseline(args.base_ckpt, cfg)

    # ── Dataset ──────────────────────────────────────────────────────────────
    from arc_data.rearc_dataset import ReARCDataset
    dataset = ReARCDataset(data_path=args.data_path)

    # ── Inference ────────────────────────────────────────────────────────────
    results, _, _ = run_slot_inference(
        slot_enc, slot_dec, base_enc, base_dec,
        dataset, cfg, slot_cfg, n_grids=args.n_grids)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📊 Generating panels...")
    plot_slot_mask_gallery(results, slot_cfg, cfg, args.out, n_show=args.n_show)
    plot_reconstruction_compare(results, args.out, n_show=min(12, args.n_show * 2))
    plot_slot_specialisation(results, slot_cfg, cfg, args.out)
    plot_prior_delta(slot_enc, slot_cfg, args.out)
    plot_slot_collapse(results, args.out)
    plot_accuracy_comparison(results, args.out)
    save_phase1_summary(results, slot_enc, args.out)

    print(f"\n✅ Phase 1 audit complete → {args.out}/")
    print(f"   scp -r eet242799@10.225.67.239:~/development/Model-Jepa/{args.out} ~/Desktop/")


if __name__ == '__main__':
    main()
