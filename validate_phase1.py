"""
validate_phase1.py — Structural Integrity Audit for Phase 1 Training
======================================================================
Checks 4 specific claims about the Phase 1 checkpoint:

  [CLAIM 1] Codebook Drift:
    'The Phase 0 codebook primitives are corrupted during Phase 1
     because codebook_shape/color are registered as nn.Parameter.'

  [CLAIM 2] Encoder Re-use:
    'The Phase 0 PatchTransformerEncoder is NOT reused in Phase 1.
     Phase 1 patches images from scratch with a random new encoder.'
    
  [CLAIM 3] Slot Collapse:
    'If claims 1 & 2 are true, the slot assignments may be trivially
     distributed (all patches map to one slot) instead of object-segmented.'

  [CLAIM 4] Reality Check:
    'The claimed 90% Pixel Accuracy is genuine learning, NOT mock-data
     overfitting, by testing on held-out real ARC grids.'

Usage:
  python validate_phase1.py \\
    --p0_vq     runs/.../phase0/frozen_vq_codebook.pth \\
    --p1_slot   runs/.../phase1/latest_slot_checkpoint.pth \\
    --data_path arc_data/re-arc/tasks

Output:
  A report in evaluation_reports/phase1_validation/
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.semantic_encoders import SemanticSlotEncoder
from modules.vq import FactorizedVectorQuantizer
from arc_data.rearc_dataset import ReARCDataset

# ── Default CFG (must match training config) ─────────────────────────────────
CFG = {
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'in_channels':     10,
    'patch_size':      5,
    'hidden_dim':      256,
    'latent_dim':      256,
    'pose_dim':        64,
    'vocab_size':      10,
    'grid_size':       30,
    'num_slots':       12,     # ← match CFG in train_phase0.py
    'slot_iters':      7,
    'slot_temperature': 0.05,
    'slot_temp_start': 1.0,
    'slot_temp_end':   0.05,
    'slot_temp_anneal':400,
    'num_shape_codes': 1024,   # ← match CFG in train_phase0.py
    'num_color_codes': 32,     # ← match CFG in train_phase0.py
    'commitment_cost': 0.25,
    'focal_gamma':     2.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(A, B):
    """Pairwise cosine similarity between rows of A and B. Returns [len(A), len(B)]."""
    A_n = F.normalize(A.float(), dim=-1)
    B_n = F.normalize(B.float(), dim=-1)
    return torch.mm(A_n, B_n.T)


def vector_displacement(A, B):
    """L2 distance between corresponding rows of A and B."""
    return (A.float() - B.float()).norm(dim=-1)


def pct_bar(label, value, lo=0.0, hi=1.0, width=40):
    pct = (value - lo) / (hi - lo)
    filled = int(pct * width)
    bar = '█' * filled + '░' * (width - filled)
    return f"  {label:<40} [{bar}] {value:.4f}"


def sep(char='─', n=80):
    return char * n


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 1: Codebook Drift
# ─────────────────────────────────────────────────────────────────────────────

def audit_codebook_drift(p0_vq_path: str, p1_slot_ckpt: dict, out_dir: str, device: str):
    """
    Loads the frozen Phase 0 VQ codebook and compares it against
    the codebook stored inside the Phase 1 slot encoder checkpoint.
    """
    print(f"\n{sep()}")
    print("CLAIM 1 — Codebook Drift Measurement")
    print(f"{sep()}")

    # Load Phase 0 VQ
    p0_state = torch.load(p0_vq_path, map_location=device, weights_only=True)

    # Extract P0 codebooks
    if 'embedding_shape.weight' in p0_state:
        p0_shape = p0_state['embedding_shape.weight'].cpu()
        p0_color = p0_state['embedding_color.weight'].cpu()
    else:
        print("  ❌ Could not find 'embedding_shape.weight' in P0 VQ checkpoint.")
        return None

    # Extract P1 codebooks from slot encoder state dict
    slot_state = p1_slot_ckpt.get('slot_enc', p1_slot_ckpt)
    if 'codebook_shape' in slot_state:
        p1_shape = slot_state['codebook_shape'].cpu()
        p1_color = slot_state['codebook_color'].cpu()
    else:
        print("  ❌ Could not find 'codebook_shape' in P1 slot encoder checkpoint.")
        print("     Keys available:", [k for k in slot_state.keys() if 'code' in k.lower()])
        return None

    # Measure shape drift
    shape_l2   = vector_displacement(p0_shape, p1_shape)
    shape_cos  = cosine_similarity_matrix(p0_shape, p1_shape).diag()  # self-similarity
    color_l2   = vector_displacement(p0_color, p1_color)
    color_cos  = cosine_similarity_matrix(p0_color, p1_color).diag()

    shape_drift_mean = shape_l2.mean().item()
    shape_drift_max  = shape_l2.max().item()
    color_drift_mean = color_l2.mean().item()

    print(f"\n  Shape Codebook ({p0_shape.shape[0]} codes × {p0_shape.shape[1]}D):")
    print(f"    Mean L2 Drift       : {shape_drift_mean:.4f}")
    print(f"    Max  L2 Drift       : {shape_drift_max:.4f}")
    print(f"    Mean Cosine Self-Sim: {shape_cos.mean().item():.4f}  (1.0 = identical)")
    print(f"    Min  Cosine Self-Sim: {shape_cos.min().item():.4f}")

    print(f"\n  Color Codebook ({p0_color.shape[0]} codes × {p0_color.shape[1]}D):")
    print(f"    Mean L2 Drift       : {color_drift_mean:.4f}")
    print(f"    Mean Cosine Self-Sim: {color_cos.mean().item():.4f}")

    # Verdict
    DRIFT_THRESHOLD = 0.05  # If mean L2 > 0.05, codebook has drifted meaningfully
    drifted = shape_drift_mean > DRIFT_THRESHOLD
    print(f"\n  {'🔴 CLAIM CONFIRMED' if drifted else '🟢 CLAIM REFUTED'}: "
          f"Shape codebook drifted by mean L2={shape_drift_mean:.4f} "
          f"(threshold={DRIFT_THRESHOLD})")

    # Plot drift histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Claim 1: Codebook Drift Audit (P0 → P1)", fontsize=14, fontweight='bold')

    axes[0].hist(shape_l2.numpy(), bins=50, color='crimson', alpha=0.7, edgecolor='black')
    axes[0].axvline(shape_drift_mean, color='gold', lw=2, linestyle='--', label=f'Mean={shape_drift_mean:.4f}')
    axes[0].axvline(DRIFT_THRESHOLD, color='cyan', lw=2, linestyle=':', label=f'Threshold={DRIFT_THRESHOLD}')
    axes[0].set_title("Shape Code L2 Drift: P0 → P1")
    axes[0].set_xlabel("L2 Distance")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    scatter_x = shape_cos.numpy()
    axes[1].scatter(range(len(scatter_x)), scatter_x, c=shape_l2.numpy(),
                    cmap='RdYlGn_r', s=8, alpha=0.7)
    axes[1].set_title("Per-code Cosine Self-Similarity (color = L2 drift)")
    axes[1].set_xlabel("Code Index")
    axes[1].set_ylabel("Cosine Similarity (1.0 = no drift)")
    cm = plt.cm.RdYlGn_r
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=shape_l2.min(), vmax=shape_l2.max()))
    plt.colorbar(sm, ax=axes[1], label='L2 Drift')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_codebook_drift.png'), dpi=150)
    plt.close()
    print(f"  📊 Plot: {os.path.join(out_dir, '1_codebook_drift.png')}")

    return {
        'shape_drift_mean': shape_drift_mean,
        'shape_drift_max': shape_drift_max,
        'shape_cos_mean': shape_cos.mean().item(),
        'color_drift_mean': color_drift_mean,
        'drifted': drifted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 2: Encoder Re-use (Weight fingerprint comparison)
# ─────────────────────────────────────────────────────────────────────────────

def audit_encoder_reuse(p0_full_ckpt_path: str, p1_slot_ckpt: dict, out_dir: str, device: str):
    """
    Compares the Phase 0 patch_embed weights against the Phase 1
    SemanticSlotEncoder patch_embed weights.
    If they share no weights, the encoder was NOT reused.
    """
    print(f"\n{sep()}")
    print("CLAIM 2 — Encoder Re-use Check")
    print(f"{sep()}")

    if not p0_full_ckpt_path or not os.path.exists(p0_full_ckpt_path):
        print("  ⚠️  Phase 0 full checkpoint (with model weights) not found.")
        print("     Provide --p0_full to check encoder weight fingerprints.")
        print("     Skipping this test.")
        return None

    p0_state = torch.load(p0_full_ckpt_path, map_location=device, weights_only=False)
    p0_model = p0_state.get('model', p0_state)

    # Phase 0 PatchTransformerEncoder uses 'encoder.patch_embed.weight'
    p0_embed_key = 'encoder.patch_embed.weight'
    if p0_embed_key not in p0_model:
        print(f"  ⚠️  Key '{p0_embed_key}' not found in P0 model.")
        print("      Available keys:", [k for k in p0_model.keys() if 'embed' in k])
        return None
    p0_embed = p0_model[p0_embed_key].cpu().float()

    # Phase 1 SemanticSlotEncoder uses 'patch_embed.weight'
    slot_state = p1_slot_ckpt.get('slot_enc', p1_slot_ckpt)
    p1_embed_key = 'patch_embed.weight'
    if p1_embed_key not in slot_state:
        print(f"  ⚠️  Key '{p1_embed_key}' not found in P1 slot encoder.")
        print("      Available keys:", [k for k in slot_state.keys() if 'embed' in k])
        return None
    p1_embed = slot_state[p1_embed_key].cpu().float()

    print(f"\n  P0 patch_embed shape : {tuple(p0_embed.shape)}")
    print(f"  P1 patch_embed shape : {tuple(p1_embed.shape)}")

    if p0_embed.shape != p1_embed.shape:
        print("  🔴 SHAPES DIFFER — Encoder was definitely NOT reused (different architecture).")
        return {'reused': False, 'same_shape': False}

    # If same shape, check if weights are identical or close
    l2_diff   = (p0_embed - p1_embed).norm().item()
    max_diff  = (p0_embed - p1_embed).abs().max().item()
    are_same  = max_diff < 1e-6

    print(f"\n  L2 difference in patch_embed weights : {l2_diff:.6f}")
    print(f"  Max absolute difference              : {max_diff:.8f}")
    print(f"\n  {'🟢 CLAIM REFUTED — weights are identical (encoder IS shared)' if are_same else '🔴 CLAIM CONFIRMED — weights diverged; encoder NOT shared from P0'}")

    # Visualize filter comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Claim 2: P0 vs P1 Patch Embed Filter Comparison", fontsize=14)
    im0 = axes[0].imshow(p0_embed.view(p0_embed.shape[0], -1).numpy(), aspect='auto', cmap='RdBu')
    axes[0].set_title("Phase 0 Encoder: patch_embed.weight")
    axes[0].set_xlabel("Kernel Values (flattened)")
    axes[0].set_ylabel("Output Channels (256)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(p1_embed.view(p1_embed.shape[0], -1).numpy(), aspect='auto', cmap='RdBu')
    axes[1].set_title("Phase 1 Encoder: patch_embed.weight")
    axes[1].set_xlabel("Kernel Values (flattened)")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_encoder_reuse.png'), dpi=150)
    plt.close()
    print(f"  📊 Plot: {os.path.join(out_dir, '2_encoder_reuse.png')}")

    return {'reused': are_same, 'same_shape': True, 'l2_diff': l2_diff}


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 3: Slot Collapse Check
# ─────────────────────────────────────────────────────────────────────────────

def audit_slot_quality(slot_enc, data_path: str, out_dir: str, device: str, n_samples: int = 32):
    """
    Runs inference on real ARC grids and checks if slots are:
    (a) diverse (different slots capture different things)
    (b) spatially non-trivial (not all mass on one slot)
    """
    print(f"\n{sep()}")
    print("CLAIM 3 — Slot Quality & Object Segmentation")
    print(f"{sep()}")

    try:
        dataset = ReARCDataset(data_path=data_path)
    except Exception as e:
        print(f"  ❌ Could not load dataset from {data_path}: {e}")
        return None

    slot_enc.eval()
    all_entropies = []
    all_max_slot_masses = []
    slot_usage_counts = torch.zeros(slot_enc.num_slots)

    with torch.no_grad():
        for _ in range(n_samples // 4):
            batch = dataset.sample(4, split='val')
            state = batch['state'].to(device)
            out = slot_enc({'state': state}, temperature=0.1)

            # masks: [B, K, H, W] — how much each slot "owns" each pixel
            masks = out['masks']  # Already softmax'd

            # 1. Slot entropy per image: high entropy = slots uniformly own pixels (good)
            # Low entropy = one slot owns everything (collapse)
            B, K, H, W = masks.shape
            mask_flat = masks.view(B, K, -1)  # [B, K, N_pixels]
            # Per-slot mass
            slot_mass = mask_flat.sum(dim=-1) / (H * W)  # [B, K]
            # Shannon entropy across slots
            entropy = -(slot_mass * (slot_mass + 1e-10).log()).sum(dim=-1)  # [B]
            all_entropies.extend(entropy.cpu().numpy().tolist())

            # 2. Max slot mass: if one slot always gets > 80% of pixels → collapse
            max_mass = slot_mass.max(dim=-1).values  # [B]
            all_max_slot_masses.extend(max_mass.cpu().numpy().tolist())

            # 3. Which slot wins most often?
            dominant_slots = slot_mass.argmax(dim=-1)  # [B]
            for s in dominant_slots.cpu():
                slot_usage_counts[s] += 1

    max_entropy = float(np.log(slot_enc.num_slots))
    mean_entropy = float(np.mean(all_entropies))
    mean_max_mass = float(np.mean(all_max_slot_masses))
    entropy_ratio = mean_entropy / max_entropy

    n_active_slots = (slot_usage_counts > 0).sum().item()
    dominant_slot_usage = slot_usage_counts.max().item() / slot_usage_counts.sum().item()

    print(f"\n  Slot Entropy (per image):")
    print(pct_bar("Mean Entropy / Max Possible", entropy_ratio, 0, 1))
    print(f"    {'🟢 Diverse slots' if entropy_ratio > 0.5 else '🔴 SLOT COLLAPSE detected'}: "
          f"{mean_entropy:.3f} / {max_entropy:.3f}")

    print(f"\n  Dominant Slot Pixel Mass (lower = better):")
    print(pct_bar("Mean max single-slot pixel ownership", mean_max_mass, 0, 1))
    print(f"    {'🔴 One slot dominates' if mean_max_mass > 0.5 else '🟢 Mass spread across slots'}: "
          f"{mean_max_mass*100:.1f}% of pixels owned by single slot")

    print(f"\n  Slot Utilization:")
    print(f"    Active slots     : {n_active_slots} / {slot_enc.num_slots}")
    print(f"    {'🔴 SLOT COLLAPSE' if n_active_slots < 3 else '🟢 Multiple slots used'}: "
          f"Dominant slot used {dominant_slot_usage*100:.1f}% of the time as the 'winning' slot")

    # Plot masks for a few samples
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("Claim 3: Slot Masks on Validation Grids", fontsize=14)

    batch = dataset.sample(4, split='val')
    state = batch['state'].to(device)
    with torch.no_grad():
        out = slot_enc({'state': state}, temperature=0.1)

    masks = out['masks'].cpu()  # [B, K, H, W]
    colors = plt.cm.tab20(np.linspace(0, 1, slot_enc.num_slots))

    for b in range(min(4, masks.shape[0])):
        ax_orig = fig.add_subplot(2, 4, b + 1)
        ax_mask = fig.add_subplot(2, 4, b + 5)

        grid = state[b, 0].cpu().numpy()
        ax_orig.imshow(grid, cmap='tab10', vmin=0, vmax=9)
        ax_orig.set_title(f"Sample {b+1}: Grid")
        ax_orig.axis('off')

        # Composite mask: color each pixel by its dominant slot
        dominant = masks[b].argmax(dim=0).numpy()  # [H, W]
        composite = np.zeros((*dominant.shape, 3))
        for s in range(slot_enc.num_slots):
            composite[dominant == s] = colors[s][:3]
        ax_mask.imshow(composite)
        ax_mask.set_title(f"Sample {b+1}: Slot Segmentation")
        ax_mask.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_slot_segmentation.png'), dpi=150)
    plt.close()
    print(f"  📊 Plot: {os.path.join(out_dir, '3_slot_segmentation.png')}")

    return {
        'mean_entropy': mean_entropy,
        'entropy_ratio': entropy_ratio,
        'mean_max_mass': mean_max_mass,
        'n_active_slots': n_active_slots,
        'collapsed': entropy_ratio < 0.3 or n_active_slots < 3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 4: Reality Check — Is 90% Accuracy Genuine?
# ─────────────────────────────────────────────────────────────────────────────

def audit_accuracy_reality(slot_enc, data_path: str, out_dir: str, device: str, n_batches: int = 20):
    """
    Checks if performance is consistent across different grid sizes
    and color distributions. Mock data would have perfectly uniform patterns.
    """
    from modules.semantic_decoders import SemanticDecoder

    print(f"\n{sep()}")
    print("CLAIM 4 — Reality Check: Is 90% Accuracy Genuine?")
    print(f"{sep()}")

    try:
        dataset = ReARCDataset(data_path=data_path)
    except Exception as e:
        print(f"  ❌ Could not load dataset: {e}")
        return None

    cfg_local = CFG.copy()

    slot_dec = SemanticDecoder(cfg_local).to(device)
    # Note: SemanticDecoder doesn't need loading — we're testing slot_enc quality
    # by checking accuracy WITHOUT the decoder (using just codebook similarity as a proxy)

    slot_enc.eval()

    # Measure accuracy split by grid area (small vs large grids)
    accs_small = []  # grids where most pixels are zero (sparse)
    accs_large = []  # grids where most pixels are non-zero (dense)

    with torch.no_grad():
        for _ in range(n_batches):
            batch = dataset.sample(8, split='val')
            state = batch['state'].to(device)

            out = slot_enc({'state': state}, temperature=0.1)
            slots = out['latent']  # [B, K, 320]

            # Measure slot diversity within each sample
            for b in range(state.shape[0]):
                s = slots[b]  # [K, 320]
                # Compute pairwise slot cosine similarity
                sim = F.cosine_similarity(s.unsqueeze(0), s.unsqueeze(1), dim=-1)  # [K, K]
                # Off-diagonal mean = inter-slot similarity (low = diverse slots)
                off_diag = sim[~torch.eye(s.shape[0], dtype=bool, device=device)].mean().item()

                # Classify grid as sparse/dense
                grid = state[b, 0]
                nonzero_ratio = (grid > 0).float().mean().item()
                if nonzero_ratio < 0.3:
                    accs_small.append(off_diag)
                else:
                    accs_large.append(off_diag)

    mean_sim_sparse = float(np.mean(accs_small)) if accs_small else float('nan')
    mean_sim_dense  = float(np.mean(accs_large)) if accs_large else float('nan')

    print(f"\n  Inter-slot Cosine Similarity (lower = more diverse = better):")
    print(f"    Sparse grids (few colors):   {mean_sim_sparse:.4f}")
    print(f"    Dense grids  (many colors):  {mean_sim_dense:.4f}")
    print(f"\n  {'🟢 Slots differ by grid density (genuine learning)' if abs(mean_sim_sparse - mean_sim_dense) > 0.02 else '🟡 Similar performance across densities'}")

    # Distribution of non-zero pixel ratios
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Claim 4: Grid Complexity vs Slot Diversity", fontsize=14)

    total_samples = len(accs_small) + len(accs_large)
    axes[0].bar(['Sparse\nGrids', 'Dense\nGrids'],
                [len(accs_small)/total_samples, len(accs_large)/total_samples],
                color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    axes[0].set_title("Dataset Composition: Sparse vs Dense Grids")
    axes[0].set_ylabel("Fraction of Val Set")

    axes[1].bar(['Sparse\nSlot Sim', 'Dense\nSlot Sim'],
                [mean_sim_sparse, mean_sim_dense],
                color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    axes[1].axhline(0.1, color='red', linestyle='--', label='Collapse threshold (>0.9)')
    axes[1].set_title("Inter-Slot Cosine Similarity by Grid Type")
    axes[1].set_ylabel("Cosine Similarity (lower = more diverse)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_accuracy_reality.png'), dpi=150)
    plt.close()
    print(f"  📊 Plot: {os.path.join(out_dir, '4_accuracy_reality.png')}")

    return {
        'inter_slot_sim_sparse': mean_sim_sparse,
        'inter_slot_sim_dense':  mean_sim_dense,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: dict, out_dir: str):
    report_path = os.path.join(out_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n{sep('═')}")
    print("SUMMARY")
    print(f"{sep('═')}")

    drift = results.get('codebook_drift', {})
    enc   = results.get('encoder_reuse', {})
    slot  = results.get('slot_quality', {})

    if drift:
        status = "🔴 DRIFTED" if drift.get('drifted') else "🟢 STABLE"
        print(f"  [Claim 1] Codebook Drift    : {status} — mean L2={drift.get('shape_drift_mean', 0):.4f}")
    if enc:
        status = "🟢 SHARED" if enc.get('reused') else "🔴 NOT REUSED"
        print(f"  [Claim 2] Encoder Re-use    : {status}")
    if slot:
        status = "🔴 COLLAPSED" if slot.get('collapsed') else "🟢 DIVERSE"
        print(f"  [Claim 3] Slot Quality      : {status} — entropy ratio={slot.get('entropy_ratio', 0):.3f}")

    print(f"\n  Full JSON report: {report_path}")
    print(f"  All plots in   : {out_dir}/")
    print(f"{sep('═')}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Structural Integrity Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--p0_vq',    required=True,
                        help='Path to frozen_vq_codebook.pth from Phase 0')
    parser.add_argument('--p0_full',  default=None,
                        help='Path to latest_checkpoint.pth (full model) from Phase 0 — needed for Claim 2')
    parser.add_argument('--p1_slot',  required=True,
                        help='Path to latest_slot_checkpoint.pth from Phase 1')
    parser.add_argument('--data_path', default='arc_data/re-arc/tasks',
                        help='Path to Re-ARC tasks directory for live inference')
    parser.add_argument('--out', default='evaluation_reports/phase1_validation',
                        help='Output directory for plots and report')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    print(f"\n{'═'*80}")
    print("  Phase 1 Structural Integrity Audit")
    print(f"{'═'*80}")
    print(f"  P0 VQ Checkpoint   : {args.p0_vq}")
    print(f"  P1 Slot Checkpoint : {args.p1_slot}")
    print(f"  Data              : {args.data_path}")
    print(f"  Device            : {device}")
    print(f"  Output            : {args.out}")

    # Load Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint...")
    p1_ckpt = torch.load(args.p1_slot, map_location=device, weights_only=False)
    print(f"  Keys in checkpoint: {list(p1_ckpt.keys())}")

    # Reconstruct slot encoder
    cfg = CFG.copy()
    cfg['device'] = device
    slot_enc = SemanticSlotEncoder(cfg)

    # Reconstruct the VQ for codebook injection
    p0_vq = FactorizedVectorQuantizer(
        num_shape_codes=cfg['num_shape_codes'],
        num_color_codes=cfg['num_color_codes'],
        embedding_dim=cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
    )
    p0_vq.load_state_dict(torch.load(args.p0_vq, map_location='cpu', weights_only=True))
    slot_enc.inject_codebook(p0_vq)

    # Load Phase 1 weights
    slot_state = p1_ckpt.get('slot_enc', p1_ckpt)
    missing, unexpected = slot_enc.load_state_dict(slot_state, strict=False)
    print(f"  ✅ Slot encoder loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")
    slot_enc = slot_enc.to(device).eval()

    # Run all claims
    results = {}
    results['codebook_drift'] = audit_codebook_drift(args.p0_vq, p1_ckpt, args.out, device)
    results['encoder_reuse']  = audit_encoder_reuse(args.p0_full, p1_ckpt, args.out, device)
    results['slot_quality']   = audit_slot_quality(slot_enc, args.data_path, args.out, device)
    results['accuracy_check'] = audit_accuracy_reality(slot_enc, args.data_path, args.out, device)

    write_report(results, args.out)


if __name__ == '__main__':
    main()
