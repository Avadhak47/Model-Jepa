"""
Codebook Capacity Audit — Phoenix Architecture

Tests whether the NMF codebook is "good enough" to serve as a dictionary
BEFORE any encoder/decoder training. This answers the question:

  "Can 1024 NMF atoms reconstruct all ARC objects with K slots?"

Method:
  For each object → find K nearest atoms by L2 distance in NMF space
  → reconstruct via weighted combination → measure pixel accuracy

If this gives >80% accuracy with K=5, the codebook is fine and the
problem is purely encoder/decoder.
If this gives <50%, the codebook needs more atoms or better quality.

Usage:
    python tools/audit_codebook.py
    python tools/audit_codebook.py --k 3      # test with K=3 slots
    python tools/audit_codebook.py --k 10     # test with K=10 slots
"""

import sys, os, argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.extract_arc_objects import PrimitiveDataset

ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
              '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
cmap = ListedColormap(ARC_COLORS)

# ── Helper: reconstruct from K nearest atoms ──────────────────────────────────

def reconstruct_from_atoms(obj_flat, atoms, k):
    """
    obj_flat: [2700]  — flattened object (one-hot color × 15 × 15 × 12? no — raw pixel grid)
    atoms:    [1024, 2700]  — NMF basis
    k:        int  — number of atoms to use

    Returns:
        recon_grid: [15, 15]  — reconstructed color grid (argmax of summed atoms)
        best_atom_ids: [k]    — which atoms were selected
        px_acc: float         — pixel accuracy vs original
    """
    # L2 distance to all atoms
    dists = ((atoms - obj_flat.unsqueeze(0)) ** 2).sum(dim=1)   # [1024]
    best_k = dists.topk(k, largest=False).indices                # [k] — nearest atoms

    # Weighted reconstruction: sum of K nearest atoms
    # Weight each atom inversely by its distance (soft combination)
    best_dists = dists[best_k]
    # Use non-negative weights proportional to inverse distance
    weights = 1.0 / (best_dists + 1e-8)
    weights = weights / weights.sum()

    recon = (atoms[best_k] * weights.unsqueeze(1)).sum(dim=0)    # [2700]
    recon_3d = recon.view(15, 15, 12)
    # Color logits from first 10 channels per pixel
    color_logits = recon_3d[:, :, :10]   # [15, 15, 10]
    recon_grid   = color_logits.argmax(dim=-1)   # [15, 15]

    return recon_grid, best_k, weights

# ── Main audit ────────────────────────────────────────────────────────────────

def audit_codebook(k=5, n_visualize=8, library_path='arc_data/primitive_library.pt',
                   basis_path='arc_data/arc_basis_nmf_1024.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load codebook
    data  = torch.load(basis_path, weights_only=False)
    atoms = data['basis'].to(device)   # [1024, 2700]
    print(f"Codebook: {atoms.shape[0]} atoms × {atoms.shape[1]} dims")

    # Load dataset
    dataset = PrimitiveDataset(library_path)
    print(f"Dataset:  {len(dataset)} objects")

    # ── Run audit on the full dataset ──
    all_accs  = []
    worst_accs, best_accs = [], []
    worst_data, best_data = [], []

    print(f"\nRunning K={k} nearest-atom reconstruction on all objects...")

    for i in range(len(dataset)):
        item  = dataset[i]
        state = item['state'].squeeze(0).long()   # [15, 15] int grid

        # Convert to one-hot [15, 15, 10] then flatten to [2700]... but NMF was
        # trained on a different representation. Use raw pixel values × color as
        # proxy: one-hot color [10] + 2-channel position mask [2] = 12 channels
        # Replicate what analyze_arc_basis.py did
        oh   = F.one_hot(state, num_classes=10).float()          # [15, 15, 10]
        mask = (state > 0).float().unsqueeze(-1)                  # [15, 15, 1]
        # 2-channel shape: binary mask + normalized intensity
        pos  = torch.cat([mask, mask * state.float().unsqueeze(-1) / 9.0], dim=-1)
        obj_12ch = torch.cat([oh, pos], dim=-1)                   # [15, 15, 12]
        obj_flat = obj_12ch.view(-1).to(device)                   # [2700]

        recon_grid, best_k, _ = reconstruct_from_atoms(obj_flat, atoms, k)

        px_acc = (recon_grid.cpu() == state).float().mean().item() * 100
        all_accs.append(px_acc)

        # Track worst and best examples for visualization
        if len(worst_accs) < n_visualize or px_acc < min(worst_accs):
            worst_accs.append(px_acc)
            worst_data.append((state.numpy(), recon_grid.cpu().numpy(), best_k.cpu().numpy(), px_acc))
            if len(worst_accs) > n_visualize:
                max_idx = worst_accs.index(max(worst_accs))
                worst_accs.pop(max_idx)
                worst_data.pop(max_idx)

        if len(best_accs) < n_visualize or px_acc > max(best_accs):
            best_accs.append(px_acc)
            best_data.append((state.numpy(), recon_grid.cpu().numpy(), best_k.cpu().numpy(), px_acc))
            if len(best_accs) > n_visualize:
                min_idx = best_accs.index(min(best_accs))
                best_accs.pop(min_idx)
                best_data.pop(min_idx)

    # ── Statistics ──
    all_accs  = np.array(all_accs)
    print(f"\n{'='*55}")
    print(f"  CODEBOOK CAPACITY AUDIT  (K={k} atoms per object)")
    print(f"{'='*55}")
    print(f"  Mean pixel accuracy  : {all_accs.mean():.1f}%")
    print(f"  Median accuracy      : {np.median(all_accs):.1f}%")
    print(f"  Best  reconstruction : {all_accs.max():.1f}%")
    print(f"  Worst reconstruction : {all_accs.min():.1f}%")
    print(f"  Objects >80% acc     : {(all_accs > 80).sum()} / {len(all_accs)}")
    print(f"  Objects >50% acc     : {(all_accs > 50).sum()} / {len(all_accs)}")
    print(f"  Objects <20% acc     : {(all_accs < 20).sum()} / {len(all_accs)}")
    print(f"{'='*55}")

    # Verdict
    mean = all_accs.mean()
    if mean >= 75:
        verdict = "✅  CODEBOOK IS SUFFICIENT — problem is encoder/decoder"
    elif mean >= 50:
        verdict = "⚠️  CODEBOOK IS MARGINAL — consider more atoms or larger K"
    else:
        verdict = "❌  CODEBOOK IS INSUFFICIENT — need better NMF or larger codebook"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*55}\n")

    # ── Visualize best and worst ──
    fig, axes = plt.subplots(2 * n_visualize, 2, figsize=(6, 3 * 2 * n_visualize))
    fig.suptitle(f"Codebook Audit (K={k})  |  Mean Acc={all_accs.mean():.1f}%", fontsize=13)

    for row, (orig, recon, atom_ids, acc) in enumerate(best_data[:n_visualize]):
        axes[row, 0].imshow(orig,  cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[row, 0].set_title(f"Original (BEST #{row+1})", fontsize=7)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(recon, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[row, 1].set_title(f"Recon {acc:.0f}%  atoms:{atom_ids}", fontsize=6)
        axes[row, 1].axis('off')

    for row, (orig, recon, atom_ids, acc) in enumerate(worst_data[:n_visualize]):
        r = n_visualize + row
        axes[r, 0].imshow(orig,  cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[r, 0].set_title(f"Original (WORST #{row+1})", fontsize=7)
        axes[r, 0].axis('off')
        axes[r, 1].imshow(recon, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[r, 1].set_title(f"Recon {acc:.0f}%  atoms:{atom_ids}", fontsize=6)
        axes[r, 1].axis('off')

    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    out = f'artifacts/codebook_audit_k{k}.png'
    plt.savefig(out, dpi=100)
    print(f"  Visualization saved → {out}")
    print(f"  scp to Mac: scp eet242799@10.225.67.239:~/development/Model-Jepa/{out} ~/Downloads/")

    # Also test K=1,2,5,10 to see how accuracy scales
    print("\n── Accuracy vs K (how many atoms needed?) ──")
    for test_k in [1, 2, 3, 5, 8, 10, 20]:
        sample_accs = []
        for i in range(0, min(500, len(dataset)), 1):
            item   = dataset[i]
            state  = item['state'].squeeze(0).long()
            oh     = F.one_hot(state, num_classes=10).float()
            mask   = (state > 0).float().unsqueeze(-1)
            pos    = torch.cat([mask, mask * state.float().unsqueeze(-1) / 9.0], dim=-1)
            obj_flat = torch.cat([oh, pos], dim=-1).view(-1).to(device)
            recon_g, _, _ = reconstruct_from_atoms(obj_flat, atoms, test_k)
            sample_accs.append((recon_g.cpu() == state).float().mean().item() * 100)
        print(f"  K={test_k:2d} → {np.mean(sample_accs):.1f}% mean acc  (sample of {len(sample_accs)})")

    return all_accs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',       type=int, default=5,  help='Number of atoms per object')
    parser.add_argument('--vis',     type=int, default=8,  help='Number of examples to visualize')
    parser.add_argument('--library', type=str, default='arc_data/primitive_library.pt')
    parser.add_argument('--basis',   type=str, default='arc_data/arc_basis_nmf_1024.pt')
    args = parser.parse_args()

    audit_codebook(k=args.k, n_visualize=args.vis,
                   library_path=args.library, basis_path=args.basis)
