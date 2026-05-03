"""
Codebook Capacity Audit — Phoenix v6 (High Fidelity)
Aligned with 2250-dim (10-channel) weighted encoding.
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

def denoise_grid(grid):
    """Matches the geometric purity filter in analyze_arc_basis.py."""
    h, w = grid.shape
    new_grid = grid.copy()
    counts = np.bincount(grid.flatten(), minlength=10)
    freq_colors = np.argsort(counts[1:])[::-1] + 1
    top_2 = freq_colors[:2]
    for c in range(1, 10):
        if c not in top_2 or counts[c] == 0:
            new_grid[grid == c] = 0
    final_grid = new_grid.copy()
    for r in range(h):
        for c in range(w):
            color = new_grid[r, c]
            if color == 0: continue
            has_neighbor = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if new_grid[nr, nc] == color:
                            has_neighbor = True
                            break
                if has_neighbor: break
            if not has_neighbor:
                final_grid[r, c] = 0
    return final_grid

def reconstruct_from_atoms(obj_grid, atoms, k):
    """
    obj_grid: [15, 15] — original object
    atoms:    [1024, 2250]  — NMF basis (15×15×10 one-hot)
    """
    oh = F.one_hot(torch.from_numpy(obj_grid).long(), num_classes=10).float()
    oh_weighted = oh.clone()
    oh_weighted[:, :, 0] *= 0.1
    obj_flat = oh_weighted.view(-1).to(atoms.device)

    dists = ((atoms - obj_flat.unsqueeze(0)) ** 2).sum(dim=1)
    best_k = dists.topk(k, largest=False).indices
    
    weights = 1.0 / (dists[best_k] + 1e-8)
    weights = weights / weights.sum()
    
    recon_flat = (atoms[best_k] * weights.unsqueeze(1)).sum(dim=0)
    recon_10ch = recon_flat.view(15, 15, 10)
    recon_viz = recon_10ch.clone()
    recon_viz[:, :, 0] *= 10.0
    recon_grid = recon_viz.argmax(dim=-1).cpu().numpy()
    
    return recon_grid, best_k, weights

def audit_codebook(k=5, n_visualize=8, library_path='arc_data/primitive_library.pt',
                   basis_path='arc_data/arc_basis_nmf_1024.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data  = torch.load(basis_path, weights_only=False)
    atoms = data['basis'].to(device)
    print(f"Codebook: {atoms.shape[0]} atoms × {atoms.shape[1]} dims")

    dataset = PrimitiveDataset(library_path)
    print(f"Dataset:  {len(dataset)} objects")

    all_accs = []
    worst_data, best_data = [], []

    print(f"\nRunning K={k} nearest-atom reconstruction on all objects...")
    for i in range(len(dataset)):
        item  = dataset[i]
        state = item['state'].squeeze(0).numpy()
        denoised = denoise_grid(state)

        recon_grid, best_k, _ = reconstruct_from_atoms(denoised, atoms, k)

        px_acc = (recon_grid == denoised).mean() * 100
        all_accs.append(px_acc)

        if len(worst_data) < n_visualize or px_acc < min([d[3] for d in worst_data]):
            worst_data.append((denoised, recon_grid, best_k.cpu().numpy(), px_acc))
            worst_data = sorted(worst_data, key=lambda x: x[3])[:n_visualize]

        if len(best_data) < n_visualize or px_acc > max([d[3] for d in best_data]):
            best_data.append((denoised, recon_grid, best_k.cpu().numpy(), px_acc))
            best_data = sorted(best_data, key=lambda x: x[3], reverse=True)[:n_visualize]

    all_accs = np.array(all_accs)
    print(f"\n{'='*55}")
    print(f"  CODEBOOK CAPACITY AUDIT  (K={k} atoms per object)")
    print(f"{'='*55}")
    print(f"  Mean pixel accuracy  : {all_accs.mean():.1f}%")
    print(f"  Median accuracy      : {np.median(all_accs):.1f}%")
    print(f"  Objects >80% acc     : {(all_accs > 80).sum()} / {len(all_accs)}")
    print(f"  Objects <20% acc     : {(all_accs < 20).sum()} / {len(all_accs)}")
    print(f"{'='*55}")

    # Visualize
    fig, axes = plt.subplots(n_visualize, 2, figsize=(6, 3 * n_visualize))
    for i in range(n_visualize):
        orig, recon, ids, acc = best_data[i]
        axes[i, 0].imshow(orig, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(recon, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[i, 1].set_title(f"Acc: {acc:.1f}%", fontsize=8)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/codebook_audit_v6.png')
    print(f"  Saved → artifacts/codebook_audit_v6.png")

if __name__ == '__main__':
    audit_codebook()
