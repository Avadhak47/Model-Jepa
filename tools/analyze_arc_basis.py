"""
analyze_arc_basis.py — Phoenix NMF Basis Extraction (v2)

Key fixes over v1:
  1. Run NMF on ALL objects (with color), not just deduplicated geometry.
     Color is part of the object identity in ARC — blue L ≠ red L.
  2. Clean one-hot encoding: 10 color channels, NO coordinate channels.
     Coordinates add noise that NMF confuses with color structure.
  3. Pre-normalize each object so large solid-color objects don't dominate.
  4. After NMF, audit reconstruction quality to verify codebook capacity.
"""

import torch
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)

def build_feature_matrix(tensors):
    """
    Convert integer grids [N, 15, 15] to foreground-only color matrix [N, 2025].
    2025 = 15 * 15 * 9  (9 foreground colors, color 0 / black EXCLUDED).

    Why exclude color 0?
      ARC objects are padded to 15×15. Most pixels are black background.
      If we include color 0, NMF collapses to 'mostly-black' atoms because
      that's the dominant signal. Excluding it forces atoms to represent
      real foreground structure instead.

    Reconstruction: pixels with no foreground channel activation → predicted black.
    """
    n = tensors.shape[0]
    X = np.zeros((n, 2025), dtype=np.float32)  # 15*15*9
    for i in range(n):
        grid = tensors[i].astype(int)  # [15, 15]
        for r in range(15):
            for c in range(15):
                color = grid[r, c]
                if color > 0:  # Skip background (color 0)
                    # Map color 1-9 to channels 0-8
                    X[i, (r * 15 + c) * 9 + (color - 1)] = 1.0
    return X

def analyze_basis(library_path, n_components=1024):
    payload = torch.load(library_path, weights_only=False)
    tensors = payload['tensors'].numpy()   # [N, 15, 15]
    n_total = tensors.shape[0]
    print(f"Loaded {n_total} objects from library.")

    # ── Build feature matrix: ALL objects (color + geometry) ──
    # No deduplication — we want color variants as separate training samples
    print(f"Building one-hot color feature matrix [{n_total} × 2250]...")
    X = build_feature_matrix(tensors)

    # Remove all-black rows (objects with no foreground pixels at all)
    row_sums = X.sum(axis=1)
    valid = row_sums > 0
    X = X[valid]
    tensors = tensors[valid]
    print(f"  After removing empty objects: {X.shape[0]} samples")
    print(f"  Feature dim: {X.shape[1]} (15×15×9, background excluded)")

    # L1-normalize each row: foreground pixel count varies per object
    row_norms = X.sum(axis=1, keepdims=True)
    X_norm = X / (row_norms + 1e-8)

    # ── Run NMF ──
    print(f"Running NMF ({n_components} components) on {X_norm.shape[0]} objects...")
    print(f"  Feature space: {X_norm.shape[1]} dims (15×15×9, foreground colors only)")
    print(f"  This may take 5–15 minutes. Error should drop well below 5.0 now.")

    nmf = NMF(
        n_components=n_components,
        init='nndsvda',
        solver='mu',
        beta_loss='kullback-leibler',
        max_iter=800,       # Run longer — was stopping at epoch 80 due to loose tol
        random_state=42,
        verbose=1,
        tol=1e-6,           # Tight tol — let it converge fully, not stop early
    )
    W = nmf.fit_transform(X_norm)   # [N, K] — object-to-atom weights
    H = nmf.components_             # [K, 2250] — the atoms

    recon_err = nmf.reconstruction_err_
    print(f"NMF reconstruction error: {recon_err:.4f}")

    # ── Post-process atoms ──
    # Scale each atom to [0, 1] for clean visualization and distance computation
    print("Normalizing atoms...")
    for i in range(n_components):
        max_v = np.max(H[i])
        if max_v > 0:
            H[i] = H[i] / max_v

    # ── Quick reconstruction audit ──
    print("\nRunning quick reconstruction audit (K=5 nearest atoms)...")
    H_tensor = torch.from_numpy(H).float()   # [1024, 2025]
    X_tensor = torch.from_numpy(X_norm).float()

    accs = []
    for i in range(min(1000, X_tensor.shape[0])):
        obj  = X_tensor[i]                              # [2025]
        dist = ((H_tensor - obj.unsqueeze(0)) ** 2).sum(dim=1)
        top5 = dist.topk(5, largest=False).indices

        dists_k = dist[top5]
        weights  = 1.0 / (dists_k + 1e-8)
        weights  = weights / weights.sum()
        recon    = (H_tensor[top5] * weights.unsqueeze(1)).sum(dim=0)   # [2025]

        # Pixel accuracy: recon is [2025] = 15×15×9 (foreground channels)
        # Argmax over 9 channels + 1 implicit background gives 10-way prediction
        recon_9ch  = recon.view(15, 15, 9)                # [15, 15, 9]
        # Add a background channel (all zeros → background wins if no foreground)
        bg_channel = torch.zeros(15, 15, 1)               # implicit black
        recon_10ch = torch.cat([bg_channel, recon_9ch], dim=-1)  # [15, 15, 10]
        recon_grid = recon_10ch.argmax(dim=-1)            # [15, 15]  colors 0-9

        # Original grid from tensors
        orig_idx = i % tensors.shape[0]
        orig_grid = torch.from_numpy(tensors[orig_idx].astype(np.int64))
        acc = (recon_grid == orig_grid).float().mean().item() * 100
        accs.append(acc)

    mean_acc = np.mean(accs)
    print(f"  Mean pixel accuracy (K=5): {mean_acc:.1f}%")
    if mean_acc >= 75:
        print(f"  ✅ Codebook quality: GOOD")
    elif mean_acc >= 50:
        print(f"  ⚠️  Codebook quality: MARGINAL — consider n_components=2048")
    else:
        print(f"  ❌  Codebook quality: POOR — check feature encoding or increase components")

    # ── Save ──
    os.makedirs('arc_data', exist_ok=True)
    torch.save({
        'basis':        torch.from_numpy(H).float(),    # [1024, 2025]
        'weights':      torch.from_numpy(W).float(),    # [N, 1024]
        'n_components': n_components,
        'feature_dim':  2025,                           # 15×15×9 (foreground only)
        'encoding':     'one_hot_foreground_only',      # color 0 excluded
        'n_objects':    X_norm.shape[0],
        'recon_err':    float(recon_err),
    }, 'arc_data/arc_basis_nmf_1024.pt')
    print(f"\nSaved → arc_data/arc_basis_nmf_1024.pt")
    print(f"  Basis shape: {H.shape}  (n_components × 2025)")

    # ── Visualize 6 sample atoms ──
    print("Generating atom preview...")
    os.makedirs('artifacts', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"NMF Atoms Preview — {n_components} components, err={recon_err:.4f}", fontsize=12)

    non_empty = [i for i in range(n_components) if np.max(H[i]) > 0.1]
    sample_ids = np.random.choice(non_empty if len(non_empty) >= 6 else n_components, 6, replace=False)

    for i, idx in enumerate(sample_ids):
        ax = axes[i // 3, i % 3]
        atom = H[idx].reshape(15, 15, 9)    # [15, 15, 9] foreground channels
        # Add implicit background channel so argmax gives colors 0-9
        atom_10ch = np.concatenate([np.zeros((15, 15, 1)), atom], axis=-1)
        grid      = np.argmax(atom_10ch, axis=-1)   # [15, 15]
        intensity = np.max(atom_10ch, axis=-1)
        grid[intensity < 0.1] = 0   # mask weak activations → black

        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(f"Atom #{idx}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('artifacts/arc_basis_nmf_preview.png', dpi=100)
    print("Preview → artifacts/arc_basis_nmf_preview.png")
    print("\nDone. Re-run audit_codebook.py to verify quality.")

if __name__ == "__main__":
    analyze_basis('arc_data/primitive_library.pt')
