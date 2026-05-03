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
import torch.nn.functional as F
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

def denoise_grid(grid):
    """
    ARC Geometric Purity Filter:
    1. Connectivity: Pixels must have a neighbor of the same color (kills static).
    2. Color-Sparsity: Only keep the top 2 colors per object (kills multicolor noise).
    """
    h, w = grid.shape
    new_grid = grid.copy()
    
    # --- 1. Color Sparsity ---
    counts = np.bincount(grid.flatten(), minlength=10)
    # Get colors 1-9 sorted by frequency
    freq_colors = np.argsort(counts[1:])[::-1] + 1
    top_2 = freq_colors[:2]
    
    # Wipe any color not in the top 2
    for c in range(1, 10):
        if c not in top_2 or counts[c] == 0:
            new_grid[grid == c] = 0
            
    # --- 2. Connectivity (8-neighborhood) ---
    final_grid = new_grid.copy()
    for r in range(h):
        for c in range(w):
            color = new_grid[r, c]
            if color == 0: continue
            
            # Check neighbors
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

def build_feature_matrix(tensors):
    """
    Convert integer grids [N, 15, 15] to 10-channel one-hot matrix [N, 2250].
    
    We include color 0 (black) but down-weight it (0.1) so padding doesn't 
    dominate, while still allowing the model to see black shapes (rings/hollows).
    """
    n = tensors.shape[0]
    X = np.zeros((n, 2250), dtype=np.float32)  # 15*15*10
    for i in range(n):
        grid = tensors[i].astype(int)  # [15, 15]
        for r in range(15):
            for c in range(15):
                color = grid[r, c]
                # Map color 0-9 to channels 0-9
                # WEIGHTING: Color 0 gets 0.1 weight
                val = 1.0 if color > 0 else 0.1
                X[i, (r * 15 + c) * 10 + color] = val
    return X

def analyze_basis(library_path, n_components=1024):
    payload = torch.load(library_path, weights_only=False)
    tensors = payload['tensors'].numpy()   # [N, 15, 15]
    n_total = tensors.shape[0]
    print(f"Loaded {n_total} objects from library.")

    # ── Denoise and Build feature matrix ──
    print(f"Denoising and building one-hot matrix [{n_total} × 2250]...")
    denoised_tensors = np.array([denoise_grid(t) for t in tensors])
    X = build_feature_matrix(denoised_tensors)

    # All rows are non-empty now because color 0 is included
    print(f"  Samples: {X.shape[0]}")
    print(f"  Feature dim: {X.shape[1]} (15×15×10, weighted)")

    # L1-normalize each row: foreground pixel count varies per object
    row_norms = X.sum(axis=1, keepdims=True)
    X_norm = X / (row_norms + 1e-8)

    # ── Run NMF ──
    print(f"Running NMF ({n_components} components) on {X_norm.shape[0]} objects...")
    print(f"  Feature space: {X_norm.shape[1]} dims (15×15×10, weighted)")
    print(f"  This may take 5–15 minutes. Error should drop well below 5.0 now.")

    nmf = NMF(
        n_components=n_components,
        init='nndsvda',
        solver='mu',         # MU is more robust to high-dim discrete data
        beta_loss='kullback-leibler', 
        max_iter=1000,       # Start with 1000, it will converge faster than CD
        random_state=42,
        verbose=1,
        tol=1e-7,            # Tight enough for good atoms
        alpha_W=0.0,      
        alpha_H=0.01,        # Stronger sparsity to force solid shapes
        l1_ratio=1.0,
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ── Simple nearest-neighbour lookup audit ──
    # (reconstruct each object using top-5 atoms from the codebook)
    oh_full  = F.one_hot(torch.from_numpy(denoised_tensors).long(), num_classes=10).float()
    # Apply the same 0.1 weighting to channel 0 for the audit
    oh_weighted = oh_full.clone()
    oh_weighted[:, :, :, 0] *= 0.1
    X_audit = oh_weighted.reshape(oh_full.shape[0], -1).to(device)

    H_torch = torch.from_numpy(H).to(device)
    dists   = torch.cdist(X_audit, H_torch)
    best_k  = dists.topk(5, largest=False).indices

    # Reconstruction: average of top 5 atoms
    # Weight them by inverse distance for a better audit
    weights = 1.0 / (dists.gather(1, best_k) + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # [N, 5, 1] * [N, 5, 2250] -> [N, 2250]
    recon_flat = (H_torch[best_k] * weights.unsqueeze(2)).sum(dim=1)
    recon_10ch = recon_flat.view(-1, 15, 15, 10)
    
    # Before argmax, un-weight the background channel to get true colors
    recon_viz = recon_10ch.clone()
    recon_viz[:, :, :, 0] *= 10.0
    
    recon_grid = recon_viz.argmax(dim=-1).cpu().numpy()
    px_acc = (recon_grid == denoised_tensors).mean() * 100
    print(f"\nQuick audit (K=5) pixel accuracy: {px_acc:.1f}%")

    mean_acc = px_acc
    if mean_acc >= 75:
        print(f"  ✅ Codebook quality: GOOD")
    elif mean_acc >= 50:
        print(f"  ⚠️  Codebook quality: MARGINAL — consider n_components=2048")
    else:
        print(f"  ❌  Codebook quality: POOR — check feature encoding or increase components")

    # ── Save ──
    os.makedirs('arc_data', exist_ok=True)
    torch.save({
        'basis':        torch.from_numpy(H).float(),    # [1024, 2250]
        'weights':      torch.from_numpy(W).float(),    # [N, 1024]
        'n_components': n_components,
        'feature_dim':  2250,                           # 15×15×10 (all colors)
        'encoding':     'one_hot_weighted_bg',          # color 0 weighted 0.1x
        'n_objects':    X_norm.shape[0],
        'recon_err':    float(recon_err),
    }, 'arc_data/arc_basis_nmf_1024.pt')
    print(f"\nSaved → arc_data/arc_basis_nmf_1024.pt")
    print(f"  Basis shape: {H.shape}  (n_components × 2250)")

    # ── Visualize 6 sample atoms ──
    print("Generating atom preview...")
    os.makedirs('artifacts', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"NMF Atoms Preview — {n_components} components, err={recon_err:.4f}", fontsize=12)

    non_empty = [i for i in range(n_components) if np.max(H[i]) > 0.1]
    sample_ids = np.random.choice(non_empty if len(non_empty) >= 6 else n_components, 6, replace=False)

    for i, idx in enumerate(sample_ids):
        ax = axes[i // 3, i % 3]
        atom = H[idx].reshape(15, 15, 10)     # [15, 15, 10] all colors
        
        # Re-weight the background channel for visualization (undo the 0.1x)
        atom_viz = atom.copy()
        atom_viz[:, :, 0] *= 10.0
        
        grid = np.argmax(atom_viz, axis=-1)   # [15, 15]
        intensity = np.max(atom_viz, axis=-1) 

        # Relative threshold for clean display
        thresh = np.max(intensity) * 0.15
        grid[intensity < thresh] = 0   

        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(f"Atom #{idx} (max={np.max(intensity):.2f})", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('artifacts/arc_basis_nmf_preview.png', dpi=100)
    print("Preview → artifacts/arc_basis_nmf_preview.png")
    print("\nDone. Re-run audit_codebook.py to verify quality.")

if __name__ == "__main__":
    # n_components must be <= 2025 for 'nndsvda' initialization
    analyze_basis('arc_data/primitive_library.pt', n_components=1024)
