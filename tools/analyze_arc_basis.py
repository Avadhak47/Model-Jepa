import torch
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import os

def analyze_basis(library_path, n_components=200):
    payload = torch.load(library_path)
    tensors = payload['tensors'].numpy() # [N, 15, 15]
    n_samples = tensors.shape[0]
    
    print(f"Constructing sparse manifold for {n_samples} objects...")
    
    manifold = np.zeros((n_samples, 15, 15, 12))
    for r in range(15):
        for c in range(15):
            manifold[:, r, c, 10] = r / 14.0
            manifold[:, r, c, 11] = c / 14.0
            
    for i in range(n_samples):
        grid = tensors[i].astype(int)
        for r in range(15):
            for c in range(15):
                color = grid[r, c]
                manifold[i, r, c, color] = 1.0
                
    X = manifold.reshape(n_samples, -1)
    
    print(f"Running Sparse NMF (L1 Penalty) to find {n_components} Sharp Parts...")
    # alpha_W and alpha_H enforce sparsity (L1)
    nmf = NMF(
        n_components=n_components, 
        init='nndsvda', 
        solver='mu', 
        beta_loss='frobenius',
        alpha_W=0.1, # Sparsity on weights
        alpha_H=0.1, # Sparsity on components (CRISP EDGES)
        l1_ratio=1.0, 
        random_state=42, 
        max_iter=2000
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    # --- POST-PROCESSING: THRESHOLD SNAPPING ---
    print("Applying Threshold Snapping to remove ghost pixels...")
    # For each atom, find the max intensity and zero out anything below 15% of max
    for i in range(n_components):
        atom = H[i]
        max_val = np.max(atom)
        if max_val > 0:
            atom[atom < 0.15 * max_val] = 0
            # Re-normalize to preserve "unit" feel
            H[i] = atom / (np.max(atom) + 1e-8)
    
    print("Saving Clean NMF Basis...")
    torch.save({
        'basis': torch.from_numpy(H).float(),
        'weights': torch.from_numpy(W).float(),
        'reconstruction_err': nmf.reconstruction_err_,
        'n_components': n_components
    }, 'arc_data/arc_basis_nmf_200.pt')
    
    print("Re-generating clean preview...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(min(6, n_components)):
        ax = axes[i//3, i%3]
        atom = H[i].reshape(15, 15, 12)
        shape_atom = atom[:, :, :10].argmax(axis=-1)
        ax.imshow(shape_atom, cmap='tab10')
        ax.set_title(f"Clean NMF Part #{i}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('artifacts/arc_basis_nmf_preview.png')

if __name__ == "__main__":
    analyze_basis('arc_data/primitive_library.pt')
