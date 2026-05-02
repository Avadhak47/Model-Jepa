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

def analyze_basis(library_path, n_components=1024):
    payload = torch.load(library_path)
    tensors = payload['tensors'].numpy()
    
    print("Deduplicating by Binary Geometry...")
    unique_geometries = []
    seen_masks = set()
    
    for i in range(tensors.shape[0]):
        mask = (tensors[i] > 0).astype(np.uint8)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows): continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = mask[rmin:rmax+1, cmin:cmax+1]
        
        h = hash(cropped.tobytes())
        if h not in seen_masks:
            seen_masks.add(h)
            unique_geometries.append(tensors[i])
            
    unique_geometries = np.array(unique_geometries)
    n_samples = unique_geometries.shape[0]
    print(f"Found {n_samples} unique geometric primitives. Running 1024-component NMF...")
    
    X = np.zeros((n_samples, 2700))
    for i in range(n_samples):
        grid = unique_geometries[i]
        for r in range(15):
            for c in range(15):
                color = int(grid[r, c])
                base_idx = (r * 15 + c) * 12
                if color > 0:
                    X[i, base_idx + color] = 1.0
                X[i, base_idx + 10] = r / 14.0
                X[i, base_idx + 11] = c / 14.0
                
    # Using Mu solver with slightly less sparsity to allow for 1024 components to spread out
    nmf = NMF(
        n_components=n_components, 
        init='random', 
        solver='mu', 
        alpha_W=0.05, 
        alpha_H=0.05, 
        random_state=42, 
        max_iter=500
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    for i in range(n_components):
        atom = H[i]
        if np.max(atom) > 0:
            H[i] = atom / (np.max(atom) + 1e-8)
            
    torch.save({
        'basis': torch.from_numpy(H).float(),
        'weights': torch.from_numpy(W).float(),
        'n_components': n_components
    }, 'arc_data/arc_basis_nmf_1024.pt')
    
    print("Generating High-Fidelity Preview...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # Randomly sample 6 atoms from the 1024 to show variety
    sample_indices = np.random.choice(n_components, 6, replace=False)
    for i, idx in enumerate(sample_indices):
        ax = axes[i//3, i%3]
        atom = H[idx].reshape(15, 15, 12)
        color_data = atom[:, :, :10]
        intensities = np.max(color_data, axis=-1)
        grid = np.argmax(color_data, axis=-1)
        grid[intensities <= 0.05] = 0
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"Atom #{idx}")
        ax.axis('off')
    plt.savefig('artifacts/arc_basis_nmf_preview.png')
    print("1024-component preview saved.")

if __name__ == "__main__":
    analyze_basis('arc_data/primitive_library.pt')
