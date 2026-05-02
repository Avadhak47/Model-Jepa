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
    print(f"Found {n_samples} unique geometric primitives.")
    
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
                
    print(f"Running Signal-Boosted NMF ({n_components} components)...")
    nmf = NMF(n_components=n_components, init='random', solver='mu', random_state=42, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    # SIGNAL BOOSTING: Force every atom to have a max value of 1.0
    print("Boosting signals and normalizing atoms...")
    for i in range(n_components):
        atom = H[i]
        max_v = np.max(atom)
        if max_v > 0:
            # Scale so the structural signal is dominant
            H[i] = atom / (max_v + 1e-8)
            
    torch.save({
        'basis': torch.from_numpy(H).float(),
        'weights': torch.from_numpy(W).float(),
        'n_components': n_components
    }, 'arc_data/arc_basis_nmf_1024.pt')
    
    print("Generating High-Visibility Preview...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # Pick 6 non-empty atoms for the preview
    non_empty = [idx for idx in range(n_components) if np.max(H[idx, :2250]) > 0.1]
    if len(non_empty) < 6:
        sample_indices = np.random.choice(n_components, 6, replace=False)
    else:
        sample_indices = np.random.choice(non_empty, 6, replace=False)
        
    for i, idx in enumerate(sample_indices):
        ax = axes[i//3, i%3]
        atom = H[idx].reshape(15, 15, 12)
        color_data = atom[:, :, :10]
        intensities = np.max(color_data, axis=-1)
        
        # Adaptive Threshold: Pick top 5% of active pixels
        thresh = np.percentile(intensities, 95)
        grid = np.argmax(color_data, axis=-1)
        # Snap to color only if it's significant relative to the atom's peak
        grid[intensities < max(0.05, thresh * 0.5)] = 0
        
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"Atom #{idx} (Boosted)")
        ax.axis('off')
    plt.savefig('artifacts/arc_basis_nmf_preview.png')
    print("Ready. Dashboard will now show clear atoms.")

if __name__ == "__main__":
    analyze_basis('arc_data/primitive_library.pt')
