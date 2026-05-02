import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from torch.utils.data import DataLoader
import os

from tools.extract_arc_objects import PrimitiveDataset
from train_basis_model import BasisEncoder, AlgebraicDecoder
from modules.basis_vq import BasisVQ

ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)

def visualize_reconstructions(checkpoint_path, n_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint and config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    
    encoder = BasisEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model'], n_basis=cfg['n_basis']).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    
    vq = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt').to(device)
    vq.snapping_gain = 50.0 # Force clean snapping for visualization
    vq.eval()
    
    decoder = AlgebraicDecoder().to(device)
    decoder.eval()
    
    dataset = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    batch = next(iter(dataloader))
    
    state = batch['state'].to(device)
    x_onehot = F.one_hot(state.squeeze(1).long(), num_classes=10).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        basis_logits = encoder(x_onehot)
        q_manifold, indices = vq(basis_logits)
        color_logits, _ = decoder(q_manifold)
        recons = torch.argmax(color_logits, dim=1).cpu().numpy()
        
    originals = state.squeeze(1).cpu().numpy()
    indices = indices.cpu().numpy() # [B, K]
    
    # Plotting
    fig, axes = plt.subplots(n_samples, 2 + cfg['k_slots'], figsize=(15, 3 * n_samples))
    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(originals[i], cmap=cmap, vmin=0, vmax=9)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # Recon
        axes[i, 1].imshow(recons[i], cmap=cmap, vmin=0, vmax=9)
        axes[i, 1].set_title("Recon")
        axes[i, 1].axis('off')
        
        # Slots
        for k in range(cfg['k_slots']):
            atom_id = indices[i, k]
            # Fetch atom from vq buffer for visualization
            atom = vq.basis_vectors[atom_id].view(15, 15, 12).cpu().numpy()
            atom_grid = np.argmax(atom[:, :, :10], axis=-1)
            # Mask background
            intensities = np.max(atom[:, :, :10], axis=-1)
            atom_grid[intensities < 0.1] = 0
            
            axes[i, 2 + k].imshow(atom_grid, cmap=cmap, vmin=0, vmax=9)
            axes[i, 2 + k].set_title(f"Slot {k}: #{atom_id}")
            axes[i, 2 + k].axis('off')
            
    plt.tight_layout()
    plt.savefig('artifacts/basis_recon_audit.png')
    print("Reconstruction audit saved to artifacts/basis_recon_audit.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_reconstructions(sys.argv[1])
    else:
        # Try latest checkpoint
        ckpts = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if ckpts:
            latest = sorted(ckpts)[-1]
            visualize_reconstructions(f"checkpoints/{latest}")
        else:
            print("No checkpoints found.")
