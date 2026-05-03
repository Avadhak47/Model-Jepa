"""
visualize_phoenix_v6.py — Final Audit for Phoenix v6 (VQ-VAE)

Loads the trained checkpoint and generates a grid comparing:
Original | Reconstruction | Atom 1 | Atom 2 | Atom 3

This verifies if the model is actually using the codebook as a dictionary.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os, sys

sys.path.insert(0, os.getcwd())
from train_basis_model import BasisEncoder, AlgebraicDecoder
from modules.basis_vq import BasisVQ
from tools.extract_arc_objects import PrimitiveDataset

ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
              '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
cmap = ListedColormap(ARC_COLORS)

@torch.no_grad()
def run_audit(checkpoint_path, n_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt['cfg']
    
    # 2. Init Models
    encoder = BasisEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model']).to(device)
    vq      = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt', d_model=cfg['d_model']).to(device)
    decoder = AlgebraicDecoder(n_slots=cfg['k_slots'], basis_dim=vq.basis_dim).to(device)
    
    encoder.load_state_dict(ckpt['encoder'])
    vq.slot_proj.load_state_dict(ckpt['vq_proj'])
    decoder.load_state_dict(ckpt['decoder'])
    
    encoder.eval(); vq.eval(); decoder.eval()
    print(f"Loaded {checkpoint_path} (Epoch {ckpt['epoch']})")

    # 3. Get Data
    dataset = PrimitiveDataset(cfg['library_path'])
    indices = torch.linspace(0, len(dataset)-1, n_samples).long()
    
    # 4. Visualize
    fig, axes = plt.subplots(n_samples, 2 + cfg['k_slots'], figsize=(15, 3 * n_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        state = item['state'].to(device).unsqueeze(0) # [1, 1, 15, 15]
        target = state.squeeze().cpu().numpy()
        
        x_oh = F.one_hot(state.squeeze(1).long(), 10).permute(0,3,1,2).float()
        
        # Pass through model
        slots = encoder(x_oh)
        q_st, atom_indices, _, _ = vq(slots)
        color_logits, _ = decoder(q_st)
        
        recon = color_logits.argmax(dim=1).squeeze().cpu().numpy()
        px_acc = (recon == target).mean() * 100
        
        # Column 0: Original
        axes[i, 0].imshow(target, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[i, 0].set_title(f"Original #{idx}", fontsize=9)
        axes[i, 0].axis('off')
        
        # Column 1: Reconstruction
        axes[i, 1].imshow(recon, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[i, 1].set_title(f"Recon ({px_acc:.1f}%)", fontsize=9)
        axes[i, 1].axis('off')
        
        # Remaining Columns: The chosen Atoms
        for k in range(cfg['k_slots']):
            atom_id = atom_indices[0, k].item()
            # De-project atom from 2025-dim to 15x15x9
            atom_vec = vq.embed[atom_id].view(15, 15, 9).cpu().numpy()
            # Add bg channel for visualization
            atom_10ch = np.concatenate([np.zeros((15, 15, 1)), atom_vec], axis=-1)
            atom_grid = np.argmax(atom_10ch, axis=-1)
            atom_mask = np.max(atom_vec, axis=-1)
            atom_grid[atom_mask < 0.1] = 0
            
            axes[i, 2+k].imshow(atom_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
            axes[i, 2+k].set_title(f"Slot {k}: Atom {atom_id}", fontsize=7)
            axes[i, 2+k].axis('off')

    out_path = f"artifacts/phoenix_v6_audit_epoch{ckpt['epoch']}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Visual audit saved to: {out_path}")
    print(f"Run this to see it: scp eet242799@10.225.67.239:~/development/Model-Jepa/{out_path} ~/Downloads/")

if __name__ == "__main__":
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pth")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    run_audit(args.checkpoint, args.samples)
