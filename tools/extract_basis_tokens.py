"""
Phoenix v5 — Extraction & Audit Script
Run on the lab server after training completes.

Usage:
    python tools/extract_basis_tokens.py                 # uses latest checkpoint
    python tools/extract_basis_tokens.py checkpoints/phoenix_v5_e1000.pth

Outputs:
    artifacts/phoenix_v5_audit.png        — Visual grid: Original vs Reconstruction vs Atoms
    arc_data/basis_token_sequences.pt     — Full token library for Phase 2 reasoning
"""

import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.extract_arc_objects import PrimitiveDataset
from train_basis_model import BasisEncoder, AlgebraicDecoder
from modules.basis_vq import BasisVQ

# ── ARC color palette ─────────────────────────────────────────────────────────
ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
              '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
cmap = ListedColormap(ARC_COLORS)

# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg    = ckpt['cfg']

    encoder = BasisEncoder(k_slots=cfg['k_slots'], d_model=cfg['d_model'],
                           n_basis=cfg['n_basis']).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    vq = BasisVQ(basis_path='arc_data/arc_basis_nmf_1024.pt').to(device)
    vq.temperature = 0.01   # Near-zero → fully discrete hard argmax
    vq.eval()

    decoder = AlgebraicDecoder(n_slots=cfg['k_slots']).to(device)
    decoder.load_state_dict(ckpt['decoder'])   # ← Load TRAINED decoder weights
    decoder.eval()

    print(f"[Loaded] {checkpoint_path}  |  epoch={ckpt.get('epoch','?')}")
    return encoder, vq, decoder, cfg, device

# ── 1. Visual Audit ───────────────────────────────────────────────────────────
def visualize_reconstructions(encoder, vq, decoder, cfg, device, n_samples=8):
    dataset    = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    batch      = next(iter(dataloader))

    state    = batch['state'].to(device)
    x_onehot = F.one_hot(state.squeeze(1).long(), 10).permute(0,3,1,2).float()

    with torch.no_grad():
        logits               = encoder(x_onehot)
        c_m, p_m, indices, _ = vq(logits)
        col_logits, _        = decoder(c_m, p_m)   # ← TRAINED decoder
        recons = col_logits.argmax(dim=1).cpu().numpy()

    originals = state.squeeze(1).cpu().numpy()
    indices   = indices.cpu().numpy()    # [n_samples, K]
    k_slots   = cfg['k_slots']

    n_cols = 2 + k_slots
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3 * n_cols, 3 * n_samples))
    fig.suptitle("Phoenix v5 — Original | Reconstruction | K Basis Slots", fontsize=14)

    for i in range(n_samples):
        axes[i, 0].imshow(originals[i], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[i, 0].set_title("Original", fontsize=8)
        axes[i, 0].axis('off')

        acc = (recons[i] == originals[i]).mean() * 100
        axes[i, 1].imshow(recons[i], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[i, 1].set_title(f"Recon ({acc:.0f}%)", fontsize=8)
        axes[i, 1].axis('off')

        for k in range(k_slots):
            atom_id   = indices[i, k]
            atom      = vq.color_basis[atom_id].view(15, 15, 10).cpu().numpy()
            atom_grid = np.argmax(atom, axis=-1)
            intensity = np.max(atom, axis=-1)
            atom_grid[intensity < 0.05] = 0   # mask near-zero → black

            axes[i, 2+k].imshow(atom_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
            axes[i, 2+k].set_title(f"S{k}: #{atom_id}", fontsize=7)
            axes[i, 2+k].axis('off')

    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    out = 'artifacts/phoenix_v5_audit.png'
    plt.savefig(out, dpi=120)
    print(f"[Saved] Visual audit → {out}")
    return recons, originals

# ── 2. Full-Dataset Token Extraction (Phase 2 Input) ─────────────────────────
def extract_all_tokens(encoder, vq, decoder, cfg, device):
    """
    Run the full primitive library through the encoder.
    Uses the TRAINED decoder to compute pixel accuracy — not a fresh random one.
    """
    dataset    = PrimitiveDataset(cfg['library_path'])
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_tokens, all_states = [], []
    correct_px, total_px   = 0, 0

    print(f"\n[Extracting tokens for {len(dataset)} objects...]")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            state = batch['state'].to(device)
            x_oh  = F.one_hot(state.squeeze(1).long(), 10).permute(0,3,1,2).float()

            logits               = encoder(x_oh)
            c_m, p_m, indices, _ = vq(logits)

            # ── TRAINED decoder — weights loaded from checkpoint ──
            col_logits, _  = decoder(c_m, p_m)
            preds          = col_logits.argmax(dim=1)    # [B, 15, 15]
            correct_px    += (preds == state.squeeze(1)).sum().item()
            total_px      += state.squeeze(1).numel()

            all_tokens.append(indices.cpu())
            all_states.append(state.squeeze(1).cpu())

    all_tokens = torch.cat(all_tokens, dim=0)    # [N, K]
    all_states = torch.cat(all_states, dim=0)    # [N, 15, 15]
    pixel_acc  = correct_px / total_px * 100

    out = {
        'tokens':    all_tokens,
        'states':    all_states,
        'pixel_acc': pixel_acc,
        'n_basis':   cfg['n_basis'],
        'k_slots':   cfg['k_slots'],
    }
    os.makedirs('arc_data', exist_ok=True)
    path = 'arc_data/basis_token_sequences.pt'
    torch.save(out, path)
    print(f"[Saved] Token sequences ({len(all_states)} objects, K={cfg['k_slots']}) → {path}")
    print(f"[Metric] TRUE Pixel Accuracy (full library): {pixel_acc:.1f}%")
    return out

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None

    if ckpt_path is None:
        ckpts = sorted([f for f in os.listdir('checkpoints') if 'phoenix_v5' in f])
        if not ckpts:
            print("No phoenix_v5 checkpoints found. Run train_basis_model.py first.")
            sys.exit(1)
        ckpt_path = f"checkpoints/{ckpts[-1]}"

    encoder, vq, decoder, cfg, device = load_model(ckpt_path)

    print("\n── Phase 1: Visual Audit ──")
    recons, originals = visualize_reconstructions(encoder, vq, decoder, cfg, device)
    sample_acc = (recons == originals).mean() * 100
    print(f"[Accuracy] Visual sample pixel accuracy: {sample_acc:.1f}%")

    print("\n── Phase 2: Token Extraction ──")
    # Pass the TRAINED decoder — fixes the 10.8% bug
    token_data = extract_all_tokens(encoder, vq, decoder, cfg, device)

    print("\n── Summary ──")
    print(f"  Objects encoded : {len(token_data['states'])}")
    print(f"  Vocabulary used : {token_data['n_basis']} atoms")
    print(f"  Slots per object: {token_data['k_slots']}")
    print(f"  Pixel accuracy  : {token_data['pixel_acc']:.1f}%")
    print(f"\n  Tokens ready for Phase 2 at: arc_data/basis_token_sequences.pt")
    print(f"  Visual audit at:             artifacts/phoenix_v5_audit.png")
    print(f"\n  To copy audit image to your Mac:")
    print(f"  scp eet242799@10.225.67.239:~/development/Model-Jepa/artifacts/phoenix_v5_audit.png ~/Downloads/")
