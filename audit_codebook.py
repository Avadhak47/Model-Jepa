"""
Codebook Audit Tool
===================
Answers: "What features are actually stored in our VQ-VAE codebook?"

Strategy — three complementary lenses:

  1. DECODE-DIRECT  : Feed each codebook vector through the PatchDecoder and
                      render what 2×2 pixel patch it reconstructs. Pure lookup.

  2. NEAREST-REAL   : Sample 10 000 real patches from the training set, encode
                      them, then for every code show the 8 REAL training patches
                      that landed closest to it in latent space. This reveals
                      what genuine grid structures the code has specialised on.

  3. GEOMETRY       : For each code compute the mean colour distribution,
                      dominant ARC colour, mean Sobel edge magnitude, and entropy.
                      Printed as a stats table and plotted as heatmaps so you can
                      tell colour codes from structure codes at a glance.

  4. USAGE & FPS    : Bar chart of code utilisation + scatter showing which 10
                      codes were selected as FPS seeds for the Phase 1 slots.

Run from the repo root:
    python audit_codebook.py --checkpoint runs/.../phase0/latest_checkpoint.pth
                             --out       evaluation_reports/codebook_audit
                             --n_batches 80   # batches of 128 to sample
"""

import argparse, os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from arc_data.rearc_dataset import ReARCDataset
from modules.encoders import PatchTransformerEncoder
from modules.decoders import PatchDecoder
from modules.vq      import FactorizedVectorQuantizer

# ── ARC colour palette ──────────────────────────────────────────────────────
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40',
    '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B',
    '#7FDBFF', '#870C25',
]
ARC_CMAP = ListedColormap(ARC_COLORS)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def load_phase0(ckpt_path: str, cfg: dict, device: str):
    """Load the full Phase 0 autoencoder from a saved checkpoint."""
    encoder = PatchTransformerEncoder(cfg).to(device)
    decoder = PatchDecoder(cfg).to(device)
    vq = FactorizedVectorQuantizer(
        num_shape_codes=cfg['num_shape_codes'],
        num_color_codes=cfg['num_color_codes'],
        embedding_dim  =cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)

    # Strip "encoder.", "vq.", "decoder." prefixes if present
    enc_state  = {k.replace('encoder.', '', 1): v for k, v in state.items() if k.startswith('encoder.')}
    vq_state   = {k.replace('vq.', '',      1): v for k, v in state.items() if k.startswith('vq.')}
    dec_state  = {k.replace('decoder.', '', 1): v for k, v in state.items() if k.startswith('decoder.')}

    encoder.load_state_dict(enc_state, strict=False)
    vq.load_state_dict(vq_state,       strict=False)
    decoder.load_state_dict(dec_state, strict=False)

    encoder.eval(); vq.eval(); decoder.eval()
    return encoder, vq, decoder


def patch_to_pixel(patch_tensor: torch.Tensor) -> np.ndarray:
    """
    Decode a single 2×2 patch latent through the decoder and return a
    (2, 2) integer ARC colour array.  We embed the patch as a 15×15 grid
    of identical vectors, decode the full 30×30 grid, then crop the top-left
    2×2 to get the patch primitive.
    """
    # patch_tensor: [128] — one codebook vector
    # PatchDecoder expects [B, N, C] where N=225 patches
    z = patch_tensor.unsqueeze(0).unsqueeze(0).expand(1, 225, -1)  # [1, 225, 128]
    return z


@torch.no_grad()
def decode_single_code(decoder, code_vec: torch.Tensor, device: str) -> np.ndarray:
    """
    Tile a single 128-d code vector across all 225 patch positions and run
    the decoder. Returns a (30, 30) integer grid showing the 'primitive' the
    code represents when the whole grid is made of that code.
    """
    z = code_vec.unsqueeze(0).unsqueeze(0).expand(1, 225, -1).to(device)   # [1, 225, 128]
    out = decoder({'latent': z})
    logits = out['reconstructed_logits']     # [1, 10, 30, 30]  — PatchDecoder key
    grid = logits.argmax(dim=1).squeeze(0).cpu().numpy()                    # (30, 30)
    return grid


@torch.no_grad()
def collect_patch_samples(encoder, vq, dataset, cfg: dict,
                           n_batches: int, device: str):
    """
    Run n_batches through the encoder+VQ and collect per-patch statistics.

    Returns:
        shape_assigns : list[int]  — shape code index for every real patch
        color_assigns : list[int]  — color code index for every real patch
        patch_pixels  : list[np.ndarray (2,2)] — actual pixel values for every patch
        shape_usage   : np.ndarray (256,) — total hits per shape code
        color_usage   : np.ndarray (16,)  — total hits per color code
    """
    shape_assigns, color_assigns, patch_pixels = [], [], []
    shape_usage = np.zeros(cfg['num_shape_codes'], dtype=np.int64)
    color_usage = np.zeros(cfg['num_color_codes'], dtype=np.int64)

    patch_size = cfg['patch_size']  # 2
    grid_size  = cfg['grid_size']   # 30
    Ph = Pw    = grid_size // patch_size   # 15

    for _ in tqdm(range(n_batches), desc="Sampling real patches"):
        batch  = dataset.sample(cfg['batch_size'])
        states = batch['state'].to(device)   # [B, 1, 30, 30]
        B      = states.shape[0]

        z  = encoder({'state': states})['latent']     # [B, N, 128]
        z4 = z.permute(0, 2, 1).view(B, 128, Ph, Pw) # [B, 128, 15, 15]

        # Distances to shape sub-codebook
        flat_z      = z4.permute(0, 2, 3, 1).contiguous().view(-1, 128)
        flat_shape  = flat_z[:, :64]
        flat_color  = flat_z[:, 64:]

        dist_shape  = (flat_shape.pow(2).sum(1, keepdim=True)
                       + vq.embedding_shape.weight.pow(2).sum(1)
                       - 2 * flat_shape @ vq.embedding_shape.weight.T)
        dist_color  = (flat_color.pow(2).sum(1, keepdim=True)
                       + vq.embedding_color.weight.pow(2).sum(1)
                       - 2 * flat_color @ vq.embedding_color.weight.T)

        s_idx = dist_shape.argmin(1).cpu().numpy()   # [B*225]
        c_idx = dist_color.argmin(1).cpu().numpy()

        # Extract actual pixel patches
        state_np = states.squeeze(1).cpu().numpy()   # [B, 30, 30]
        for b in range(B):
            for ph in range(Ph):
                for pw in range(Pw):
                    loc = b * Ph * Pw + ph * Pw + pw
                    pixel_patch = state_np[b,
                                           ph*patch_size:(ph+1)*patch_size,
                                           pw*patch_size:(pw+1)*patch_size]  # (2,2)
                    # Skip pure padding patches (all zeros)
                    if pixel_patch.max() == 0:
                        continue
                    shape_assigns.append(s_idx[loc])
                    color_assigns.append(c_idx[loc])
                    patch_pixels.append(pixel_patch.astype(np.int32))

        shape_usage += np.bincount(s_idx, minlength=cfg['num_shape_codes'])
        color_usage += np.bincount(c_idx, minlength=cfg['num_color_codes'])

    return shape_assigns, color_assigns, patch_pixels, shape_usage, color_usage


def compute_code_stats(assigns, patch_pixels, num_codes, label="Shape"):
    """
    For each code, compute:
      - dominant ARC colour
      - colour entropy
      - mean Sobel edge magnitude (proxy for structure complexity)
    """
    from scipy.ndimage import generic_filter

    stats = []
    assigns = np.array(assigns)
    for c in range(num_codes):
        mask    = assigns == c
        patches = [patch_pixels[i] for i in np.where(mask)[0]]
        n_hits  = len(patches)

        if n_hits == 0:
            stats.append({'code': c, 'hits': 0, 'dominant_color': -1,
                          'entropy': 0.0, 'edge_mag': 0.0})
            continue

        all_pixels = np.concatenate([p.flatten() for p in patches])
        counts     = np.bincount(all_pixels.clip(0, 9), minlength=10)
        probs      = counts / counts.sum()
        entropy    = -np.sum(probs * np.log(probs + 1e-10))
        dominant   = int(np.argmax(counts))

        # Sobel proxy: variance over the 2×2 patch colours ≈ edge-ness
        edge_vals = [np.std(p.astype(float)) for p in patches]
        edge_mag  = float(np.mean(edge_vals))

        stats.append({'code': c, 'hits': n_hits, 'dominant_color': dominant,
                      'entropy': entropy, 'edge_mag': edge_mag})
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_decoded_primitives(decoder, vq, device, out_dir):
    """Panel 1: decoded 30×30 grid for every shape code (256 tiles)."""
    print("🎨 Decoding shape codebook primitives...")
    ncols = 32
    nrows = (vq.num_shape_codes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    fig.suptitle("Shape Codebook — Decoded Primitives\n"
                 "(Each tile = what the decoder produces when the whole grid is made of this code)",
                 fontsize=13, y=1.01)

    for i, ax in enumerate(axes.flat):
        if i >= vq.num_shape_codes:
            ax.axis('off')
            continue
        # Full code vector: shape code + first color code
        shape_vec = vq.embedding_shape.weight[i]         # [64]
        color_vec = vq.embedding_color.weight[0]         # [64]  use code-0 as neutral color
        code_vec  = torch.cat([shape_vec, color_vec])    # [128]
        grid      = decode_single_code(decoder, code_vec, device)
        ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(f'{i}', fontsize=5, pad=1)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, '1_shape_codebook_primitives.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_color_primitives(decoder, vq, device, out_dir):
    """Panel 2: decoded primitives for all 16 color codes."""
    print("🎨 Decoding color codebook primitives...")
    ncols = 8
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
    fig.suptitle("Color Codebook — Decoded Primitives (16 entries)\n"
                 "(Shape code fixed to code-0; only color varies)",
                 fontsize=13)

    shape_vec = vq.embedding_shape.weight[0]  # neutral shape
    for i, ax in enumerate(axes.flat):
        if i >= vq.num_color_codes:
            ax.axis('off')
            continue
        color_vec = vq.embedding_color.weight[i]
        code_vec  = torch.cat([shape_vec, color_vec])
        grid      = decode_single_code(decoder, code_vec, device)
        ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(f'Color-{i}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, '2_color_codebook_primitives.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_usage_bars(shape_usage, color_usage, fps_shape_ids, out_dir):
    """Panel 3: usage histogram + FPS seed markers."""
    print("📊 Plotting usage histograms...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))

    colors_s = ['#e74c3c' if i in fps_shape_ids else '#3498db'
                for i in range(len(shape_usage))]
    ax1.bar(range(len(shape_usage)), shape_usage, color=colors_s, width=1.0)
    ax1.set_title(f'Shape Codebook Usage  (red = FPS seed for Phase 1 slots)\n'
                  f'Active codes: {(shape_usage > 0).sum()}/{len(shape_usage)}', fontsize=12)
    ax1.set_xlabel('Code Index')
    ax1.set_ylabel('Total assignements')

    ax2.bar(range(len(color_usage)), color_usage, color='#2ecc71', width=0.8)
    ax2.set_title(f'Color Codebook Usage\n'
                  f'Active codes: {(color_usage > 0).sum()}/{len(color_usage)}', fontsize=12)
    ax2.set_xlabel('Code Index')
    ax2.set_ylabel('Total assignments')

    plt.tight_layout()
    path = os.path.join(out_dir, '3_usage_histograms.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_nearest_real_patches(shape_assigns, patch_pixels, shape_usage, out_dir,
                               top_n=40, n_examples=8):
    """Panel 4: For the top-N most-used shape codes, show real training patches nearest to them."""
    print(f"🔍 Plotting {n_examples} real patches for top-{top_n} used shape codes...")
    top_codes = np.argsort(shape_usage)[::-1][:top_n]
    assigns   = np.array(shape_assigns)

    ncols = n_examples + 1   # code index label + n example patches
    nrows = top_n

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 0.9, nrows * 0.9))
    fig.suptitle(f"Top-{top_n} Most-Used Shape Codes → Real Training Patches",
                 fontsize=14, y=1.005)

    for row, code in enumerate(top_codes):
        mask    = assigns == code
        patches = [patch_pixels[i] for i in np.where(mask)[0][:n_examples]]

        # Label column
        axes[row, 0].text(0.5, 0.5, f'S-{code}\n({shape_usage[code]:,})',
                          ha='center', va='center', fontsize=6,
                          transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')

        for col in range(1, ncols):
            ax = axes[row, col]
            if col - 1 < len(patches):
                ax.imshow(patches[col-1], cmap=ARC_CMAP,
                           vmin=0, vmax=9, interpolation='nearest')
            ax.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, '4_top_codes_real_patches.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_geometry_heatmaps(shape_stats, out_dir):
    """Panel 5: per-code geometry stats as heatmaps (hits, entropy, edge magnitude, dominant colour)."""
    print("🗺️  Plotting geometry heatmaps...")
    N      = len(shape_stats)
    side   = 16   # 16 × 16 = 256
    hits   = np.array([s['hits']          for s in shape_stats]).reshape(side, side)
    ent    = np.array([s['entropy']        for s in shape_stats]).reshape(side, side)
    edge   = np.array([s['edge_mag']       for s in shape_stats]).reshape(side, side)
    dom    = np.array([s['dominant_color'] for s in shape_stats]).reshape(side, side).astype(float)
    dom[dom < 0] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Shape Codebook Geometry Analysis (16×16 layout of all 256 codes)",
                 fontsize=14)

    def hmap(ax, data, title, cmap):
        im = ax.imshow(data, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    hmap(axes[0,0], hits, 'Utilisation (total patch hits)', 'Blues')
    hmap(axes[0,1], ent,  'Colour Entropy\n(high = multi-colour patches)',  'RdYlGn')
    hmap(axes[1,0], edge, 'Mean Edge Magnitude\n(high = structured detail)', 'hot')
    hmap(axes[1,1], dom,  'Dominant ARC Colour Index\n(0=black…9=maroon)',   ARC_CMAP)

    plt.tight_layout()
    path = os.path.join(out_dir, '5_shape_geometry_heatmaps.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_fps_seeds(vq, fps_shape_ids, out_dir):
    """Panel 6: PCA of shape codebook vectors with FPS seeds highlighted."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("⚠️  sklearn not available — skipping PCA plot")
        return

    print("🔵 Plotting FPS seeds in PCA space...")
    W   = vq.embedding_shape.weight.detach().cpu().numpy()   # [256, 64]
    pca = PCA(n_components=2)
    W2  = pca.fit_transform(W)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(W2[:, 0], W2[:, 1], c='#95a5a6', s=18, alpha=0.6, label='All codes')
    fps_pts = W2[fps_shape_ids]
    ax.scatter(fps_pts[:, 0], fps_pts[:, 1], c='#e74c3c', s=120,
               zorder=5, marker='*', label='FPS seeds (Phase 1 slots)')
    for i, idx in enumerate(fps_shape_ids):
        ax.annotate(f'Slot {i}\n(S-{idx})', W2[idx],
                    fontsize=7, ha='center', xytext=(0, 8),
                    textcoords='offset points')

    ax.set_title(f'Shape Codebook — PCA Projection\n'
                 f'Var explained: PC1={pca.explained_variance_ratio_[0]:.1%}  '
                 f'PC2={pca.explained_variance_ratio_[1]:.1%}',
                 fontsize=12)
    ax.legend()
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')

    path = os.path.join(out_dir, '6_fps_seeds_pca.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def save_stats_json(shape_stats, color_stats, fps_shape_ids, out_dir):
    """Save a JSON summary for easy programmatic inspection."""
    summary = {
        'fps_shape_codes': [int(x) for x in fps_shape_ids],
        'shape_codebook': shape_stats,
        'color_codebook': color_stats,
    }
    path = os.path.join(out_dir, 'codebook_stats.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  → {path}")

    # Print quick terminal summary
    active_shapes = sum(1 for s in shape_stats if s['hits'] > 0)
    active_colors = sum(1 for s in color_stats  if s['hits'] > 0)
    print(f"\n{'='*55}")
    print(f"  Shape codebook:  {active_shapes}/{len(shape_stats)} codes active")
    print(f"  Color codebook:  {active_colors}/{len(color_stats)} codes active")
    print(f"  FPS shape seeds: {fps_shape_ids}")
    dead_shapes = [s['code'] for s in shape_stats if s['hits'] == 0]
    if dead_shapes:
        print(f"  Dead shape codes ({len(dead_shapes)}): {dead_shapes[:20]}{'...' if len(dead_shapes)>20 else ''}")
    print(f"{'='*55}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Audit the VQ-VAE codebook')
    parser.add_argument('--checkpoint', type=str,
                        default='runs/FactorizedFPS-v3_2026-04-23_14-40-34/phase0/latest_checkpoint.pth',
                        help='Path to Phase 0 checkpoint (.pth)')
    parser.add_argument('--out',        type=str,
                        default='evaluation_reports/codebook_audit',
                        help='Output directory for plots and JSON')
    parser.add_argument('--n_batches',  type=int, default=80,
                        help='Number of batches of 128 to sample from training set')
    parser.add_argument('--data_path',  type=str, default='arc_data/re-arc')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"🔬 Codebook Audit")
    print(f"   Checkpoint : {args.checkpoint}")
    print(f"   Output dir : {args.out}")
    print(f"   Device     : {args.device}")

    CFG = {
        'device':          args.device,
        'in_channels':     10,
        'patch_size':      2,
        'hidden_dim':      128,
        'latent_dim':      128,
        'vocab_size':      10,
        'grid_size':       30,
        'focal_gamma':     2.0,
        'num_shape_codes': 256,
        'num_color_codes': 16,
        'commitment_cost': 0.25,
        'batch_size':      128,
    }

    # ── Load model ───────────────────────────────────────────────────────────
    encoder, vq, decoder = load_phase0(args.checkpoint, CFG, args.device)

    # ── FPS seeds (same deterministic call as the training script) ──────────
    fps_slots          = vq.get_farthest_point_samples(target_slots=10)  # [1,10,128]
    fps_shape_latents  = fps_slots[0, :, :64]   # [10, 64]
    W                  = vq.embedding_shape.weight.data   # [256, 64]
    dists              = torch.cdist(fps_shape_latents, W)
    fps_shape_ids      = dists.argmin(dim=1).cpu().tolist()
    print(f"\n🎯 FPS seed → shape codes: {fps_shape_ids}")

    # ── Sample real patches ──────────────────────────────────────────────────
    dataset = ReARCDataset(data_path=args.data_path)
    (shape_assigns, color_assigns, patch_pixels,
     shape_usage, color_usage) = collect_patch_samples(
         encoder, vq, dataset, CFG, args.n_batches, args.device)

    print(f"\n✅ Collected {len(patch_pixels):,} real (non-padded) patch samples")

    # ── Per-code stats ───────────────────────────────────────────────────────
    print("📈 Computing per-code geometry stats...")
    shape_stats = compute_code_stats(shape_assigns, patch_pixels,
                                     CFG['num_shape_codes'], "Shape")
    color_stats = compute_code_stats(color_assigns,  patch_pixels,
                                     CFG['num_color_codes'], "Color")

    # ── Generate all panels ──────────────────────────────────────────────────
    with torch.no_grad():
        plot_decoded_primitives(decoder, vq, args.device, args.out)
        plot_color_primitives(decoder, vq, args.device, args.out)

    plot_usage_bars(shape_usage, color_usage, fps_shape_ids, args.out)
    plot_nearest_real_patches(shape_assigns, patch_pixels, shape_usage, args.out)
    plot_geometry_heatmaps(shape_stats, args.out)
    plot_fps_seeds(vq, fps_shape_ids, args.out)
    save_stats_json(shape_stats, color_stats, fps_shape_ids, args.out)

    print(f"\n✅ Audit complete. All outputs in: {args.out}/")
    print("   Run `scp` to copy to your Mac:")
    print(f"   scp -r eet242799@10.225.67.239:~/development/Model-Jepa/{args.out} ~/Desktop/")


if __name__ == '__main__':
    main()
