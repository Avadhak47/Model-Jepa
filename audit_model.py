"""
Full Model Audit — Phase 0 Autoencoder Validation
===================================================
Works with ANY checkpoint regardless of hidden_dim, patch_size, num_shape_codes, etc.
All architecture dimensions are auto-detected from the checkpoint weights.

Panels generated:
  1. Reconstruction Gallery    — input vs decoded grid side-by-side, error mask
  2. Pixel-Level Confusion     — 10×10 ARC colour confusion matrix
  3. Per-Task Accuracy         — which ARC tasks are hardest to reconstruct
  4. VQ Commitment Distribution — how far encoder outputs are from their nearest code
  5. Patch Latent t-SNE        — patch embeddings in 2-D, coloured by ARC colour
  6. Codebook Load Balancing   — both shape and colour utilisation bars
  7. Accuracy Distribution     — histogram of per-grid pixel accuracy

Usage:
    python audit_model.py --checkpoint runs/.../phase0/latest_checkpoint.pth
                          --out evaluation_reports/model_audit
                          --n_grids 200          # grids to reconstruct
                          --n_tsne  2000         # patches for t-SNE (expensive)

The script does NOT require any CFG flags for architecture dims — it reads everything
from the checkpoint itself.
"""

import argparse, os, sys, json, warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

# ── ARC colour palette ──────────────────────────────────────────────────────
ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
               '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
ARC_CMAP   = ListedColormap(ARC_COLORS)
ARC_NAMES  = ['Black','Blue','Red','Green','Yellow',
               'Grey','Magenta','Orange','Cyan','Maroon']

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT ARCHITECTURE FROM CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════
def detect_arch(ckpt_path: str) -> dict:
    """
    Inspect checkpoint weight shapes to infer the full model configuration.
    Returns a CFG dict that can be passed directly to the module constructors.
    """
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)

    arch = {
        'patch_size':      2,
        'hidden_dim':      128,
        'latent_dim':      128,
        'num_shape_codes': 256,
        'num_color_codes': 16,
        'vocab_size':      10,
        'grid_size':       30,
        'in_channels':     10,
        'focal_gamma':     2.0,
        'commitment_cost': 0.25,
        'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    }

    for key, val in state.items():
        shape = tuple(val.shape)

        # patch_size & hidden_dim from encoder patch_embed
        # shape: [hidden_dim, in_channels, patch_size, patch_size]
        if 'encoder.patch_embed.weight' in key and val.dim() == 4:
            arch['hidden_dim']  = shape[0]
            arch['patch_size']  = shape[2]

        # latent_dim from projector last linear weight [latent_dim, hidden_dim]
        if 'encoder.projector' in key and '3.weight' in key and val.dim() == 2:
            arch['latent_dim'] = shape[0]

        # VQ codebook sizes
        if 'vq.embedding_shape.weight' in key and val.dim() == 2:
            arch['num_shape_codes'] = shape[0]
            # half_dim = shape[1]  (embedding_dim // 2)

        if 'vq.embedding_color.weight' in key and val.dim() == 2:
            arch['num_color_codes'] = shape[0]

    num_patches = (arch['grid_size'] // arch['patch_size']) ** 2

    print(f"\n{'─'*55}")
    print(f"  Detected architecture:")
    print(f"    patch_size     : {arch['patch_size']}")
    print(f"    hidden_dim     : {arch['hidden_dim']}")
    print(f"    latent_dim     : {arch['latent_dim']}")
    print(f"    num_patches    : {num_patches} ({arch['grid_size']//arch['patch_size']}×{arch['grid_size']//arch['patch_size']})")
    print(f"    num_shape_codes: {arch['num_shape_codes']}")
    print(f"    num_color_codes: {arch['num_color_codes']}")
    print(f"{'─'*55}\n")

    return arch


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_model(ckpt_path: str, cfg: dict):
    from modules.encoders import PatchTransformerEncoder
    from modules.decoders import PatchDecoder
    from modules.vq       import FactorizedVectorQuantizer

    device = cfg['device']

    encoder = PatchTransformerEncoder(cfg).to(device)
    decoder = PatchDecoder(cfg).to(device)
    vq = FactorizedVectorQuantizer(
        num_shape_codes=cfg['num_shape_codes'],
        num_color_codes=cfg['num_color_codes'],
        embedding_dim  =cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
    ).to(device)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)

    enc_state = {k.replace('encoder.','',1): v for k,v in state.items() if k.startswith('encoder.')}
    dec_state = {k.replace('decoder.','',1): v for k,v in state.items() if k.startswith('decoder.')}
    vq_state  = {k.replace('vq.','',1):      v for k,v in state.items() if k.startswith('vq.')}

    miss_e = encoder.load_state_dict(enc_state, strict=False).missing_keys
    miss_d = decoder.load_state_dict(dec_state, strict=False).missing_keys
    miss_v = vq.load_state_dict(vq_state,       strict=False).missing_keys

    if miss_e: print(f"  ⚠ Encoder missing: {miss_e}")
    if miss_d: print(f"  ⚠ Decoder missing: {miss_d}")
    if miss_v: print(f"  ⚠ VQ missing:      {miss_v}")

    encoder.eval(); decoder.eval(); vq.eval()
    return encoder, vq, decoder


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PASS
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(encoder, vq, decoder, dataset, cfg: dict,
                  n_grids: int, n_tsne: int):
    """
    Run n_grids samples through the full pipeline.

    Returns:
        results    : list of dict per grid with keys:
                     task_id, input_np, recon_np, px_acc, perfect
        cm         : np.ndarray (10,10) confusion matrix
        patch_embs : np.ndarray (N, latent_dim)  — for t-SNE
        patch_cols : np.ndarray (N,)             — ARC colour index of each patch
        commit_errs: np.ndarray (M,)             — VQ commitment errors per patch
        shape_usage: np.ndarray (num_shape_codes,)
        color_usage: np.ndarray (num_color_codes,)
    """
    device     = cfg['device']
    patch_size = cfg['patch_size']
    grid_size  = cfg['grid_size']
    Ph = Pw    = grid_size // patch_size
    num_patches = Ph * Pw

    results     = []
    cm          = np.zeros((10, 10), dtype=np.int64)
    patch_embs  = []
    patch_cols  = []
    commit_errs = []
    shape_usage = np.zeros(cfg['num_shape_codes'], dtype=np.int64)
    color_usage = np.zeros(cfg['num_color_codes'], dtype=np.int64)

    batch_size  = min(32, n_grids)
    n_iters     = (n_grids + batch_size - 1) // batch_size

    for it in tqdm(range(n_iters), desc="Running inference"):
        batch  = dataset.sample(batch_size)
        states = batch['state'].to(device)          # [B, 1, H, W]
        B      = states.shape[0]

        # ── Forward pass ────────────────────────────────────────────────────
        z_raw   = encoder({'state': states})['latent']   # [B, N, latent_dim]
        z4      = z_raw.permute(0,2,1).view(B, cfg['latent_dim'], Ph, Pw)
        z_q, _, _, _ = vq(z4)                           # [B, latent_dim, Ph, Pw]
        z_q_seq = z_q.permute(0,2,3,1).view(B, num_patches, cfg['latent_dim'])

        out     = decoder({'latent': z_q_seq})
        logits  = out['reconstructed_logits']       # [B, 10, H, W]
        recon   = logits.argmax(1)                  # [B, H, W]
        target  = states.squeeze(1).long()          # [B, H, W]

        # ── Per-grid metrics ─────────────────────────────────────────────────
        state_np = target.cpu().numpy()
        recon_np = recon.cpu().numpy()
        for b in range(B):
            if len(results) >= n_grids:
                break
            tgt  = state_np[b]     # (30,30)
            rec  = recon_np[b]
            acc  = (tgt == rec).mean()
            perf = (tgt == rec).all()
            task_id = batch.get('task_id', [f'task_{it*batch_size+b}'] * B)[b] if isinstance(batch, dict) else f'task_{it*batch_size+b}'
            results.append({'task_id': task_id,
                             'input_np': tgt, 'recon_np': rec,
                             'px_acc': float(acc), 'perfect': bool(perf)})
            # Confusion matrix
            for true_c in range(10):
                mask = tgt == true_c
                if mask.any():
                    pred_counts = np.bincount(rec[mask], minlength=10)
                    cm[true_c] += pred_counts

        # ── Patch-level stats ────────────────────────────────────────────────
        flat_z     = z_raw.view(-1, cfg['latent_dim']).cpu().numpy()
        flat_shape = torch.from_numpy(flat_z[:, :cfg['latent_dim']//2]).to(device)
        flat_color = torch.from_numpy(flat_z[:, cfg['latent_dim']//2:]).to(device)

        # VQ distances → commitment error = min distance
        d_s = (flat_shape.pow(2).sum(1, keepdim=True)
               + vq.embedding_shape.weight.pow(2).sum(1)
               - 2 * flat_shape @ vq.embedding_shape.weight.T)
        d_c = (flat_color.pow(2).sum(1, keepdim=True)
               + vq.embedding_color.weight.pow(2).sum(1)
               - 2 * flat_color @ vq.embedding_color.weight.T)

        s_idx = d_s.argmin(1); c_idx = d_c.argmin(1)
        min_d_s = d_s.min(1).values; min_d_c = d_c.min(1).values
        commit_errs.append((min_d_s + min_d_c).cpu().numpy())

        shape_usage += np.bincount(s_idx.cpu().numpy(), minlength=cfg['num_shape_codes'])
        color_usage += np.bincount(c_idx.cpu().numpy(), minlength=cfg['num_color_codes'])

        # ── Patch embeddings for t-SNE ───────────────────────────────────────
        if len(patch_embs) * num_patches < n_tsne:
            patch_embs.append(flat_z)
            # Compute dominant colour per patch
            sn = states.squeeze(1).cpu().numpy()   # [B,30,30]
            for b in range(B):
                for ph in range(Ph):
                    for pw in range(Pw):
                        px = sn[b, ph*patch_size:(ph+1)*patch_size,
                                   pw*patch_size:(pw+1)*patch_size].flatten()
                        vals, counts = np.unique(px, return_counts=True)
                        dom = int(vals[counts.argmax()])
                        patch_cols.append(dom)

    return (results,
            cm,
            np.vstack(patch_embs)[:n_tsne] if patch_embs else np.zeros((1, cfg['latent_dim'])),
            np.array(patch_cols[:n_tsne]),
            np.concatenate(commit_errs),
            shape_usage,
            color_usage)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reconstruction_gallery(results, out_dir, n_show=20, cols=4):
    """Panel 1: input | reconstruction | error mask for first n_show grids."""
    print("🎨 Reconstruction gallery...")
    n_show  = min(n_show, len(results))
    rows    = (n_show + cols - 1) // cols
    fig_w   = cols * 6
    fig_h   = rows * 3.5

    fig, axes = plt.subplots(rows * 3, cols, figsize=(fig_w, fig_h))
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_show):
        r        = results[idx]
        row_base = (idx // cols) * 3
        col      = idx % cols

        ax_in  = axes[row_base,     col]
        ax_rec = axes[row_base + 1, col]
        ax_err = axes[row_base + 2, col]

        ax_in.imshow(r['input_np'],  cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
        ax_rec.imshow(r['recon_np'], cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')

        err_mask = (r['input_np'] != r['recon_np']).astype(float)
        ax_err.imshow(err_mask, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')

        acc_pct = r['px_acc'] * 100
        perf    = '✓' if r['perfect'] else ''
        ax_in.set_title(f'Input  {perf}', fontsize=7)
        ax_rec.set_title(f'Recon  {acc_pct:.1f}%', fontsize=7)
        ax_err.set_title('Errors', fontsize=7)

        for ax in (ax_in, ax_rec, ax_err):
            ax.axis('off')

    for idx in range(n_show, rows * cols):
        for rb in range(3):
            axes[(idx // cols) * 3 + rb, idx % cols].axis('off')

    fig.suptitle(f'Reconstruction Gallery — {n_show} samples\n'
                 f'Mean accuracy: {np.mean([r["px_acc"] for r in results[:n_show]])*100:.1f}%  |  '
                 f'Perfect: {sum(r["perfect"] for r in results[:n_show])}/{n_show}',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, '1_reconstruction_gallery.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_confusion_matrix(cm, out_dir):
    """Panel 2: ARC colour confusion matrix."""
    print("📊 Confusion matrix...")
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.where(cm.sum(1, keepdims=True) > 0,
                           cm / cm.sum(1, keepdims=True), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, title in zip(axes,
                                 [cm_norm, cm],
                                 ['Normalised (row = true class)', 'Raw counts']):
        im = ax.imshow(data, cmap='Blues', aspect='auto')
        ax.set_xticks(range(10)); ax.set_xticklabels(ARC_NAMES, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(ARC_NAMES, fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(10):
            for j in range(10):
                val = f'{data[i,j]:.2f}' if data.dtype == float else str(int(data[i,j]))
                ax.text(j, i, val, ha='center', va='center',
                        fontsize=6, color='white' if data[i,j] > data.max()*0.6 else 'black')

    fig.suptitle('Pixel-Level ARC Colour Confusion Matrix', fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, '2_colour_confusion.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_per_task_accuracy(results, out_dir, top_n=40):
    """Panel 3: per-task accuracy — best / worst tasks."""
    print("📈 Per-task accuracy...")
    task_acc = defaultdict(list)
    for r in results:
        task_acc[str(r['task_id'])].append(r['px_acc'])

    task_means = {t: np.mean(v) for t, v in task_acc.items()}
    sorted_tasks = sorted(task_means.items(), key=lambda x: x[1])
    worst  = sorted_tasks[:top_n//2]
    best   = sorted_tasks[-(top_n//2):]
    shown  = worst + best

    names  = [t for t,_ in shown]
    accs   = [a*100 for _,a in shown]
    colors = ['#e74c3c'] * len(worst) + ['#2ecc71'] * len(best)

    fig, ax = plt.subplots(figsize=(14, max(4, len(shown) * 0.35)))
    ax.barh(range(len(shown)), accs, color=colors)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(np.mean(accs), color='#f39c12', lw=2, ls='--', label=f'Mean {np.mean(accs):.1f}%')
    ax.set_xlabel('Pixel Accuracy (%)')
    ax.set_title(f'Per-Task Accuracy  (worst {len(worst)} red, best {len(best)} green)', fontsize=12)
    ax.legend(fontsize=9)
    legend_elements = [Patch(color='#e74c3c', label='Worst tasks'),
                        Patch(color='#2ecc71', label='Best tasks')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, '3_per_task_accuracy.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_commit_distribution(commit_errs, out_dir):
    """Panel 4: histogram of per-patch VQ commitment errors."""
    print("📉 Commitment error distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    q95 = np.percentile(commit_errs, 95)
    clipped = commit_errs[commit_errs <= q95 * 2]  # remove extreme outliers for viz
    ax.hist(clipped, bins=80, color='#3498db', edgecolor='none', alpha=0.8)
    ax.axvline(np.median(commit_errs), color='#e74c3c', lw=2,
               label=f'Median: {np.median(commit_errs):.3f}')
    ax.axvline(np.mean(commit_errs),   color='#f39c12', lw=2, ls='--',
               label=f'Mean:   {np.mean(commit_errs):.3f}')
    ax.set_xlabel('VQ Commitment Error (shape + colour distance to nearest code)')
    ax.set_ylabel('Patch count')
    ax.set_title('VQ Commitment Error Distribution\n'
                 '(lower = encoder outputs are close to codebook entries = well-trained VQ)',
                 fontsize=11)
    ax.legend()

    # Inset: long-tail log scale
    ax2 = ax.inset_axes([0.65, 0.45, 0.32, 0.45])
    ax2.hist(commit_errs, bins=80, color='#3498db', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_title('Log scale', fontsize=8)
    ax2.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, '4_commitment_errors.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_tsne(patch_embs, patch_cols, cfg, out_dir, max_pts=3000):
    """Panel 5: t-SNE of patch latent vectors coloured by ARC colour."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("⚠️  sklearn not available — skipping t-SNE. Install with: pip install scikit-learn")
        return

    print(f"🔵 t-SNE of {min(len(patch_embs), max_pts)} patch embeddings...")
    idx = np.random.choice(len(patch_embs), min(max_pts, len(patch_embs)), replace=False)
    X   = patch_embs[idx]
    C   = patch_cols[idx]

    # PCA first to 50 dims for speed
    n_pca = min(50, X.shape[1])
    X_pca = PCA(n_components=n_pca).fit_transform(X)
    X_2d  = TSNE(n_components=2, perplexity=40, n_iter=800,
                 init='pca', random_state=42).fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(10, 9))
    for c in range(10):
        mask = C == c
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=ARC_COLORS[c], s=6, alpha=0.6, label=ARC_NAMES[c])
    ax.set_title(f't-SNE of Patch Latent Space  (patch_size={cfg["patch_size"]})\n'
                 f'Colour = dominant ARC colour in that patch',
                 fontsize=12)
    ax.legend(fontsize=8, markerscale=3, ncol=2)
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    plt.tight_layout()
    path = os.path.join(out_dir, '5_patch_tsne.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_codebook_load(shape_usage, color_usage, out_dir):
    """Panel 6: shape and colour codebook utilisation."""
    print("📊 Codebook load balancing...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))

    s_norm = shape_usage / shape_usage.sum() if shape_usage.sum() > 0 else shape_usage
    c_norm = color_usage / color_usage.sum() if color_usage.sum() > 0 else color_usage
    s_ent  = -np.sum(s_norm[s_norm>0] * np.log(s_norm[s_norm>0]))
    c_ent  = -np.sum(c_norm[c_norm>0] * np.log(c_norm[c_norm>0]))
    s_max_ent = np.log(len(shape_usage))
    c_max_ent = np.log(len(color_usage))

    ax1.bar(range(len(shape_usage)), shape_usage, color='#3498db', width=1.0)
    ax1.set_title(f'Shape Codebook Utilisation  '
                  f'Active: {(shape_usage>0).sum()}/{len(shape_usage)}  |  '
                  f'Entropy: {s_ent:.2f}/{s_max_ent:.2f} = {s_ent/s_max_ent:.1%}',
                  fontsize=11)
    ax1.set_xlabel('Code index'); ax1.set_ylabel('Patch assignments')

    ax2.bar(range(len(color_usage)), color_usage, color='#2ecc71', width=0.8)
    ax2.set_title(f'Colour Codebook Utilisation  '
                  f'Active: {(color_usage>0).sum()}/{len(color_usage)}  |  '
                  f'Entropy: {c_ent:.2f}/{c_max_ent:.2f} = {c_ent/c_max_ent:.1%}',
                  fontsize=11)
    ax2.set_xlabel('Code index'); ax2.set_ylabel('Patch assignments')

    plt.tight_layout()
    path = os.path.join(out_dir, '6_codebook_load.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_accuracy_distribution(results, out_dir):
    """Panel 7: histogram of per-grid pixel accuracy."""
    print("📊 Accuracy distribution...")
    accs = [r['px_acc'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(accs, bins=30, color='#9b59b6', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(accs),   color='#e74c3c', lw=2.5, label=f'Mean  {np.mean(accs):.1f}%')
    ax.axvline(np.median(accs), color='#f39c12', lw=2.5, ls='--',
               label=f'Median {np.median(accs):.1f}%')
    ax.axvline(90, color='gray', lw=1, ls=':', label='90% line')
    ax.set_xlabel('Per-Grid Pixel Accuracy (%)')
    ax.set_ylabel('Number of grids')
    ax.set_title(f'Reconstruction Accuracy Distribution  (n={len(results)})\n'
                 f'Perfect reconstructions: {sum(r["perfect"] for r in results)}/{len(results)} '
                 f'({sum(r["perfect"] for r in results)/len(results):.1%})',
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, '7_accuracy_distribution.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def save_summary(results, cm, shape_usage, color_usage, cfg, arch_info, out_dir):
    """Save JSON summary + print terminal report."""
    accs    = [r['px_acc'] for r in results]
    perfect = sum(r['perfect'] for r in results)
    n       = len(results)

    # Per-colour accuracy from confusion matrix
    colour_acc = {}
    for c in range(10):
        total = cm[c].sum()
        colour_acc[ARC_NAMES[c]] = float(cm[c, c] / total) if total > 0 else 0.0

    summary = {
        'architecture':        arch_info,
        'n_grids_evaluated':   n,
        'mean_pixel_acc':      float(np.mean(accs)),
        'median_pixel_acc':    float(np.median(accs)),
        'std_pixel_acc':       float(np.std(accs)),
        'perfect_reconstructions': perfect,
        'perfect_pct':         float(perfect / n),
        'per_colour_accuracy': colour_acc,
        'shape_codes_active':  int((shape_usage > 0).sum()),
        'color_codes_active':  int((color_usage > 0).sum()),
        'shape_codebook_size': int(len(shape_usage)),
        'color_codebook_size': int(len(color_usage)),
    }

    path = os.path.join(out_dir, 'model_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"  AUDIT SUMMARY")
    print(f"{'─'*55}")
    print(f"  Grids evaluated  : {n}")
    print(f"  Mean accuracy    : {np.mean(accs)*100:.2f}%")
    print(f"  Median accuracy  : {np.median(accs)*100:.2f}%")
    print(f"  Perfect grids    : {perfect}/{n}  ({perfect/n:.1%})")
    print(f"  Shape codes used : {(shape_usage>0).sum()}/{len(shape_usage)}")
    print(f"  Colour codes used: {(color_usage>0).sum()}/{len(color_usage)}")
    print(f"\n  Per-colour accuracy:")
    for name, acc in sorted(colour_acc.items(), key=lambda x: x[1]):
        bar = '█' * int(acc * 20)
        print(f"    {name:<10} {acc*100:5.1f}%  {bar}")
    print(f"{'═'*55}\n")
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Full Phase-0 model validation and visualization audit',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Phase 0 checkpoint (.pth)')
    parser.add_argument('--out',        type=str,
                        default='evaluation_reports/model_audit',
                        help='Output directory for plots and JSON')
    parser.add_argument('--n_grids',   type=int, default=200,
                        help='Number of grids to reconstruct and evaluate')
    parser.add_argument('--n_tsne',    type=int, default=3000,
                        help='Number of patch embeddings for t-SNE (expensive)')
    parser.add_argument('--n_show',    type=int, default=20,
                        help='Number of grids to show in reconstruction gallery')
    parser.add_argument('--data_path', type=str, default='arc_data/re-arc')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"{'═'*55}")
    print(f"  Full Model Audit")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output     : {args.out}")
    print(f"  Grids      : {args.n_grids}")
    print(f"{'═'*55}")

    # ── Auto-detect architecture ─────────────────────────────────────────────
    cfg = detect_arch(args.checkpoint)
    cfg['data_path'] = args.data_path

    # ── Load model ───────────────────────────────────────────────────────────
    print("⚙️  Loading model...")
    encoder, vq, decoder = load_model(args.checkpoint, cfg)

    # ── Load dataset ─────────────────────────────────────────────────────────
    from arc_data.rearc_dataset import ReARCDataset
    dataset = ReARCDataset(data_path=args.data_path)
    print(f"  Dataset: {len(dataset)} samples")

    # ── Inference ────────────────────────────────────────────────────────────
    results, cm, patch_embs, patch_cols, commit_errs, shape_usage, color_usage = \
        run_inference(encoder, vq, decoder, dataset, cfg,
                      n_grids=args.n_grids, n_tsne=args.n_tsne)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📊 Generating plots...")
    plot_reconstruction_gallery(results, args.out, n_show=args.n_show)
    plot_confusion_matrix(cm, args.out)
    plot_per_task_accuracy(results, args.out)
    plot_commit_distribution(commit_errs, args.out)
    plot_tsne(patch_embs, patch_cols, cfg, args.out, max_pts=args.n_tsne)
    plot_codebook_load(shape_usage, color_usage, args.out)
    plot_accuracy_distribution(results, args.out)
    save_summary(results, cm, shape_usage, color_usage, cfg, cfg, args.out)

    print(f"\n✅ Audit complete → {args.out}/")
    print(f"   scp -r eet242799@10.225.67.239:~/development/Model-Jepa/{args.out} ~/Desktop/")


if __name__ == '__main__':
    main()
