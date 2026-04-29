import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from matplotlib.colors import ListedColormap

# ── ARC colour palette ──────────────────────────────────────────────────────
ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
               '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
ARC_CMAP   = ListedColormap(ARC_COLORS)
import math

from arc_data.rearc_dataset import ReARCDataset
from train_phase0 import Phase0Autoencoder
from modules.semantic_encoders import SemanticSlotEncoder
from modules.semantic_decoders import SemanticDecoder

def entropy(counts):
    total = sum(counts)
    if total == 0: return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True, help="Path to the run directory (e.g., runs/ObjectCodebook-v1_...)")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batches', type=int, default=5, help="Number of validation batches to run for the audit")
    parser.add_argument('--out-dir', type=str, default='audit_results', help="Where to save audit plots/logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Starting Codebook Audit on device: {device}")

    # 1. Load config
    config_path = os.path.join(args.run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    cfg['device'] = device

    # 2. Instantiate Models
    print("📦 Loading models...")
    p0_model = Phase0Autoencoder(cfg).to(device)
    p1_enc = SemanticSlotEncoder(cfg).to(device)
    p1_dec = SemanticDecoder(cfg).to(device)

    # 3. Load Checkpoints
    p0_ckpt = os.path.join(args.run_dir, 'phase0', 'latest_checkpoint.pth')
    p1_ckpt = os.path.join(args.run_dir, 'phase1', 'latest_slot_checkpoint.pth')
    
    print(f"🔎 Checking P0: {p0_ckpt}")
    print(f"🔎 Checking P1: {p1_ckpt}")
    
    if os.path.exists(p0_ckpt):
        p0_model.load_state_dict(torch.load(p0_ckpt, map_location=device)['model'])
        print("✅ Phase 0 Checkpoint loaded.")
        # We MUST inject the codebook into Phase 1 encoder so the parameters exist for loading
        print("💉 Injecting codebook into Phase 1 encoder...")
        p1_enc.inject_codebook(p0_model.vq)
    else:
        print("⚠️  Phase 0 Checkpoint not found! Skipping Phase 0 audit.")

    if os.path.exists(p1_ckpt):
        state1 = torch.load(p1_ckpt, map_location=device)
        p1_enc.load_state_dict(state1['slot_enc'])
        p1_dec.load_state_dict(state1['slot_dec'])
        print("✅ Phase 1 Checkpoint loaded.")
    else:
        print("⚠️  Phase 1 Checkpoint not found! Skipping Phase 1 audit.")

    p0_model.eval()
    p1_enc.eval()
    p1_dec.eval()

    # 4. Load Data
    dataset = ReARCDataset(data_path=cfg.get('data_path', 'data/rearc'))

    # 5. Track metrics
    p0_usage = Counter()
    p1_usage = Counter()
    p1_color_usage = Counter()
    p1_poses = []

    print(f"🏃 Running {args.batches} batches to collect codebook statistics...")
    with torch.no_grad():
        for i in tqdm(range(args.batches)):
            batch = dataset.sample(args.batch_size)
            states = batch['state'].to(device)  # [B, 1, H, W]
            
            # --- Phase 0 (Patch Alphabet) ---
            if os.path.exists(p0_ckpt):
                # Run Phase 0 forward
                # out, vq_loss, p_shape, p_color, z_vq, shape_idx
                res = p0_model({'state': states}, temperature=0.1)
                shape_idx = res[5]
                p0_usage.update(shape_idx.flatten().cpu().numpy().tolist())
                
            # --- Phase 1 (Object Vocabulary) ---
            if os.path.exists(p1_ckpt):
                out1 = p1_enc({'state': states})
                slots = out1['latent']  # [B, K, 320]
                masks = out1['masks']   # [B, K, H, W]
                
                # A. Shape Usage (0:128)
                slot_shape = slots[:, :, :128]
                cb_s = p1_enc.codebook_shape
                s_n = F.normalize(slot_shape.reshape(-1, 128), dim=-1)
                c_s_n = F.normalize(cb_s, dim=-1)
                sims_s = s_n @ c_s_n.T
                p1_usage.update(sims_s.argmax(dim=-1).cpu().numpy().tolist())

                # B. Color Usage (128:256)
                slot_color = slots[:, :, 128:256]
                cb_c = p1_enc.codebook_color
                c_n = F.normalize(slot_color.reshape(-1, 128), dim=-1)
                c_c_n = F.normalize(cb_c, dim=-1)
                sims_c = c_n @ c_c_n.T
                p1_color_usage.update(sims_c.argmax(dim=-1).cpu().numpy().tolist())

                # C. Pose Centroids
                for b in range(masks.shape[0]):
                    for k in range(masks.shape[1]):
                        mask = masks[b, k]
                        if mask.sum() > 0.1:
                            # Weighted average for centroid
                            gh, gw = mask.shape
                            y_indices, x_indices = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
                            y_indices = y_indices.to(device).float()
                            x_indices = x_indices.to(device).float()
                            
                            y_c = (mask * y_indices).sum() / mask.sum()
                            x_c = (mask * x_indices).sum() / mask.sum()
                            
                            # Scale to 30x30 coordinates
                            patch_size = 30 // gh
                            p1_poses.append([y_c.item() * patch_size, x_c.item() * patch_size])

    # 6. --- Affine Invariance Test ---
    if os.path.exists(p1_ckpt):
        print("\n🔄 Running Affine Invariance Test (Rotations)...")
        invariance_hits = 0
        total_objects = 0
        
        # Take a single sample for rotation test
        batch = dataset.sample(1)
        states = batch['state'].to(device) # [B, 1, H, W]
        
        def get_codes(x):
            with torch.no_grad():
                out = p1_enc({'state': x})
                slots = out['latent'] # [B, K, 320]
                shape_p = slots[:, :, :128]
                cb = p1_enc.codebook_shape
                s_n = F.normalize(shape_p.reshape(-1, 128), dim=-1)
                c_n = F.normalize(cb, dim=-1)
                indices = (s_n @ c_n.T).argmax(dim=-1).reshape(slots.shape[0], slots.shape[1])
                return indices.cpu().numpy()

        orig_codes = get_codes(states) # [B, K]
        
        # Rotate 90 degrees
        states_90 = torch.rot90(states, k=1, dims=[2, 3])
        rot_codes = get_codes(states_90) # [B, K]
        
        # In Slot Attention, slots are permutation invariant, so we check set intersection
        for b in range(states.shape[0]):
            set_orig = set(orig_codes[b])
            set_rot = set(rot_codes[b])
            # Intersection tells us which codes survived the rotation
            hits = len(set_orig.intersection(set_rot))
            invariance_hits += hits
            total_objects += len(set_orig)
            
        inv_ratio = (invariance_hits / total_objects) * 100 if total_objects > 0 else 0
        print(f"✅ Invariance Match: {inv_ratio:.1f}% of object codes survived 90° rotation.")

    # 7. --- Visual Surgery (Full Shape Codebook Map) ---
    if os.path.exists(p1_ckpt):
        print(f"\n🔪 Generating Full Shape Codebook Map (128 Primitives)...")
        vocab_size = cfg.get('num_shape_codes', 128)
        num_to_show = min(vocab_size, 128)
        cols = 16
        rows = (num_to_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = axes.flatten()

        with torch.no_grad():
            for i in range(num_to_show):
                # Construct latent: Shape(i) + Neutral Color(5) + Zero Pose
                # Color 5 is Grey in ARC
                full_latent = torch.zeros(1, 1, 320, device=device)
                if i < p1_enc.codebook_shape.shape[0]:
                    full_latent[0, 0, :128] = p1_enc.codebook_shape[i]
                
                # We need a valid color embedding. We'll use the first one if available.
                if hasattr(p1_enc, 'codebook_color'):
                    full_latent[0, 0, 128:256] = p1_enc.codebook_color[min(5, p1_enc.codebook_color.shape[0]-1)]

                # Decode
                decoded = p1_dec({'latent': full_latent})
                recon = decoded['reconstruction'].argmax(dim=1).cpu().numpy()[0] # [H, W]
                
                ax = axes[i]
                ax.imshow(recon, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
                ax.set_title(f"#{i}", fontsize=8)
                ax.axis('off')

        # Hide unused subplots
        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Phase 1: Full Shape Codebook Map (Total Codes: {vocab_size})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        map_path = os.path.join(args.out_dir, "full_shape_codebook.png")
        plt.savefig(map_path, dpi=120)
        plt.close()
        print(f"✅ Full codebook map saved to {map_path}")

    # 8. --- Generate Final Audit Plots ---
    print(f"\n📈 Generating Final Audit Plots in {args.out_dir}...")
    vocab_size = cfg.get('num_shape_codes', 128)
    
    # Plot 1: Shape Utilization
    if p1_usage:
        plt.figure(figsize=(12, 5))
        counts = [p1_usage[i] for i in range(vocab_size)]
        plt.bar(range(vocab_size), counts, color='teal', alpha=0.7)
        plt.title("Phase 1: Shape Codebook Utilization")
        plt.savefig(os.path.join(args.out_dir, "shape_utilization.png"))
        plt.close()

    # Plot 2: Color Utilization
    if p1_color_usage:
        plt.figure(figsize=(8, 5))
        n_colors = cfg.get('num_color_codes', 16)
        counts = [p1_color_usage[i] for i in range(n_colors)]
        plt.bar(range(n_colors), counts, color='magenta', alpha=0.7)
        plt.title("Phase 1: Color Codebook Utilization")
        plt.savefig(os.path.join(args.out_dir, "color_utilization.png"))
        plt.close()

    # Plot 3: Pose Heatmap
    if p1_poses:
        plt.figure(figsize=(6, 6))
        p_arr = np.array(p1_poses)
        plt.hexbin(p_arr[:, 1], p_arr[:, 0], gridsize=15, cmap='inferno')
        plt.xlim(0, 30); plt.ylim(30, 0) # Invert Y for grid coordinates
        plt.title("Phase 1: Object Pose Heatmap (Spatial Centroids)")
        plt.savefig(os.path.join(args.out_dir, "pose_usage_heatmap.png"))
        plt.close()

    # Plot 4: Invariance Summary
    if os.path.exists(p1_ckpt):
        plt.figure(figsize=(6, 5))
        plt.bar(['Invariance Score'], [inv_ratio], color='orange', alpha=0.8)
        plt.ylim(0, 100)
        plt.ylabel("Percentage (%)")
        plt.title("Affine Invariance Test (90° Rotation)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(args.out_dir, "affine_invariance_score.png"))
        plt.close()

    # 9. Report Summary
    print("\n" + "="*50)
    print("📊 CODEBOOK AUDIT RESULTS SUMMARY")
    print("="*50)
    
    vocab_size = cfg.get('num_shape_codes', 128)
    
    if os.path.exists(p0_ckpt):
        p0_ent = entropy([p0_usage[i] for i in range(vocab_size)])
        print(f"Phase 0 Entropy: {p0_ent:.3f} bits")

    if os.path.exists(p1_ckpt):
        p1_ent = entropy([p1_usage[i] for i in range(vocab_size)])
        print(f"Phase 1 Entropy: {p1_ent:.3f} bits")
        print(f"Invariance Score: {inv_ratio:.1f}%")
        
    print("\n💡 Final Analysis:")
    if os.path.exists(p1_ckpt) and inv_ratio > 80:
        print("🌟 EXCELLENT: High Invariance Score confirms objects are assigned stable codes regardless of pose.")
    elif os.path.exists(p1_ckpt) and inv_ratio < 40:
        print("⚠️  POOR: Low Invariance Score suggests the model is 'memorizing' pose+shape together. Check VICReg weights.")
    
    print("="*50)

if __name__ == '__main__':
    main()
