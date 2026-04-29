import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
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

    print(f"🏃 Running {args.batches} batches to collect codebook statistics...")
    with torch.no_grad():
        for i in range(args.batches):
            batch = dataset.sample(args.batch_size)
            states = batch['state'].to(device)  # [B, 1, H, W]
            
            # --- Phase 0 (Patch Alphabet) ---
            if os.path.exists(p0_ckpt):
                # Run Phase 0 forward
                # out, vq_loss, p_shape, p_color, z_vq, shape_idx
                _, _, _, _, _, shape_idx = p0_model({'state': states}, valid_mask=None, temperature=0.1)
                p0_usage.update(shape_idx.flatten().cpu().numpy().tolist())
                
            # --- Phase 1 (Slot Vocabulary) ---
            if os.path.exists(p1_ckpt):
                out1 = p1_enc({'state': states})
                slots = out1['latent']  # [B, K, 320]
                
                # Extract shape part and map to codebook
                slot_shape = slots[:, :, :128]  # [B, K, 128]
                if hasattr(p1_enc, 'codebook_shape'):
                    cb = p1_enc.codebook_shape      # [V, 128]
                    # Cosine similarity matching
                    slot_shape_n = F.normalize(slot_shape.reshape(-1, 128), dim=-1)
                    cb_n = F.normalize(cb, dim=-1)
                    sims = slot_shape_n @ cb_n.T  # [B*K, V]
                    nearest_idx = sims.argmax(dim=-1)  # [B*K]
                    
                    p1_usage.update(nearest_idx.cpu().numpy().tolist())

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

    # 7. --- Visual Surgery (Decoding Top Codes) ---
    if os.path.exists(p1_ckpt):
        print("\n🔪 Running Visual Surgery (Decoding Top 5 Phase 1 Codes)...")
        top_codes = [c for c, count in p1_usage.most_common(5)]
        
        with torch.no_grad():
            cb = p1_enc.codebook_shape # [V, 128]
            for idx, code_id in enumerate(top_codes):
                # Create a synthetic slot: Shape from codebook, neutral Pose/Color
                shape_emb = cb[code_id].unsqueeze(0).unsqueeze(0) # [1, 1, 128]
                # Fill remaining 192 dims (color + pose) with zeros
                dummy_latent = torch.zeros((1, 1, 192), device=device)
                full_latent = torch.cat([shape_emb, dummy_latent], dim=-1) # [1, 1, 320]
                
                # Decode
                decoded = p1_dec({'latent': full_latent})
                recon = decoded['reconstruction'].argmax(dim=1).cpu().numpy()[0] # [H, W]
                
                # Print a small ASCII representation for the terminal audit
                print(f"Code #{code_id} Visual Primitive (High-res stored in {args.out_dir}):")
                for row in recon[:10, :10]: # show 10x10 slice
                    print(" ".join(str(int(p)) if p > 0 else "." for p in row))
                print("-" * 20)

    # 8. Report Summary
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
