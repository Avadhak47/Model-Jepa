import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from sklearn.manifold import TSNE
import warnings

# Try to load HDBSCAN either from sklearn (new) or external library
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan
        HDBSCAN = hdbscan.HDBSCAN
    except ImportError:
        warnings.warn("HDBSCAN not found in sklearn and 'hdbscan' package is missing. Fallback to DBSCAN.")
        from sklearn.cluster import DBSCAN
        # Wrapper to make DBSCAN act slightly like HDBSCAN syntax
        class HDBSCAN(DBSCAN):
            def __init__(self, min_cluster_size=5):
                super().__init__(eps=0.6, min_samples=min_cluster_size)

from arc_data.arc_dataset import ARCDataset
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder as Decoder

class DeepTransformerEncoder(TransformerEncoder):
    def __init__(self, config):
        cfg = dict(config)
        cfg['_enc_depth'] = cfg.get('enc_depth', 4)
        super().__init__(cfg)
        embed_dim = cfg.get('hidden_dim', 256)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4,
            batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=cfg['_enc_depth'])

BASE_CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'input_dim': 64,  
    'in_channels': 1,
    'patch_size': 14,
    'hidden_dim': 256,
    'latent_dim': 128,
    'enc_depth': 4,
    'vocab_size': 50,
}

def swap_colors_fn(tensor):
    state_val = tensor.cpu().numpy()
    unique_c = np.unique(state_val)
    c1, c2 = 0, 1
    if len(unique_c) >= 2:
        c1, c2 = int(unique_c[0]), int(unique_c[1])
    if len(unique_c) >= 3 and 0 in unique_c:
        c1, c2 = int(unique_c[1]), int(unique_c[2])
    
    swapped = tensor.clone()
    mask1 = tensor == c1
    mask2 = tensor == c2
    swapped[mask1] = c2
    swapped[mask2] = c1
    return swapped

def main():
    device = BASE_CFG['device']
    print(f"Using device: {device}")

    modules = {
        'encoder': DeepTransformerEncoder(BASE_CFG).to(device),
        'decoder': Decoder(BASE_CFG).to(device)
    }

    api = wandb.Api()
    run = api.run("avadheshkumarajay-indian-institute-of-technology-delhi/NS-ARC-Scaling/q24acjes")
    print("Downloading weights...")
    for mod_name in ['encoder', 'decoder']:
        artifact = api.artifact(f"{run.entity}/{run.project}/NSARC-32_{mod_name}:latest")
        artifact_dir = artifact.download()
        for f in os.listdir(artifact_dir):
            if f.endswith('.pt'):
                state_dict = torch.load(os.path.join(artifact_dir, f), map_location=device, weights_only=True)
                modules[mod_name].load_state_dict(state_dict)
                modules[mod_name].eval()

    # Load 500 grids from training, 500 grids from eval
    train_dataset = ARCDataset(data_path="ARC-AGI/data/training")
    eval_dataset = ARCDataset(data_path="ARC-AGI/data/evaluation")
    print(f"Dataset Size => Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    X_latents = []
    meta_split = [] # 0: Train, 1: Eval
    meta_transform = [] # 0: Orig, 1: HFlip, 2: Rot90, 3: ColorSwap
    meta_base_id = [] # Traces transformed versions back to base grid ID
    
    base_idx_counter = 0

    def process_split(dataset, split_id, max_samples=400):
        nonlocal base_idx_counter
        with torch.no_grad():
            for i in range(min(max_samples, len(dataset))):
                batch = dataset.sample(1)
                state = batch['state'].to(device)
                
                state_hflip = torch.flip(state, dims=[3])
                state_rot90 = torch.rot90(state, k=1, dims=[2, 3])
                state_color = swap_colors_fn(state)

                # Encode
                z_o = modules['encoder']({'state': state})['latent']
                z_h = modules['encoder']({'state': state_hflip})['latent']
                z_r = modules['encoder']({'state': state_rot90})['latent']
                z_c = modules['encoder']({'state': state_color})['latent']

                # Store vectors and metadata
                for trans_id, vector in enumerate([z_o, z_h, z_r, z_c]):
                    X_latents.append(vector.cpu().numpy().flatten())
                    meta_split.append(split_id)
                    meta_transform.append(trans_id)
                    meta_base_id.append(base_idx_counter)
                
                base_idx_counter += 1

    print("Encoding matrices across all dimensions and distributions...")
    process_split(train_dataset, 0, max_samples=400)
    process_split(eval_dataset, 1, max_samples=400)

    X_latents = np.array(X_latents)
    meta_split = np.array(meta_split)
    meta_transform = np.array(meta_transform)
    meta_base_id = np.array(meta_base_id)

    print(f"Extracted Grand Total Matrix: {X_latents.shape} (Dimensions: {X_latents.shape[1]})")

    # ========================================================
    # 1. N-Dimensional Unsupervised Clustering (HDBSCAN)
    # ========================================================
    print("Initiating pure N-dimensional HDBSCAN Density Clustering...")
    # min_cluster_size=4 because a base element usually has 4 variations (Orig, Flip, Rot, Color)
    clusterer = HDBSCAN(min_cluster_size=4)
    cluster_labels = clusterer.fit_predict(X_latents)

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = list(cluster_labels).count(-1)
    
    print(f"\n--- Topology Report ---")
    print(f"Algorithm isolated {num_clusters} dense neural clusters natively in the 128D space.")
    print(f"Data points flagged as structureless noise (Outliers): {noise_count} / {len(X_latents)} ({noise_count/len(X_latents)*100:.1f}%)")

    # ========================================================
    # 2. Relational Cross-Analysis
    # ========================================================
    # Assess semantic disjointedness (Do Evaluation tasks map to specific Training clusters?)
    for cluster_id in set(cluster_labels):
        if cluster_id == -1: continue
        c_mask = (cluster_labels == cluster_id)
        c_train = np.sum(meta_split[c_mask] == 0)
        c_eval = np.sum(meta_split[c_mask] == 1)
        # print specific unique info directly to file or log

    # Evaluate Orbits: Do transforms stay in their parent cluster?
    parent_preservation = 0
    total_valid_pairs = 0
    # Group by base_id
    for bid in set(meta_base_id):
        idx_mask = np.where(meta_base_id == bid)[0]
        # the first one is transformation 0 (orig)
        orig_cluster = cluster_labels[idx_mask[0]]
        if orig_cluster != -1:
            # Check if derived copies mutated out of the dense zone
            for other_idx in idx_mask[1:]:
                if cluster_labels[other_idx] == orig_cluster:
                    parent_preservation += 1
                total_valid_pairs += 1

    if total_valid_pairs > 0:
        print(f"Geometrical Cluster Invariance: {parent_preservation/total_valid_pairs*100:.1f}%")
        print("\t(Percentage of transformed grids that the Model natively perceived as still belonging to the exact same dense cluster hierarchy as its original source grid).")

    # ========================================================
    # 3. Visualization mapping
    # ========================================================
    print("Compressing representations for high-fidelity 2D visualization rendering...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_latents)

    os.makedirs("evaluation_reports/clustering", exist_ok=True)
    
    # Exclude noise (-1) for clearer viewing of structured classes
    non_noise_mask = cluster_labels != -1

    sns.set_style('darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"HDBSCAN Cluster Topography (128 Dimensions -> 2D Projection)\n{num_clusters} semantic nodes isolated", fontsize=16)

    # Panel 1: Split Coloring
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=meta_split, cmap='coolwarm', alpha=0.6, s=15)
    axes[0].set_title("Manifold Density (Blue: Train, Red: Eval)")
    axes[0].axis('off')

    # Panel 2: Transformation Coloring
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=meta_transform, cmap='Set1', alpha=0.6, s=15)
    axes[1].set_title("Transformation Vectors\n(Red: Orig, Blue: Flip, Green: Rot, Purp: Color)")
    axes[1].axis('off')

    # Panel 3: HDBSCAN Native Clusters
    scatter3 = axes[2].scatter(X_2d[non_noise_mask, 0], X_2d[non_noise_mask, 1], 
                               c=cluster_labels[non_noise_mask], cmap='tab20', alpha=0.8, s=20)
    
    # Plot Noise vaguely in background
    axes[2].scatter(X_2d[~non_noise_mask, 0], X_2d[~non_noise_mask, 1], 
                    color='gray', alpha=0.2, s=5, label='Noise / Disassociated')
    
    axes[2].set_title("Raw Cluster Topography mapped purely by HDBSCAN")
    axes[2].axis('off')

    plt.tight_layout()
    map_path = "evaluation_reports/clustering/overall_manifold_map.png"
    plt.savefig(map_path, dpi=200)
    plt.close()
    
    # Write explicit statistical mapping logs to a local file
    with open("evaluation_reports/clustering/cluster_math_summary.txt", "w") as f:
        f.write("HDBSCAN Unsupervised Metric Breakdown\n")
        f.write("======================================\n")
        f.write(f"Total Isolated Semantic Clusters: {num_clusters}\n")
        f.write(f"Background Noise Ratio: {noise_count/len(X_latents)*100:.1f}%\n")
        
        if total_valid_pairs > 0:
            f.write(f"\nTransformation Cohesion Score: {parent_preservation/total_valid_pairs*100:.1f}%\n")
            f.write(" - Note: Measures whether applying physical symmetries knocks vectors out of their neural category.\n")

if __name__ == "__main__":
    main()
