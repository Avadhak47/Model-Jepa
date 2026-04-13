import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import wandb
import seaborn as sns

from arc_data.arc_dataset import ARCDataset
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder as Decoder
from analysis.plot_utils import plot_reconstruction_dashboard

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

def analyze_latent_space():
    device = BASE_CFG['device']
    print(f"Using device: {device}")

    # 1. Initialize Models
    modules = {
        'encoder': DeepTransformerEncoder(BASE_CFG).to(device),
        'decoder': Decoder(BASE_CFG).to(device)
    }

    # 2. Download from W&B
    api = wandb.Api()
    run_path = "avadheshkumarajay-indian-institute-of-technology-delhi/NS-ARC-Scaling/q24acjes"
    model_name = "NSARC-32"
    run = api.run(run_path)
    
    print("Downloading weights...")
    for mod_name in ['encoder', 'decoder']:
        artifact_name = f"{model_name}_{mod_name}:latest"
        artifact = api.artifact(f"{run.entity}/{run.project}/{artifact_name}")
        artifact_dir = artifact.download()
        for f in os.listdir(artifact_dir):
            if f.endswith('.pt'):
                state_dict = torch.load(os.path.join(artifact_dir, f), map_location=device, weights_only=True)
                modules[mod_name].load_state_dict(state_dict)
                modules[mod_name].eval()

    # 3. Load Dataset
    data_path = "ARC-AGI/data/evaluation"
    dataset = ARCDataset(data_path=data_path)
    
    # 4. Extract Concepts and Latent Vectors
    print("Extracting representations...")
    latents = []
    unique_colors = []
    grid_mass = []  # number of non-zero pixels
    
    original_grids = []
    reconstructed_grids = []
    latent_tensors = []

    with torch.no_grad():
        for i in range(min(500, len(dataset))):  # Process up to 500 samples
            # Directly sample to iterate through tasks
            batch = dataset.sample(1)
            state = batch['state'].to(device)
            target = batch['target_latent'].to(device) # it's out_pad! Same shape
            
            z_dict = modules['encoder']({'state': state})
            z = z_dict['latent']  # [1, 128]
            
            # Reconstruction
            out = modules['decoder']({'latent': z, 'state': state})
            recon = out.get('reconstruction', out.get('reconstructed_logits')) # [1, 50, 30, 30]
            
            latents.append(z.cpu().numpy().flatten())
            latent_tensors.append(z)
            
            # Record Original and Recon
            original_grids.append(state)
            reconstructed_grids.append(recon)
            
            # Extract concepts from input array
            arr = state.cpu().numpy().flatten()
            unique_colors.append(len(np.unique(arr)))
            grid_mass.append(np.count_nonzero(arr))

    latents = np.array(latents)
    unique_colors = np.array(unique_colors)
    grid_mass = np.array(grid_mass)
    
    print(f"Extracted {latents.shape[0]} latent representations.")

    # 5. Dimensionality Reduction (PCA & t-SNE)
    os.makedirs("evaluation_reports/explore", exist_ok=True)
    
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # Plot PCA colored by 'Number of Unique Colors'
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=unique_colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Unique Colors')
    plt.title('Latent Space (PCA)\nColored by Palette Complexity')
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    
    # Plot PCA colored by 'Grid Mass (Non-Zero Pixels)'
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=grid_mass, cmap='plasma', alpha=0.7)
    plt.colorbar(scatter2, label='Grid Mass (Non-Zero Pixels)')
    plt.title('Latent Space (PCA)\nColored by Structural Density')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    
    plt.tight_layout()
    pca_path = "evaluation_reports/explore/latent_pca_concepts.png"
    plt.savefig(pca_path, dpi=150)
    plt.close()
    
    print(f"Saved Latent Space PCA visualization to {pca_path}")

    # 6. Generate Error Heatmap Dashboards for 3 Samples
    print("Generating Error Heatmaps...")
    for i in range(3):
        idx = np.random.randint(0, len(original_grids))
        save_path = f"evaluation_reports/explore/error_heatmap_sample_{i+1}.png"
        
        # Recon is [1, 50, 30, 30], we need to argmax to get [30, 30]
        recon_grid_np = reconstructed_grids[idx][0].argmax(dim=0).cpu().numpy()
        orig_grid_np = original_grids[idx][0, 0].cpu().numpy()
        
        plot_reconstruction_dashboard(
            torch.from_numpy(orig_grid_np), 
            torch.from_numpy(recon_grid_np), 
            latent_tensors[idx][0], 
            epoch="Eval", 
            save_path=save_path
        )
        print(f"Saved Error Heatmap to {save_path}")

if __name__ == "__main__":
    analyze_latent_space()
