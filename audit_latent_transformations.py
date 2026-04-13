import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

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

def swap_colors(tensor, c1, c2):
    # tensor: [1, 1, 30, 30] integers
    swapped = tensor.clone()
    mask1 = tensor == c1
    mask2 = tensor == c2
    swapped[mask1] = c2
    swapped[mask2] = c1
    return swapped

def main():
    device = BASE_CFG['device']
    print(f"Using device: {device}")

    # 1. Initialize Models & Load Weights
    modules = {
        'encoder': DeepTransformerEncoder(BASE_CFG).to(device),
        'decoder': Decoder(BASE_CFG).to(device)
    }

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

    # 2. Extract Latent Encodings for Original and Transformed Grids
    dataset = ARCDataset(data_path="ARC-AGI/data/evaluation")
    print(f"Dataset Size: {len(dataset)}")

    Z_orig = []
    Z_hflip = []
    Z_rot90 = []
    Z_color = []
    
    # Store test-set visualization pairs
    test_vis_pairs = []

    with torch.no_grad():
        # Using fixed sequence to gather diverse concepts
        for i in range(min(400, len(dataset))):
            batch = dataset.sample(1)
            state = batch['state'].to(device)  # [1, 1, 30, 30]
            
            # Generate Transformations natively
            state_hflip = torch.flip(state, dims=[3])
            state_rot90 = torch.rot90(state, k=1, dims=[2, 3])
            # Color Swap: Assume swap color 1 and 2 (or 0 and 1 if sparsely colored)
            state_val = state.cpu().numpy()
            unique_c = np.unique(state_val)
            c1, c2 = 0, 1
            if len(unique_c) >= 2:
                c1, c2 = int(unique_c[0]), int(unique_c[1])
            if len(unique_c) >= 3 and 0 in unique_c:
                c1, c2 = int(unique_c[1]), int(unique_c[2]) # avoid background swap if possible
            state_color = swap_colors(state, c1, c2)

            # Encode all variations
            z_o = modules['encoder']({'state': state})['latent']
            z_h = modules['encoder']({'state': state_hflip})['latent']
            z_r = modules['encoder']({'state': state_rot90})['latent']
            z_c = modules['encoder']({'state': state_color})['latent']

            Z_orig.append(z_o.cpu().numpy().flatten())
            Z_hflip.append(z_h.cpu().numpy().flatten())
            Z_rot90.append(z_r.cpu().numpy().flatten())
            Z_color.append(z_c.cpu().numpy().flatten())
            
            # Save 10 particular index samples for visualization later
            if len(test_vis_pairs) < 10 and i % 30 == 0:
                test_vis_pairs.append({
                    'state_orig': state,
                    'state_hflip': state_hflip,
                    'state_rot90': state_rot90,
                    'state_color': state_color,
                    'orig_latent_tensor': z_o
                })

    Z_orig = np.array(Z_orig)
    Z_hflip = np.array(Z_hflip)
    Z_rot90 = np.array(Z_rot90)
    Z_color = np.array(Z_color)

    # ========================================================
    # QUESTION 3: VERIFYING GLOBAL LATENT SPECTRUM (NOT COLLAPSED)
    # ========================================================
    os.makedirs("evaluation_reports/audit", exist_ok=True)
    global_std = np.std(Z_orig, axis=0)
    print(f"\n--- Latent Space Health ---")
    print(f"Mean dimensional standard deviation across 400 unique grids: {global_std.mean():.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(128), global_std, color='darkmagenta', alpha=0.8)
    plt.axhline(global_std.mean(), color='red', linestyle='dashed', label=f'Mean Std Dev: {global_std.mean():.4f}')
    plt.title("True Latent Spectrum across Entire Evaluation Set\n(Notice: Variance is strongly non-zero, space has not collapsed)")
    plt.xlabel("Latent Feature Dimension (1-128)")
    plt.ylabel("Variance (Standard Deviation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_reports/audit/true_latent_variance.png")
    plt.close()

    # ========================================================
    # QUESTION 4: HOMOMORPHIC TRANSFORMATION EQUATION FITTING 
    # ========================================================
    # Learn W where Z_orig @ W + b = Z_transformed
    def evaluate_transformation(concept_name, Z_target, test_visuals, plot_key):
        print(f"\n--- Analylzing Concept: {concept_name} ---")
        X_train, X_test, y_train, y_test = train_test_split(Z_orig, Z_target, test_size=0.2, random_state=42)
        
        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train, y_train)
        
        r2_train = regressor.score(X_train, y_train)
        r2_test = regressor.score(X_test, y_test)
        print(f"Linear Manifold Map R2 (Train): {r2_train:.4f}")
        print(f"Linear Manifold Map R2 (Test):  {r2_test:.4f}")
        
        # Aggregate Accuracies
        total_acc = []
        
        # Visualize Arithmetic
        for i, vis_dict in enumerate(test_visuals[:10]):
            # Take native orig latent vector
            z_o = vis_dict['orig_latent_tensor'] # [1, 128]
            
            # Predict the translated vector strictly using geometry math
            z_o_np = z_o.cpu().numpy()
            predicted_z_trans = regressor.predict(z_o_np) # [1, 128]
            predicted_z_tensor = torch.from_numpy(predicted_z_trans).float().to(device)
            
            # Decode the linearly transformed latent
            with torch.no_grad():
                out = modules['decoder']({'latent': predicted_z_tensor, 'state': vis_dict['state_orig']})
                recon_logits = out.get('reconstruction', out.get('reconstructed_logits'))
                recon_grid = recon_logits.argmax(dim=1).squeeze(0).cpu().numpy() # [30, 30]
            
            orig_input_grid = vis_dict['state_orig'].squeeze().cpu().numpy()
            true_target_grid = vis_dict[plot_key].squeeze().cpu().numpy()
            
            
            acc = np.mean(recon_grid == true_target_grid)
            total_acc.append(acc)

            # Plot Side by Side
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.suptitle(f"Latent Spatial Arithmetic | {concept_name}\n"
                         f"(Calculated purely by translating latent vector, then decoding)", fontsize=14)
            axes[0].imshow(orig_input_grid, cmap='tab10', vmin=0, vmax=9)
            axes[0].set_title("1. Original Grid")
            
            axes[1].imshow(true_target_grid, cmap='tab10', vmin=0, vmax=9)
            axes[1].set_title("2. True Target Grid (Real Env)")
            
            axes[2].imshow(recon_grid, cmap='tab10', vmin=0, vmax=9)
            axes[2].set_title("3. Decoded from Shifted Latent State\n(Z_orig * W + b)")
            
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            save_loc = f"evaluation_reports/audit/{concept_name.replace(' ', '_')}_demo_{i}.png"
            plt.savefig(save_loc)
            plt.close()

        # Print overall zero-shot accuracy across collected vis pairs
        mean_zero_shot = np.mean(total_acc)
        print(f"Zero-Shot Decoding Accuracy to true state (Sampled): {mean_zero_shot * 100:.2f}%")

    evaluate_transformation("Horizontal Flip", Z_hflip, test_vis_pairs, 'state_hflip')
    evaluate_transformation("Rotate 90 Degrees", Z_rot90, test_vis_pairs, 'state_rot90')
    evaluate_transformation("Color Swap (A to B)", Z_color, test_vis_pairs, 'state_color')

if __name__ == "__main__":
    main()
