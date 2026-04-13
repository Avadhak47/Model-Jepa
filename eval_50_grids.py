import os
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt

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

def main():
    device = BASE_CFG['device']
    print(f"Using device: {device}")

    # Initialize and Load
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

    # Load unseen grids
    dataset = ARCDataset(data_path="ARC-AGI/data/evaluation")
    
    os.makedirs("evaluation_reports/50_grid_analysis", exist_ok=True)
    
    perfect_count = 0
    total_samples = 50
    
    print("\nEvaluating 50 unseen grids...")
    
    with torch.no_grad():
        for i in range(total_samples):
            # strict sample size of 1
            batch = dataset.sample(1)
            state = batch['state'].to(device) # [1, 1, 30, 30]
            
            z = modules['encoder']({'state': state})['latent']
            out = modules['decoder']({'latent': z, 'state': state})
            
            recon_logits = out.get('reconstruction', out.get('reconstructed_logits')) # [1, 50, 30, 30]
            recon_grid = recon_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            orig_grid = state.squeeze(0).squeeze(0).cpu().numpy()
            
            # Count perfect pixel matches
            errors = np.sum(recon_grid != orig_grid)
            if errors == 0:
                perfect_count += 1
                
            # Generate visualization
            save_path = f"evaluation_reports/50_grid_analysis/eval_grid_{i+1:02d}.png"
            plot_reconstruction_dashboard(
                torch.from_numpy(orig_grid), 
                torch.from_numpy(recon_grid), 
                z[0], 
                epoch="Final Benchmark", 
                save_path=save_path
            )

    print("\n" + "="*50)
    print("🎯 EXACT PIXEL-PERFECT RECONSTRUCTION METRICS")
    print("="*50)
    print(f"Total Grids Analyzed: {total_samples}")
    print(f"0-Error Perfect Reconstructions: {perfect_count} ({perfect_count/total_samples*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
