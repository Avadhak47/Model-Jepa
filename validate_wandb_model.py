import os
import argparse
import torch
import wandb
from arc_data.arc_dataset import ARCDataset
from analysis.evaluator import run_validation_epoch

# Import your model structures
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder as Decoder
from train_small_slotted import HarmonicSlotEncoder, HarmonicDecoder

# Define the local DeepTransformerEncoder wrapper used in your experiment notebook
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

# Basic config to match NSARC-32
BASE_CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'input_dim': 64,  
    'in_channels': 1,
    'patch_size': 14,
    'hidden_dim': 256,
    'latent_dim': 128,
    'enc_depth': 4,
    'vocab_size': 10, # default, overwritten by argument
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 Autoencoder performance from a trained W&B model.")
    parser.add_argument("--run_path", type=str, required=True, help="W&B run path (e.g., entity/project/run_id)")
    parser.add_argument("--model_name", type=str, default="NSARC-32", help="Model name identifier prefixed to the W&B artifact (default: NSARC-32)")
    parser.add_argument("--data_path", type=str, default="arc_data/evaluation", help="Path to the ARC eval JSON subset directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=10, help="Vocabulary size (e.g. 10 or 50) corresponding to Decoder's output dimension.")
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "slotted"], help="Architecture to test")
    args = parser.parse_args()

    BASE_CFG['vocab_size'] = args.vocab_size
    device = BASE_CFG['device']
    print(f"Using device: {device}")

    # 1. Initialize Phase 1 Modules (Encoder and Decoder)
    if args.arch == "transformer":
        modules = {
            'encoder': DeepTransformerEncoder(BASE_CFG).to(device),
            'decoder': Decoder(BASE_CFG).to(device)
        }
    elif args.arch == "slotted":
        # Overwrite with slotted config
        SLOT_CFG = dict(BASE_CFG)
        SLOT_CFG.update({
            'hidden_dim': 128,
            'patch_size': 2,
            'num_slots': 10,
            'slot_iters': 5,
            'slot_temperature': 0.1,
            'grid_size': 30, # default from train_small_slotted
        })
        modules = {
            'encoder': HarmonicSlotEncoder(SLOT_CFG).to(device),
            'decoder': HarmonicDecoder(SLOT_CFG).to(device)
        }

    # 2. Download Weights from W&B
    api = wandb.Api()
    print(f"Looking up W&B Run: {args.run_path}")
    try:
        run = api.run(args.run_path)
    except Exception as e:
        print(f"❌ Failed to find run {args.run_path}. Check permissions or run path format. Error: {e}")
        return

    print(f"Downloading artifacts...")
    for mod_name in ['encoder', 'decoder']:
        # Format matching how your save_and_upload function generates artifact names
        artifact_name = f"{args.model_name}_{mod_name}:latest"
        try:
            artifact = api.artifact(f"{run.entity}/{run.project}/{artifact_name}")
            artifact_dir = artifact.download()
            
            # Find and load the specific .pt file in the downloaded directory
            for f in os.listdir(artifact_dir):
                if f.endswith('.pt'):
                    state_dict = torch.load(os.path.join(artifact_dir, f), map_location=device)
                    modules[mod_name].load_state_dict(state_dict)
                    print(f"✅ Successfully loaded {mod_name} weights from W&B.")
                    break
        except Exception as e:
            print(f"⚠️ Failed to load artifact {artifact_name}: {e}\n(Is the model name correct?)")

    # 3. Load Evaluation Dataset 
    # (Assuming you created an 'arc_data/evaluation' folder manually with the test subset)
    print(f"\nLoading evaluation dataset from {args.data_path}...")
    dataset = ARCDataset(data_path=args.data_path)
    if len(dataset) == 0:
        print("❌ Dataset is empty or path doesn't exist. Please point --data_path to your ARC eval JSONs.")
        return

    # 4. Trigger Validation Code
    print("\nRunning Phase 1 (Autoencoding) Evaluation sequence...")
    val_loss, val_acc, val_perf = run_validation_epoch(modules, dataset, phase='ae', batch_size=args.batch_size, device=device)

    # 5. Output Formal Verdict
    print("\n==================================================")
    print("📊 PHASE 1 EVALUATION RESULTS")
    print("==================================================")
    print(f"W&B Run      : {args.run_path}")
    print(f"Dataset      : {args.data_path}")
    print(f"Tasks Loaded : {len(dataset)}")
    print("--------------------------------------------------")
    print(f"Average Recon Loss : {val_loss:.5f}")
    if val_acc > 0 or hasattr(dataset, 'tasks'):
        print(f"Strict Pixel Acc   : {val_acc*100:.2f}%")
        print(f"Perfect Regen Grid : {val_perf*100:.2f}%")
    print("==================================================")
    
    # 6. Visualizations
    from analysis.plot_utils import plot_reconstruction_dashboard, plot_slot_masks
    print("\nGenerating visual diagnostics for ONE batch...")
    
    # Sample a tiny batch just for visual
    batch = dataset.sample(4, split='val')
    states = batch['state'].to(device)
    with torch.no_grad():
        z_dict = modules['encoder']({'state': states})
        out = modules['decoder']({'latent': z_dict['latent'], 'state': states})
        recon_tensor = out.get('reconstruction', out.get('reconstructed_logits'))
    
    # Dashboard requires standard dimensional inputs [H, W] and potentially class extraction
    os.makedirs("evaluation_reports", exist_ok=True)
    save_path = plot_reconstruction_dashboard(
        original=states[0],
        reconstructed=recon_tensor[0],
        latent=z_dict['latent'][0].flatten() if hasattr(z_dict['latent'][0], 'flatten') else z_dict['latent'][0],
        epoch="Validation",
        save_path="evaluation_reports/val_reconstruction.png"
    )
    print(f"✅ Saved reconstruction dashboard to {save_path}")

    # Check for slotted decoder output (like slot attention alphas generated out of components)
    if 'alphas' in out:
        mask_path = plot_slot_masks(out['alphas'], "Validation", "evaluation_reports/val_slot_masks.png")
        print(f"✅ Saved slot attention maps to {mask_path}")
    elif 'masks' in z_dict:
        # Sometimes slot encoders return masks directly
        mask_path = plot_slot_masks(z_dict['masks'].unsqueeze(2), "Validation", "evaluation_reports/val_slot_masks.png")
        print(f"✅ Saved slot attention maps to {mask_path}")

if __name__ == "__main__":
    main()
