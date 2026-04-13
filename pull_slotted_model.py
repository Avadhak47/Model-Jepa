import wandb
import torch
import os

def pull_last_harmonic_model(encoder, decoder, run_id, project="NS-ARC-Scaling", entity="avadheshkumarajay-indian-institute-of-technology-delhi"):
    """
    Finds the absolute highest version (the very last save) for the Harmonic 
    Encoder and Decoder in a specific run and loads them into your models.
    """
    api = wandb.Api()
    
    # 1. Access the run
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        run_name = run.name
        print(f"📡 Locating last saved state for Run: {run_name} ({run_id})")
    except Exception as e:
        print(f"❌ Error finding run: {e}")
        return

    # 2. Modules mapping strictly for the small slotted experiment
    modules = {
        'encoder': encoder,
        'decoder': decoder
    }
    
    for mod_name, mod_obj in modules.items():
        try:
            artifact_name = f"{run_name}_{mod_name}:latest"
            
            # Use api.artifact() instead of deprecated artifact_versions()
            artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
            print(f"  📥 Found artifact: {artifact_name}")
            
            # Download to a clean artifact sub-directory
            save_dir = f"./artifacts/RESUME_SLOTTED/{mod_name}"
            os.makedirs(save_dir, exist_ok=True)
            datadir = artifact.download(root=save_dir)
            
            # The .pt file matches the prefix we theoretically saved it under
            weights_path = os.path.join(datadir, f"{mod_name}.pt")
            
            # Load into your local model objects
            mod_obj.load_state_dict(torch.load(weights_path, map_location=mod_obj.device))
            print(f"  ✅ {mod_name} restored successfully from {artifact_name}")
                
        except Exception as e:
            print(f"  ⚠️ Could not load {mod_name}: {e}")
            continue

# ==========================================
# HOW TO RUN IT IN THE NOTEBOOK
# ==========================================
# Assuming 'encoder' and 'decoder' are already instantiated:
# pull_last_harmonic_model(encoder, decoder, run_id="YOUR_RUN_ID_HERE")
