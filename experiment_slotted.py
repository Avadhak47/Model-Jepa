"""
# 🧠 NS-ARC Object-Centric (Slotted) Scaling Experiment
This script utilizes the new parallel slotted architecture (`SlotTransformerEncoder`, `SlotDecoder`, `SlotWorldModel`).
It breaks the ARC grid into independent objects (slots) and reasons about their spatio-temporal interactions.
"""

import sys
import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pathlib
import shutil

# --- 1. Kaggle Path Setup ---
KAGGLE_REPO_PATH = '/kaggle/working/Model-Jepa'
if os.path.exists(KAGGLE_REPO_PATH):
    print("✅ Kaggle Repository Structure Detected.")
    if KAGGLE_REPO_PATH not in sys.path:
        sys.path.insert(0, KAGGLE_REPO_PATH)
        print(f"✅ Added {KAGGLE_REPO_PATH} to sys.path")
else:
    sys.path.insert(0, os.path.abspath('.'))

# --- 2. Device Config ---
if torch.cuda.is_available(): DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

print(f'Using device: {DEVICE}')
torch.manual_seed(42)
np.random.seed(42)

# --- 3. W&B Setup ---
import wandb
WANDB_KEY = os.environ.get('WANDB_API_KEY', '')
if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
WANDB_PROJECT = "NS-ARC-Slotted"

# --- 4. Model Config ---
from modules.encoders import SlotTransformerEncoder, SlotDecoder
from modules.world_models import SlotWorldModel32, SlotWorldModel64, SlotWorldModel128
from modules.policies import SlotPPOPolicy
from modules.curiosity import SlotRNDCuriosity
from arc_data.rearc_dataset import ReARCDataset

MODELS_CONFIG = {
    'Slotted-NSARC-32':  {'world_model': SlotWorldModel32, 'encoder_layers': 4},
    'Slotted-NSARC-64':  {'world_model': SlotWorldModel64, 'encoder_layers': 8},
    'Slotted-NSARC-128': {'world_model': SlotWorldModel128, 'encoder_layers': 12}
}

BASE_CFG = {
    'device': DEVICE,
    'input_dim': 64,  
    'in_channels': 1,
    'patch_size': 14,
    'hidden_dim': 256,
    'latent_dim': 128,
    'action_dim': 10,
    'num_slots': 16, # The Object-Centric Bottleneck
    'slot_iters': 3
}

MODELS = {}
PROFILES = {}

for name, setup in MODELS_CONFIG.items():
    cfg = BASE_CFG.copy()
    cfg['encoder_layers'] = setup['encoder_layers']
    PROFILES[name] = cfg
    
    MODELS[name] = {
        'encoder': SlotTransformerEncoder(cfg).to(DEVICE),
        'decoder': SlotDecoder(cfg).to(DEVICE),
        'world_model': setup['world_model'](cfg).to(DEVICE),
        'policy': SlotPPOPolicy(cfg).to(DEVICE),
        'curiosity': SlotRNDCuriosity(cfg).to(DEVICE)
    }

print("✅ Initiated Object-Centric Modules for all 3 Profiles.")

# --- 5. Data & Tracking ---
# Real ReARCDataset integration
REARC_PATH = '/kaggle/working/Model-Jepa/arc_data/re-arc' if os.path.exists('/kaggle/working/Model-Jepa') else 'arc_data/re-arc'
dataset = ReARCDataset(data_path=REARC_PATH)

class RunTracker:
    def __init__(self):
        self.history = {}
    def log(self, phase, step, metrics_dict):
        for k, v in metrics_dict.items():
            key = f"{phase}/{k}"
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, v.item() if hasattr(v, 'item') else v))
    def get(self, key):
        return [v for _, v in self.history.get(key, [])]

TRACKERS = {m: RunTracker() for m in MODELS.keys()}
SAVE_DIR = pathlib.Path('./checkpoints')
SAVE_DIR.mkdir(exist_ok=True)

# --- 6. Slotted Training Utilities ---
from analysis.plot_utils import plot_reconstruction_dashboard

def train_slotted_phase(phase, modules, cfg, tracker, current_step, wb_run, n_batches, batch_size, model_name="base", epoch=0):
    if phase == 'ae':
        opt = torch.optim.AdamW(list(modules['encoder'].parameters()) + list(modules['decoder'].parameters()), lr=1e-3)
    elif phase == 'wm':
        opt = torch.optim.AdamW(list(modules['world_model'].parameters()) + list(modules['encoder'].parameters()), lr=1e-4)
    else: 
        opt = torch.optim.AdamW([p for m in modules.values() for p in m.parameters()], lr=1e-5)

    for b in range(n_batches):
        batch = dataset.sample(batch_size)
        states = batch['state'].to(cfg['device'])
        
        # --- Visualization Logic ---
        if phase == 'ae' and epoch % 5 == 0 and b == 0:
            modules['encoder'].eval(); modules['decoder'].eval()
            with torch.no_grad():
                z_dict = modules['encoder']({'state': states})
                recon_out = modules['decoder']({'latent': z_dict['latent'], 'state': states})
                
                orig = states[0, 0]
                recon = recon_out['reconstruction'][0].squeeze(0) # [H, W]
                os.makedirs("evaluation_reports/plots", exist_ok=True)
                viz_path = f"evaluation_reports/plots/slot_diag_{model_name}_ep_{epoch}.png"
                plot_reconstruction_dashboard(orig, recon, z_dict['latent'][0], epoch, viz_path)
                
                if wb_run:
                    wb_run.log({f"diagnostics/{phase}_slots": wandb.Image(viz_path)}, step=current_step)
            modules['encoder'].train(); modules['decoder'].train()
        # ---------------------------

        if phase == 'ae':
            z_dict = modules['encoder']({'state': states})
            out = modules['decoder']({'latent': z_dict['latent'], 'state': states})
            loss_dict = modules['decoder'].loss({'state': states, 'latent': z_dict['latent']}, out)
        
        elif phase == 'wm':
            z_start = modules['encoder']({'state': states})['latent'].detach() # [B, S, D]
            action = torch.randn(batch_size, cfg['action_dim'], device=cfg['device']) 
            out = modules['world_model']({'latent': z_start, 'action': action})
            
            # Encode true next state to get target latent for MSE comparison
            with torch.no_grad():
                target_states = batch['target_state'].to(cfg['device'])
                target_z = modules['encoder']({'state': target_states})['latent']
            
            loss_dict = modules['world_model'].loss(
                {'target_latent': target_z, 'target_reward': batch['target_reward'].to(cfg['device'])}, 
                out
            )
        else: # Unrunnable dummy
            loss_dict = {'loss': torch.tensor(0.0, requires_grad=True, device=cfg['device'])}

        opt.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_([p for m in modules.values() for p in m.parameters()], 1.0)
        opt.step()
        
        step = current_step + b
        tracker.log(phase, step, loss_dict)
        if wb_run: wb_run.log({f'{phase}/{k}': (v.item() if hasattr(v,'item') else v) for k,v in loss_dict.items()}, step=step)

    return current_step + n_batches

def save_and_upload_slotted(model_name: str, modules: dict, epoch: int, wb_run=None):
    model_dir = SAVE_DIR / model_name
    epoch_dir = model_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for mod_name, mod in modules.items():
        torch.save(mod.state_dict(), epoch_dir / f'{mod_name}.pt')
        
    latest_dir = model_dir / "latest"
    if latest_dir.exists(): shutil.rmtree(latest_dir)
    shutil.copytree(epoch_dir, latest_dir)

    all_epochs = sorted([int(d.name.split('_')[1]) for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")])
    if len(all_epochs) > 5:
        for old_epoch in all_epochs[:-5]:
            shutil.rmtree(model_dir / f"epoch_{old_epoch}")

    if wb_run:
        for mod_name in modules.keys():
            art = wandb.Artifact(f'{model_name}_{mod_name}_slotted', type='model')
            art.add_file(str(latest_dir / f'{mod_name}.pt'))
            wb_run.log_artifact(art)

# --- 7. Execution Loop ---
if __name__ == "__main__":
    N_EPOCHS = 200
    BATCHES_PER_PHASE = 200
    BATCH_SIZE = 16
    AE_LOSS_THRESHOLD = 0.3

    for model_name, modules in MODELS.items():
        cfg = PROFILES[model_name]
        wb_run = wandb.init(project=WANDB_PROJECT, name=model_name, config=cfg, reinit=True)
        
        print(f"\n{'='*55}\n Training {model_name}\n{'='*55}")
        current_step = 0
        
        for epoch in range(N_EPOCHS):
            current_step = train_slotted_phase('ae',  modules, cfg, TRACKERS[model_name], current_step, wb_run, BATCHES_PER_PHASE, BATCH_SIZE, model_name, epoch)
            
            recent_ae_losses = TRACKERS[model_name].get('ae/loss')[-BATCHES_PER_PHASE:]
            mean_ae_loss = sum(recent_ae_losses) / len(recent_ae_losses) if recent_ae_losses else float('inf')
            
            if mean_ae_loss > AE_LOSS_THRESHOLD:
                print(f'  Epoch {epoch+1:>2}/{N_EPOCHS} | Stage 1 ({mean_ae_loss:.4f}): Slotted Encoder unstable.')
                save_and_upload_slotted(model_name, modules, epoch, wb_run=wb_run)
                continue
                
            print(f'  Epoch {epoch+1:>2}/{N_EPOCHS} | Phase 1 ({mean_ae_loss:.4f}) <= {AE_LOSS_THRESHOLD}. Downstream!')
            current_step = train_slotted_phase('wm',  modules, cfg, TRACKERS[model_name], current_step, wb_run, BATCHES_PER_PHASE, BATCH_SIZE, model_name, epoch)
            save_and_upload_slotted(model_name, modules, epoch, wb_run=wb_run)
            
        if wb_run: wb_run.finish()
