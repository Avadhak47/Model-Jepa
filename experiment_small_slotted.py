# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # 🧠 NS-ARC Small Slot-JEPA Experiment
#
# Based on Harmonic Frequency Initialization, Patch/Decoder MLPs, and deep object compositionality theories.
# Now augmented with VICReg regularizers and Vector Quantization (VQ) bottlenecks for pure discrete routing.

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import sys, os, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Try Kaggle Path
KAGGLE_REPO_PATH = '/kaggle/working/Model-Jepa'
if os.path.exists(KAGGLE_REPO_PATH):
    print('✅ Kaggle detected.')
    if KAGGLE_REPO_PATH not in sys.path:
        sys.path.insert(0, KAGGLE_REPO_PATH)
else:
    sys.path.insert(0, os.path.abspath('.'))
    print('✅ Local mode.')

if torch.cuda.is_available(): DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'
print(f'Device: {DEVICE}')


# %% [markdown]
# ## 2. Model Configurations & Regularizers

# %%
BASE_CFG = {
    'device': DEVICE,
    'in_channels': 1,
    'input_dim': 64,
    'patch_size': 2,
    'hidden_dim': 128,          
    'latent_dim': 128,
    'num_slots': 10,            
    'slot_iters': 3,            # Kept to 3 for 1000-epoch speed efficiency
    'slot_temperature': 0.1,    
    'vocab_size': 10,           
    'grid_size': 30,            
    'focal_gamma': 2.0,         
}

def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z, std_coeff=25.0, cov_coeff=1.0):
    """Variance, Invariance, Covariance Regularization for Slot-heads."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std))
    cov = (z.T @ z) / (z.shape[0] - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum() / z.shape[1]
    return std_coeff * std_loss + cov_coeff * cov_loss

class VectorQuantizer(nn.Module):
    """Standard VQ Bottleneck with Straight-Through Estimator (STE)."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
                    
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

# %% [markdown]
# ## 3. Harmonic Slot Attention & Core Backbone

# %%
class HarmonicSlotEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.embed_dim = config['hidden_dim']
        self.num_slots = config['num_slots']
        self.slot_iters = config['slot_iters']
        self.temperature = config['slot_temperature']
        
        self.patch_embed = nn.Conv2d(config['in_channels'], self.embed_dim, kernel_size=config['patch_size'], stride=config['patch_size'])
        self.patch_pos_embed = nn.Parameter(torch.randn(1, 1024, self.embed_dim))
        
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        self.harmonic_priors = nn.Parameter(self._build_harmonic_priors(self.num_slots, self.embed_dim), requires_grad=False)
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
        self.slot_logsigma = nn.Parameter(torch.ones(1, 1, self.embed_dim) * -2.0)
        
        self.norm_inputs, self.norm_slots, self.norm_mlp = nn.LayerNorm(self.embed_dim), nn.LayerNorm(self.embed_dim), nn.LayerNorm(self.embed_dim)
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.gru = nn.GRUCell(self.embed_dim, self.embed_dim)
        
        self.slot_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        self.to(self.device)

    def _build_harmonic_priors(self, num_slots, dim):
        priors = torch.zeros(1, num_slots, dim)
        position = torch.arange(0, num_slots, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        priors[0, :, 0::2] = torch.sin(position * div_term)
        priors[0, :, 1::2] = torch.cos(position * div_term)
        return priors

    def forward(self, inputs):
        img = inputs["state"].float().to(self.device)
        B = img.shape[0]
        
        x = self.patch_embed(img) 
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        N = x.shape[1]
        
        x = x + self.patch_pos_embed[:, :N, :]
        x = self.patch_mlp(x) 
        x = self.norm_inputs(x)
        
        k, v = self.k_proj(x), self.v_proj(x)
        
        slots = self.harmonic_priors.expand(B, -1, -1) + self.slot_mu
        if self.training:
            slots = slots + torch.randn_like(slots) * self.slot_logsigma.exp()
            
        for _ in range(self.slot_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.q_proj(slots_norm)

            attn_logits = torch.bmm(k, q.transpose(1, 2)) * (self.embed_dim ** -0.5)
            attn = F.softmax(attn_logits / self.temperature, dim=-1)
            attn_norm = attn + 1e-8
            attn_norm = attn_norm / attn_norm.sum(dim=1, keepdim=True)

            updates = torch.bmm(attn_norm.transpose(1, 2), v)
            slots = self.gru(updates.reshape(-1, self.embed_dim), slots_prev.reshape(-1, self.embed_dim)).reshape(B, self.num_slots, self.embed_dim)
            slots = slots + self.slot_mlp(self.norm_mlp(slots))

        masks = attn.transpose(1, 2).view(B, self.num_slots, H, W).detach()
        return {"latent": slots, "masks": masks, "slot_logsigma": self.slot_logsigma}

# %% [markdown]
# ## 4. Decoder with VQ Bottleneck

# %%
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.register_buffer("grid", self._build_grid(resolution))

    @staticmethod
    def _build_grid(resolution):
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1)
        grid = torch.cat([grid, 1.0 - grid], dim=-1)
        return grid.unsqueeze(0)

    def forward(self, x):
        pos = self.dense(self.grid).permute(0, 3, 1, 2)
        return x + pos

class HarmonicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.embed_dim = config['hidden_dim']
        self.vocab_size = config['vocab_size']
        self.grid_size = config['grid_size']
        
        self.decoder_pos = SoftPositionEmbed(self.embed_dim, (self.grid_size, self.grid_size))
        
        self.spatial_broadcast = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        # Vector Quantization on the 32-dim features
        self.vq_bottleneck = VectorQuantizer(num_embeddings=128, embedding_dim=32, commitment_cost=0.25)
        
        self.color_mlp = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, self.vocab_size, 3, padding=1)
        )
        self.alpha_mlp = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        self.to(self.device)

    def forward(self, inputs):
        z = inputs["latent"] 
        B, num_slots, D = z.shape
        H, W = self.grid_size, self.grid_size
        
        z_tiled = z.view(B * num_slots, D, 1, 1).expand(-1, -1, H, W)
        z_tiled = self.decoder_pos(z_tiled)
        
        decoded_features = self.spatial_broadcast(z_tiled)

        features_perm = decoded_features.permute(0, 2, 3, 1) # [B*S, H, W, 32]
        vq_features, vq_loss = self.vq_bottleneck(features_perm)
        vq_features = vq_features.permute(0, 3, 1, 2).contiguous() # back to [B*S, 32, H, W]

        colors = self.color_mlp(vq_features).view(B, num_slots, self.vocab_size, H, W)
        alphas = self.alpha_mlp(decoded_features).view(B, num_slots, 1, H, W)
        
        alphas_normalized = F.softmax(alphas, dim=1) 
        reconstruction = torch.sum(alphas_normalized * colors, dim=1)
        
        return {"reconstruction": reconstruction, "alphas": alphas_normalized, "vq_loss": vq_loss}

    def loss(self, inputs, outputs):
        target = inputs["state"].to(self.device).long().squeeze()
        if target.dim() == 2: target = target.unsqueeze(0)
        recon_logits = outputs["reconstruction"] 
        
        # 1. Focal / Pixel Loss
        ce_loss = F.cross_entropy(recon_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        recon_loss = (((1 - pt) ** 2.0) * ce_loss).mean()
        
        # 2. VQ Commitment Loss
        vq_loss = outputs.get("vq_loss", 0.0)

        # 3. VICReg Regularization
        z = inputs["latent"] 
        B, S, D = z.shape
        z_flat = z.view(B * S, D)
        vic_loss = vicreg_loss(z_flat, std_coeff=10.0, cov_coeff=1.0) 
        
        total_loss = recon_loss + vq_loss + (0.5 * vic_loss)
        return {"loss": total_loss, "recon_loss": recon_loss, "vq_loss": vq_loss, "vic_loss": vic_loss}

# %% [markdown]
# ## 5. Baseline Models for Control Comparison
# Pure Continuous Autoencoder (No Slots, No VQ)

# %%
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder

class DeepTransformerEncoder(TransformerEncoder):
    """Deep Continuous Baseline without slots."""
    def __init__(self, config):
        cfg = dict(config)
        cfg['_enc_depth'] = cfg.get('enc_depth', 4)
        super().__init__(cfg)
        embed_dim = cfg.get('hidden_dim', 128)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4,
            batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=cfg['_enc_depth'])

# %% [markdown]
# ## 6. 1000-Epoch Dual Curriculum: Target vs Baseline

# %%
import wandb
from tqdm import tqdm
from arc_data.rearc_dataset import ReARCDataset
from arc_data.arc_dataset import ARCDataset
from analysis.evaluator import run_validation_epoch
from analysis.plot_utils import plot_reconstruction_dashboard, plot_slot_masks

# Initialize Slotted Models (Test)
slotted_encoder = HarmonicSlotEncoder(BASE_CFG)
slotted_decoder = HarmonicDecoder(BASE_CFG)
opt_slotted = torch.optim.AdamW(list(slotted_encoder.parameters()) + list(slotted_decoder.parameters()), lr=1e-4)

# Initialize Baseline Models (Control)
baseline_encoder = DeepTransformerEncoder(BASE_CFG).to(DEVICE)
baseline_decoder = TransformerDecoder(BASE_CFG).to(DEVICE)
opt_baseline = torch.optim.AdamW(list(baseline_encoder.parameters()) + list(baseline_decoder.parameters()), lr=1e-4)

# Load ARC Datasets
train_dataset = ReARCDataset(data_path='/kaggle/working/data/re-arc' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'arc_data/re-arc')
eval_dataset = ARCDataset(data_path='arc_data/re-arc/arc_original/evaluation')

# Ensure directories exist
os.makedirs("evaluation_reports/plots", exist_ok=True)

# Initialize Weights and Biases Tracking
if os.environ.get('WANDB_API_KEY'):
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
else:
    wandb.login()

wb_run = wandb.init(project="NS-ARC-Scaling", name="VICReg-VQ-vs-Baseline", config=BASE_CFG, reinit=True)

# Training Curriculum
EPOCHS = 1000
BATCH_SIZE = 32
STEPS_PER_EPOCH = 50

print("🚀 Commencing 1000-Epoch Dual Training (VICReg/VQ Slots vs Baseline)...")

for epoch in range(1, EPOCHS + 1):
    slotted_encoder.train(); slotted_decoder.train()
    baseline_encoder.train(); baseline_decoder.train()
    
    epoch_losses, epoch_recon, epoch_vicreg, epoch_vq = [], [], [], []
    base_epoch_losses = []
    
    for _ in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}", leave=False):
        batch = train_dataset.sample(BATCH_SIZE)
        state_batch = batch['state'].to(DEVICE)
        
        # --- 1. RUN SLOTTED ITERATION ---
        z_dict = slotted_encoder({"state": state_batch})
        out_dict = slotted_decoder({"latent": z_dict["latent"], "state": state_batch})
        
        loss_dict = slotted_decoder.loss({"state": state_batch, "latent": z_dict["latent"]}, out_dict)
        opt_slotted.zero_grad()
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(slotted_encoder.parameters()) + list(slotted_decoder.parameters()), 1.0)
        opt_slotted.step()
        
        epoch_losses.append(loss_dict["loss"].item())
        epoch_recon.append(loss_dict["recon_loss"].item())
        epoch_vicreg.append(loss_dict["vic_loss"].item())
        epoch_vq.append(loss_dict["vq_loss"].item() if isinstance(loss_dict["vq_loss"], torch.Tensor) else loss_dict["vq_loss"])
        
        # --- 2. RUN BASELINE ITERATION ---
        b_z_dict = baseline_encoder({"state": state_batch})
        b_out_dict = baseline_decoder({"latent": b_z_dict["latent"], "state": state_batch})
        
        b_loss_dict = baseline_decoder.loss({"state": state_batch}, b_out_dict)
        opt_baseline.zero_grad()
        b_loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(baseline_encoder.parameters()) + list(baseline_decoder.parameters()), 1.0)
        opt_baseline.step()
        
        base_epoch_losses.append(b_loss_dict["loss"].item())
        
    avg_loss = np.mean(epoch_losses)
    avg_recon = np.mean(epoch_recon)
    avg_vicreg = np.mean(epoch_vicreg)
    avg_vq = np.mean(epoch_vq)
    avg_base_loss = np.mean(base_epoch_losses)

    # 3. LOG TRAINING RESULTS
    wb_run.log({
        "Train_Slot/Total_Loss": avg_loss, 
        "Train_Slot/Recon_Loss": avg_recon, 
        "Train_Slot/VICReg": avg_vicreg,
        "Train_Slot/VQ_Loss": avg_vq,
        "Train_Base/Total_Loss": avg_base_loss
    }, step=epoch)
    
    # 4. RUN PERIODIC VALIDATION (Every 25 Epochs)
    if epoch % 25 == 0 or epoch == 1:
        print(f"\n[Epoch {epoch:03d}] Slot_Recon: {avg_recon:.4f} | Base_Recon: {avg_base_loss:.4f} | VICReg: {avg_vicreg:.4f} | VQ: {avg_vq:.4f}")
        
        modules_slotted = {
            'encoder': slotted_encoder,
            'decoder': slotted_decoder,
            'config': BASE_CFG
        }
        
        modules_base = {
            'encoder': baseline_encoder,
            'decoder': baseline_decoder,
            'config': BASE_CFG
        }
        
        # Run standard validation routine
        val_loss, val_acc, val_perf = run_validation_epoch(
            modules_slotted, eval_dataset, phase='ae', batch_size=8, device=DEVICE
        )
        base_val_loss, base_val_acc, base_val_perf = run_validation_epoch(
            modules_base, eval_dataset, phase='ae', batch_size=8, device=DEVICE
        )
        
        wb_run.log({
            "Val_Slot/Recon": val_loss,
            "Val_Slot/Pixel_Acc": val_acc,
            "Val_Slot/Perfect_Regen": val_perf,
            "Val_Base/Recon": base_val_loss,
            "Val_Base/Pixel_Acc": base_val_acc,
            "Val_Base/Perfect_Regen": base_val_perf,
        }, step=epoch)

        print(f"--> [Slot Val]  Pixel.Acc: {val_acc:.2f}% | Perfect_Regen: {val_perf:.2f}%")
        print(f"--> [Base Val]  Pixel.Acc: {base_val_acc:.2f}% | Perfect_Regen: {base_val_perf:.2f}%\n")
        
        # 5. GENERATE DIAGNOSTIC VISUALS
        slotted_encoder.eval(); slotted_decoder.eval()
        with torch.no_grad():
            val_batch = eval_dataset.sample(4)
            val_state = val_batch['state'].to(DEVICE)
            z_val = slotted_encoder({"state": val_state})
            out_val = slotted_decoder({"latent": z_val["latent"], "state": val_state})
            
            z_flat = z_val["latent"].view(4 * BASE_CFG['num_slots'], -1).cpu().numpy()
            recon_grid = out_val["reconstruction"].argmax(dim=1).float()
            
            viz_path = f"evaluation_reports/plots/recon_epoch_{epoch}.png"
            plot_reconstruction_dashboard(val_state[0, 0].cpu(), recon_grid[0].cpu(), torch.tensor(z_flat), epoch, viz_path)
            
            mask_path = f"evaluation_reports/plots/masks_epoch_{epoch}.png"
            plot_slot_masks(out_val["alphas"], epoch, mask_path)
            
            wb_run.log({
                "Visuals/Slot_Reconstruction": wandb.Image(viz_path),
                "Visuals/Slot_Masks": wandb.Image(mask_path)
            }, step=epoch)

wb_run.finish()
print("✅ 1000-Epoch Dual Training Complete.")
