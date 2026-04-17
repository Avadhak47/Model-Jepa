import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
import umap
import pandas as pd

import streamlit as st
import plotly.express as px

from arc_data.arc_dataset import ARCDataset

st.set_page_config(page_title="NS-ARC Latent Explorer", layout="wide")

# ========================================================
# 1. CORE MODEL DEFINITIONS (From Harmonic Slot Experiment)
# ========================================================
BASE_CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
    'in_channels': 1, 'input_dim': 64, 'patch_size': 2,
    'hidden_dim': 128, 'latent_dim': 128, 'num_slots': 10,
    'slot_iters': 5, 'slot_temperature': 0.1,    
    'vocab_size': 10, 'grid_size': 30, 'focal_gamma': 2.0,         
}

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
        self.patch_mlp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.LayerNorm(self.embed_dim * 2), nn.GELU(), nn.Linear(self.embed_dim * 2, self.embed_dim))
        self.harmonic_priors = nn.Parameter(self._build_harmonic_priors(self.num_slots, self.embed_dim), requires_grad=False)
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.slot_logsigma = nn.Parameter(torch.ones(1, 1, self.embed_dim) * -2.0)
        self.norm_inputs = nn.LayerNorm(self.embed_dim)
        self.norm_slots  = nn.LayerNorm(self.embed_dim)
        self.norm_mlp    = nn.LayerNorm(self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.gru = nn.GRUCell(self.embed_dim, self.embed_dim)
        self.slot_mlp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.GELU(), nn.Linear(self.embed_dim * 2, self.embed_dim))
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
        k = self.k_proj(x)
        v = self.v_proj(x)
        slots = self.harmonic_priors.expand(B, -1, -1) + self.slot_mu
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
        self.spatial_broadcast = nn.Sequential(nn.ConvTranspose2d(self.embed_dim, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.color_mlp = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, self.vocab_size, 3, padding=1))
        self.alpha_mlp = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 1, 3, padding=1))
        self.to(self.device)

    def forward(self, inputs):
        z = inputs["latent"]
        B, num_slots, D = z.shape
        H, W = self.grid_size, self.grid_size
        z_tiled = z.view(B * num_slots, D, 1, 1).expand(-1, -1, H, W)
        z_tiled = self.decoder_pos(z_tiled)
        decoded_features = self.spatial_broadcast(z_tiled)
        colors = self.color_mlp(decoded_features).view(B, num_slots, self.vocab_size, H, W)
        alphas = self.alpha_mlp(decoded_features).view(B, num_slots, 1, H, W)
        alphas_normalized = F.softmax(alphas, dim=1)
        reconstruction = torch.sum(alphas_normalized * colors, dim=1)
        return {"reconstruction": reconstruction, "alphas": alphas_normalized}

# ========================================================
# 2. DATA PROCESSING & CACHING PIPELINE
# ========================================================

@st.cache_resource
def load_models_from_wandb(run_id, entity="avadheshkumarajay-indian-institute-of-technology-delhi", project="NS-ARC-Scaling"):
    encoder = HarmonicSlotEncoder(BASE_CFG)
    decoder = HarmonicDecoder(BASE_CFG)
    
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        for mod, obj in zip(['encoder', 'decoder'], [encoder, decoder]):
            artifact_name = f"{run.name}_{mod}:latest"
            artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
            datadir = artifact.download(root=f"./artifacts/DASHBOARD/{mod}")
            obj.load_state_dict(torch.load(os.path.join(datadir, f"{mod}.pt"), map_location=BASE_CFG['device'], weights_only=True))
        st.sidebar.success(f"Successfully connected to Run: {run.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to load weights: {e}")
        
    encoder.eval()
    decoder.eval()
    return encoder, decoder

@st.cache_data
def encode_dataset_samples(_encoder, _decoder, split_name="Train", max_samples=100):
    dataset_path = "ARC-AGI/data/training" if split_name == "Train" else "ARC-AGI/data/evaluation"
    dataset = ARCDataset(data_path=dataset_path)
    
    latent_slots = []      # Shape: [N*K, 128]
    latent_grids = []      # Shape: [N, 128*K] -> Flattened slots representing whole true grid
    
    meta_slots = []        # Metadata for slot df
    meta_grids = []        # Metadata for grid df
    
    grid_visuals = {}      # dict mapping base_id -> (true_grid, recon_grid, raw_alphas)
    
    device = BASE_CFG['device']
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            batch = dataset.sample(1)
            state = batch['state'].to(device)
            z_dict = _encoder({'state': state})
            out_dict = _decoder({'latent': z_dict['latent']})
            
            z = z_dict['latent'].cpu().numpy() # [1, 10, 128]
            alphas = out_dict['alphas'].cpu().numpy() # [1, 10, 1, 30, 30]
            recon = torch.argmax(out_dict['reconstruction'], dim=1).cpu().numpy() # [1, 30, 30]
            true_g = state.squeeze().cpu().numpy()
            
            grid_visuals[i] = {
                'true': true_g,
                'recon': recon.squeeze(),
                'alphas': alphas.squeeze() # [10, 30, 30]
            }
            
            # Record Global Grid Vector
            flat_z = z.flatten()
            latent_grids.append(flat_z)
            meta_grids.append({'grid_id': i, 'split': split_name})
            
            # Record Individual Slot Vectors
            for k in range(z.shape[1]):
                latent_slots.append(z[0, k, :])
                meta_slots.append({'grid_id': i, 'slot_id': k, 'idx_ref': f"{i}_{k}"})
                
    return np.array(latent_slots), meta_slots, np.array(latent_grids), meta_grids, grid_visuals

@st.cache_data
def run_cluster_analysis(X_dat, algo="UMAP"):
    # 1. Dimensionality Reduction to 2D for Click Interactivity
    if algo == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X_dat)
    
    # 2. HDBSCAN Density Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(X_dat)
    
    return X_2d, labels

# ========================================================
# 3. STREAMLIT UI RENDERER
# ========================================================

def plot_arcade_grid(grid, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_alpha_mask(mask, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(mask, cmap='magma', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# Sidebar Setup
st.sidebar.title("ARC Latent EDA")
run_id = st.sidebar.text_input("Enter W&B Run ID:", value="rbkzy98w")
n_samples = st.sidebar.slider("Samples to Load per Dataset:", 10, 200, 50)
proj_algo = st.sidebar.selectbox("3D Projection Algorithm:", ["UMAP", "PCA"])

if run_id:
    enc, dec = load_models_from_wandb(run_id)
    
    # Process
    with st.spinner("Extracting latents..."):
        sL_t, sM_t, gL_t, gM_t, vis_t = encode_dataset_samples(enc, dec, "Train", n_samples)
        sL_e, sM_e, gL_e, gM_e, vis_e = encode_dataset_samples(enc, dec, "Eval", n_samples)
        
        # Combine
        X_slots = np.vstack([sL_t, sL_e])
        meta_slots = sM_t + sM_e
        
        X_grids = np.vstack([gL_t, gL_e])
        meta_grids = gM_t + gM_e
        
        # We need a unified vis dictionary lookup. We'll differentiate IDs by split
        all_vis = {"Train_" + str(k): v for k, v in vis_t.items()}
        all_vis.update({"Eval_" + str(k): v for k, v in vis_e.items()})

    tab1, tab2 = st.tabs(["Individual Slot Explorer", "Unified Grid Explorer"])
    
    # --- TAB 1: INDIVIDUAL SLOTS ---
    with tab1:
        st.markdown("### Slot Level Analysis (Finding Object Archetypes)")
        if True: # Executing automatically because buttons wipe state on graph interaction!
            with st.spinner("Clustering Slot Manifold..."):
                X_2d_s, labels_s = run_cluster_analysis(X_slots, proj_algo)
                
                # Plotly definition
                import pandas as pd
                df_s = pd.DataFrame(X_2d_s, columns=['x', 'y'])
                df_s['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels_s]
                # Embed the absolute split and grid IDs so click events can retrieve them
                df_s['Split'] = [s['idx_ref'].split('_')[0] for s in meta_slots]
                df_s['Grid_ID'] = [("Train_" if i < len(meta_slots)//2 else "Eval_") + str(s['grid_id']) for i, s in enumerate(meta_slots)]
                df_s['Slot_ID'] = [s['slot_id'] for s in meta_slots]
                
                fig_s = px.scatter(df_s, x='x', y='y', color='Cluster', 
                                    hover_data=['Grid_ID', 'Slot_ID'], title=f"Slot Space Topography ({proj_algo})" , opacity=0.8)
                fig_s.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
                fig_s.update_layout(clickmode='event+select', dragmode='lasso')
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Rerun on select
                    selection_s = st.plotly_chart(fig_s, on_select="rerun", selection_mode=("points", "box", "lasso"), height=700)
                
                with col2:
                    st.markdown("### Microscope View")
                    st.write("Raw Selection Event Data:", selection_s)
                    try:
                        if len(selection_s.get('selection', {}).get('points', [])) > 0:
                            pt = selection_s['selection']['points'][-1] # The last clicked point
                            
                            st.write("Extracting customdata:", pt.get('customdata', 'MISSING_CUSTOMDATA'))
                            # Plotly splits traces by color, so pointIndex breaks. We use customdata mapping embedded via hover_data!
                            grid_key = pt['customdata'][0]
                            slot_k = pt['customdata'][1]
                            
                            clicked_row = df_s[(df_s['Grid_ID'] == grid_key) & (df_s['Slot_ID'] == slot_k)].iloc[0]
                            cluster_k = clicked_row['Cluster']
                            
                            vis_dict = all_vis[grid_key]
                            
                            st.write(f"**Origin:** {grid_key} | **Slot:** {slot_k}")
                            st.write(f"**HDBSCAN Cluster:** {cluster_k}")
                            
                            img1 = plot_arcade_grid(vis_dict['true'], "True Grid")
                            st.image(img1, use_column_width=True)
                            
                            img2 = plot_alpha_mask(vis_dict['alphas'][slot_k], f"Slot {slot_k} Mask")
                            st.image(img2, use_column_width=True)
                            
                            img3 = plot_arcade_grid(vis_dict['recon'], "Full Model Reconstruction")
                            st.image(img3, use_column_width=True)
                        else:
                            st.info("Click any node in the 2D plane to retrieve its real mask and grid. (Make sure you actually click precisely on a dot!)")
                    except Exception as e:
                        st.error(f"Error drawing microscope view: {e}")

    # --- TAB 2: UNIFIED GRIDS ---
    with tab2:
        st.markdown("### Grid Level Analysis (Finding Global Scene Archetypes)")
        if True: # Executing automatically because buttons wipe state on graph interaction!
            with st.spinner("Clustering Grid Manifold..."):
                X_2d_g, labels_g = run_cluster_analysis(X_grids, proj_algo)
                
                df_g = pd.DataFrame(X_2d_g, columns=['x', 'y'])
                df_g['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels_g]
                df_g['Grid_ID'] = [("Train_" if s['split']=='Train' else "Eval_") + str(s['grid_id']) for s in meta_grids]
                
                fig_g = px.scatter(df_g, x='x', y='y', color='Cluster', 
                                    hover_data=['Grid_ID'], title=f"Global Representation Topography ({proj_algo})" , opacity=0.8)
                fig_g.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
                fig_g.update_layout(clickmode='event+select', dragmode='lasso')
                
                colA, colB = st.columns([3, 1])
                with colA:
                    selection_g = st.plotly_chart(fig_g, on_select="rerun", selection_mode=("points", "box", "lasso"), height=700)
                
                with colB:
                    st.markdown("### Microscope View")
                    st.write("Raw Selection Event Data:", selection_g)
                    try:
                        if len(selection_g.get('selection', {}).get('points', [])) > 0:
                            pt = selection_g['selection']['points'][-1]
                            
                            # Safely route through customdata mapped via hover_data
                            grid_key = pt['customdata'][0]
                            
                            clicked_row = df_g[df_g['Grid_ID'] == grid_key].iloc[0]
                            cluster_k = clicked_row['Cluster']
                            
                            vis_dict = all_vis[grid_key]
                            
                            st.write(f"**Grid Identity:** {grid_key}")
                            st.write(f"**HDBSCAN Cluster:** {cluster_k}")
                            
                            img1 = plot_arcade_grid(vis_dict['true'], "True Target Grid")
                            st.image(img1, use_column_width=True)
                            
                            img3 = plot_arcade_grid(vis_dict['recon'], "Model Reconstruction")
                            st.image(img3, use_column_width=True)
                        else:
                            st.info("Click any node in the 3D void to retrieve the True vs Target Grid.")
                    except Exception as e:
                        st.error(f"Error drawing grid microscope view: {e}")
