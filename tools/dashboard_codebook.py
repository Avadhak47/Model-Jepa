import os
import sys

# Ensure the root directory is in the Python path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

import torch
import torch.nn.functional as F
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from train_object_codebook import ObjectDecoder
from modules.vq import StructureAwareDynamicVQ

st.set_page_config(page_title="Phase 0: Codebook Audit", layout="wide")

ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(base_dir, 'checkpoints', 'object_codebook_e100.pth')
    
    if not os.path.exists(ckpt_path):
        return None, None, None
        
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    cfg = checkpoint['cfg']
    
    # Initialize Models
    decoder = ObjectDecoder(latent_dim=cfg['latent_dim'], pose_dim=cfg['pose_dim'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    
    vq = StructureAwareDynamicVQ(
        max_shape_codes=cfg['num_shape_codes'],
        max_color_codes=cfg['num_color_codes'],
        embedding_dim=cfg['latent_dim'],
        commitment_cost=cfg['commitment_cost'],
        novelty_threshold=cfg.get('novelty_threshold', 2.0),
        repulsion_weight=cfg.get('repulsion_weight', 0.1)
    )
    vq.load_state_dict(checkpoint['vq'])
    vq.eval()
    
    return decoder, vq, cfg

def plot_object(tensor_grid, title=""):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(tensor_grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_xticks(np.arange(-.5, tensor_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, tensor_grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return fig

def main():
    st.title("📖 Phase 0: Object Codebook Audit")
    st.markdown("Decode the discrete vectors stored in the VQ-VAE codebook back into visual space to see exactly what shape atoms the network learned.")
    
    decoder, vq, cfg = load_models()
    if decoder is None:
        st.error("Could not find `checkpoints/object_codebook_e100.pth`. Ensure the checkpoint is downloaded to the correct folder.")
        return
        
    st.sidebar.header("Codebook Parameters")
    
    # User selects which shape code range to visualize
    num_shapes = cfg['num_shape_codes']
    num_colors = cfg['num_color_codes']
    
    color_idx = st.sidebar.slider("Select Output Color (1-9)", min_value=1, max_value=9, value=1)
    
    st.sidebar.markdown(f"**Shape Codes:** {num_shapes}")
    st.sidebar.markdown(f"**Color Codes:** {num_colors}")
    
    st.subheader(f"Visualizing Shape Codes (Color forced to {color_idx})")
    
    # Generate images from the codebook
    with torch.no_grad():
        # Get shape embedding weight
        shape_codes = vq.embedding_shape.weight # [num_shape_codes, 64]
        
        # Get color embedding weight for the selected color
        # (Assuming the VQ learned color indexes matching roughly 1-9, though they might be permuted)
        # We can just pick the color code at color_idx.
        color_code = vq.embedding_color.weight[color_idx].unsqueeze(0).expand(num_shapes, -1) # [1024, 64]
        
        # Combine
        latent_vq = torch.cat([shape_codes, color_code], dim=-1) # [1024, 128]
        
        # Add zero-pose
        pose_zero = torch.zeros(num_shapes, cfg['pose_dim'])
        latent_combined = torch.cat([latent_vq, pose_zero], dim=-1).unsqueeze(1) # [1024, 1, 192]
        
        # Decode
        logits = decoder(latent_combined) # [1024, 10, 15, 15]
        predictions = logits.argmax(dim=1) # [1024, 15, 15]
    
    # Pagination
    items_per_page = 36
    total_pages = num_shapes // items_per_page + (1 if num_shapes % items_per_page > 0 else 0)
    page = st.slider("Page", 1, total_pages, 1)
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, num_shapes)
    
    cols = st.columns(6)
    for i in range(start_idx, end_idx):
        col = cols[(i - start_idx) % 6]
        with col:
            grid = predictions[i].numpy()
            fig = plot_object(grid, title=f"Code #{i}")
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
