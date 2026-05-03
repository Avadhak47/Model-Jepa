import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="ARC NMF Basis Audit", layout="wide")

ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)

@st.cache_resource
def load_nmf_data():
    path = 'arc_data/arc_basis_nmf_1024.pt'
    data = torch.load(path, weights_only=False)
    basis = data['basis'].numpy()       # [1024, 2250]
    weights = data['weights'].numpy()   # [N, 1024]
    return basis, weights

def plot_atom(atom_flat, threshold=0.1, title=""):
    # Reshape to [15, 15, 10]
    atom = atom_flat.reshape(15, 15, 10)
    
    # UN-WEIGHT the background channel (color 0) which was 0.1x in training
    atom_viz = atom.copy()
    atom_viz[:, :, 0] *= 10.0
    
    # Adaptive Masking
    intensities = np.max(atom_viz, axis=-1)
    
    if threshold is None:
        actual_thresh = np.percentile(intensities, 95)
    else:
        actual_thresh = threshold * np.max(intensities) if np.max(intensities) > 0 else 0
        
    grid = np.argmax(atom_viz, axis=-1)
    grid[intensities <= max(1e-5, actual_thresh)] = 0
    
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8)
    return fig

def main():
    st.title("🧩 ARC NMF Basis Audit v2")
    st.markdown("Auditing the 200 Structural Atoms with Adaptive Masking.")
    
    basis, weights = load_nmf_data()
    n_components = basis.shape[0]
    
    st.sidebar.header("Global Controls")
    global_thresh = st.sidebar.slider("Snap Threshold (%)", 0, 100, 15) / 100.0
    
    tabs = st.tabs(["📚 Basis Library", "🧪 Synthesis Lab", "🔍 Decomposition"])
    
    with tabs[0]:
        st.header("The 1024 Atoms of ARC")
        items_per_page = 40
        n_pages = (n_components // items_per_page) + 1
        page = st.number_input("Page", min_value=1, max_value=n_pages, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, n_components)
        
        cols = st.columns(8)
        for i in range(start_idx, end_idx):
            with cols[(i - start_idx) % 8]:
                fig = plot_atom(basis[i], threshold=global_thresh, title=f"Atom #{i}")
                st.pyplot(fig)
                plt.close(fig)
                
    with tabs[1]:
        st.header("Synthesis: Build an Object")
        selected_atoms = st.multiselect("Select Atoms to Combine", options=list(range(n_components)), default=[0, 1, 2])
        
        if selected_atoms:
            combined = np.sum(basis[selected_atoms], axis=0)
            fig = plot_atom(combined, threshold=global_thresh, title="Synthesized Grid")
            st.pyplot(fig)
            plt.close(fig)
            
    with tabs[2]:
        st.header("Decomposition: Analyze an Object")
        sample_idx = st.slider("Select Sample Object from Library", 0, weights.shape[0] - 1, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original (NMF Reconstruction)")
            reconstructed = weights[sample_idx] @ basis
            fig_orig = plot_atom(reconstructed, threshold=global_thresh, title=f"Object #{sample_idx}")
            st.pyplot(fig_orig)
            plt.close(fig_orig)
            
        with col2:
            st.subheader("Top Contributing Atoms")
            sample_weights = weights[sample_idx]
            top_indices = np.argsort(sample_weights)[-5:][::-1]
            
            sub_cols = st.columns(5)
            for i, idx in enumerate(top_indices):
                with sub_cols[i]:
                    fig = plot_atom(basis[idx], threshold=global_thresh, title=f"Atom #{idx}\nWeight: {sample_weights[idx]:.2f}")
                    st.pyplot(fig)
                    plt.close(fig)

if __name__ == "__main__":
    main()
