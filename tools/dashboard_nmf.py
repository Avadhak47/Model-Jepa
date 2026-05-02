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
    data = torch.load('arc_data/arc_basis_nmf_200.pt', weights_only=False)
    basis = data['basis'].numpy() # [200, 2700]
    weights = data['weights'].numpy() # [N, 200]
    return basis, weights

def plot_atom(atom_flat, title=""):
    atom = atom_flat.reshape(15, 15, 12)
    # Get the dominant color for each pixel
    # We ignore the pos channels [10, 11] for simple visualization
    grid = np.argmax(atom[:, :, :10], axis=-1)
    
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8)
    return fig

def main():
    st.title("🧩 ARC NMF Basis Audit")
    st.markdown("Auditing the 200 fundamental 'Basis Atoms' discovered via Non-negative Matrix Factorization.")
    
    basis, weights = load_nmf_data()
    n_components = basis.shape[0]
    
    tabs = st.tabs(["📚 Basis Library", "🧪 Synthesis Lab", "🔍 Decomposition"])
    
    with tabs[0]:
        st.header("The 200 Atoms of ARC")
        items_per_page = 40
        page = st.number_input("Page", min_value=1, max_value=5, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, n_components)
        
        cols = st.columns(8)
        for i in range(start_idx, end_idx):
            with cols[(i - start_idx) % 8]:
                fig = plot_atom(basis[i], title=f"Atom #{i}")
                st.pyplot(fig)
                plt.close(fig)
                
    with tabs[1]:
        st.header("Synthesis: Build an Object")
        st.markdown("Manually combine components to see how they form complex structures.")
        
        selected_atoms = st.multiselect("Select Atoms to Combine", options=list(range(n_components)), default=[0, 1, 2])
        
        if selected_atoms:
            combined = np.sum(basis[selected_atoms], axis=0)
            fig = plot_atom(combined, title="Synthesized Grid")
            st.pyplot(fig)
            plt.close(fig)
            
    with tabs[2]:
        st.header("Decomposition: Analyze an Object")
        sample_idx = st.slider("Select Sample Object from Library", 0, weights.shape[0] - 1, 0)
        
        col1, col2 = st.columns(2)
        
        # Original (Approximate from NMF weight reconstruction)
        with col1:
            st.subheader("Original (Approx)")
            reconstructed = weights[sample_idx] @ basis
            fig_orig = plot_atom(reconstructed, title=f"Object #{sample_idx} (NMF Recon)")
            st.pyplot(fig_orig)
            plt.close(fig_orig)
            
        with col2:
            st.subheader("Top Contributing Atoms")
            sample_weights = weights[sample_idx]
            top_indices = np.argsort(sample_weights)[-5:][::-1]
            
            sub_cols = st.columns(5)
            for i, idx in enumerate(top_indices):
                with sub_cols[i]:
                    weight_val = sample_weights[idx]
                    fig = plot_atom(basis[idx], title=f"Atom #{idx}\nWeight: {weight_val:.2f}")
                    st.pyplot(fig)
                    plt.close(fig)

if __name__ == "__main__":
    main()
