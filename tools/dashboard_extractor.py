import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

st.set_page_config(page_title="ARC Object Vocabulary", layout="wide")

# ARC Standard Colors
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)

@st.cache_resource
def load_library(path):
    if not os.path.exists(path):
        return None
    return torch.load(path)

def plot_object(tensor_grid, title=""):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(tensor_grid, cmap=cmap, vmin=0, vmax=9)
    
    # Draw grid lines
    ax.set_xticks(np.arange(-.5, tensor_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, tensor_grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return fig

def main():
    st.title("🧩 ARC Object Vocabulary Explorer")
    st.markdown("Explore the discrete semantic primitives extracted by the Multi-Hypothesis Extractor.")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    library_path = os.path.join(base_dir, 'arc_data', 'primitive_library.pt')
    
    payload = load_library(library_path)
    if payload is None:
        st.error(f"Library not found at {library_path}. Please run the extractor first.")
        return
        
    tensors = payload['tensors'].numpy()
    metadata = payload['metadata']
    
    st.sidebar.header("Filter Objects")
    
    # Extract unique hypotheses and count them
    from collections import Counter
    all_hypotheses = [m['hypothesis'] for m in metadata]
    counts = Counter(all_hypotheses)
    
    st.sidebar.markdown("### 📊 Dataset Breakdown")
    for h, c in counts.items():
        st.sidebar.markdown(f"- **{h}**: {c}")
        
    st.sidebar.divider()
    
    hypotheses = list(counts.keys())
    selected_hypo = st.sidebar.multiselect("Select Hypothesis to View", hypotheses, default=hypotheses)
    
    # Allow filtering by Task ID
    all_tasks = list(set([m['task_id'] for m in metadata]))
    search_task = st.sidebar.text_input("Search Task ID (leave empty for all)")
    
    # Filter the dataset
    filtered_indices = []
    for i, meta in enumerate(metadata):
        match_hypo = meta['hypothesis'] in selected_hypo
        match_task = (search_task in meta['task_id']) if search_task else True
        if match_hypo and match_task:
            filtered_indices.append(i)
            
    st.subheader(f"Showing {len(filtered_indices)} of {len(tensors)} Total Objects")
    
    # Pagination
    items_per_page = 24
    total_pages = max(1, len(filtered_indices) // items_per_page + (1 if len(filtered_indices) % items_per_page > 0 else 0))
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
        
    # Reset page if out of bounds (e.g. after filtering)
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1
        
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous Page"):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
    with col2:
        st.markdown(f"<div style='text-align: center;'><b>Page {st.session_state.current_page} of {total_pages}</b></div>", unsafe_allow_html=True)
    with col3:
        if st.button("Next Page ➡️"):
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()
    
    page = st.session_state.current_page
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_indices))
    
    current_indices = filtered_indices[start_idx:end_idx]
    
    # Display in a grid
    cols = st.columns(4)
    for idx, obj_idx in enumerate(current_indices):
        col = cols[idx % 4]
        with col:
            tensor = tensors[obj_idx]
            meta = metadata[obj_idx]
            
            fig = plot_object(tensor)
            st.pyplot(fig)
            
            st.markdown(f"**Hypothesis**: `{meta['hypothesis']}`")
            st.markdown(f"**Task**: `{meta['task_id']}`")
            st.markdown(f"**Background**: Color {meta['bg_color']}")
            st.divider()

if __name__ == "__main__":
    main()
