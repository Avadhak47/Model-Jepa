import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def project_latents(latents: np.ndarray, method="PCA", n_components=2):
    """
    Projection: Reduce high-dimensional continuous latent states into 2D/3D visual mappings.
    latents: [N, latent_dim] numpy array
    """
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, learning_rate='auto', init='random')
    elif method == "UMAP":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP library is not installed. Run `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError(f"Unknown projection method {method}")
        
    return reducer.fit_transform(latents)

def plot_clusters(latents_2d: np.ndarray, labels: np.ndarray, title="Latent Space Clustering"):
    """
    Generates a scatter plot of projected latent vectors colored by associated environment task or goal IDs.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Task / Goal ID")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_trajectory(latents_2d: np.ndarray, title="Continuous Latent Trajectory"):
    """
    Draws a visual path representing the sequential evolution of latent vectors across a single episode rollout.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(latents_2d[:, 0], latents_2d[:, 1], marker='o', linestyle='-', alpha=0.6, color='royalblue')
    
    # Emphasize terminal zones
    plt.scatter(latents_2d[0, 0], latents_2d[0, 1], color='forestgreen', s=120, label="Episode Start", zorder=5)
    plt.scatter(latents_2d[-1, 0], latents_2d[-1, 1], color='crimson', s=120, label="Episode End", zorder=5)
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
