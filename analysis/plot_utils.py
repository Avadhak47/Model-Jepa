import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_reconstruction_dashboard(original, reconstructed, latent, epoch, save_path):
    """
    4-panel diagnostic dashboard to monitor Phase 1 (Representation Learning).
    
    1. Grid Reconstruction (Original vs Predicted)
    2. Error Heatmap (Spatial failing)
    3. Color Histogram (Palette matching)
    4. Latent Spectrum (Anti-collapse check)
    """
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    latent = latent.cpu().detach().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Phase 1 Diagnostic Dashboard - Epoch {epoch}", fontsize=16)

    # Panel 1: Reconstruction
    combined = np.hstack([original, reconstructed])
    axes[0, 0].imshow(combined, cmap='tab10', vmin=0, vmax=9)
    axes[0, 0].set_title("Original Grid (Left) vs. Reconstructed (Right)")
    axes[0, 0].axis('off')

    # Panel 2: Error Heatmap
    error = np.abs(original - reconstructed)
    sns.heatmap(error, ax=axes[0, 1], cmap="YlOrRd", cbar=True)
    axes[0, 1].set_title("Pixel-wise Error Heatmap")

    # Panel 3: Color Histogram Comparison
    target_counts = np.bincount(original.flatten().astype(int), minlength=10) / original.size
    recon_counts = np.bincount(reconstructed.flatten().astype(int), minlength=10) / reconstructed.size
    
    x = np.arange(10)
    axes[1, 0].bar(x - 0.2, target_counts, width=0.4, label='True', color='blue', alpha=0.6)
    axes[1, 0].bar(x + 0.2, recon_counts, width=0.4, label='Recon', color='red', alpha=0.6)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_title("Color Distribution Mapping")
    axes[1, 0].legend()

    # Panel 4: Latent Spectrum (Anti-collapse check)
    # std deviation across the latent dimensions
    latent_std = np.std(latent, axis=0) # [latent_dim]
    axes[1, 1].bar(range(len(latent_std)), latent_std, color='purple', alpha=0.7)
    axes[1, 1].set_title(f"Latent Spectrum (Mean Std: {latent_std.mean():.4f})")
    axes[1, 1].set_xlabel("Latent Dimension (1-128)")
    axes[1, 1].set_ylabel("Std Dev")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    return save_path
