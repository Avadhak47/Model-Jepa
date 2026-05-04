import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os

ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
              '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
cmap = ListedColormap(ARC_COLORS)

def preview_library():
    path = 'arc_data/primitive_library.pt'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    data = torch.load(path, weights_only=False)
    tensors = data['tensors'] # [N, 15, 15]
    n_total = tensors.shape[0]
    print(f"Total objects in library: {n_total}")

    # Pick 64 random objects to show
    indices = np.random.choice(n_total, 64, replace=False)
    samples = tensors[indices]

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle("Raw Segmented Objects (The Input to NMF)", fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/raw_segmented_objects.png')
    print("Saved preview to artifacts/raw_segmented_objects.png")

if __name__ == "__main__":
    preview_library()
