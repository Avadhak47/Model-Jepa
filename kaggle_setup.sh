#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# NS-ARC Kaggle Bootstrap Script
# Run this in the first cell of your Kaggle notebook:
#   import subprocess; subprocess.run(['bash', 'kaggle_setup.sh'])
# ─────────────────────────────────────────────────────────────────────────

set -e

echo "=== NS-ARC Kaggle Setup ==="

# 1. Install Python dependencies
pip install -q wandb umap-learn scikit-learn tqdm

# 2. Download ARC-AGI data (if not already present)
if [ ! -d "data/arc-agi" ]; then
    echo "Cloning ARC-AGI dataset..."
    git clone --quiet https://github.com/fchollet/ARC-AGI data/arc-agi
fi

# 3. Set W&B to offline mode first (update key below)
# export WANDB_API_KEY="YOUR_KEY_HERE"

echo "=== Setup complete ==="
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name:      $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')"
