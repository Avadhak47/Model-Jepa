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

# 2. Clone Re-ARC dataset (40k+ procedurally generated pairs)
#    Re-ARC is recommended over ARC-AGI-1 for training as it has
#    100 generated grid pairs per task (400 tasks = 40,000+ pairs).
if [ ! -d "data/re-arc" ]; then
    echo "Cloning Re-ARC dataset (~40k generated grid pairs)..."
    git clone --quiet https://github.com/michaelhodel/re-arc data/re-arc
    echo "Re-ARC cloned: $(ls data/re-arc/*.json 2>/dev/null | wc -l | tr -d ' ') task files"
fi

# 3. Set W&B key (update below or set WANDB_API_KEY env var before running)
# export WANDB_API_KEY="YOUR_KEY_HERE"

echo "=== Setup complete ==="
echo "GPU  : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Name : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""
echo "Dataset: set REARC_DATA_PATH=data/re-arc in notebook Cell 2 (setup)"
