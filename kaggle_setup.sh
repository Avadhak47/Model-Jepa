#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# NS-ARC Kaggle Bootstrap Script
# Run this in the first cell of your Kaggle notebook:
#   import subprocess; subprocess.run(['bash', 'kaggle_setup.sh'])
# ─────────────────────────────────────────────────────────────────────────

set -e

echo "=== NS-ARC Kaggle Setup ==="

# 1. Fetch Repository Code if running on Kaggle
if [ ! -f "data/rearc_dataset.py" ]; then
    echo "Kaggle environment detected. Fetching project code..."
    git clone --quiet https://github.com/Avadhak47/Model-Jepa .tmp_repo
    cp -rn .tmp_repo/* .
    rm -rf .tmp_repo
fi

# 2. Install Python dependencies
pip install -q wandb umap-learn scikit-learn tqdm seaborn

# 2. Clone Re-ARC dataset (~40k generated grid pairs)
if [ ! -d "data/re-arc" ]; then
    echo "Cloning Re-ARC repository..."
    git clone --quiet https://github.com/michaelhodel/re-arc data/re-arc
fi

# 3. Handle Zip Extraction (Puzzles are in re_arc.zip)
cd data/re-arc
if [ ! -d "re_arc/tasks" ]; then
    echo "Extracting re_arc.zip..."
    unzip -q re_arc.zip
    # This creates data/re-arc/re_arc/tasks/*.json
fi

if [ ! -d "arc_original/tasks" ]; then
    echo "Extracting arc_original.zip..."
    unzip -q arc_original.zip
    # This creates data/re-arc/arc_original/tasks/*.json
fi
cd ../..

echo "Setup complete. Puzzles located in data/re-arc/re_arc/tasks/"

# 4. Verification
echo "Re-ARC Tasks: $(ls data/re-arc/re_arc/tasks/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "=== Setup complete ==="
