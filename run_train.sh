#!/bin/bash

# Configuration
PROJECT="NS-ARC-Scaling"
RUN_NAME="Harmonic-Slot-Small-Server"
EPOCHS=500
BATCH_SIZE=128
DATA_PATH="./arc_data/re-arc"

# Ensure W&B API Key is set if available
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY is not set. You may need to log in manually."
fi

# Run training
python3 train_small_slotted.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --data_path $DATA_PATH \
    --wandb_project $PROJECT \
    --run_name $RUN_NAME \
    --resume
