# NS-ARC — Neuro-Symbolic RL Framework

A modular, research-grade PyTorch library for curiosity-driven, model-based RL on ARC-AGI tasks.

## Architecture

```
modules/
  encoders.py       — CNN / Transformer (ViT) encoders
  world_models.py   — MLP / Gaussian / Transformer world models (32/64/128-layer AttnRes)
  policies.py       — PPO, DQN, Decision Transformer
  planners.py       — MCTS (MuZero-style), CEM
  curiosity.py      — RND, Prediction Error, Ensemble Disagreement
  symbolic.py       — Program Generator, Constraint Mask
training/
  trainer.py        — Isolated + E2E training loops
  replay_buffer.py  — O(1) circular replay buffer
analysis/
  latent_analysis.py — PCA / t-SNE / UMAP projections
  plotting.py        — Rollout + attention entropy dashboards
```

## Quick Start

```bash
pip install -r requirements.txt

# Base model (4-layer WM)
python main.py --mode debug --profile base

# Deep 32-layer Transformer (AttnRes)
python main.py --mode train --profile deep32 --arc-data /path/to/arc-agi/training

# Three-model depth comparison experiment
jupyter notebook experiment.ipynb
```

## Depth Profiles

| Profile | WM Layers | Enc Depth | MCTS Sims | ~Params |
|---------|-----------|-----------|-----------|---------|
| `base`  | 4         | 4         | 25        | 3.5M    |
| `deep32`| 32        | 4         | 25        | 34M     |
| `deep64`| 64        | 8         | 50        | 67M     |
| `deep128`| 128      | 12        | 100       | 134M    |

## ARC Data

```bash
git clone https://github.com/fchollet/ARC-AGI ./data/arc-agi
```

## W&B Tracking

Set `WANDB_API_KEY` environment variable or call `wandb.login()` in the notebook.  
All training metrics — WM loss, SIGReg, attention entropy, policy entropy, RND — are logged automatically.

## References

- [Attention Residuals (arXiv:2603.15031)](https://arxiv.org/abs/2603.15031)
- [ARC-AGI Challenge](https://github.com/fchollet/ARC-AGI)
