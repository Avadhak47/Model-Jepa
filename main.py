import argparse
import sys
import torch
import yaml

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ─────────────────────────── Module Factory ────────────────────────────────
ENCODER_REGISTRY = {
    "MLPEncoder":         ("modules.encoders",    "MLPEncoder"),
    "CNNEncoder":         ("modules.encoders",    "CNNEncoder"),
    "TransformerEncoder": ("modules.encoders",    "TransformerEncoder"),
}

WORLD_MODEL_REGISTRY = {
    "MLPDynamicsModel":         ("modules.world_models", "MLPDynamicsModel"),
    "GaussianDynamicsModel":    ("modules.world_models", "GaussianDynamicsModel"),
    "TransformerWorldModel":    ("modules.world_models", "TransformerWorldModel"),
    "TransformerWorldModel32":  ("modules.world_models", "TransformerWorldModel32"),
    "TransformerWorldModel64":  ("modules.world_models", "TransformerWorldModel64"),
    "TransformerWorldModel128": ("modules.world_models", "TransformerWorldModel128"),
}

POLICY_REGISTRY = {
    "PPOPolicy":                  ("modules.policies", "PPOPolicy"),
    "DQNPolicy":                  ("modules.policies", "DQNPolicy"),
    "DecisionTransformerPolicy":  ("modules.policies", "DecisionTransformerPolicy"),
}

CURIOSITY_REGISTRY = {
    "RNDCuriosity":               ("modules.curiosity", "RNDCuriosity"),
    "PredErrCuriosity":           ("modules.curiosity", "PredErrCuriosity"),
    "EnsembleDisagreementCuriosity": ("modules.curiosity", "EnsembleDisagreementCuriosity"),
}

DATASET_REGISTRY = {
    "ARC":     ("arc_data.arc_dataset",     "ARCDataset"),
    "ReARC":   ("arc_data.rearc_dataset",   "ReARCDataset"),
    "Terrain": ("arc_data.terrain_dataset", "TerrainDataset"),
}

def _load_class(module_path, class_name):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)

def build_module(registry: dict, key: str, config: dict):
    if key not in registry:
        raise KeyError(f"Unknown module key '{key}'. Available: {list(registry.keys())}")
    mod_path, cls_name = registry[key]
    cls = _load_class(mod_path, cls_name)
    return cls(config)

# ─────────────────────────── CLI ────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="NS-ARC Framework Driver")
    parser.add_argument("--config", type=str, default="/kaggle/working/Model-Jepa/config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["train", "eval", "validate", "debug", "comprehensive_eval"], default="train")
    parser.add_argument("--arc-data", type=str, default=None, help="Path to ARC JSON folder")
    parser.add_argument("--profile", choices=["base", "deep32", "deep64", "deep128"], default="base",
                        help="Model depth profile override")
    return parser.parse_args()

# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    args = parse_args()
    config = load_yaml(args.config)

    # Apply profile overrides
    PROFILES = {
        "base":    {"world_model": "TransformerWorldModel",    "num_layers": 4},
        "deep32":  {"world_model": "TransformerWorldModel32",  "num_layers": 32},
        "deep64":  {"world_model": "TransformerWorldModel64",  "num_layers": 64},
        "deep128": {"world_model": "TransformerWorldModel128", "num_layers": 128},
    }
    config.update(PROFILES[args.profile])
    if args.arc_data:
        config["arc_data_path"] = args.arc_data

    device_str = config.get("device", "cpu")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    config["device"] = device_str
    device = torch.device(device_str)

    torch.manual_seed(config.get("seed", 42))
    print(f"\n{'='*55}")
    print(f"  NS-ARC Framework  |  profile={args.profile}  |  device={device}")
    print(f"  mode={args.mode}  |  num_layers={config['num_layers']}")
    print(f"{'='*55}")

    # ─── Build Dataset ────────────────────────────────────────
    dataset_key = config.get("dataset", "ARC")
    arc_path = config.get("arc_data_path", "./data/arc-agi/training")
    if dataset_key == "ARC":
        Dataset = _load_class("arc_data.arc_dataset", "ARCDataset")
        dataset = Dataset(arc_path)
    elif dataset_key == "Terrain":
        Dataset = _load_class("arc_data.terrain_dataset", "TerrainDataset")
        dataset = Dataset(config)
    else:
        Dataset = _load_class("arc_data.rearc_dataset", "ReARCDataset")
        dataset = Dataset(arc_path)

    # ─── Build Modules ────────────────────────────────────────
    wm_key   = config.get("world_model", "TransformerWorldModel")
    enc_key  = config.get("encoder",     "CNNEncoder")
    pol_key  = config.get("policy",      "PPOPolicy")

    modules = {}
    modules["encoder"]     = build_module(ENCODER_REGISTRY,      enc_key,  config)
    modules["world_model"] = build_module(WORLD_MODEL_REGISTRY,  wm_key,   config)
    modules["policy"]      = build_module(POLICY_REGISTRY,       pol_key,  config)

    total_params = sum(p.numel() for m in modules.values() for p in m.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ─── Build Trainer ────────────────────────────────────────
    from training.trainer import Trainer
    from envs.arc_env import ARCEnvironment
    env = ARCEnvironment(config)
    trainer = Trainer(config, env, modules)

    # ─── Run ──────────────────────────────────────────────────
    from training.replay_buffer import ReplayBuffer
    replay = ReplayBuffer(capacity=config.get("replay_capacity", 10000), device=device_str)

    if args.mode == "train":
        epochs = config.get("epochs", 100)
        train_mode = config.get("train_mode", "world_model")
        trainer.train(epochs, replay, mode=train_mode)
    elif args.mode == "eval":
        result = trainer.evaluate()
        print(f"Eval total reward: {result}")
    elif args.mode == "validate":
        result = trainer.validate(replay)
        print(f"Validation metrics: {result}")
    elif args.mode == "comprehensive_eval":
        from analysis.evaluate_model import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator(config, env, modules, dataset)
        evaluator.run_full_diagnostics(n_episodes=10)
    else:
        sample = dataset.sample(4)
        print("Debug sample shapes:", {k: v.shape for k, v in sample.items()})

if __name__ == "__main__":
    main()
