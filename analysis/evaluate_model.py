import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from datetime import datetime

class ComprehensiveEvaluator:
    """
    Comprehensive pipeline to evaluate, validate, and visualize the 
    under-the-hood workings of the NS-ARC model.
    """
    def __init__(self, config: dict, env, modules: dict, dataset):
        self.config = config
        self.env = env
        self.modules = sorted_modules(modules)
        self.dataset = dataset
        self.device = config.get("device", "cpu")
        self.save_dir = "evaluation_reports"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Timestamp for current evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for m in self.modules.values():
            m.eval()

    def _encode(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).to(self.device)
            if s.dim() == 2: s = s.unsqueeze(0).unsqueeze(0)
            elif s.dim() == 3: s = s.unsqueeze(0)
            return self.modules["encoder"]({"state": s})

    def run_full_diagnostics(self, n_episodes=5):
        print(f"\n[{self.timestamp}] Starting Comprehensive Evaluation...")
        metrics = {}
        
        # 1. Evaluate Latent Manifold & Encoder
        metrics["encoder"] = self.evaluate_encoder_and_manifold()
        
        # 2. Evaluate World Model Dynamics
        metrics["world_model"] = self.evaluate_world_model()
        
        # 3. Evaluate End-to-End Policy Performance
        metrics["policy"] = self.evaluate_policy_performance(n_episodes)
        
        self._print_summary(metrics)
        return metrics

    def evaluate_encoder_and_manifold(self):
        print("\n=> Evaluating Encoder & Latent Manifold...")
        latents = []
        labels = [] # We use the 'target_reward' as dummy labels if task IDs aren't available
        
        batch = self.dataset.sample(min(100, len(self.dataset)))
        states = batch["state"].to(self.device).float()
        
        with torch.no_grad():
            # Get latents
            z_dict = self.modules["encoder"]({"state": states})
            z = z_dict["latent"]
            
            # Reconstruction via Decoder (if it exists)
            recon_mse = None
            if "decoder" in self.modules:
                recon = self.modules["decoder"]({**batch, "latent": z})
                logits = recon["reconstructed_logits"]
                pred = logits.argmax(dim=1).float()
                target = states.squeeze(1).float()
                recon_mse = torch.nn.functional.mse_loss(pred, target).item()
                self._plot_reconstruction(target[0], pred[0])

        z_np = z.cpu().numpy()
        
        # Calculate latent spread (std dev across batch) to check for collapse
        latent_std = np.std(z_np, axis=0).mean()
        
        # Visualize Manifold
        self._plot_umap_manifold(z_np)
        
        return {
            "latent_std": latent_std,
            "latent_collapse": latent_std < 0.1,
            "reconstruction_mse": recon_mse
        }

    def evaluate_world_model(self):
        print("=> Evaluating World Model Dynamics...")
        batch = self.dataset.sample(32)
        states = batch["state"].to(self.device).float()
        actions = torch.randn(32, self.config.get("action_dim", 10)).to(self.device)
        
        with torch.no_grad():
            z = self.modules["encoder"]({"state": states})["latent"]
            
            # If autoregressive, test 1 step prediction MSE
            wm_out = self.modules["world_model"]({"latent": z, "action": actions})
            pred_z = wm_out.get("next_latent", z)
            
            # Note: We don't have true next_z since it's a generated action.
            # We measure magnitude of change as a sanity check.
            z_delta = torch.norm(pred_z - z, dim=-1).mean().item()
            attn_ent = wm_out.get("attention_entropy", 0.0)

        return {
            "mean_latent_shift": z_delta,
            "attention_entropy": attn_ent
        }

    def evaluate_policy_performance(self, n_episodes):
        print("=> Evaluating Policy in Environment...")
        total_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs = self.env.reset()
            ep_reward = 0
            steps = 0
            rewards_over_time = []
            
            for _ in range(self.config.get("max_steps", 50)):
                z = self._encode(obs["state"])["latent"]
                
                with torch.no_grad():
                    action_out = self.modules["policy"]({"latent": z})
                
                action = action_out["action"][0].cpu().numpy()
                if action.ndim == 0:
                    a_vec = np.zeros(self.config.get("action_dim", 10))
                    a_vec[int(action) % len(a_vec)] = 1.0
                    action = a_vec
                    
                obs, r, done, info = self.env.step(action)
                ep_reward += r
                steps += 1
                rewards_over_time.append(r)
                if done: break
                
            total_rewards.append(ep_reward)
            episode_lengths.append(steps)
            self._plot_episode_reward(rewards_over_time, ep)

        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": sum(r > 0.9 for r in total_rewards) / n_episodes
        }

    # ─── Plotting Methods ───────────────────────────────────────────────
    def _plot_reconstruction(self, original, reconstructed):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original.cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[0].set_title("Original State")
        axes[1].imshow(reconstructed.cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[1].set_title("Decoder Reconstruction")
        fig.savefig(f"{self.save_dir}/reconstruction_{self.timestamp}.png")
        plt.close(fig)

    def _plot_umap_manifold(self, z_np):
        if len(z_np) < 5: return # Not enough data for UMAP
        reducer = UMAP(n_neighbors=5, random_state=42)
        embedding = reducer.fit_transform(z_np)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        ax.set_title("UMAP Projection of Latent Manifold")
        fig.savefig(f"{self.save_dir}/latent_manifold_{self.timestamp}.png")
        plt.close(fig)

    def _plot_episode_reward(self, rewards, ep_idx):
        if not rewards: return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rewards, marker='o')
        ax.set_title(f"Reward Tragectory - Episode {ep_idx}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        fig.savefig(f"{self.save_dir}/reward_ep{ep_idx}_{self.timestamp}.png")
        plt.close(fig)

    def _print_summary(self, metrics):
        print("\n" + "="*50)
        print("📊 COMPREHENSIVE EVALUATION SUMMARY")
        print("="*50)
        print(f" Encoder Latent Std : {metrics['encoder']['latent_std']:.4f}")
        print(f" Latent Collapse?   : {'YES ⚠️' if metrics['encoder']['latent_collapse'] else 'NO ✅'}")
        if metrics['encoder']['reconstruction_mse']:
            print(f" Decoder Recon MSE  : {metrics['encoder']['reconstruction_mse']:.4f}")
        
        print(f" WM Extracted Shift : {metrics['world_model']['mean_latent_shift']:.4f}")
        print(f" WM Attn Entropy    : {metrics['world_model']['attention_entropy']}")
        
        print(f" Policy Mean Reward : {metrics['policy']['mean_reward']:.4f} ± {metrics['policy']['std_reward']:.4f}")
        print(f" Policy Mean Length : {metrics['policy']['mean_length']:.1f} steps")
        print(f" Policy Win Rate    : {metrics['policy']['success_rate']*100:.1f}%")
        print("="*50)
        print(f"Plots saved to: {os.path.abspath(self.save_dir)}")

def sorted_modules(modules_dict):
    """Ensure modules order mapping for strict requirements."""
    return modules_dict
