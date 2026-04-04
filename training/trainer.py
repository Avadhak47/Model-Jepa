import os
import shutil
import torch
import torch.nn.functional as F
from datetime import datetime
from training.utils import compute_gae
from analysis.plot_utils import plot_reconstruction_dashboard

class Trainer:
    """Orchestrates structured RL routines conforming to the TensorDict framework."""
    def __init__(self, config: dict, env, modules: dict, model_name: str = "base"):
        self.config = config
        self.env = env
        self.modules = modules
        self.model_name = model_name
        self.device = config.get("device", "cpu")
        self.clip_eps = config.get("clip_eps", 0.2)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        self.ckpt_dir = f"./checkpoints/{self.model_name}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs("evaluation_reports/plots", exist_ok=True)

        self.optimizers = {}
        for name, module in self.modules.items():
            if hasattr(module, 'parameters') and len(list(module.parameters())) > 0:
                self.optimizers[name] = torch.optim.AdamW(
                    module.parameters(),
                    lr=config.get("learning_rate", 1e-3),
                    weight_decay=config.get("weight_decay", 1e-4)
                )

    # ─── utility ────────────────────────────────────────────────────────────
    def _encode(self, state_tensor):
        """Encode a raw state tensor to a latent dict via encoder module."""
        enc = self.modules.get("encoder")
        if enc is None:
            raise RuntimeError("No 'encoder' module registered in Trainer.")
        s = state_tensor.float().to(self.device)
        if s.dim() == 2:                   # [B, D] → add channel dims for CNN encoder
            s = s.unsqueeze(1).unsqueeze(1)
        elif s.dim() == 3:                 # [B, H, W] → add channel
            s = s.unsqueeze(1)
        return enc({"state": s})

    # ─── representation learning ─────────────────────────────────────────────
    def train_representation_learning_step(self, batch: dict) -> dict:
        """
        Autoencoding loop: encode the state to a latent, decode back to pixels,
        and minimise reconstruction MSE.
        """
        encoder = self.modules.get("encoder")
        decoder = self.modules.get("decoder")
        if encoder is None or decoder is None:
            return {}

        enc_opt = self.optimizers["encoder"]
        dec_opt = self.optimizers["decoder"]
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        z_dict = self._encode(batch["state"])
        merged = {**batch, **z_dict}
        recon_dict = decoder(merged)
        loss_dict = decoder.loss(merged, recon_dict)
        loss_dict["loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            self.max_grad_norm
        )
        enc_opt.step()
        dec_opt.step()
        return {"recon_loss": loss_dict["loss"].item()}

    # ─── world model ────────────────────────────────────────────────────────
    def train_world_model_step(self, batch: dict) -> dict:
        """
        Teacher-forced dynamics prediction:
          latent_t + action_t → predict latent_{t+1} + reward_t
        """
        wm  = self.modules["world_model"]
        opt = self.optimizers["world_model"]

        # Encode state if raw pixels supplied
        if "latent" not in batch:
            z_dict = self._encode(batch["state"])
            batch  = {**batch, **z_dict}

        opt.zero_grad()
        outputs   = wm(batch)
        loss_dict = wm.loss(batch, outputs)
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), self.max_grad_norm)
        opt.step()

        def _item(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v or 0.0)

        return {
            "wm_loss":             _item(loss_dict.get("loss")),
            "wm_z_loss":           _item(loss_dict.get("z_loss")),
            "wm_r_loss":           _item(loss_dict.get("r_loss")),
            "wm_sigreg_loss":      _item(loss_dict.get("sigreg_loss")),
            "wm_attn_entropy":     _item(loss_dict.get("attention_entropy")),
            "wm_grad_norm":        grad_norm.item(),
        }

    # ─── policy (PPO) ────────────────────────────────────────────────────────
    def train_policy_step(self, batch: dict) -> dict:
        """
        Full PPO surrogate step with clipping, value regression, and entropy bonus.
        Requires batch to contain: old_log_probs, advantages, returns, taken_actions.
        """
        policy = self.modules["policy"]
        opt    = self.optimizers["policy"]

        if "latent" not in batch:
            with torch.no_grad():
                z_dict = self._encode(batch["state"])
            batch = {**batch, **z_dict}

        opt.zero_grad()
        outputs   = policy(batch)
        loss_dict = policy.loss(batch, outputs)
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()

        def _item(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v or 0.0)

        return {
            "policy_loss":   _item(loss_dict.get("policy_loss")),
            "value_loss":    _item(loss_dict.get("value_loss")),
            "entropy":       _item(loss_dict.get("entropy")),
            "policy_grad":   grad_norm.item(),
        }

    # ─── curiosity ──────────────────────────────────────────────────────────
    def train_curiosity_step(self, batch: dict) -> dict:
        """RND/Pred-Err curiosity distillation step."""
        cur = self.modules.get("curiosity")
        if cur is None:
            return {}
        opt = self.optimizers["curiosity"]

        if "latent" not in batch:
            with torch.no_grad():
                z_dict = self._encode(batch["state"])
            batch = {**batch, **z_dict}

        opt.zero_grad()
        cur_out  = cur(batch)
        loss_dict = cur.loss(batch, cur_out)
        loss_dict["loss"].backward()
        opt.step()
        return {
            "curiosity_loss":     loss_dict["loss"].item(),
            "intrinsic_reward":   cur_out["intrinsic_reward"].mean().item(),
        }

    # ─── end-to-end ─────────────────────────────────────────────────────────
    def train_end_to_end_step(self, batch: dict) -> dict:
        """
        Joint gradient step:
          1. Encode state → z
          2. WM predicts next z and reward  (loss flows back through encoder)
          3. Policy maximizes returns + entropy  (stopped gradient from encoder)
          4. Curiosity distillation
        """
        enc = self.modules.get("encoder")
        wm  = self.modules["world_model"]
        pol = self.modules["policy"]
        cur = self.modules.get("curiosity")

        all_params = [p for m in self.modules.values() for p in m.parameters()]
        for opt in self.optimizers.values():
            opt.zero_grad()

        # 1. Encode
        state  = batch["state"].float().to(self.device)
        if state.dim() == 3:
            state = state.unsqueeze(1)
        z_dict = enc({"state": state})
        z      = z_dict["latent"]

        # 2. World model
        wm_inputs = {**batch, "latent": z}
        wm_out    = wm(wm_inputs)
        wm_loss   = wm.loss(wm_inputs, wm_out)["loss"]

        # 3. Policy (stop grad from encoder so WM and policy compete fairly)
        pol_out  = pol({"latent": z.detach()})
        pol_loss = -pol_out["entropy"]          # entropy maximisation proxy

        # 4. Curiosity
        cur_loss = torch.tensor(0.0, device=self.device)
        if cur is not None:
            cur_out  = cur({"latent": z.detach()})
            cur_loss = cur.loss({}, cur_out)["loss"]

        total = wm_loss + 0.1 * pol_loss + 0.05 * cur_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        for opt in self.optimizers.values():
            opt.step()

        def _v(t): return t.item() if isinstance(t, torch.Tensor) else float(t)
        return {
            "e2e_loss":   _v(total),
            "wm_loss":    _v(wm_loss),
            "pol_loss":   _v(pol_loss),
            "cur_loss":   _v(cur_loss),
            "attn_ent":   _v(wm_out.get("attention_entropy", 0.0)),
            "pol_entropy": _v(pol_out["entropy"]),
        }

    # ─── main loop ──────────────────────────────────────────────────────────
    def train(self, num_epochs: int, replay_buffer, mode: str = "end_to_end"):
        """Main training loop. mode ∈ {representation, world_model, policy, curiosity, end_to_end}."""
        print(f"Starting {mode} training for {num_epochs} epochs…")
        batch_size = self.config.get("batch_size", 32)
        dispatch   = {
            "representation": self.train_representation_learning_step,
            "world_model":    self.train_world_model_step,
            "policy":         self.train_policy_step,
            "curiosity":      self.train_curiosity_step,
            "end_to_end":     self.train_end_to_end_step,
        }
        step_fn = dispatch.get(mode, self.train_end_to_end_step)

        for epoch in range(num_epochs):
            batch   = replay_buffer.sample(batch_size)
            metrics = step_fn(batch)
            
            # 1. Visualization (Every 5 epochs for Stage 1 / Representation)
            if mode == "representation" and epoch % 5 == 0:
                self._visualize_diagnostics(batch, epoch)

            # 2. Metrics Logging
            if epoch % max(1, num_epochs // 10) == 0:
                metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"  Epoch {epoch:>5}/{num_epochs}  {metric_str}")
            
            # 3. Rolling Checkpointing (Every epoch)
            self._save_checkpoint_rolling(epoch)

        print(f"Training complete. Checkpoints saved to {self.ckpt_dir}")

    def _save_checkpoint_rolling(self, epoch):
        """Save weights every epoch, keeping only the last 5."""
        epoch_dir = os.path.join(self.ckpt_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        for name, module in self.modules.items():
            torch.save(module.state_dict(), os.path.join(epoch_dir, f"{name}.pt"))
        
        # Also update 'latest'
        latest_dir = os.path.join(self.ckpt_dir, "latest")
        if os.path.exists(latest_dir):
            if os.path.islink(latest_dir): os.unlink(latest_dir)
            else: shutil.rmtree(latest_dir)
        
        # Use a copy instead of symlink for better Kaggle/Windows compatibility
        shutil.copytree(epoch_dir, latest_dir)

        # Rotate: Keep only last 5
        history_size = 5
        all_epochs = sorted([
            int(d.split('_')[1]) for d in os.listdir(self.ckpt_dir) 
            if d.startswith("epoch_") and os.path.isdir(os.path.join(self.ckpt_dir, d))
        ])
        
        if len(all_epochs) > history_size:
            for old_epoch in all_epochs[:-history_size]:
                old_dir = os.path.join(self.ckpt_dir, f"epoch_{old_epoch}")
                shutil.rmtree(old_dir)

    def _visualize_diagnostics(self, batch, epoch):
        """Generates the 4-panel Stage 1 dashboard."""
        encoder = self.modules.get("encoder")
        decoder = self.modules.get("decoder")
        if encoder is None or decoder is None: return

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            z_dict = self._encode(batch["state"])
            recon_dict = decoder({**batch, **z_dict})
            
            # Sample first item in batch for visualization
            orig = batch["state"][0, 0] # [H, W]
            recon_logits = recon_dict.get("reconstructed_logits")
            if recon_logits is not None:
                recon = recon_logits[0].argmax(dim=0) # [H, W]
            else:
                # If using MSE decoder directly on pixels
                recon = recon_dict["reconstruction"][0].view(orig.shape)
            
            save_path = f"evaluation_reports/plots/diag_epoch_{epoch}.png"
            plot_reconstruction_dashboard(orig, recon, z_dict["latent"], epoch, save_path)
            
        encoder.train(); decoder.train()

    # ─── evaluation ─────────────────────────────────────────────────────────
    def evaluate(self, max_steps: int = 200) -> float:
        """
        Run one full episode using encoder → policy, return total reward.
        Properly encodes each raw state before feeding to policy.
        """
        enc    = self.modules.get("encoder")
        policy = self.modules["policy"]
        enc.eval(); policy.eval()

        with torch.no_grad():
            obs        = self.env.reset()
            state      = torch.tensor(obs["state"], dtype=torch.float32).to(self.device)
            if state.dim() == 2:
                state = state.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            elif state.dim() == 3:
                state = state.unsqueeze(0)

            total_reward = 0.0
            for _ in range(max_steps):
                z      = enc({"state": state})["latent"]
                action = policy({"latent": z})["action"]
                a_np   = action[0].cpu().numpy()
                # Convert to one-hot if scalar
                if a_np.ndim == 0:
                    a_vec = [0.0] * self.config.get("action_dim", 10)
                    a_vec[int(a_np) % len(a_vec)] = 1.0
                    a_np = a_vec
                obs, reward, done, _ = self.env.step(a_np)
                total_reward += reward
                state = torch.tensor(obs["state"], dtype=torch.float32).to(self.device)
                if state.dim() == 2:
                    state = state.unsqueeze(0).unsqueeze(0)
                elif state.dim() == 3:
                    state = state.unsqueeze(0)
                if done:
                    break

        enc.train(); policy.train()
        return total_reward

    # ─── validation ─────────────────────────────────────────────────────────
    def validate(self, replay_buffer, n_batches: int = 10) -> dict:
        """
        Compute held-out WM prediction loss on n_batches of replay data
        without gradient updates.
        """
        wm  = self.modules["world_model"]
        enc = self.modules.get("encoder")
        wm.eval()
        if enc: enc.eval()

        total_z_loss = 0.0
        total_r_loss = 0.0
        bs = self.config.get("batch_size", 32)

        with torch.no_grad():
            for _ in range(n_batches):
                batch = replay_buffer.sample(bs)
                if "latent" not in batch:
                    z_dict = self._encode(batch["state"])
                    batch  = {**batch, **z_dict}
                outputs   = wm(batch)
                loss_dict = wm.loss(batch, outputs)
                total_z_loss += loss_dict.get("z_loss", torch.tensor(0.)).item()
                total_r_loss += loss_dict.get("r_loss", torch.tensor(0.)).item()

        wm.train()
        if enc: enc.train()
        return {
            "val_z_loss": total_z_loss / n_batches,
            "val_r_loss": total_r_loss / n_batches,
        }
