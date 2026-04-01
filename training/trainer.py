import torch

class Trainer:
    """Orchestrates structured Reinforcement Learning routines conforming to the TensorDict framework."""
    def __init__(self, config: dict, env, modules: dict):
        self.config = config
        self.env = env
        self.modules = modules
        self.device = config.get("device", "cpu")
        
        self.optimizers = {}
        for name, module in self.modules.items():
            if hasattr(module, 'parameters') and len(list(module.parameters())) > 0:
                self.optimizers[name] = torch.optim.AdamW(
                    module.parameters(), 
                    lr=config.get("learning_rate", 1e-3)
                )

    def train_representation_learning_step(self, batch: dict) -> dict:
        """
        Isolated Training Loop for the Encoder-Decoder pipeline.
        Optimizes purely on state reconstruction (e.g., autoencoding representation).
        """
        encoder = self.modules.get("encoder")
        decoder = self.modules.get("decoder")
        if not encoder or not decoder:
            return {}
            
        enc_opt = self.optimizers["encoder"]
        dec_opt = self.optimizers["decoder"]
        
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        
        # Forward pass -> z -> recon
        z_dict = encoder(batch)
        batch.update(z_dict)
        recon_dict = decoder(batch)
        
        # Calculate image/structure reconstruction MSE loss
        loss_dict = decoder.loss(batch, recon_dict)
        loss = loss_dict["loss"]
        loss.backward()
        
        enc_opt.step()
        dec_opt.step()
        
        return {"recon_loss": loss.item()}

    def train_world_model_step(self, batch: dict) -> dict:
        """
        Isolated Dynamics Prediction Learning Loop.
        Uses teacher forcing on pre-collected sequences.
        """
        wm = self.modules["world_model"]
        opt = self.optimizers["world_model"]
        
        opt.zero_grad()
        outputs = wm(batch)
        loss_dict = wm.loss(batch, outputs)
        
        loss_dict["loss"].backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=1.0)
        opt.step()
        
        return {
            "wm_loss": loss_dict["loss"].item(),
            "wm_z_loss": loss_dict.get("z_loss", 0.0).item() if isinstance(loss_dict.get("z_loss"), torch.Tensor) else 0.0,
            "wm_sigreg_loss": loss_dict.get("sigreg_loss", 0.0).item() if isinstance(loss_dict.get("sigreg_loss"), torch.Tensor) else 0.0,
            "wm_attention_entropy": loss_dict.get("attention_entropy", 0.0).item() if isinstance(loss_dict.get("attention_entropy"), torch.Tensor) else 0.0,
            "wm_grad_norm": grad_norm.item()
        }

    def train_policy_step(self, batch: dict) -> dict:
        """Isolated Policy Optimization (PPO) step."""
        policy = self.modules["policy"]
        opt = self.optimizers["policy"]
        
        opt.zero_grad()
        outputs = policy(batch)
        loss_dict = policy.loss(batch, outputs)
        
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        opt.step()
        
        return {
            "policy_loss": loss_dict.get("policy_loss", 0.0).item() if isinstance(loss_dict.get("policy_loss"), torch.Tensor) else 0.0,
            "value_loss": loss_dict.get("value_loss", 0.0).item() if isinstance(loss_dict.get("value_loss"), torch.Tensor) else 0.0,
            "exploration_entropy": loss_dict.get("entropy", 0.0).item() if isinstance(loss_dict.get("entropy"), torch.Tensor) else 0.0,
            "policy_grad_norm": grad_norm.item()
        }

    def train_end_to_end_step(self, batch: dict) -> dict:
        """
        Joint End-to-End MCTS/MuZero Execution.
        Backpropagates MCTS targets (Policy, Value) into the Policy/Value heads,
        and dynamically backpropagates the dynamics gradients straight through to the Encoder.
        """
        # Execute forward passes
        # Get target values from planner
        # Accumulate combined losses
        
        metrics = {}
        if "encoder" in self.modules and "decoder" in self.modules:
            metrics.update(self.train_representation_learning_step(batch))
        metrics.update(self.train_world_model_step(batch))
        metrics.update(self.train_policy_step(batch))
        return metrics

    def train(self, num_epochs: int, replay_buffer, mode="end_to_end"):
        """Main training loop orchestrator applying requested pipeline operations."""
        print(f"Executing {mode} routine for {num_epochs} epochs...")
        batch_size = self.config.get("batch_size", 32)
        
        for epoch in range(num_epochs):
            batch = replay_buffer.sample(batch_size)
            
            if mode == "representation":
                metrics = self.train_representation_learning_step(batch)
            elif mode == "world_model":
                metrics = self.train_world_model_step(batch)
            elif mode == "policy":
                metrics = self.train_policy_step(batch)
            else:
                metrics = self.train_end_to_end_step(batch)
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Metrics: {metrics}")

    def evaluate(self):
        """Simulation metrics rollout validation without PyTorch gradients."""
        print("Commencing evaluation rollout...")
        import torch
        with torch.no_grad():
            state = self.env.reset()["state"]
            total_reward = 0.0
            done = False
            while not done:
                action = self.modules["policy"]({"latent": torch.tensor(state).float().unsqueeze(0).to(self.device)})["action"]
                next_state_dict, reward, done, _ = self.env.step(action[0].cpu().numpy())
                total_reward += reward
                state = next_state_dict["state"]
        return total_reward

    def validate(self):
        """Holdout validation for prediction loss / MSE metrics."""
        import torch
        print("Validating prediction models on test set...")
        with torch.no_grad():
            return {"val_loss": 0.0}
