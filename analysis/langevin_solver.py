import torch
import torch.nn as nn
import torch.nn.functional as F

class LangevinGridSculptor:
    """
    Optimizes a randomly initialized grid so its latent representation 
    matches a target latent, effectively replacing pixel decoders.
    """
    def __init__(self, target_encoder: nn.Module, device: str):
        self.encoder = target_encoder
        self.encoder.eval() 
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.device = device
        
    def sculpt(self, target_latent: torch.Tensor, init_shape: tuple, steps: int = 1500, lr: float = 0.1, noise_scale: float = 0.05):
        """
        Args:
            target_latent: [B, num_slots, latent_dim] the solved solution representations.
            init_shape: (B, 1, H, W) of the grid to hallucinate (e.g., 30x30 for ARC).
            steps: Optimization iterations.
            lr: Learning rate for pixel intensity shifts.
            noise_scale: Langevin noise injection to break local geometric minima.
            
        Returns:
            The final snapped discrete grid.
        """
        # Detach the target latent so PyTorch doesn't try to backpropagate 
        # into the encoder that originally generated it!
        target_latent = target_latent.detach()
        
        # Initialize continuous grid canvas
        grid_continuous = nn.Parameter(torch.rand(init_shape, device=self.device) * 9.0)
        
        optimizer = torch.optim.AdamW([grid_continuous], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Predict latent state of our current canvas
            encoded_dict = self.encoder({'state': grid_continuous})
            z_cand = encoded_dict['latent']
            
            loss_mse = F.mse_loss(z_cand, target_latent)
            
            # Total Variation (TV) Loss to penalize high-frequency static
            tv_h = torch.mean(torch.abs(grid_continuous[:, :, 1:, :] - grid_continuous[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(grid_continuous[:, :, :, 1:] - grid_continuous[:, :, :, :-1]))
            loss_tv = tv_h + tv_w
            
            loss = loss_mse + 0.1 * loss_tv
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_([grid_continuous], 1.0)
            optimizer.step()
            
            # Langevin Noise Injection 
            decay = max(0.01, 1.0 - (i / steps))
            with torch.no_grad():
                grid_continuous.add_(torch.randn_like(grid_continuous) * noise_scale * decay)
                # Keep strictly inside the valid ARC color category range (0 to 9)
                grid_continuous.clamp_(0.0, 9.0)
                
        # Final Rounding / Discretization
        with torch.no_grad():
            final_discrete_grid = torch.round(grid_continuous)
            
        return final_discrete_grid.detach()
