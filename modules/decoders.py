import torch
import torch.nn as nn
from core.base_module import BaseTrainableModule

class TransformerDecoder(BaseTrainableModule):
    def __init__(self, config):
        super().__init__(config)
        
        self.latent_dim = config.get('latent_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.vocab_size = config.get('vocab_size', 10)  # ARC colors 0-9
        
        # Project from latent bottleneck back to sequence length for grid
        # 30x30 = 900 spatial positions
        self.seq_len = 30 * 30
        
        self.latent_to_seq = nn.Linear(self.latent_dim, self.seq_len * self.hidden_dim)
        
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=config.get('nhead', 4), 
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=2, enable_nested_tensor=False)
        
        # Predict the exact color category (0-9) for each pixel
        self.head = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs['latent']  # [B, latent_dim]
        
        if z.dim() == 3 and z.size(1) == 1:
            z = z.squeeze(1)  # Remove sequence dim if present
            
        bsz = z.size(0)
        
        # [B, seq_len * hidden_dim] -> [B, seq_len, hidden_dim]
        x = self.latent_to_seq(z)
        x = x.view(bsz, self.seq_len, self.hidden_dim)
        
        # Transform
        x = self.transformer(x)
        
        # [B, seq_len, vocab_size]
        logits = self.head(x)
        
        # Reshape to 30x30 grid of logits
        # [B, vocab_size, 30, 30]
        logits = logits.transpose(1, 2).view(bsz, self.vocab_size, 30, 30)
        
        return {'reconstructed_logits': logits}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        logits = outputs['reconstructed_logits']
        # Original state is [B, 1, 30, 30] integers 0-9
        target = inputs['state'].squeeze(1).long()
        
        # Standard Cross-Entropy for autoencoding
        ce_loss = nn.functional.cross_entropy(logits, target, reduction='none')
        
        # --- ARC FOCAL LOSS INJECTION ---
        # If 'focal_gamma' > 0, we heavily penalize mispredicted pixels (tiny dots)
        gamma = self.config.get('focal_gamma', 2.0)
        if gamma > 0:
            pt = torch.exp(-ce_loss) # Recover the probability of the true class
            focal_loss_matrix = ((1 - pt) ** gamma) * ce_loss
            recon_loss = focal_loss_matrix.mean()
        else:
            recon_loss = ce_loss.mean()
        # --- RiJEPA Energy-Based Constraints (EBC) ---
        # 1. Color Invariance (Pixel distribution should roughly match)
        # We enforce that the predicted color histogram matches the target histogram
        pred_probs = torch.softmax(logits, dim=1)  # [B, 10, 30, 30]
        pred_hist = pred_probs.mean(dim=(2, 3))    # [B, 10]
        
        target_one_hot = nn.functional.one_hot(target, num_classes=self.vocab_size).float()
        target_hist = target_one_hot.mean(dim=(1, 2)) # [B, 10]
        
        # Use KL-Div instead of MSE to properly penalize rare colors
        color_ebc_loss = nn.functional.kl_div(
            torch.log(pred_hist + 1e-8), 
            target_hist, 
            reduction='batchmean'
        )
        
        total_loss = recon_loss + (0.5 * color_ebc_loss)
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "color_ebc_loss": color_ebc_loss.detach()
        }

class PatchDecoder(TransformerDecoder):
    """
    SLATE-style Patch-Based Decoder.
    Takes a sequence of quantized patch tokens [B, N, D] and upsamples them
    back to the raw pixel space [B, Vocab, H, W].
    """
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.get('patch_size', 2)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=2, enable_nested_tensor=False)

        self.pixel_generator = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.vocab_size, kernel_size=3, padding=1)
        )
        self.to(self.device)

    @property
    def num_patches_per_grid(self) -> int:
        """Number of patch tokens expected by this decoder (e.g. 225 for patch_size=2, 25 for patch_size=6)."""
        grid_size = self.config.get('grid_size', 30)
        side = grid_size // self.patch_size
        return side * side

    def forward(self, inputs: dict) -> dict:
        z = inputs['latent']  # [B, N, D]
        B, N, D = z.shape
        
        # Compute dynamic spatial extent of the patches. For 30x30 with size 2, N=225, H=15
        num_patches_side = int(N ** 0.5)
        
        x = self.transformer(z) # Contextualize patches
        
        # Reshape to 2D patch grid [B, D, H', W']
        x = x.transpose(1, 2).view(B, D, num_patches_side, num_patches_side)
        
        # Deconvolve back directly into pixel grid
        logits = self.pixel_generator(x) # [B, 10, 30, 30]
        
        return {'reconstructed_logits': logits}

    def loss(self, inputs: dict, outputs: dict, w_fg=50.0, gamma=2.0) -> dict:
        """
        Foreground-Weighted Focal Loss implementation to prevent Background Collapse (Lazy Minima).
        """
        state = inputs['state'].long() # [B, 1, 30, 30]
        state = state.squeeze(1) # [B, 30, 30]
        
        recon_logits = outputs['reconstructed_logits'] # [B, 10, 30, 30]
        
        # 1. Base Cross Entropy
        ce_loss = torch.nn.functional.cross_entropy(recon_logits, state, reduction='none')
        
        # 2. Foreground Weighting Mask (Target > 0 gets w_fg weight)
        mask_fg = (state > 0).float()
        weight_matrix = 1.0 + (mask_fg * (w_fg - 1.0))
        
        # 3. Focal Modulation
        pt = torch.exp(-ce_loss)
        recon_loss = (((1 - pt) ** gamma) * ce_loss * weight_matrix).mean()
        
        return {
            "loss": recon_loss,
            "recon_loss": recon_loss.detach()
        }
