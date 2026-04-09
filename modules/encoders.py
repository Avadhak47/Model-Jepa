import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.interfaces import BaseEncoder

class MLPEncoder(BaseEncoder):
    """Simple continuous state encoder."""
    def __init__(self, config: dict):
        super().__init__(config)
        in_dim = config.get("input_dim", 64)
        latent_dim = config.get("latent_dim", 128)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        x = inputs["state"].to(self.device).float()
        x = x.view(x.shape[0], -1)  # flatten any spatial dims
        return {"latent": self.net(x)}

class CNNEncoder(BaseEncoder):
    """Extracts spatial priors from visual grid inputs."""
    def __init__(self, config: dict):
        super().__init__(config)
        in_channels = config.get("in_channels", 1)  # ARC grids are single-channel
        latent_dim = config.get("latent_dim", 128)
        # For small grids like ARC
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.projector = nn.Linear(128, latent_dim)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        img = inputs["state"].float().to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        z = self.projector(self.conv(img))
        return {"latent": z}

class TransformerEncoder(BaseEncoder):
    """Vision Transformer (ViT) backbone with CLS token projection designed to prevent representation collapse."""
    def __init__(self, config: dict):
        super().__init__(config)
        in_channels = config.get("in_channels", 1)
        patch_size = config.get("patch_size", 14)
        embed_dim = config.get("hidden_dim", 256)
        latent_dim = config.get("latent_dim", 128)
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Tiny ViT config: 12 layers, 3 heads
        # num_heads must divide embed_dim; derive the largest valid one from config
        raw_nhead = config.get("nhead", 4)
        nhead = max(1, raw_nhead) if embed_dim % raw_nhead == 0 else 1
        # Fall back to 1 if nothing works, then try common divisors
        for h in [raw_nhead, 4, 2, 1]:
            if embed_dim % h == 0:
                nhead = h
                break
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.get("encoder_layers", 4))
        
        # Crucial anti-collapse projection (LeWM Section 3.1)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim)
        )
        self.to(self.device)
        
    def forward(self, inputs: dict) -> dict:
        img = inputs["state"].float().to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        x = self.patch_embed(img) # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, embed_dim]
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x)
        cls_out = x[:, 0] # Extract [CLS]
        z = self.projector(cls_out)
        return {"latent": z}

class Decoder(BaseEncoder):
    """Inverts the latent space back to pixel space for autencoding metrics."""
    def __init__(self, config: dict):
        super().__init__(config)
        latent_dim = config.get("latent_dim", 128)
        out_dim = config.get("input_dim", 64) # flattened
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        recon = self.net(z)
        return {"reconstruction": recon}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        target = inputs["state"].to(self.device)
        target = target.view(target.shape[0], -1) 
        recon = outputs["reconstruction"]
        return {"loss": F.mse_loss(recon, target)}

class SlotTransformerEncoder(BaseEncoder):
    """
    Object-Centric Encoder using Slot Attention.
    Replaces global CLS reasoning with discrete structural extraction over K slots.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        in_channels = config.get("in_channels", 1)
        patch_size = config.get("patch_size", 14)
        self.embed_dim = config.get("hidden_dim", 256)
        self.latent_dim = config.get("latent_dim", 128)
        self.num_slots = config.get("num_slots", 16)
        self.slot_iters = config.get("slot_iters", 3)
        self.temperature = config.get("slot_temperature", 0.1) # Prevents fuzzy slot sharing
        
        # Grid -> Patch Embeddings
        self.patch_embed = nn.Conv2d(in_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings for flat patches. Assuming 224x224 input -> 16x16 patches = 256. ARC grids are smaller, we'll make it dynamic or large enough
        self.patch_pos_embed = nn.Parameter(torch.randn(1, 1024, self.embed_dim)) # Over-provisioned
        
        # ── Slot prior: shared Gaussian, initialised with glorot_uniform per paper ──
        # shape [1, 1, embed_dim] — shared across ALL slots (paper §3.1)
        self.slots_mu = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.slots_logsigma = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        nn.init.xavier_uniform_(self.slots_mu.data.view(1, self.embed_dim).unsqueeze(0))
        nn.init.xavier_uniform_(self.slots_logsigma.data.view(1, self.embed_dim).unsqueeze(0))

        # ── Slot Attention Projections (no bias on Q/K per paper) ──
        self.norm_inputs = nn.LayerNorm(self.embed_dim)
        self.norm_slots  = nn.LayerNorm(self.embed_dim)
        self.norm_mlp    = nn.LayerNorm(self.embed_dim)   # norm BEFORE slot MLP

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.gru = nn.GRUCell(self.embed_dim, self.embed_dim)

        # ── Slot MLP (CRITICAL — paper adds residual MLP after every GRU step) ──
        self.slot_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        # ── Relational Self-Attention between slots ──
        self.relational_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)

        # ── Projection to latent slot size ──
        self.projector = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.latent_dim)
        )
        self.to(self.device)
        
    def forward(self, inputs: dict) -> dict:
        img = inputs["state"].float().to(self.device)
        if img.dim() == 3: img = img.unsqueeze(0)
        
        x = self.patch_embed(img) # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, embed_dim]
        N = x.shape[1]
        
        # Add spatial position
        x = x + self.patch_pos_embed[:, :N, :]
        x = self.norm_inputs(x)
        
        # Keys and Values from grid patches
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Initialize slots stochastically to break symmetry
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)
        if self.training:
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = mu
        
        # ── Iterative Slot Attention (following paper §3.1 exactly) ──
        # attn shape: [B, N, num_slots]  (patches × slots)
        for it in range(self.slot_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)       # [B, S, D]
            q = self.q_proj(slots_norm)               # [B, S, D]

            # Attention logits: [B, N, S]
            scale = self.embed_dim ** -0.5
            attn_logits = torch.bmm(k, q.transpose(1, 2)) * scale  # [B, N, S]

            # Step 1: softmax over SLOTS (dim=-1) per paper — slots compete for each patch
            attn = F.softmax(attn_logits / self.temperature, dim=-1)   # [B, N, S]

            # Step 2: normalise over PATCHES so each slot's weights sum to 1
            # Add epsilon for stability (paper uses epsilon=1e-8)
            attn_norm = attn + 1e-8
            attn_norm = attn_norm / attn_norm.sum(dim=1, keepdim=True)  # [B, N, S]

            # Weighted mean: updates shape [B, S, D]
            updates = torch.bmm(attn_norm.transpose(1, 2), v)           # [B, S, D]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.embed_dim),
                slots_prev.reshape(-1, self.embed_dim)
            ).reshape(B, self.num_slots, self.embed_dim)

            # Residual slot MLP (CRITICAL — paper §3.1, missing in previous code)
            slots = slots + self.slot_mlp(self.norm_mlp(slots))
            
        # Relational Extraction (Slots talk to slots)
        slots_rel, _ = self.relational_attn(slots, slots, slots)
        slots = slots + slots_rel

        # Project to target latent dimensions
        z = self.projector(slots)  # [B, num_slots, latent_dim]

        # attn is [B, N, S] — reshape to spatial mask [B, S, H, W]
        masks = attn.transpose(1, 2).view(B, self.num_slots, H, W).detach()

        # Expose slot prior parameters for KL loss (Slot-VAE style)
        # Gradients flow through z, mu/logsigma are for the training loop's KL calculation
        slot_mu = self.slots_mu.expand(B, self.num_slots, -1)
        slot_logsigma = self.slots_logsigma.expand(B, self.num_slots, -1)

        return {
            "latent": z,
            "masks": masks,
            "slot_mu": slot_mu,
            "slot_logsigma": slot_logsigma
        }

    @torch.no_grad()
    def update_ema(self, student_model: nn.Module, momentum: float = 0.996):
        """Updates this Target Encoder's weights via Exponential Moving Average of the Student Encoder."""
        for param_t, param_s in zip(self.parameters(), student_model.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

class SoftPositionEmbed(nn.Module):
    """
    Positional encoding from the Slot Attention paper (§3.2).
    Builds a coordinate grid [x, y, 1-x, 1-y] and projects it through a
    learned linear layer, then adds to the input features.
    This gives the decoder spatial awareness — each pixel position gets a
    unique signal so the slot vector can produce different outputs at
    different locations.
    """
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.register_buffer("grid", self._build_grid(resolution))

    @staticmethod
    def _build_grid(resolution):
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1)                    # [H, W, 2]
        grid = torch.cat([grid, 1.0 - grid], dim=-1)       # [H, W, 4]
        return grid.unsqueeze(0)                            # [1, H, W, 4]

    def forward(self, x):
        # x: [B, C, H, W] (channels-first PyTorch convention)
        # grid: [1, H, W, 4]
        pos = self.dense(self.grid)             # [1, H, W, hidden_size]
        pos = pos.permute(0, 3, 1, 2)          # [1, hidden_size, H, W]
        return x + pos


class SlotDecoder(BaseEncoder):
    """
    Decodes K independent slots into a single grid via Alpha-Mask Compositing.
    Now includes SoftPositionEmbed (matching the paper) so the decoder can
    produce spatially-varying output per slot — forcing z to encode position,
    shape, and color (not just color).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = config.get("latent_dim", 128)
        self.vocab_size = config.get("vocab_size", 10)  # 10 ARC Colors
        self.grid_size = config.get("grid_size", 30)
        self.num_slots = config.get("num_slots", 16)

        # Positional encoding for decoder (paper §3.2 — the missing piece)
        self.decoder_pos = SoftPositionEmbed(self.latent_dim, (self.grid_size, self.grid_size))

        # Spatial Broadcast CNN (deeper to match paper's 6-layer decoder)
        self.spatial_broadcast = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )

        # Color Map (Logits) and Alpha Mask predictors
        # Paper outputs channels+1 from final conv; we keep separate heads for clarity
        self.color_predictor = nn.Conv2d(32, self.vocab_size, kernel_size=3, padding=1)
        self.alpha_predictor = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)  # [B, num_slots, latent_dim]
        B, num_slots, D = z.shape

        target_size = inputs["state"].shape[-2:] if "state" in inputs else (self.grid_size, self.grid_size)
        H, W = target_size

        # Spatial Broadcast: Tile each slot vector across an HxW grid
        z_flat = z.view(B * num_slots, D, 1, 1)
        z_tiled = z_flat.expand(-1, -1, H, W)  # [B*num_slots, D, H, W]

        # ── ADD POSITIONAL ENCODING (paper §3.2) ──
        # Each pixel now gets: slot_vector + pos(x, y)
        # This lets the decoder produce different output at different positions
        z_tiled = self.decoder_pos(z_tiled)

        # Decode components
        decoded_features = self.spatial_broadcast(z_tiled)

        colors = self.color_predictor(decoded_features)  # [B*num_slots, C, H, W]
        alphas = self.alpha_predictor(decoded_features)  # [B*num_slots, 1, H, W]

        # Reshape to slot dimension for competition
        C = colors.shape[1]
        colors = colors.view(B, num_slots, C, H, W)
        alphas = alphas.view(B, num_slots, 1, H, W)

        # Soft alpha competition across slots per pixel
        alphas_normalized = F.softmax(alphas, dim=1)  # [B, num_slots, 1, H, W]

        # Composite final image
        reconstruction = torch.sum(alphas_normalized * colors, dim=1)  # [B, C, H, W]
        
        return {
            "reconstruction": reconstruction,
            "alphas": alphas_normalized, # Saved for visualization plotting
            "colors": colors
        }

    def loss(self, inputs: dict, outputs: dict) -> dict:
        target = inputs["state"].to(self.device).long()
        if target.dim() == 4 and target.size(1) == 1: 
            target = target.squeeze(1) # [B, H, W]
        recon_logits = outputs["reconstruction"] # [B, 10, H, W]
        
        # Focal Loss across discrete color categories, replacing broken contiguous MSE
        ce_loss = F.cross_entropy(recon_logits, target, reduction='none')
        gamma = self.config.get('focal_gamma', 2.0)
        
        if gamma > 0:
            pt = torch.exp(-ce_loss)
            recon_loss = (((1 - pt) ** gamma) * ce_loss).mean()
        else:
            recon_loss = ce_loss.mean()
        
        # Hinge-Based Similarity Penalty (Soft Orthogonality)
        z = inputs["latent"] # [B, num_slots, latent_dim]
        B, num_slots, _ = z.shape
        
        z_norm = F.normalize(z, dim=-1)
        # Cosine similarity matrix [B, num_slots, num_slots]
        sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(num_slots, device=self.device).unsqueeze(0).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1.0)
        
        # Hinge penalty: we dynamically relax the competitive constraint
        # as accuracy increases. 
        # Extremely Strict at start (min_t), more relaxed at end (max_t).
        t = inputs.get("hinge_threshold", 0.95)
        hinge_loss = F.relu(sim_matrix - t).mean()
        
        # Slot Regularization: Prevent slot decay
        total_loss = recon_loss + 0.1 * hinge_loss
        return {
            "loss": total_loss, 
            "recon_loss": recon_loss.detach(),
            "hinge_loss": hinge_loss.detach(),
            "hinge_threshold": float(t)
        }
