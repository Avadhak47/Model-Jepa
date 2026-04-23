import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_2d_rope(grid_size, head_dim, device="cpu"):
    """
    Generates 2D Rotary Positional Embeddings.
    """
    half_dim = head_dim // 2
    y = torch.arange(grid_size, device=device).float()
    x = torch.arange(grid_size, device=device).float()
    
    # Frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    
    pos_x = torch.einsum('i,j->ij', x, inv_freq)
    pos_y = torch.einsum('i,j->ij', y, inv_freq)
    
    emb_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)
    emb_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
    
    # Expand to grid [grid_size, grid_size, half_dim]
    emb_x = emb_x.unsqueeze(0).expand(grid_size, grid_size, -1)
    emb_y = emb_y.unsqueeze(1).expand(grid_size, grid_size, -1)
    
    emb = torch.cat([emb_y, emb_x], dim=-1) # [grid_size, grid_size, head_dim]
    return emb.view(grid_size * grid_size, head_dim) # [N, head_dim]

def apply_rope(x, cos, sin):
    # x: [B, N, num_heads, head_dim]
    # Rotate half
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # Note: RoPE is typically [-x2, x1]
    rotated_x = torch.cat([-x2, x1], dim=-1)
    
    # Just standard multiplication (if cos/sin already interlaced)
    return (x * cos) + (rotated_x * sin)

class RoPESelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, grid_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_size = grid_size
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        rope_emb = generate_2d_rope(grid_size, self.head_dim) # [N, head_dim]
        # Interlace sin/cos for easy apply
        self.register_buffer("cos_emb", rope_emb.cos().view(1, grid_size*grid_size, 1, self.head_dim))
        self.register_buffer("sin_emb", rope_emb.sin().view(1, grid_size*grid_size, 1, self.head_dim))
        
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4) # [3, B, N, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE directly to Q and K patches
        q = apply_rope(q, self.cos_emb, self.sin_emb).transpose(1, 2) # [B, num_heads, N, head_dim]
        k = apply_rope(k, self.cos_emb, self.sin_emb).transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)

class SemanticSlotEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.embed_dim = config['hidden_dim']
        self.num_slots = config['num_slots']
        self.slot_iters = config['slot_iters']
        self.temperature = config['slot_temperature']
        self.patch_size = config['patch_size'] # 2
        self.grid_size = config['grid_size'] // config['patch_size'] # 30 / 2 = 15
        
        self.vocab_size = config.get('vocab_size', 10)
        
        # Sobel Filters (Fixed)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        
        # In Channels: 10 (OneHot) + 2 (Sobel X/Y) = 12
        self.patch_embed = nn.Conv2d(12, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        self.patch_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        # Relational Patch Pre-Contextualization!
        self.patch_rope_attn = RoPESelfAttention(self.embed_dim, num_heads=4, grid_size=self.grid_size)
        
        self.norm_inputs = nn.LayerNorm(self.embed_dim)
        self.norm_slots = nn.LayerNorm(self.embed_dim)
        self.norm_mlp = nn.LayerNorm(self.embed_dim)
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.gru = nn.GRUCell(self.embed_dim, self.embed_dim)
        
        self.slot_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        self.to(self.device)

    def inject_semantic_priors(self, fps_slots):
        """
        Called between Phase 0 and Phase 1 to inject sampled Codebook modes.
        fps_slots: [1, num_slots, embed_dim]

        Design: keeps the FPS vector frozen as a semantic anchor, but registers
        a learnable `prior_delta` (initialised to zero) on top of it.
        - At epoch 0: delta=0, so initialisation is identical to pure FPS.
        - With gradient flow: delta sculpts each slot's starting position toward
          the clusters its patches actually live in, without disrupting the
          codebook-derived semantic meaning of the base prior.
        """
        self.semantic_priors = nn.Parameter(fps_slots.to(self.device), requires_grad=False)
        # Learnable offset — zero-initialised so it doesn't change anything on day 1
        self.prior_delta = nn.Parameter(
            torch.zeros_like(fps_slots, device=self.device), requires_grad=True
        )

    def forward(self, inputs, temperature: float = None):
        img = inputs["state"].float().to(self.device) # [B, 1, H, W]
        B, _, H, W = img.shape

        # Use externally-annealed temperature if provided, else use self.temperature
        temp = temperature if temperature is not None else self.temperature

        # 1. Sobel Edges (pad to keep same H,W)
        padded_img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        sx = F.conv2d(padded_img, self.sobel_x)
        sy = F.conv2d(padded_img, self.sobel_y)

        # 2. One-Hot Colors
        onehot = F.one_hot(img.long().squeeze(1), num_classes=self.vocab_size) # [B, H, W, 10]
        onehot = onehot.permute(0, 3, 1, 2).float() # [B, 10, H, W]

        # Composite! [B, 12, H, W]
        comp_img = torch.cat([onehot, sx, sy], dim=1)

        # 3. Patching
        x = self.patch_embed(comp_img) # [B, embed_dim, 15, 15]
        x = x.flatten(2).transpose(1, 2) # [B, 225, embed_dim]

        # Patch Self-Attention with 2D RoPE
        x = self.patch_mlp(x)
        x = x + self.patch_rope_attn(x)
        x = self.norm_inputs(x)

        k, v = self.k_proj(x), self.v_proj(x)

        # Semantic Slot Initialization:
        # base (frozen FPS anchor) + delta (learnable sculpting offset, starts at 0)
        slots = (self.semantic_priors + self.prior_delta).expand(B, -1, -1)

        for _ in range(self.slot_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.q_proj(slots_norm)

            attn_logits = torch.bmm(k, q.transpose(1, 2)) * (self.embed_dim ** -0.5)
            # Divide by temperature: high temp → soft (exploratory), low → sharp (specialised)
            attn = F.softmax(attn_logits / temp, dim=-1)
            attn_norm = attn + 1e-8
            attn_norm = attn_norm / attn_norm.sum(dim=1, keepdim=True)

            updates = torch.bmm(attn_norm.transpose(1, 2), v)
            slots = self.gru(
                updates.reshape(-1, self.embed_dim),
                slots_prev.reshape(-1, self.embed_dim)
            ).reshape(B, self.num_slots, self.embed_dim)
            slots = slots + self.slot_mlp(self.norm_mlp(slots))

        masks = attn.transpose(1, 2).view(B, self.num_slots, self.grid_size, self.grid_size).detach()
        return {"latent": slots, "masks": masks}
