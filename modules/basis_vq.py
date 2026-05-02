import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisVQ(nn.Module):
    """
    Phoenix v6 BasisVQ — Proper VQ-VAE Straight-Through.

    The key insight: the ENCODER'S slot features (not raw NMF atom vectors)
    are what get passed to the decoder. The NMF atoms define WHICH structure
    is selected, but color information stays in the encoder representation.

    Forward pass (training):
        q = z + (e_i - z).detach()     ← straight-through
        decoder sees q, which equals z in the forward pass (encoder features)
        but pulls gradients back through the atom selection

    Inference:
        Same formula, but z ≈ e_i because the encoder has learned to
        align its output with the correct atom.
    """
    def __init__(self, basis_path='arc_data/arc_basis_nmf_1024.pt',
                 d_model=256, n_basis=1024):
        super().__init__()
        data = torch.load(basis_path, weights_only=False)
        basis = data['basis']                           # [1024, 2700]

        self.register_buffer('basis_vectors', basis)    # frozen NMF atoms
        self.num_codes  = basis.shape[0]
        self.basis_dim  = basis.shape[1]                # 2700
        self.d_model    = d_model

        # Learnable projection: encoder slot (d_model) → NMF manifold space (2700)
        # This aligns the encoder representation with the NMF atoms for comparison
        self.slot_proj = nn.Linear(d_model, self.basis_dim)

        self.register_buffer('usage_counts', torch.zeros(self.num_codes))
        self.register_buffer('ema_usage',    torch.zeros(self.num_codes))

        # Commitment loss weight (encourages encoder to commit to selected atom)
        self.beta = 0.25

    def forward(self, slot_features):
        """
        slot_features: [B, K, d_model]  ← raw encoder slot features (contain color!)

        Returns:
            q_st:      [B, K, 2700]  ← straight-through quantized features for decoder
            indices:   [B, K]        ← selected atom IDs (the symbolic vocabulary)
            vq_loss:   scalar        ← commitment + embedding loss for training
            entropy:   scalar        ← codebook diversity bonus
        """
        b, k, _ = slot_features.shape

        # Project encoder slots into NMF manifold space for nearest-neighbour lookup
        z_e = self.slot_proj(slot_features)             # [B, K, 2700]

        # ── Nearest-neighbour lookup in NMF basis ──
        # Distances: ||z_e - e_i||^2  (broadcasting)
        z_flat = z_e.view(b * k, self.basis_dim)        # [B*K, 2700]
        # ||z||^2 - 2*z·e + ||e||^2
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.basis_vectors.T
            + self.basis_vectors.pow(2).sum(dim=1, keepdim=True).T
        )                                               # [B*K, 1024]
        indices = dist.argmin(dim=-1)                   # [B*K]
        indices = indices.view(b, k)                    # [B, K]

        # ── Selected atom vectors ──
        e_i = self.basis_vectors[indices.view(-1)].view(b, k, self.basis_dim)  # [B, K, 2700]

        # ── VQ-VAE Straight-Through Estimator ──
        # Forward: pass z_e (contains encoder's color info!) to decoder
        # Backward: gradients flow through z_e (not through frozen basis)
        q_st = z_e + (e_i - z_e).detach()              # [B, K, 2700]

        # ── VQ Loss ──
        # Commitment loss: encoder output stays close to selected atom
        commit_loss = F.mse_loss(z_e, e_i.detach())
        # Embedding loss: (skipped — basis is frozen, can't update atoms)
        vq_loss = self.beta * commit_loss

        # ── Diversity / Entropy bonus ──
        # Encourage uniform usage across the 1024 atoms
        # Use soft distances as "probabilities" for entropy calculation
        logits = -dist.view(b, k, self.num_codes)       # [B, K, 1024]
        avg_probs = F.softmax(logits, dim=-1).view(-1, self.num_codes).mean(dim=0)
        entropy   = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        # ── Usage tracking ──
        if self.training:
            with torch.no_grad():
                usage = torch.zeros(self.num_codes, device=slot_features.device)
                usage.scatter_add_(0, indices.view(-1),
                                   torch.ones(b * k, device=slot_features.device))
                self.usage_counts += usage
                self.ema_usage = 0.99 * self.ema_usage + 0.01 * usage

        return q_st, indices, vq_loss, entropy

    @torch.no_grad()
    def revive_dead_atoms(self):
        """Reset EMA tracking for dead atoms to give them a fresh chance."""
        dead_mask = self.ema_usage < 0.1
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            self.ema_usage[dead_mask] = 0.5
            self.usage_counts.zero_()
        return n_dead
