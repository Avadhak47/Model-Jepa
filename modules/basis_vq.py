import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisVQ(nn.Module):
    """
    Phoenix v6 BasisVQ — VQ-VAE with EMA Codebook Updates.

    Codebook is initialized from NMF basis (good geometric starting point),
    then updated each training step via Exponential Moving Average (EMA).
    This lets atoms specialize to encode BOTH geometry and color over time.

    Standard VQ-VAE (van den Oord et al. 2017) approach:
      - No codebook gradient needed (EMA handles the update)
      - Only commitment loss needed for the encoder
      - Straight-through estimator for decoder gradient flow
    """
    def __init__(self, basis_path='arc_data/arc_basis_nmf_1024.pt',
                 d_model=256, n_basis=1024, decay=0.99, epsilon=1e-5):
        super().__init__()
        data  = torch.load(basis_path, weights_only=False)
        basis = data['basis']   # [1024, 2700]

        self.num_codes = basis.shape[0]       # 1024
        self.basis_dim = basis.shape[1]       # 2700
        self.d_model   = d_model
        self.decay     = decay                # EMA decay rate (0.99 = slow, stable update)
        self.epsilon   = epsilon              # Laplace smoothing

        # Learnable projection: encoder slot (d_model) → codebook space (basis_dim)
        self.slot_proj = nn.Linear(d_model, self.basis_dim)

        # ── Codebook ──
        # embed: the actual atom vectors (updated via EMA, not gradient)
        # cluster_size: EMA count of how many encoder outputs map to each atom
        # embed_avg: EMA sum of encoder outputs per atom (used to compute new embed)
        self.register_buffer('embed',        basis.clone())        # [1024, 2700]
        self.register_buffer('cluster_size', torch.ones(self.num_codes))
        self.register_buffer('embed_avg',    basis.clone())        # [1024, 2700]

        # Usage tracking
        self.register_buffer('usage_counts', torch.zeros(self.num_codes))

    @torch.no_grad()
    def _ema_update(self, z_flat, indices_flat):
        """
        Update codebook atoms via EMA given the encoder outputs of this batch.
        z_flat:       [B*K, 2700]  — projected encoder slots
        indices_flat: [B*K]        — which atom each slot mapped to
        """
        # One-hot encoding of assignments: [B*K, 1024]
        one_hot = F.one_hot(indices_flat, num_classes=self.num_codes).float()

        # Count how many encoder outputs mapped to each atom this batch
        batch_cluster_size = one_hot.sum(dim=0)                        # [1024]
        # Sum of encoder outputs per atom
        batch_embed_sum = one_hot.T @ z_flat                           # [1024, 2700]

        # EMA update
        self.cluster_size = (
            self.decay * self.cluster_size + (1 - self.decay) * batch_cluster_size
        )
        self.embed_avg = (
            self.decay * self.embed_avg + (1 - self.decay) * batch_embed_sum
        )

        # Laplace-smoothed atom = avg of encoder outputs assigned to it
        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.epsilon)
            / (n + self.num_codes * self.epsilon)
            * n
        )
        self.embed = self.embed_avg / smoothed.unsqueeze(1)

    def forward(self, slot_features):
        """
        slot_features: [B, K, d_model]  — raw encoder slot features

        Returns:
            q_st:    [B, K, 2700]  — straight-through for decoder
            indices: [B, K]        — atom IDs (the symbolic tokens)
            vq_loss: scalar        — commitment loss (encoder toward atom)
            entropy: scalar        — diversity bonus
        """
        b, k, _ = slot_features.shape

        # Project encoder slots into codebook space
        z_e   = self.slot_proj(slot_features)      # [B, K, 2700]
        z_flat = z_e.view(b * k, self.basis_dim)   # [B*K, 2700]

        # ── Nearest-neighbour lookup ──
        # ||z - e||^2 = ||z||^2 - 2*z·e + ||e||^2
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * (z_flat @ self.embed.T)
            + self.embed.pow(2).sum(dim=1, keepdim=True).T
        )                                           # [B*K, 1024]
        indices_flat = dist.argmin(dim=-1)          # [B*K]
        indices      = indices_flat.view(b, k)      # [B, K]

        # ── EMA codebook update (training only, no gradient needed) ──
        if self.training:
            self._ema_update(z_flat.detach(), indices_flat.detach())

        # ── Selected atom vectors (from updated codebook) ──
        e_i = self.embed[indices_flat].view(b, k, self.basis_dim)   # [B, K, 2700]

        # ── Straight-Through Estimator ──
        # Forward:  passes e_i to decoder (clean codebook vector)
        # Backward: gradient flows through z_e (encoder retains color info)
        q_st = z_e + (e_i - z_e).detach()          # [B, K, 2700]

        # ── Commitment loss (encoder → atom, no atom gradient needed with EMA) ──
        vq_loss = F.mse_loss(z_e, e_i.detach())

        # ── Entropy / diversity bonus ──
        logits    = -dist.view(b, k, self.num_codes)
        avg_probs = F.softmax(logits, dim=-1).view(-1, self.num_codes).mean(dim=0)
        entropy   = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        # ── Usage tracking ──
        if self.training:
            with torch.no_grad():
                usage = torch.zeros(self.num_codes, device=slot_features.device)
                usage.scatter_add_(0, indices_flat,
                                   torch.ones(b * k, device=slot_features.device))
                self.usage_counts += usage

        return q_st, indices, vq_loss, entropy

    @torch.no_grad()
    def revive_dead_atoms(self):
        """
        Reset dead atoms (never selected) by cloning a high-usage donor atom
        + small noise. Prevents codebook collapse over long training.
        """
        dead = self.usage_counts < 1
        n_dead = dead.sum().item()
        if n_dead == 0:
            self.usage_counts.zero_()
            return 0

        alive = (~dead).nonzero(as_tuple=False).squeeze(1)
        if len(alive) == 0:
            self.usage_counts.zero_()
            return n_dead

        # Clone donors + tiny noise so dead atoms get a fresh start
        donors = alive[torch.randint(len(alive), (n_dead,))]
        self.embed[dead]     = self.embed[donors] + 0.01 * torch.randn_like(self.embed[donors])
        self.embed_avg[dead] = self.embed_avg[donors]
        self.cluster_size[dead] = 1.0

        self.usage_counts.zero_()
        return n_dead
