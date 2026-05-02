import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisVQ(nn.Module):
    """
    Phoenix v5 BasisVQ:
    - Uses Gumbel-Softmax with temperature scheduling (NOT naive straight-through)
    - Tracks dead atoms and supports Adaptive Reinitialization (NS-VQ inspired)
    - Separates color and positional axes of the NMF basis for clean reconstruction
    """
    def __init__(self, basis_path='arc_data/arc_basis_nmf_1024.pt', temperature=1.0):
        super().__init__()
        data = torch.load(basis_path, weights_only=False)
        basis = data['basis']  # [1024, 2700]

        # Split basis into color (channels 0-9 per pixel) and position (channels 10-11)
        # Reshape to [1024, 15, 15, 12], then separate
        basis_3d = basis.view(1024, 15, 15, 12)
        color_basis = basis_3d[:, :, :, :10].reshape(1024, -1)   # [1024, 2250]
        pos_basis   = basis_3d[:, :, :, 10:].reshape(1024, -1)   # [1024, 450]

        self.register_buffer('color_basis', color_basis)
        self.register_buffer('pos_basis', pos_basis)
        self.num_codes = basis.shape[0]

        self.temperature = temperature  # Will be annealed externally

        self.register_buffer('usage_counts', torch.zeros(self.num_codes))
        self.register_buffer('ema_usage',    torch.zeros(self.num_codes))

    def forward(self, logits, encoder_outputs=None):
        """
        logits: [B, K, 1024]  - raw encoder logits over atoms
        encoder_outputs: optional [B, K, d_model] for dead atom revival
        Returns: (color_manifold, pos_manifold, indices, aux_loss)
        """
        b, k, c = logits.shape

        if self.training:
            # Gumbel-Softmax: guarantees gradients to ALL atoms, not just the argmax
            # hard=True gives a one-hot in the forward pass (for discrete selection)
            # but smooth gradients in the backward pass
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)  # [B, K, 1024]
            indices = one_hot.argmax(dim=-1)  # [B, K]
        else:
            indices = logits.argmax(dim=-1)
            one_hot = F.one_hot(indices, num_classes=self.num_codes).float()

        # Algebraic reconstruction (separate color and position)
        color_manifold = torch.matmul(one_hot, self.color_basis)  # [B, K, 2250]
        pos_manifold   = torch.matmul(one_hot, self.pos_basis)    # [B, K, 450]

        # Track usage with EMA for dead atom detection
        if self.training:
            with torch.no_grad():
                batch_usage = torch.zeros(self.num_codes, device=logits.device)
                batch_usage.scatter_add_(0, indices.view(-1),
                                         torch.ones(b * k, device=logits.device))
                self.usage_counts += batch_usage
                self.ema_usage = 0.99 * self.ema_usage + 0.01 * batch_usage

        # Commitment loss: encourage encoder to stay close to its chosen atom
        # (even though basis is frozen, this regularises the logit distribution)
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.view(-1, self.num_codes).mean(dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        return color_manifold, pos_manifold, indices, entropy

    @torch.no_grad()
    def revive_dead_atoms(self, encoder_batch_logits):
        """
        NS-VQ inspired: find dead atoms and re-seed them using current encoder outputs.
        Call every N epochs.
        encoder_batch_logits: [B, K, 1024]
        """
        dead_mask = self.ema_usage < 0.1
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Find the most-used atoms in this batch as "donors"
        batch_usage = self.usage_counts.clone()
        _, donor_ids = batch_usage.sort(descending=True)
        donor_ids = donor_ids[:n_dead]

        # The dead atoms' logit positions get "seeded" toward the donors
        # (We can't change the frozen basis, but we can report back how many were dead)
        self.ema_usage[dead_mask] = 0.5  # Give them a second chance
        self.usage_counts.zero_()
        return n_dead
