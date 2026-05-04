import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisVQ(nn.Module):
    """
    Phoenix v6 Soft-Basis Module (No Collapse).
    
    Instead of hard VQ (which collapses to Act=1), this uses a Top-K
    interpolation over the NMF dictionary. Every slot is represented
    as a sparse combination of NMF atoms.
    """
    def __init__(self, basis_path='arc_data/arc_basis_nmf_1024.pt',
                 d_model=256, n_basis=1024, k_top=8):
        super().__init__()
        data  = torch.load(basis_path, weights_only=False)
        basis = data['basis']   # [1024, 2250]

        self.num_codes = basis.shape[0]
        self.basis_dim = basis.shape[1]       
        self.d_model   = d_model
        self.k_top     = k_top  # Interpolate over top-8 atoms per slot

        # Project slot features into the NMF space
        self.slot_proj = nn.Linear(d_model, self.basis_dim)

        # Static Codebook (Learned via NMF, kept fixed to preserve geometry)
        self.register_buffer('embed', basis.clone()) # [1024, 2250]

        # Usage tracking
        self.register_buffer('usage_counts', torch.zeros(self.num_codes))

    def forward(self, slot_features):
        """
        slot_features: [B, K, d_model]
        """
        b, k, _ = slot_features.shape

        # 1. Project to NMF space
        z_e = self.slot_proj(slot_features)      # [B, K, 2250]
        z_flat = z_e.view(b * k, self.basis_dim) # [B*K, 2250]

        # 2. Similarity search (Negative L2 Distance)
        # We use cosine similarity or negative L2. Negative L2 is better for NMF.
        dist = torch.cdist(z_flat, self.embed)   # [B*K, 1024]
        
        # 3. Top-K Interpolation
        # Instead of 1, we use K=8 atoms to represent each slot
        values, indices_flat = dist.topk(self.k_top, largest=False) # [B*K, 8]
        
        # Softmax over the Top-K distances to get weights
        # We use a temperature of 0.1 to keep it relatively sparse
        weights = F.softmax(-values / 0.1, dim=-1) # [B*K, 8]
        
        # 4. Reconstruct slot from atoms
        # [B*K, 8, 1] * [B*K, 8, 2250] -> [B*K, 2250]
        selected_atoms = self.embed[indices_flat] # [B*K, 8, 2250]
        q_flat = (selected_atoms * weights.unsqueeze(-1)).sum(dim=1)
        q_st = q_flat.view(b, k, self.basis_dim)
        
        # 5. Losses
        # VQ Loss: Encoder moves toward its reconstruction
        vq_loss = F.mse_loss(z_e, q_st.detach())
        
        # Diversity: We want the average usage across the batch to be uniform
        # We use the softmax over all atoms for the entropy calculation
        all_weights = torch.zeros(b * k, self.num_codes, device=slot_features.device)
        all_weights.scatter_(1, indices_flat, weights)
        
        avg_usage = all_weights.mean(dim=0)
        entropy = -torch.sum(avg_usage * torch.log(avg_usage + 1e-8))

        # 6. Usage Tracking
        if self.training:
            with torch.no_grad():
                usage = torch.zeros(self.num_codes, device=slot_features.device)
                usage.scatter_add_(0, indices_flat.view(-1), torch.ones_like(indices_flat).view(-1).float())
                self.usage_counts += usage

        # For symbolic reasoning, we return the #1 most dominant index
        indices = indices_flat[:, 0].view(b, k)
        
        return q_st, indices, vq_loss, entropy

    def revive_dead_atoms(self):
        """Fixed dictionary doesn't need revival, just reset counts."""
        n_dead = (self.usage_counts < 1).sum().item()
        self.usage_counts.zero_()
        return n_dead
