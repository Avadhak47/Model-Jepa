import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisVQ(nn.Module):
    """
    A VQ module that uses a FIXED 1024-component NMF basis as its codebook.
    Instead of learning embeddings, it learns to map inputs to these 
    pre-discovered 'Atoms of ARC'.
    """
    def __init__(self, basis_path='arc_data/arc_basis_nmf_1024.pt', snapping_gain=30.0):
        super().__init__()
        # Load the 1024-component NMF basis
        data = torch.load(basis_path, weights_only=False)
        # basis: [1024, 2700]
        self.register_buffer('basis_vectors', data['basis']) 
        self.num_codes = self.basis_vectors.shape[0]
        self.basis_dim = self.basis_vectors.shape[1]
        
        self.snapping_gain = snapping_gain
        
        # Track usage to audit the 'Active Alphabet'
        self.register_buffer('usage_counts', torch.zeros(self.num_codes))

    def forward(self, latent_coeffs):
        """
        latent_coeffs: [B, K, 1024] - Encoder's prediction of basis weights or logits
        """
        b, k, c = latent_coeffs.shape
        
        # 1. SOFTMAX SNAPPING
        # We treat the encoder output as logits over the 1024 basis atoms
        probs = F.softmax(latent_coeffs * self.snapping_gain, dim=-1)
        
        if self.training:
            # Gumbel-Softmax for differentiable discrete selection
            indices = torch.argmax(probs, dim=-1)
            # Straight-through
            one_hot = F.one_hot(indices, num_classes=self.num_codes).float()
            one_hot = probs + (one_hot - probs).detach()
        else:
            indices = torch.argmax(probs, dim=-1)
            one_hot = F.one_hot(indices, num_classes=self.num_codes).float()
            
        # 2. ALGEBRAIC RECONSTRUCTION
        # Project the selected atoms back to the 2700-dim manifold
        # [B, K, 1024] @ [1024, 2700] -> [B, K, 2700]
        quantized_manifold = torch.matmul(one_hot, self.basis_vectors)
        
        # 3. AUDIT
        if self.training:
            self.usage_counts[indices.view(-1)] += 1
            
        return quantized_manifold, indices
