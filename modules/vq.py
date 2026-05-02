import torch
import torch.nn as nn
import torch.nn.functional as F

class NMFInformedVQ(nn.Module):
    """
    A VQ module where the latent space is the 200-dimensional NMF coefficient space.
    The codebook is initialized with the fundamental NMF basis parts.
    """
    def __init__(self, num_codes=1024, basis_dim=200, commitment_cost=0.25, snapping_gain=30.0):
        super().__init__()
        self.num_codes = num_codes
        self.basis_dim = basis_dim
        self.commitment_cost = commitment_cost
        self.snapping_gain = snapping_gain
        
        # The codebook entries are now 200-dimensional NMF coefficient recipes
        self.embedding = nn.Embedding(num_codes, basis_dim)
        
        # Initialize first 200 codes as "Pure Atoms" (One-hot recipes)
        # This ensures the model starts with the fundamental alphabet.
        with torch.no_grad():
            self.embedding.weight.fill_(0)
            eye = torch.eye(basis_dim)
            self.embedding.weight[:basis_dim] = eye
            # Fill the rest with small random combinations
            if num_codes > basis_dim:
                self.embedding.weight[basis_dim:].uniform_(0, 0.05)
                
        self.register_buffer('active_codes', torch.tensor(basis_dim, dtype=torch.long))

    def forward(self, inputs):
        # inputs: [B, K, 200]
        b, k, d = inputs.shape
        flat_input = inputs.reshape(-1, d)
        
        # L2 Normalize for Cosine Snapping
        flat_input_norm = F.normalize(flat_input, dim=-1)
        weights_norm = F.normalize(self.embedding.weight[:self.active_codes.item()], dim=-1)
        
        # Similarity and Snapping
        sim = torch.matmul(flat_input_norm, weights_norm.t())
        logits = sim * self.snapping_gain
        indices = torch.argmax(logits, dim=-1)
        
        quantized = self.embedding(indices)
        
        # Straight-through
        quantized = inputs + (quantized.view(b, k, d) - inputs).detach()
        
        vq_loss = F.mse_loss(quantized.detach(), inputs) * self.commitment_cost
        
        return quantized, vq_loss, indices.view(b, k)
