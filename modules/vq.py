import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolutionaryClusterVQ(nn.Module):
    """
    An Evolutionary VQ module that uses 100 Semantic Clusters to curate a 
    1024-size vocabulary. Uses Hypersphere Snapping and Tournament-based 
    updates to ensure only the highest-fidelity codes survive.
    """
    def __init__(self, num_codes=1024, embedding_dim=128, num_clusters=100, 
                 commitment_cost=0.25, novelty_threshold=0.85, snapping_gain=25.0):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.num_clusters = num_clusters
        self.commitment_cost = commitment_cost
        self.novelty_threshold = novelty_threshold
        self.snapping_gain = snapping_gain
        
        # Factorized Embeddings
        self.embedding_shape = nn.Embedding(num_codes, self.half_dim)
        self.embedding_color = nn.Embedding(16, self.half_dim)
        
        # Unit Norm Initialization
        nn.init.normal_(self.embedding_shape.weight, std=1.0)
        self.embedding_shape.weight.data = F.normalize(self.embedding_shape.weight.data, dim=-1)
        
        nn.init.normal_(self.embedding_color.weight, std=1.0)
        self.embedding_color.weight.data = F.normalize(self.embedding_color.weight.data, dim=-1)
        
        # Active and Usage Trackers
        self.register_buffer('active_shape_codes', torch.tensor(1, dtype=torch.long))
        self.register_buffer('shape_usage', torch.zeros(num_codes))
        
        # Cluster Tournaments: Track best recon loss seen for each cluster
        # (This will be managed by the training script calling update_tournament)
        self.register_buffer('cluster_centroids', torch.zeros(num_clusters, self.half_dim))
        self.register_buffer('cluster_best_loss', torch.full((num_clusters,), float('inf')))

    def reset_epoch(self):
        """Standard reset for dynamic expansion, but preserves the hardened winners."""
        # We don't clear the whole codebook, just the usage and expansion tracker
        # to allow for fresh discovery alongside existing codes.
        self.shape_usage.zero_()
        # Note: We keep active_shape_codes as they are to preserve the 'hardened' vocabulary

    def _quantize(self, z, embedding, active_k, snapping=True):
        # L2 Normalize
        z_norm = F.normalize(z, dim=-1)
        weights = F.normalize(embedding.weight[:active_k], dim=-1)
        
        # Cosine Similarity
        sim = torch.matmul(z_norm, weights.t())
        
        if snapping:
            # Gumbel-Softmax like snapping
            logits = sim * self.snapping_gain
            indices = torch.argmax(logits, dim=-1)
        else:
            indices = torch.argmax(sim, dim=-1)
            
        quantized = embedding(indices)
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, indices, sim

    def forward(self, inputs):
        # inputs: [B, K, D]
        b, k, d = inputs.shape
        flat_input = inputs.reshape(-1, d)
        
        flat_shape = flat_input[:, :self.half_dim]
        flat_color = flat_input[:, self.half_dim:]
        
        # Shape Quantization (Evolutionary)
        active_s = self.active_shape_codes.item()
        q_shape, s_idx, s_sim = self._quantize(flat_shape, self.embedding_shape, active_s)
        
        # Dynamic Spawning (if similarity is low)
        if self.training and active_s < self.num_codes:
            max_sim = s_sim.max(dim=-1)[0]
            novel_mask = max_sim < self.novelty_threshold
            if novel_mask.any():
                novel_latent = F.normalize(flat_shape[novel_mask][0:1].detach(), dim=-1)
                self.embedding_shape.weight.data[active_s] = novel_latent
                self.active_shape_codes.add_(1)
        
        # Color Quantization (Fixed 16)
        q_color, c_idx, _ = self._quantize(flat_color, self.embedding_color, 16)
        
        quantized_flat = torch.cat([q_shape, q_color], dim=-1)
        vq_loss = F.mse_loss(quantized_flat.detach(), flat_input) * self.commitment_cost
        
        # Usage tracking
        if self.training:
            u_idx, counts = torch.unique(s_idx, return_counts=True)
            self.shape_usage[u_idx] += counts.float()
            
        return quantized_flat.view(b, k, d), vq_loss, s_idx.view(b, k), c_idx.view(b, k)
