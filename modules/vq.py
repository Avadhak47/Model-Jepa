import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedVectorQuantizer(nn.Module):
    """
    Factorized VQ Bottleneck.
    Splits the continuous vector into two distinct dictionaries to force
    unsupervised categorization of high-variance (Shape) and low-variance (Color/Texture).
    """
    def __init__(self, num_shape_codes=256, num_color_codes=16, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        
        self.num_shape_codes = num_shape_codes
        self.num_color_codes = num_color_codes
        self.commitment_cost = commitment_cost
        
        # Codebook A: Shape (High Capacity)
        self.embedding_shape = nn.Embedding(self.num_shape_codes, self.half_dim)
        self.embedding_shape.weight.data.uniform_(-1/self.num_shape_codes, 1/self.num_shape_codes)
        
        # Codebook B: Color (Low Capacity)
        self.embedding_color = nn.Embedding(self.num_color_codes, self.half_dim)
        self.embedding_color.weight.data.uniform_(-1/self.num_color_codes, 1/self.num_color_codes)

    def _quantize(self, flat_input, embedding_layer, num_codes):
        # Calculate Euclidean distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(embedding_layer.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, embedding_layer.weight.t()))
                    
        # Encode
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], num_codes, device=flat_input.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, embedding_layer.weight)
        
        # Perplexity (Utilization Metric)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, perplexity

    def forward(self, inputs):
        """
        Expects inputs of shape [B, C, H, W] where C = embedding_dim.
        OR [B, N, C]
        """
        is_spatial = inputs.dim() == 4
        if is_spatial:
            # Convert [B, C, H, W] to [B, H, W, C]
            inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            inputs_perm = inputs.contiguous()
            
        flat_input = inputs_perm.view(-1, self.embedding_dim)
        
        # Factorize the vector
        flat_shape = flat_input[:, :self.half_dim]
        flat_color = flat_input[:, self.half_dim:]
        
        # Quantize independently
        quantized_shape, perp_shape = self._quantize(flat_shape, self.embedding_shape, self.num_shape_codes)
        quantized_color, perp_color = self._quantize(flat_color, self.embedding_color, self.num_color_codes)
        
        # Concat back together
        quantized_flat = torch.cat([quantized_shape, quantized_color], dim=1)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized_flat.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized_flat, flat_input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized_flat = flat_input + (quantized_flat - flat_input).detach()
        quantized = quantized_flat.view(inputs_perm.shape)
        
        if is_spatial:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
        return quantized, loss, perp_shape, perp_color

    @torch.no_grad()
    def get_farthest_point_samples(self, target_slots=10):
        """
        Samples the initial slots from the pre-trained Codebooks using FPS.
        Since we have two codebooks, we extract from the Shape codebook for semantic structure
        and randomly pair with the color codebook.
        """
        # FPS on Shape Codebook
        points = self.embedding_shape.weight.data # [256, 64]
        N = points.shape[0]
        
        # Pick random first point
        selected_idx = [torch.randint(0, N, (1,)).item()]
        distances = torch.sum((points - points[selected_idx[0]])**2, dim=1)
        
        for _ in range(1, target_slots):
            # Select max distant point
            farthest = torch.argmax(distances).item()
            selected_idx.append(farthest)
            # Update minimum distances
            new_distances = torch.sum((points - points[farthest])**2, dim=1)
            distances = torch.min(distances, new_distances)
            
        sampled_shapes = points[selected_idx] # [10, 64]
        
        # Just grab random color textures to pair with them, or FPS those too.
        # For simplicity, grab first 10 colors.
        sampled_colors = self.embedding_color.weight.data[:target_slots] # [10, 64]
        
        semantic_slots = torch.cat([sampled_shapes, sampled_colors], dim=1) # [10, 128]
        return semantic_slots.unsqueeze(0) # [1, 10, 128]
