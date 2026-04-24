import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedVectorQuantizer(nn.Module):
    """
    Factorized VQ Bottleneck.
    Splits the continuous vector into two distinct dictionaries to force
    unsupervised categorization of high-variance (Shape) and low-variance (Color/Texture).
    
    Includes:
    - Padding-masked perplexity: excludes padded zero-patches from entropy calculation
    - Padding-filtered surgery pool: dead code reset only samples real content patches
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
        
        # Utilization Tracking for Codebook Resurrection
        self.register_buffer('shape_usage', torch.zeros(self.num_shape_codes))
        self.register_buffer('color_usage', torch.zeros(self.num_color_codes))

    def _quantize(self, flat_input, embedding_layer, num_codes, usage_buffer=None, valid_mask=None, temperature=1.0):
        """

        valid_mask: Optional bool tensor [N_total] — True for real patches, False for padding.
        When provided, perplexity is calculated ONLY over real (non-padded) patches.
        This prevents padded zeros from dominating entropy and falsely suppressing perplexity.
        """
        # Calculate Euclidean distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(embedding_layer.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, embedding_layer.weight.t()))
                    
        # Encode using Gumbel-Softmax (Straight-Through Estimator)
        # Logits are negative distances (closest code = highest logit)
        logits = -distances
        encodings = F.gumbel_softmax(logits, tau=temperature, hard=True)
        # For tracking true argmin hits (needed for usage buffers)
        encoding_indices = encodings.argmax(dim=1).unsqueeze(1)
        
        # Quantize
        quantized = torch.matmul(encodings, embedding_layer.weight)
        
        # Usage tracking: only count real patches toward utilization
        if usage_buffer is not None and self.training:
            if valid_mask is not None:
                real_indices = encoding_indices.view(-1)[valid_mask]
            else:
                real_indices = encoding_indices.view(-1)
            usage_buffer.scatter_add_(0, real_indices, torch.ones_like(real_indices, dtype=torch.float))
            
        # Perplexity: computed ONLY over real (non-padded) patches
        # This is the research-recommended fix for padding-inflated perplexity suppression
        if valid_mask is not None:
            real_encodings = encodings[valid_mask]
        else:
            real_encodings = encodings
            
        avg_probs = torch.mean(real_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, perplexity

    def forward(self, inputs, valid_mask=None, temperature=1.0):
        """
        Expects inputs of shape [B, C, H, W] where C = embedding_dim, OR [B, N, C].
        
        valid_mask: Optional bool tensor [B, N] where True = real content patch,
                    False = zero-padding patch. Used to compute unbiased perplexity.
        temperature: Gumbel-Softmax temperature for soft-to-hard routing.
        """
        is_spatial = inputs.dim() == 4
        if is_spatial:
            # Convert [B, C, H, W] to [B, H, W, C]
            inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            inputs_perm = inputs.contiguous()
            
        flat_input = inputs_perm.view(-1, self.embedding_dim)
        
        # Flatten valid_mask to [B*N] to align with flat_input
        flat_valid_mask = valid_mask.view(-1) if valid_mask is not None else None
        
        # Factorize the vector
        flat_shape = flat_input[:, :self.half_dim]
        flat_color = flat_input[:, self.half_dim:]
        
        # Quantize independently (both receive the same spatial valid_mask)
        quantized_shape, perp_shape = self._quantize(
            flat_shape, self.embedding_shape, self.num_shape_codes,
            self.shape_usage, flat_valid_mask, temperature=temperature
        )
        quantized_color, perp_color = self._quantize(
            flat_color, self.embedding_color, self.num_color_codes,
            self.color_usage, flat_valid_mask, temperature=temperature
        )
        
        # Concat back together
        quantized_flat = torch.cat([quantized_shape, quantized_color], dim=1)
        
        # Losses — only over real patches if mask provided
        if flat_valid_mask is not None:
            flat_input_real = flat_input[flat_valid_mask]
            quantized_flat_real = quantized_flat[flat_valid_mask]
            e_latent_loss = F.mse_loss(quantized_flat_real.detach(), flat_input_real)
            q_latent_loss = F.mse_loss(quantized_flat_real, flat_input_real.detach())
        else:
            e_latent_loss = F.mse_loss(quantized_flat.detach(), flat_input)
            q_latent_loss = F.mse_loss(quantized_flat, flat_input.detach())
            
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # With Gumbel-Softmax, we no longer need the non-differentiable STE trick.
        # The gradients flow naturally through `quantized_flat` (matrix multiplication
        # of the soft one-hot encodings and the codebook).
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

    @torch.no_grad()
    def resurrect_dead_codes(self, inputs, valid_mask=None, aggression_quantile=0.25):
        """
        Gently resurrects underutilized codes using a soft quantile threshold.
        Only the BOTTOM 25% by usage count are replaced — prevents 150 simultaneous
        destabilizing injections that cause escalating VQ Loss spikes.

        valid_mask: Optional bool tensor [B, N] — filters out zero-padding patches
                    from the surgery pool so dead codes are seeded with real content.
        aggression_quantile: Fraction of codebook to replace per surgery call (default 0.25).
        """
        is_spatial = inputs.dim() == 4
        if is_spatial:
            inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            inputs_perm = inputs.contiguous()
            
        flat_input = inputs_perm.view(-1, self.embedding_dim)
        
        # Filter pool: only keep REAL (non-padded) patches for surgery candidates
        if valid_mask is not None:
            flat_valid = valid_mask.view(-1)
            real_flat = flat_input[flat_valid]
            if real_flat.size(0) == 0:
                self.shape_usage.zero_()
                self.color_usage.zero_()
                return 0, 0
        else:
            real_flat = flat_input
            
        real_shape = real_flat[:, :self.half_dim]
        real_color = real_flat[:, self.half_dim:]
        n_real = real_shape.size(0)

        # --- Shape Codebook Surgery ---
        # Only replace bottom aggression_quantile of codes by usage frequency
        shape_threshold = torch.quantile(self.shape_usage.float(), aggression_quantile)
        dead_shape_mask = self.shape_usage <= shape_threshold
        num_dead_shapes = dead_shape_mask.sum().item()
        if num_dead_shapes > 0:
            rand_indices = torch.randint(0, n_real, (num_dead_shapes,), device=inputs.device)
            self.embedding_shape.weight.data[dead_shape_mask] = \
                real_shape[rand_indices].clone().to(self.embedding_shape.weight.dtype)
            
        # --- Color Codebook Surgery ---
        color_threshold = torch.quantile(self.color_usage.float(), aggression_quantile)
        dead_color_mask = self.color_usage <= color_threshold
        num_dead_colors = dead_color_mask.sum().item()
        if num_dead_colors > 0:
            rand_indices = torch.randint(0, n_real, (num_dead_colors,), device=inputs.device)
            self.embedding_color.weight.data[dead_color_mask] = \
                real_color[rand_indices].clone().to(self.embedding_color.weight.dtype)
            
        # Reset counters for the next interval
        self.shape_usage.zero_()
        self.color_usage.zero_()
        
        return int(num_dead_shapes), int(num_dead_colors)

