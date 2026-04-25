import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedVectorQuantizer(nn.Module):
    """
    Factorized VQ Bottleneck.
    Splits the continuous vector into two distinct dictionaries to force
    unsupervised categorization of high-variance (Shape) and low-variance (Color/Texture).
    
    Semantic Partitioning:
      - Shape codes 0..(N_BG-1)  → reserved for background (zero) patches
      - Shape codes N_BG..511    → foreground object primitives
    
    Includes:
    - Padding-masked perplexity: excludes padded zero-patches from entropy calculation
    - Padding-filtered surgery pool: dead code reset only samples real content patches
    - Affinity loss: patches from same connected component share same shape code
    """
    N_BG_CODES = 16  # First 16 shape codes are reserved for background

    def __init__(self, num_shape_codes=512, num_color_codes=16, embedding_dim=128, commitment_cost=0.25):
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

    def _quantize(self, flat_input, embedding_layer, num_codes, usage_buffer=None,
                  valid_mask=None, temperature=1.0, force_bg_mask=None):
        """
        valid_mask    : bool [N_total] — True for real patches, False for padding.
        force_bg_mask : bool [N_total] — True for background (all-zero) patches.
                        If set, these patches are restricted to choose only from
                        codes 0..N_BG_CODES-1, reserving upper codes for foreground.
        """
        # Calculate Euclidean distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(embedding_layer.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, embedding_layer.weight.t()))
        
        logits = -distances  # closest code = highest logit
        
        # Semantic Partitioning: Mask foreground codes for background patches.
        # Background patches ONLY compete among the first N_BG_CODES.
        # This reserves codes [N_BG..num_codes) exclusively for foreground objects.
        if force_bg_mask is not None and force_bg_mask.any() and num_codes == self.num_shape_codes:
            fg_mask = torch.ones(num_codes, dtype=torch.bool, device=flat_input.device)
            fg_mask[:self.N_BG_CODES] = False  # first N_BG are BG-only
            # For background patches: set foreground code logits to -inf
            logits[force_bg_mask, :] = logits[force_bg_mask, :].masked_fill(fg_mask, float('-inf'))
            # For foreground patches: set background code logits to -inf
            if (~force_bg_mask).any():
                bg_mask = ~fg_mask
                logits[~force_bg_mask, :] = logits[~force_bg_mask, :].masked_fill(bg_mask, float('-inf'))
                     
        # Encode using Gumbel-Softmax (Straight-Through Estimator)
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
        if valid_mask is not None:
            real_encodings = encodings[valid_mask]
        else:
            real_encodings = encodings
            
        avg_probs = torch.mean(real_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, perplexity, encoding_indices.view(-1)

    @staticmethod
    def affinity_loss(shape_indices, state, patch_size, grid_hw):
        """
        Connected Component Affinity Loss.
        Patches from the same connected foreground object should share the same shape code.
        Patches from different objects should use different shape codes.

        Args:
            shape_indices : LongTensor [B*N] — argmax shape code per patch
            state         : FloatTensor [B, 1, H, W] — original ARC grid (values 0–9)
            patch_size    : int
            grid_hw       : int — number of patches per side (H//patch_size)
        Returns:
            scalar loss
        """
        import numpy as np
        from scipy import ndimage

        B_N = shape_indices.shape[0]
        B   = state.shape[0]
        N   = B_N // B  # patches per image
        device = shape_indices.device

        pos_pairs = []  # (i, j) same object → codes should match
        neg_pairs = []  # (i, j) different objects → codes should differ

        for b in range(B):
            grid = state[b, 0].cpu().numpy().astype(int)  # [H, W]
            # Label connected components for each non-zero color separately
            label_map = np.zeros_like(grid, dtype=int)
            next_label = 1
            for color in range(1, 10):
                mask = (grid == color)
                if not mask.any():
                    continue
                labeled, n_comp = ndimage.label(mask)
                label_map[mask] = labeled[mask] + next_label - 1
                next_label += n_comp

            # Assign each patch its dominant label
            offset = b * N
            patch_labels = []
            for pi in range(grid_hw):
                for pj in range(grid_hw):
                    r0, c0 = pi * patch_size, pj * patch_size
                    patch = label_map[r0:r0+patch_size, c0:c0+patch_size]
                    vals, counts = np.unique(patch.flatten(), return_counts=True)
                    dominant = vals[counts.argmax()]
                    patch_labels.append(dominant)

            # Build pairs
            for i in range(N):
                for j in range(i + 1, N):
                    li, lj = patch_labels[i], patch_labels[j]
                    if li == 0 or lj == 0:
                        continue  # skip background-background pairs
                    if li == lj:
                        pos_pairs.append((offset + i, offset + j))
                    else:
                        neg_pairs.append((offset + i, offset + j))

        if not pos_pairs and not neg_pairs:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Limit pair count to avoid O(N^2) explosion
        rng = np.random.default_rng()
        max_pairs = 512
        if len(pos_pairs) > max_pairs:
            pos_pairs = [pos_pairs[i] for i in rng.choice(len(pos_pairs), max_pairs, replace=False)]
        if len(neg_pairs) > max_pairs:
            neg_pairs = [neg_pairs[i] for i in rng.choice(len(neg_pairs), max_pairs, replace=False)]

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        # Same-object pairs: code indices should be the same → cross-entropy-like push
        for (i, j) in pos_pairs:
            ci = shape_indices[i].unsqueeze(0)  # [1]
            cj = shape_indices[j].unsqueeze(0)
            # Soft push: penalise if they chose different codes
            if ci.item() != cj.item():
                total_loss = total_loss + 1.0
                count += 1

        # Different-object pairs: code indices should differ → penalise if same
        for (i, j) in neg_pairs:
            ci = shape_indices[i].unsqueeze(0)
            cj = shape_indices[j].unsqueeze(0)
            if ci.item() == cj.item():
                total_loss = total_loss + 1.0
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / count

    def forward(self, inputs, valid_mask=None, temperature=1.0):
        """
        Expects inputs of shape [B, C, H, W] where C = embedding_dim, OR [B, N, C].
        
        valid_mask: Optional bool tensor [B, N] where True = real content patch,
                    False = zero-padding patch. Used to compute unbiased perplexity.
        temperature: Gumbel-Softmax temperature for soft-to-hard routing.
        """
        is_spatial = inputs.dim() == 4
        if is_spatial:
            inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            inputs_perm = inputs.contiguous()
            
        flat_input = inputs_perm.view(-1, self.embedding_dim)
        
        # Flatten valid_mask to [B*N] to align with flat_input
        flat_valid_mask = valid_mask.view(-1) if valid_mask is not None else None
        
        # Background mask: patches with ALL zero values (padding OR empty ARC cell)
        flat_bg_mask = (flat_input.abs().sum(dim=-1) < 1e-6)
        
        # Factorize the vector
        flat_shape = flat_input[:, :self.half_dim]
        flat_color = flat_input[:, self.half_dim:]
        
        # Quantize independently (both receive the same spatial valid_mask)
        quantized_shape, perp_shape, shape_idx = self._quantize(
            flat_shape, self.embedding_shape, self.num_shape_codes,
            self.shape_usage, flat_valid_mask, temperature=temperature,
            force_bg_mask=flat_bg_mask
        )
        quantized_color, perp_color, _ = self._quantize(
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
        
        quantized = quantized_flat.view(inputs_perm.shape)
        
        if is_spatial:
            quantized = quantized.permute(0, 2, 3, 1  if False else 1).contiguous()
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
        return quantized, loss, perp_shape, perp_color, shape_idx

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

