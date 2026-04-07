import torch
import torch.nn as nn

class RuleEncoder(nn.Module):
    """
    Translates symbolic rule dictionaries into continuous Energy-Based Constraints (EBC).
    Embeds categorical and geometric parameters into the same dense [latent_dim] space as visual Slots.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.latent_dim = config.get("latent_dim", 128)
        
        # Properties from ReARCDataset: type (0-2), color (1-10), r, c, w, h
        self.type_embed = nn.Embedding(5, 32)
        self.color_embed = nn.Embedding(15, 32)
        
        self.geom_map = nn.Linear(4, 32)
        
        self.projector = nn.Sequential(
            nn.Linear(32 + 32 + 32, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, batch_rules: list) -> list:
        """
        Takes a list of lists of rule dictionaries (one list of rules per batch item).
        Returns a list of tensors containing the valid Rule embeddings for each batch item.
        """
        encoded_batch = []
        for rules in batch_rules:
            if not rules:
                encoded_batch.append(torch.empty(0, self.latent_dim, device=self.device))
                continue
                
            types = torch.tensor([r["type"] for r in rules], device=self.device)
            colors = torch.tensor([r["color"] for r in rules], device=self.device)
            
            # Normalize geometries roughly
            geoms = torch.tensor([
                [r["r"]/30.0, r["c"]/30.0, r["w"]/30.0, r["h"]/30.0] for r in rules
            ], device=self.device, dtype=torch.float32)
            
            e_t = self.type_embed(types)
            e_c = self.color_embed(colors)
            e_g = self.geom_map(geoms)
            
            combined = torch.cat([e_t, e_c, e_g], dim=-1)
            z_rule = self.projector(combined)
            
            encoded_batch.append(z_rule)
            
        return encoded_batch
