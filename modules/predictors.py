       cp67--
    ₹aimport torch
import torch.nn as nn

class SlotJEPAPredictor(nn.Module):
    """
    Latent Predictor for Slot-JEPA.
    Maps Context Slots to Target Slots without pixel decoding.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config.get("latent_dim", 128)
        self.num_slots = config.get("num_slots", 16)
        
        # We use a transformer to allow slots to interact and resolve predictions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=4, 
            dim_feedforward=self.embed_dim * 4, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.get("predictor_layers", 2)
        )
        
        # Final projection 
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, context_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_slots: [B, num_slots, latent_dim] (from the Student Encoder)
        Returns:
            predicted_target_slots: [B, num_slots, latent_dim] (to compare with EMA Target Encoder)
        """
        # Self-attention over slots to contextualize predictions
        x = self.transformer(context_slots) 
        z_pred = self.projector(x)
        return z_pred

class CrossAttentionTaskConditioner(nn.Module):
    """
    Fuses multiple [In, Out] latent pairs into a single Task Embedding.
    Uses Cross-Attention to allow the test input to 'query' the task rules.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.latent_dim = config.get("latent_dim", 128)
        self.embed_dim = self.latent_dim * 2 # Concatenated In+Out
        
        # Project pairs to a relation space
        self.relation_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        # Cross-Attention: Query (Test Input) vs Keys/Values (Example Relations)
        self.attn = nn.MultiheadAttention(self.latent_dim, nhead=8, batch_first=True)
        self.norm = nn.LayerNorm(self.latent_dim)
        
    def forward(self, z_test_in, z_examples_in, z_examples_out):
        """
        Args:
            z_test_in: [B, 1, D]
            z_examples_in: [B, K, D] (K examples)
            z_examples_out: [B, K, D]
        """
        if z_test_in.dim() == 2:
            z_test_in = z_test_in.unsqueeze(1)
            
        B, K, D = z_examples_in.shape
        
        # Construct Relation Vectors for each example
        # [B, K, 2D]
        relations = torch.cat([z_examples_in, z_examples_out], dim=-1)
        # [B, K, D]
        r_embeddings = self.relation_proj(relations)
        
        # Cross-Attend: Test Input queries Example Relations
        # Query: [B, 1, D], Keys/Values: [B, K, D]
        z_task, _ = self.attn(z_test_in, r_embeddings, r_embeddings)
        
        return self.norm(z_task + z_test_in)

class TransformerJEPAPredictor(nn.Module):
    """
    Predicts the target latent z_y from (z_x, z_task).
    """
    def __init__(self, config: dict):
        super().__init__()
        self.latent_dim = config.get("latent_dim", 128)
        
        # Transformer backbone for latent reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, 
            nhead=8, 
            dim_feedforward=self.latent_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.projector = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z_test_in, z_task):
        """
        Args:
            z_test_in: [B, 1, D]
            z_task: [B, 1, D] (Conditioned relation)
        """
        if z_test_in.dim() == 2:
            z_test_in = z_test_in.unsqueeze(1)
            
        # Sum/Concatenate task info and test state
        x = z_test_in + z_task
        
        z_pred = self.transformer(x)
        return self.projector(z_pred)
