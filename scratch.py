import torch
from modules.encoders import TransformerEncoder

class DeepTransformerEncoder(TransformerEncoder):
    def __init__(self, config):
        cfg = dict(config)
        cfg['_enc_depth'] = cfg.get('enc_depth', 4)
        super().__init__(cfg)
        embed_dim = cfg.get('hidden_dim', 256)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4,
            batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=cfg['_enc_depth'])

cfg = dict(
    latent_dim   = 128,
    hidden_dim   = 256,
    in_channels  = 1,
    enc_depth = 4,
    device = 'cpu',
)
enc = DeepTransformerEncoder(cfg)
img = torch.randn(16, 1, 30, 30)
out = enc({"state": img})
print("z shape:", out['latent'].shape)
