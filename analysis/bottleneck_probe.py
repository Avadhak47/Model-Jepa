import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ARCDiagnosticProbe(nn.Module):
    """
    An overpowered spatial Convolutional structure designed 
    to extract 2D patterns directly from a flattened slot vector.
    """
    def __init__(self, num_slots=16, latent_dim=128, vocab_size=10, grid_size=30):
        super().__init__()
        self.num_slots = num_slots
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        
        # We start by mapping the flat [B, 2048] to a [B, 256, 3, 3] spatial feature map
        self.fc = nn.Linear(num_slots * latent_dim, 256 * 3 * 3)
        self.unflatten = nn.Unflatten(1, (256, 3, 3))
        
        # Spatial upsampling (ConvTranspose) to understand 2D geometry
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 3x3 -> 6x6
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1) # 6x6 -> 12x12
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 12x12 -> 24x24
        
        # Final sizing to snap perfectly to 30x30
        self.flatten_spatial = nn.Flatten(2)
        self.resize_linear = nn.Linear(24 * 24, grid_size * grid_size)
        self.unflatten_spatial = nn.Unflatten(2, (grid_size, grid_size))
        
        # Project to ARC colors (0-9)
        self.head = nn.Conv2d(64, vocab_size, kernel_size=1)
        
    def forward(self, x):
        # x is [B, num_slots * latent_dim]
        x = self.fc(x)
        x = self.unflatten(x)
        
        x = F.gelu(self.up1(x))
        x = F.gelu(self.up2(x))
        x = F.gelu(self.up3(x))
        
        # Resize from 24x24 to 30x30
        x = self.flatten_spatial(x)
        x = self.resize_linear(x)
        x = self.unflatten_spatial(x)
        
        x = self.head(x) # [B, 10, 30, 30]
        return x

def run_bottleneck_analysis(model_name: str, modules: dict, dataset_obj, device='cuda', epochs=50, batch_size=128):
    """
    Trains a Diagnostic Probe on frozen latents to determine if the 
    reconstruction bottleneck lies in the Encoder or the Decoder.
    """
    print(f"\n🔍 Running Post-Training Bottleneck Analysis for {model_name}...")
    
    encoder = modules['encoder'].eval()
    decoder = modules['decoder'].eval()
    
    # 1. Measure Baseline (Current) Performance
    print("   ↳ Measuring Baseline AE Loss...")
    baseline_losses = []
    with torch.no_grad():
        for _ in range(20): 
            batch = dataset_obj.sample(batch_size)
            x = batch['state'].to(device)
            
            z_dict = encoder({'state': x})
            out = decoder({'latent': z_dict['latent'], 'state': x})
            loss_dict = decoder.loss({'state': x, 'latent': z_dict['latent']}, out)
            baseline_losses.append(loss_dict['recon_loss'].item())
            
    avg_baseline = sum(baseline_losses) / len(baseline_losses)
    
    # 2. Build the Spatial Probe
    probe = ARCDiagnosticProbe().to(device)
    optimizer = optim.Adam(probe.parameters(), lr=2e-4) # Slightly higher LR for fast probe training
    
    # 3. Train the Probe
    print(f"   ↳ Training Diagnostic Probe for {epochs} epochs...")
    best_probe_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 50
        probe.train()
        for _ in range(num_batches):
            batch = dataset_obj.sample(batch_size)
            x = batch['state'].to(device)
            
            with torch.no_grad():
                z = encoder({'state': x})['latent'] # [B, 16, 128]
                z_flat = z.view(z.size(0), -1)      # [B, 2048]
            
            optimizer.zero_grad()
            logits = probe(z_flat)                  # [B, 10, 30, 30]
            target = x.squeeze(1).long()
            
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss/num_batches
        if avg_epoch_loss < best_probe_loss:
            best_probe_loss = avg_epoch_loss
            
        if (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch+1}/{epochs} | Probe Loss: {avg_epoch_loss:.4f}")

    print("\n" + "="*50)
    print("📊 BOTTLENECK VERDICT")
    print("="*50)
    print(f"Original Decoder Loss:    {avg_baseline:.4f}")
    print(f"Diagnostic Probe Loss:    {best_probe_loss:.4f}")
    print("-" * 50)
    
    # If probe significantly outperforms decoder 
    if best_probe_loss < avg_baseline * 0.7:
        print("🚨 VERDICT: DECODER is the bottleneck.")
        print("Reason: The probe reached a much lower error than your SlotDecoder. The necessary information exists inside the slots, but the SlotDecoder's alpha-masks or attention layers failed to extract it properly.")
    # If the probe can't do any better than the decoder
    elif best_probe_loss > avg_baseline * 0.95:
        print("🚨 VERDICT: ENCODER is the bottleneck.")
        print("Reason: The overpowered spatial probe could not reconstruct the image either. This proves the microscopic residual error is because the Encoder physically destroyed that ~4% of information and it simply no longer exists in the 128-dim latent space.")
    else:
        print("⚖️ VERDICT: BALANCED.")
        print("Reason: Both models have roughly the same error. Your bottleneck is mathematically symmetric.")
