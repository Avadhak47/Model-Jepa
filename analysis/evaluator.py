import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def check_accuracy(pred_logits, target_grid):
    """
    Computes strict pixel-wise accuracy between prediction and target.
    Requires exactly matching colors.
    """
    # pred_logits: [B, 10, H, W] unnormalized classifications
    # target_grid: [B, 1, H, W] class indices (0-9)
    if pred_logits.dim() == 4 and target_grid.dim() == 4:
        pred_classes = torch.argmax(pred_logits, dim=1)
        target_classes = target_grid.squeeze(1).long()
        
        correct = (pred_classes == target_classes).float()
        return correct.mean().item()
    return 0.0

def run_validation_epoch(modules: dict, dataset, phase: str, batch_size=32, device='cuda'):
    """
    Runs a single validation epoch across the active modules.
    Returns the average validation loss and accuracy.
    """
    # Phase 1: Freeze everything for evaluation
    for m in modules.values(): m.eval()
    
    val_losses = []
    val_accs = []
    
    with torch.no_grad():
        # Sample exactly 10 batches to form a solid statistical average
        for _ in range(10): 
            batch = dataset.sample(batch_size, split='val')
            states = batch['state'].to(device)
            
            if phase == 'ae':
                z_dict = modules['encoder']({'state': states})
                out = modules['decoder']({'latent': z_dict['latent'], 'state': states})
                loss_dict = modules['decoder'].loss({'state': states, 'latent': z_dict['latent']}, out)
                
                # Track pixel-level accuracy
                recon_tensor = out.get('reconstruction', out.get('reconstructed_logits'))
                acc = check_accuracy(recon_tensor, states)
                
                # Robustly fetch loss: fallback to 'loss' if specific sub-losses are missing
                l_val = loss_dict.get('recon_loss', loss_dict.get('mse_loss', loss_dict['loss']))
                val_losses.append(l_val.item() if hasattr(l_val, 'item') else float(l_val))
                val_accs.append(acc)
                
            elif phase == 'wm':
                z_start = modules['encoder']({'state': states})['latent']
                
                # Need random actions formatted for actual environment limits
                action_dim = modules['world_model'].action_dim if hasattr(modules['world_model'], 'action_dim') else 3
                action = torch.randn(batch_size, action_dim, device=device)
                
                out = modules['world_model']({'latent': z_start, 'action': action})
                
                target_states = batch['target_state'].to(device)
                target_z = modules['encoder']({'state': target_states})['latent']
                
                # Fetch target_reward cleanly
                target_reward = batch.get('target_reward', torch.zeros(batch_size)).to(device)
                
                loss_dict = modules['world_model'].loss(
                    {'target_latent': target_z, 'target_reward': target_reward}, 
                    out
                )
                
                # Accuracy is undefined for pure latent regression
                l_val = loss_dict.get('z_loss', loss_dict.get('loss', 0.0))
                val_losses.append(l_val.item() if hasattr(l_val, 'item') else float(l_val))
                val_accs.append(0.0) 
                
    # Phase 2: Unfreeze architecture and return to training
    for m in modules.values(): m.train()
    
    avg_loss = sum(val_losses) / len(val_losses)
    avg_acc = sum(val_accs) / max(1, len([a for a in val_accs if a > 0]))
    
    return avg_loss, avg_acc

def plot_loss_curves(train_history, val_history, phase_name, save_path="evaluation_reports/plots/train_val_curve.png"):
    """
    Plots Train vs Val loss to visually intercept Memorization (Overfitting).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Scale epochs to match length
    epochs = range(1, len(train_history) + 1)
    
    plt.plot(epochs, train_history, label='Train Loss', color='royalblue', linewidth=2.5)
    plt.plot(epochs, val_history, label='Validation Loss', color='darkorange', linewidth=2.5, linestyle='--')
    
    # Mathematical Heuristic for Overfitting Check
    if len(val_history) > 5:
        recent_val = sum(val_history[-3:]) / 3.0
        old_val = sum(val_history[-6:-3]) / 3.0
        # If val loss is climbing while train loss is dropping
        if recent_val > old_val * 1.05 and val_history[-1] > train_history[-1] * 1.2:
            plt.title(f"{phase_name} Phase | 🚨 OVERFITTING DETECTED", color='red', fontsize=14, fontweight='bold')
        else:
            plt.title(f"{phase_name} Phase | Healthy Learning Curve", color='forestgreen', fontsize=14, fontweight='bold')
    else:
        plt.title(f"{phase_name} Validation Spline", fontsize=14)
        
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE/CrossEntropy)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def get_dynamic_threshold(accuracy, min_t=0.5, max_t=0.98):
    """
    Computes a Hinge threshold that scales linearly with accuracy.
    Start at min_t (strict), end at max_t (relaxed).
    """
    # Clamp accuracy between 0 and 1
    a = max(0, min(1, accuracy))
    return min_t + (a * (max_t - min_t))
