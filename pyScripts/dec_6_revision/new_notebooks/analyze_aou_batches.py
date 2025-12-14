#!/usr/bin/env python3
"""
Analyze AOU batch training results:
1. Visualize loss convergence
2. Load and average phi across all batches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def parse_losses_from_file(loss_file_path):
    """Parse loss values from the losses file."""
    losses_by_batch = {}
    current_batch = None
    
    with open(loss_file_path, 'r') as f:
        for line in f:
            # Check for batch header
            batch_match = re.search(r'BATCH (\d+) /', line)
            if batch_match:
                current_batch = int(batch_match.group(1)) - 1  # 0-indexed
                losses_by_batch[current_batch] = []
            
            # Check for loss value
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            if loss_match and current_batch is not None:
                loss_val = float(loss_match.group(1))
                losses_by_batch[current_batch].append(loss_val)
    
    return losses_by_batch


def plot_loss_convergence(losses_by_batch, output_path=None):
    """Plot loss convergence for all batches."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: All batches overlaid
    ax1 = axes[0]
    for batch_idx, losses in sorted(losses_by_batch.items()):
        epochs = np.arange(len(losses))
        ax1.plot(epochs, losses, alpha=0.6, linewidth=1, label=f'Batch {batch_idx+1}')
    
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence Across All Batches', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to better see convergence
    
    # Plot 2: Average loss across batches
    ax2 = axes[1]
    max_epochs = max(len(losses) for losses in losses_by_batch.values())
    avg_losses = []
    std_losses = []
    
    for epoch in range(max_epochs):
        epoch_losses = []
        for losses in losses_by_batch.values():
            if epoch < len(losses):
                epoch_losses.append(losses[epoch])
        if epoch_losses:
            avg_losses.append(np.mean(epoch_losses))
            std_losses.append(np.std(epoch_losses))
    
    epochs_avg = np.arange(len(avg_losses))
    ax2.plot(epochs_avg, avg_losses, linewidth=2, color='darkblue', label='Mean')
    ax2.fill_between(epochs_avg, 
                     np.array(avg_losses) - np.array(std_losses),
                     np.array(avg_losses) + np.array(std_losses),
                     alpha=0.3, color='blue', label='±1 SD')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Average Loss Across Batches (with std dev)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved loss plot to: {output_path}")
    else:
        plt.show()
    
    return fig


def load_and_average_phi(batch_dir, n_batches=25, batch_size=10000):
    """Load phi from all batch checkpoints and average them."""
    print(f"\n{'='*80}")
    print("LOADING PHI FROM ALL BATCHES")
    print(f"{'='*80}")
    
    all_phis = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, 243303)
        
        batch_file = Path(batch_dir) / f'aou_model_batch_{batch_idx}_{start_idx}_{end_idx}.pt'
        
        if not batch_file.exists():
            print(f"  Warning: Batch {batch_idx+1} file not found: {batch_file}")
            continue
        
        print(f"  Loading batch {batch_idx+1}...")
        ckpt = torch.load(batch_file, map_location='cpu', weights_only=False)
        
        # Extract phi from model state dict
        if 'model_state_dict' in ckpt:
            phi = ckpt['model_state_dict']['phi']
        elif 'phi' in ckpt:
            phi = ckpt['phi']
        else:
            print(f"  Warning: No phi found in batch {batch_idx+1}")
            continue
        
        if torch.is_tensor(phi):
            phi = phi.detach().cpu().numpy()
        
        all_phis.append(phi)
        print(f"    Phi shape: {phi.shape}")
    
    if len(all_phis) == 0:
        raise ValueError("No phi values found in any batch!")
    
    # Average phi across batches
    print(f"\n  Averaging phi across {len(all_phis)} batches...")
    phi_avg = np.mean(all_phis, axis=0)
    phi_std = np.std(all_phis, axis=0)
    
    print(f"  Average phi shape: {phi_avg.shape}")
    print(f"  Phi range: [{phi_avg.min():.4f}, {phi_avg.max():.4f}]")
    print(f"  Phi mean: {phi_avg.mean():.4f}")
    print(f"  Phi std (across batches): [{phi_std.min():.4f}, {phi_std.max():.4f}]")
    
    return phi_avg, phi_std, all_phis


def main():
    # Parse losses
    loss_file = '/Users/sarahurbut/aladynoulli2/aou_losses'
    print("Parsing losses from file...")
    losses_by_batch = parse_losses_from_file(loss_file)
    
    print(f"\nFound losses for {len(losses_by_batch)} batches")
    for batch_idx, losses in sorted(losses_by_batch.items()):
        print(f"  Batch {batch_idx+1}: {len(losses)} epochs, final loss: {losses[-1]:.4f}")
    
    # Plot loss convergence
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/aou_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loss_plot_path = output_dir / 'aou_loss_convergence.pdf'
    plot_loss_convergence(losses_by_batch, output_path=str(loss_plot_path))
    
    # Load and average phi
    batch_dir = '/Users/sarahurbut/aladynoulli2'
    phi_avg, phi_std, all_phis = load_and_average_phi(batch_dir, n_batches=25)
    
    # Save averaged phi
    phi_save_path = output_dir / 'aou_phi_averaged.pt'
    torch.save({
        'phi_avg': torch.tensor(phi_avg),
        'phi_std': torch.tensor(phi_std),
        'n_batches': len(all_phis),
        'batch_indices': list(range(len(all_phis)))
    }, phi_save_path)
    print(f"\n✓ Saved averaged phi to: {phi_save_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"  Loss convergence plot: {loss_plot_path}")
    print(f"  Averaged phi: {phi_save_path}")
    print(f"  Shape: {phi_avg.shape}")


if __name__ == "__main__":
    main()

