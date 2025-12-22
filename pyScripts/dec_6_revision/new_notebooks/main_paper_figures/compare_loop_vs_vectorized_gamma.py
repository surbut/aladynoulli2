#!/usr/bin/env python3
"""
Simple script to compare gamma values from loop vs vectorized versions.
Creates a scatter plot showing they produce identical results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_gamma_loop_vs_vectorized():
    """Compare gamma from loop and vectorized versions."""
    
    # Paths
    loop_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_loop/enrollment_model_LOOP_W0.0001_batch_0_10000.pt'
    vec_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt'
    
    # Load checkpoints
    print("Loading loop version...")
    loop_ckpt = torch.load(loop_path, map_location='cpu', weights_only=False)
    loop_gamma = loop_ckpt['model_state_dict']['gamma'].detach().cpu().numpy()
    
    print("Loading vectorized version...")
    vec_ckpt = torch.load(vec_path, map_location='cpu', weights_only=False)
    vec_gamma = vec_ckpt['model_state_dict']['gamma'].detach().cpu().numpy()
    
    print(f"\nLoop gamma shape: {loop_gamma.shape}")
    print(f"Vectorized gamma shape: {vec_gamma.shape}")
    
    # Flatten for comparison
    loop_flat = loop_gamma.flatten()
    vec_flat = vec_gamma.flatten()
    
    # Statistics
    print(f"\nLoop gamma stats:")
    print(f"  Mean: {loop_flat.mean():.6f}")
    print(f"  Std: {loop_flat.std():.6f}")
    print(f"  Min: {loop_flat.min():.6f}, Max: {loop_flat.max():.6f}")
    
    print(f"\nVectorized gamma stats:")
    print(f"  Mean: {vec_flat.mean():.6f}")
    print(f"  Std: {vec_flat.std():.6f}")
    print(f"  Min: {vec_flat.min():.6f}, Max: {vec_flat.max():.6f}")
    
    # Differences
    diff = loop_flat - vec_flat
    print(f"\nDifferences:")
    print(f"  Max absolute difference: {np.abs(diff).max():.2e}")
    print(f"  Mean absolute difference: {np.abs(diff).mean():.2e}")
    print(f"  Are they close? {np.allclose(loop_flat, vec_flat, rtol=1e-5, atol=1e-6)}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(loop_flat, vec_flat, alpha=0.5, s=10)
    
    # Add diagonal line
    min_val = min(loop_flat.min(), vec_flat.min())
    max_val = max(loop_flat.max(), vec_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    ax.set_xlabel('Loop Version Gamma', fontsize=12)
    ax.set_ylabel('Vectorized Version Gamma', fontsize=12)
    ax.set_title('Gamma Comparison: Loop vs Vectorized', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(loop_flat, vec_flat)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.10f}', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'loop_vs_vectorized_gamma_scatter.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    compare_gamma_loop_vs_vectorized()



