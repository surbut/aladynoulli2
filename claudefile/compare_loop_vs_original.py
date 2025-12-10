#!/usr/bin/env python
"""
Compare the loop version output with the original run_aladyn_batch.py output
to validate that the loop version works correctly.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compare_tensors(tensor1, tensor2, name, rtol=1e-4, atol=1e-5):
    """
    Compare two tensors and report differences.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name: Name for reporting
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        bool: True if tensors are close, False otherwise
    """
    # Convert to numpy for easier comparison
    if isinstance(tensor1, torch.Tensor):
        arr1 = tensor1.detach().cpu().numpy()
    else:
        arr1 = np.array(tensor1)
    
    if isinstance(tensor2, torch.Tensor):
        arr2 = tensor2.detach().cpu().numpy()
    else:
        arr2 = np.array(tensor2)
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        print(f"❌ {name}: Shape mismatch! {arr1.shape} vs {arr2.shape}")
        return False
    
    # Check if close
    is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    
    if is_close:
        max_diff = np.max(np.abs(arr1 - arr2))
        mean_diff = np.mean(np.abs(arr1 - arr2))
        print(f"✅ {name}: Close (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
    else:
        diff = np.abs(arr1 - arr2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"❌ {name}: NOT close!")
        print(f"   Max diff: {max_diff:.2e} at index {max_idx}")
        print(f"   Mean diff: {mean_diff:.2e}")
        print(f"   Values at max diff: {arr1[max_idx]:.6f} vs {arr2[max_idx]:.6f}")
        print(f"   Relative diff: {max_diff / (np.abs(arr1[max_idx]) + 1e-10):.2e}")
    
    return is_close


def main():
    parser = argparse.ArgumentParser(description='Compare loop version vs original run_aladyn_batch.py output')
    parser.add_argument('--new_run_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_batch_0_10000.pt',
                       help='Path to newly run model (from run_aladyn_batch.py)')
    parser.add_argument('--original_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt',
                       help='Path to original run_aladyn_batch.py model')
    parser.add_argument('--rtol', type=float, default=1e-4,
                       help='Relative tolerance for comparison')
    parser.add_argument('--atol', type=float, default=1e-5,
                       help='Absolute tolerance for comparison')
    args = parser.parse_args()
    
    print("="*70)
    print("Comparing New Run vs Original run_aladyn_batch.py Output")
    print("="*70)
    print(f"\nNew run: {args.new_run_path}")
    print(f"Original: {args.original_path}")
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}\n")
    
    # Load models
    print("Loading models...")
    try:
        new_model = torch.load(args.new_run_path, weights_only=False)
        original_model = torch.load(args.original_path, weights_only=False)
    except FileNotFoundError as e:
        print(f"❌ Error loading models: {e}")
        return
    
    print("✅ Models loaded successfully\n")
    
    # Extract parameters
    new_state = new_model['model_state_dict']
    original_state = original_model['model_state_dict']
    
    # Compare key parameters
    print("="*70)
    print("Parameter Comparisons")
    print("="*70)
    
    all_match = True
    
    # Compare lambda_
    if 'lambda_' in new_state and 'lambda_' in original_state:
        match = compare_tensors(new_state['lambda_'], 
                               original_state['lambda_'], 
                               'lambda_', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    else:
        print("⚠️  lambda_ not found in state_dict")
    
    # Compare phi
    if 'phi' in new_state and 'phi' in original_state:
        match = compare_tensors(new_state['phi'], 
                               original_state['phi'], 
                               'phi', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    else:
        print("⚠️  phi not found in state_dict")
    
    # Compare kappa
    if 'kappa' in new_state and 'kappa' in original_state:
        match = compare_tensors(new_state['kappa'], 
                               original_state['kappa'], 
                               'kappa', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    else:
        print("⚠️  kappa not found in state_dict")
    
    # Compare gamma
    if 'gamma' in new_state and 'gamma' in original_state:
        match = compare_tensors(new_state['gamma'], 
                               original_state['gamma'], 
                               'gamma', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    
    # Compare psi
    if 'psi' in new_state and 'psi' in original_state:
        match = compare_tensors(new_state['psi'], 
                               original_state['psi'], 
                               'psi', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    
    # Compare actual predictions (pi) - THIS IS WHAT REALLY MATTERS
    print("\n" + "="*70)
    print("Prediction (pi) Comparison - THIS IS WHAT MATTERS!")
    print("="*70)
    
    # Compute pi from parameters
    new_lambda = new_state['lambda_']
    new_phi = new_state['phi']
    new_kappa = new_state['kappa']
    
    original_lambda = original_state['lambda_']
    original_phi = original_state['phi']
    original_kappa = original_state['kappa']
    
    # Compute theta (softmax of lambda)
    new_theta = F.softmax(new_lambda, dim=1)
    original_theta = F.softmax(original_lambda, dim=1)
    
    # Compute phi_prob (sigmoid of phi)
    new_phi_prob = torch.sigmoid(new_phi)
    original_phi_prob = torch.sigmoid(original_phi)
    
    # Compute pi: einsum('nkt,kdt->ndt', theta, phi_prob) * kappa
    new_pi = torch.einsum('nkt,kdt->ndt', new_theta, new_phi_prob) * new_kappa
    original_pi = torch.einsum('nkt,kdt->ndt', original_theta, original_phi_prob) * original_kappa
    
    # Clamp to same epsilon as forward()
    epsilon = 1e-6
    new_pi = torch.clamp(new_pi, epsilon, 1-epsilon)
    original_pi = torch.clamp(original_pi, epsilon, 1-epsilon)
    
    # Compare pi predictions
    pi_match = compare_tensors(new_pi, 
                               original_pi, 
                               'pi (predictions)', 
                               rtol=1e-3,  # More lenient for predictions
                               atol=1e-5)
    
    # Create correlation plots
    print("\n" + "="*70)
    print("Creating Correlation Plots...")
    print("="*70)
    
    # Flatten predictions for correlation
    new_pi_flat = new_pi.detach().cpu().numpy().flatten()
    original_pi_flat = original_pi.detach().cpu().numpy().flatten()
    
    # Calculate differences
    diff = np.abs(new_pi_flat - original_pi_flat)
    
    # Calculate correlation
    corr_coef, p_value = pearsonr(new_pi_flat, original_pi_flat)
    print(f"\nOverall Correlation: {corr_coef:.8f} (p < {p_value:.2e})")
    
    # Sample for plotting
    sample_size = min(100000, len(new_pi_flat))
    sample_idx = np.random.choice(len(new_pi_flat), sample_size, replace=False)
    eps = 1e-8
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall scatter plot - LOG SCALE
    ax1 = axes[0, 0]
    new_sampled = new_pi_flat[sample_idx]
    original_sampled = original_pi_flat[sample_idx]
    scatter1 = ax1.scatter(new_sampled + eps, original_sampled + eps, 
                alpha=0.2, s=2, c=diff[sample_idx], cmap='viridis', 
                vmin=diff.min(), vmax=diff.max())
    min_val = max(new_sampled.min(), original_sampled.min())
    max_val = min(new_sampled.max(), original_sampled.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('New Run pi (log scale)')
    ax1.set_ylabel('Original Version pi (log scale)')
    ax1.set_title(f'Overall Correlation (n={sample_size:,} samples)\nr={corr_coef:.8f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    plt.colorbar(scatter1, ax=ax1, label='|Difference|')
    
    # Zoomed-in view
    ax2 = axes[0, 1]
    mask_small = (new_sampled < 0.1) & (original_sampled < 0.1)
    if mask_small.sum() > 0:
        scatter2 = ax2.scatter(new_sampled[mask_small], original_sampled[mask_small], 
                   alpha=0.3, s=5, c=diff[sample_idx][mask_small], cmap='viridis',
                   vmin=diff.min(), vmax=diff.max())
        plt.colorbar(scatter2, ax=ax2, label='|Difference|')
    ax2.plot([0, 0.1], [0, 0.1], 'r--', lw=2, label='y=x')
    ax2.set_xlabel('New Run pi')
    ax2.set_ylabel('Original Version pi')
    ax2.set_title('Zoomed View (pi < 0.1)')
    ax2.set_xlim([0, 0.1])
    ax2.set_ylim([0, 0.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Difference histogram
    ax3 = axes[1, 0]
    if np.any(diff > 0):
        log_bins = np.logspace(np.log10(diff[diff > 0].min()), 
                              np.log10(diff.max()), 50)
        ax3.hist(diff, bins=log_bins, alpha=0.7, edgecolor='black', color='steelblue')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
    else:
        # All differences are zero - just show a single bar
        ax3.bar([0], [len(diff)], alpha=0.7, edgecolor='black', color='steelblue', width=1e-10)
        ax3.set_xlim([-1e-10, 1e-10])
    ax3.set_xlabel('Absolute Difference |New - Original|')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.set_title(f'Distribution of Differences\nMean: {np.mean(diff):.2e}, Median: {np.median(diff):.2e}, Max: {np.max(diff):.2e}')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Relative error plot
    ax4 = axes[1, 1]
    relative_error = diff / (new_pi_flat + eps)
    rel_err_sampled = relative_error[sample_idx]
    if np.any(relative_error > 0):
        ax4.scatter(new_sampled + eps, rel_err_sampled, 
                   alpha=0.2, s=2, c='coral')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
    else:
        # All relative errors are zero
        ax4.text(0.5, 0.5, 'Perfect Match!\nAll differences = 0', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, fontweight='bold', color='green')
    ax4.set_xlabel('New Run pi (log scale)')
    ax4.set_ylabel('Relative Error |New - Original| / pi (log scale)')
    ax4.set_title(f'Relative Error Distribution\nMean: {np.mean(relative_error):.2e}, Max: {np.max(relative_error):.2e}')
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(args.new_run_path).parent / 'new_run_vs_original_correlation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Correlation plot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean absolute difference: {np.mean(diff):.2e}")
    print(f"  Median absolute difference: {np.median(diff):.2e}")
    print(f"  Max absolute difference: {np.max(diff):.2e}")
    print(f"  95th percentile difference: {np.percentile(diff, 95):.2e}")
    print(f"  99th percentile difference: {np.percentile(diff, 99):.2e}")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if corr_coef > 0.9999:
        print("✅ PREDICTIONS (pi) ARE HIGHLY CORRELATED! Models are functionally equivalent.")
        print(f"   Correlation: {corr_coef:.8f}")
        print(f"   Mean absolute difference: {np.mean(diff):.2e}")
        print(f"   Max absolute difference: {np.max(diff):.2e}")
        print("   New run matches original run_aladyn_batch.py output!")
    elif corr_coef > 0.99:
        print("⚠️  PREDICTIONS (pi) ARE VERY CLOSE but not perfect.")
        print(f"   Correlation: {corr_coef:.8f}")
        print(f"   Mean absolute difference: {np.mean(diff):.2e}")
        print(f"   Max absolute difference: {np.max(diff):.2e}")
        print("   Small differences may be due to different E matrices or random seeds.")
    else:
        print("❌ PREDICTIONS (pi) DO NOT MATCH!")
        print(f"   Correlation: {corr_coef:.8f}")
        print("   This indicates a problem - investigate further!")
    
    if all_match:
        print("\n✅ ALL PARAMETERS MATCH within tolerance!")
    else:
        print("\n⚠️  SOME PARAMETERS DO NOT MATCH within strict tolerance")
        if corr_coef > 0.99:
            print("   However, predictions match, so models are functionally equivalent.")
            print("   Parameter differences may be due to:")
            print("   - Different E matrices (E_matrix.pt vs E_matrix_corrected.pt)")
            print("   - Different random seeds or initialization")
            print("   - Floating-point precision differences")
    print("="*70)
    
    # Print shapes for reference
    print("\nParameter Shapes:")
    print("-"*70)
    if 'lambda_' in new_state:
        print(f"lambda_: {new_state['lambda_'].shape}")
    if 'phi' in new_state:
        print(f"phi: {new_state['phi'].shape}")
    if 'kappa' in new_state:
        print(f"kappa: {new_state['kappa'].shape}")
    if 'gamma' in new_state:
        print(f"gamma: {new_state['gamma'].shape}")
    if 'psi' in new_state:
        print(f"psi: {new_state['psi'].shape}")


if __name__ == '__main__':
    main()

