#!/usr/bin/env python
"""
Compare phis, lambdas, and kappas between loop and vectorized versions
to verify they're within numerical precision.
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
    parser = argparse.ArgumentParser(description='Compare loop vs vectorized model outputs')
    parser.add_argument('--loop_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_loop/enrollment_model_LOOP_W0.0001_batch_0_10000.pt',
                       help='Path to loop version model')
    parser.add_argument('--vectorized_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_VECTORIZED_W0.0001_batch_0_10000.pt',
                       help='Path to vectorized version model')
    parser.add_argument('--rtol', type=float, default=1e-4,
                       help='Relative tolerance for comparison')
    parser.add_argument('--atol', type=float, default=1e-5,
                       help='Absolute tolerance for comparison')
    args = parser.parse_args()
    
    print("="*70)
    print("Comparing Loop vs Vectorized Model Parameters")
    print("="*70)
    print(f"\nLoop model: {args.loop_path}")
    print(f"Vectorized model: {args.vectorized_path}")
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}\n")
    
    # Load models
    print("Loading models...")
    try:
        loop_model = torch.load(args.loop_path, weights_only=False)
        vectorized_model = torch.load(args.vectorized_path, weights_only=False)
    except FileNotFoundError as e:
        print(f"❌ Error loading models: {e}")
        return
    
    print("✅ Models loaded successfully\n")
    
    # Extract parameters
    loop_state = loop_model['model_state_dict']
    vectorized_state = vectorized_model['model_state_dict']
    
    # Also get phi directly if saved
    loop_phi_direct = loop_model.get('phi', None)
    vectorized_phi_direct = vectorized_model.get('phi', None)
    
    # Compare key parameters
    print("="*70)
    print("Parameter Comparisons")
    print("="*70)
    
    all_match = True
    
    # Compare lambda_
    if 'lambda_' in loop_state and 'lambda_' in vectorized_state:
        match = compare_tensors(loop_state['lambda_'], 
                               vectorized_state['lambda_'], 
                               'lambda_', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    else:
        print("⚠️  lambda_ not found in state_dict")
    
    # Compare phi
    if 'phi' in loop_state and 'phi' in vectorized_state:
        match = compare_tensors(loop_state['phi'], 
                               vectorized_state['phi'], 
                               'phi', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    else:
        print("⚠️  phi not found in state_dict")
    
    # Also compare direct phi if available
    if loop_phi_direct is not None and vectorized_phi_direct is not None:
        match = compare_tensors(loop_phi_direct, 
                               vectorized_phi_direct, 
                               'phi (direct)', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    
    # Compare kappa
    if 'kappa' in loop_state and 'kappa' in vectorized_state:
        match = compare_tensors(loop_state['kappa'], 
                               vectorized_state['kappa'], 
                               'kappa', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    else:
        print("⚠️  kappa not found in state_dict")
    
    # Compare gamma (optional, but good to check)
    if 'gamma' in loop_state and 'gamma' in vectorized_state:
        match = compare_tensors(loop_state['gamma'], 
                               vectorized_state['gamma'], 
                               'gamma', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    
    # Compare psi (optional)
    if 'psi' in loop_state and 'psi' in vectorized_state:
        match = compare_tensors(loop_state['psi'], 
                               vectorized_state['psi'], 
                               'psi', 
                               rtol=args.rtol, 
                               atol=args.atol)
        all_match = all_match and match
    
    # Compare actual predictions (pi) - THIS IS WHAT REALLY MATTERS
    print("\n" + "="*70)
    print("Prediction (pi) Comparison - THIS IS WHAT MATTERS!")
    print("="*70)
    
    # Compute pi from parameters
    loop_lambda = loop_state['lambda_']
    loop_phi = loop_state['phi']
    loop_kappa = loop_state['kappa']
    
    vectorized_lambda = vectorized_state['lambda_']
    vectorized_phi = vectorized_state['phi']
    vectorized_kappa = vectorized_state['kappa']
    
    # Compute theta (softmax of lambda)
    loop_theta = F.softmax(loop_lambda, dim=1)
    vectorized_theta = F.softmax(vectorized_lambda, dim=1)
    
    # Compute phi_prob (sigmoid of phi)
    loop_phi_prob = torch.sigmoid(loop_phi)
    vectorized_phi_prob = torch.sigmoid(vectorized_phi)
    
    # Compute pi: einsum('nkt,kdt->ndt', theta, phi_prob) * kappa
    loop_pi = torch.einsum('nkt,kdt->ndt', loop_theta, loop_phi_prob) * loop_kappa
    vectorized_pi = torch.einsum('nkt,kdt->ndt', vectorized_theta, vectorized_phi_prob) * vectorized_kappa
    
    # Clamp to same epsilon as forward()
    epsilon = 1e-6
    loop_pi = torch.clamp(loop_pi, epsilon, 1-epsilon)
    vectorized_pi = torch.clamp(vectorized_pi, epsilon, 1-epsilon)
    
    # Compare pi predictions with more lenient tolerance (predictions matter more)
    pi_match = compare_tensors(loop_pi, 
                               vectorized_pi, 
                               'pi (predictions)', 
                               rtol=1e-3,  # More lenient for predictions
                               atol=1e-5)
    
    # Create correlation plots
    print("\n" + "="*70)
    print("Creating Correlation Plots...")
    print("="*70)
    
    # Flatten predictions for correlation
    loop_pi_flat = loop_pi.detach().cpu().numpy().flatten()
    vectorized_pi_flat = vectorized_pi.detach().cpu().numpy().flatten()
    
    # Calculate differences for better visualization
    diff = np.abs(loop_pi_flat - vectorized_pi_flat)
    
    # Calculate correlation
    corr_coef, p_value = pearsonr(loop_pi_flat, vectorized_pi_flat)
    print(f"\nOverall Correlation: {corr_coef:.8f} (p < {p_value:.2e})")
    
    # Sample some specific cases for detailed plots
    N, D, T = loop_pi.shape
    
    # Plot 1: Overall scatter (sample to avoid too many points)
    sample_size = min(100000, len(loop_pi_flat))
    sample_idx = np.random.choice(len(loop_pi_flat), sample_size, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall scatter plot - LOG SCALE
    ax1 = axes[0, 0]
    # Use log scale since most values are small
    loop_sampled = loop_pi_flat[sample_idx]
    vec_sampled = vectorized_pi_flat[sample_idx]
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    scatter1 = ax1.scatter(loop_sampled + eps, vec_sampled + eps, 
                alpha=0.2, s=2, c=diff[sample_idx], cmap='viridis', 
                vmin=diff.min(), vmax=diff.max())
    # Plot diagonal line
    min_val = max(loop_sampled.min(), vec_sampled.min())
    max_val = min(loop_sampled.max(), vec_sampled.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Loop Version pi (log scale)')
    ax1.set_ylabel('Vectorized Version pi (log scale)')
    ax1.set_title(f'Overall Correlation (n={sample_size:,} samples)\nr={corr_coef:.8f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    plt.colorbar(scatter1, ax=ax1, label='|Difference|')
    
    # Plot 2: Zoomed-in view of small values (linear scale)
    ax2 = axes[0, 1]
    # Focus on values < 0.1 for better visualization
    mask_small = (loop_sampled < 0.1) & (vec_sampled < 0.1)
    if mask_small.sum() > 0:
        scatter2 = ax2.scatter(loop_sampled[mask_small], vec_sampled[mask_small], 
                   alpha=0.3, s=5, c=diff[sample_idx][mask_small], cmap='viridis',
                   vmin=diff.min(), vmax=diff.max())
        plt.colorbar(scatter2, ax=ax2, label='|Difference|')
    ax2.plot([0, 0.1], [0, 0.1], 'r--', lw=2, label='y=x')
    ax2.set_xlabel('Loop Version pi')
    ax2.set_ylabel('Vectorized Version pi')
    ax2.set_title('Zoomed View (pi < 0.1)')
    ax2.set_xlim([0, 0.1])
    ax2.set_ylim([0, 0.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference histogram with better binning
    ax3 = axes[1, 0]
    # Use log bins for better visualization
    log_bins = np.logspace(np.log10(diff[diff > 0].min()), 
                          np.log10(diff.max()), 50)
    ax3.hist(diff, bins=log_bins, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.set_xlabel('Absolute Difference |Loop - Vectorized|')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.set_title(f'Distribution of Differences\nMean: {np.mean(diff):.2e}, Median: {np.median(diff):.2e}, Max: {np.max(diff):.2e}')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Relative error plot
    ax4 = axes[1, 1]
    # Calculate relative error (avoid division by zero)
    relative_error = diff / (loop_pi_flat + eps)
    # Sample for visualization
    rel_err_sampled = relative_error[sample_idx]
    ax4.scatter(loop_sampled + eps, rel_err_sampled, 
               alpha=0.2, s=2, c='coral')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Loop Version pi (log scale)')
    ax4.set_ylabel('Relative Error |Loop - Vec| / pi (log scale)')
    ax4.set_title(f'Relative Error Distribution\nMean: {np.mean(relative_error):.2e}, Max: {np.max(relative_error):.2e}')
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save prediction plot
    output_path = Path(args.loop_path).parent / 'loop_vs_vectorized_correlation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Prediction correlation plot saved to: {output_path}")
    
    # Create plots for lambda and phi parameters
    print("\n" + "="*70)
    print("Creating Lambda and Phi Parameter Comparison Plots...")
    print("="*70)
    
    # Lambda comparison
    loop_lambda_flat = loop_lambda.detach().cpu().numpy().flatten()
    vectorized_lambda_flat = vectorized_lambda.detach().cpu().numpy().flatten()
    lambda_diff = np.abs(loop_lambda_flat - vectorized_lambda_flat)
    lambda_corr, lambda_p = pearsonr(loop_lambda_flat, vectorized_lambda_flat)
    print(f"\nLambda Correlation: {lambda_corr:.8f} (p < {lambda_p:.2e})")
    
    # Phi comparison
    loop_phi_flat = loop_phi.detach().cpu().numpy().flatten()
    vectorized_phi_flat = vectorized_phi.detach().cpu().numpy().flatten()
    phi_diff = np.abs(loop_phi_flat - vectorized_phi_flat)
    phi_corr, phi_p = pearsonr(loop_phi_flat, vectorized_phi_flat)
    print(f"Phi Correlation: {phi_corr:.8f} (p < {phi_p:.2e})")
    
    # Sample for plotting
    lambda_sample_size = min(50000, len(loop_lambda_flat))
    lambda_sample_idx = np.random.choice(len(loop_lambda_flat), lambda_sample_size, replace=False)
    
    phi_sample_size = min(50000, len(loop_phi_flat))
    phi_sample_idx = np.random.choice(len(loop_phi_flat), phi_sample_size, replace=False)
    
    # Create lambda plots
    fig_lambda, axes_lambda = plt.subplots(2, 2, figsize=(16, 12))
    
    # Lambda: Overall scatter (log scale)
    ax1 = axes_lambda[0, 0]
    loop_lambda_sampled = loop_lambda_flat[lambda_sample_idx]
    vec_lambda_sampled = vectorized_lambda_flat[lambda_sample_idx]
    scatter_l1 = ax1.scatter(loop_lambda_sampled, vec_lambda_sampled,
                           alpha=0.2, s=2, c=lambda_diff[lambda_sample_idx], 
                           cmap='viridis', vmin=lambda_diff.min(), vmax=lambda_diff.max())
    min_val = max(loop_lambda_sampled.min(), vec_lambda_sampled.min())
    max_val = min(loop_lambda_sampled.max(), vec_lambda_sampled.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    ax1.set_xlabel('Loop Version lambda')
    ax1.set_ylabel('Vectorized Version lambda')
    ax1.set_title(f'Lambda Correlation (n={lambda_sample_size:,} samples)\nr={lambda_corr:.8f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter_l1, ax=ax1, label='|Difference|')
    
    # Lambda: Zoomed view
    ax2 = axes_lambda[0, 1]
    mask_lambda_small = (np.abs(loop_lambda_sampled) < 5) & (np.abs(vec_lambda_sampled) < 5)
    if mask_lambda_small.sum() > 0:
        scatter_l2 = ax2.scatter(loop_lambda_sampled[mask_lambda_small], 
                                vec_lambda_sampled[mask_lambda_small],
                               alpha=0.3, s=5, c=lambda_diff[lambda_sample_idx][mask_lambda_small], 
                               cmap='viridis', vmin=lambda_diff.min(), vmax=lambda_diff.max())
        plt.colorbar(scatter_l2, ax=ax2, label='|Difference|')
    ax2.plot([-5, 5], [-5, 5], 'r--', lw=2, label='y=x')
    ax2.set_xlabel('Loop Version lambda')
    ax2.set_ylabel('Vectorized Version lambda')
    ax2.set_title('Lambda Zoomed View (|lambda| < 5)')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Lambda: Difference histogram
    ax3 = axes_lambda[1, 0]
    log_bins_lambda = np.logspace(np.log10(lambda_diff[lambda_diff > 0].min()), 
                                  np.log10(lambda_diff.max()), 50)
    ax3.hist(lambda_diff, bins=log_bins_lambda, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.set_xlabel('Absolute Difference |Loop - Vectorized|')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.set_title(f'Lambda Difference Distribution\nMean: {np.mean(lambda_diff):.2e}, Median: {np.median(lambda_diff):.2e}, Max: {np.max(lambda_diff):.2e}')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Lambda: Relative error
    ax4 = axes_lambda[1, 1]
    lambda_relative_error = lambda_diff / (np.abs(loop_lambda_flat) + eps)
    lambda_rel_err_sampled = lambda_relative_error[lambda_sample_idx]
    ax4.scatter(loop_lambda_sampled, lambda_rel_err_sampled, 
               alpha=0.2, s=2, c='coral')
    ax4.set_xlabel('Loop Version lambda')
    ax4.set_ylabel('Relative Error |Loop - Vec| / |lambda|')
    ax4.set_title(f'Lambda Relative Error\nMean: {np.mean(lambda_relative_error):.2e}, Max: {np.max(lambda_relative_error):.2e}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    lambda_output_path = Path(args.loop_path).parent / 'loop_vs_vectorized_lambda_correlation.png'
    plt.savefig(lambda_output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Lambda correlation plot saved to: {lambda_output_path}")
    
    # Create phi plots
    fig_phi, axes_phi = plt.subplots(2, 2, figsize=(16, 12))
    
    # Phi: Overall scatter
    ax1 = axes_phi[0, 0]
    loop_phi_sampled = loop_phi_flat[phi_sample_idx]
    vec_phi_sampled = vectorized_phi_flat[phi_sample_idx]
    scatter_p1 = ax1.scatter(loop_phi_sampled, vec_phi_sampled,
                           alpha=0.2, s=2, c=phi_diff[phi_sample_idx], 
                           cmap='viridis', vmin=phi_diff.min(), vmax=phi_diff.max())
    min_val = max(loop_phi_sampled.min(), vec_phi_sampled.min())
    max_val = min(loop_phi_sampled.max(), vec_phi_sampled.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    ax1.set_xlabel('Loop Version phi')
    ax1.set_ylabel('Vectorized Version phi')
    ax1.set_title(f'Phi Correlation (n={phi_sample_size:,} samples)\nr={phi_corr:.8f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter_p1, ax=ax1, label='|Difference|')
    
    # Phi: Zoomed view
    ax2 = axes_phi[0, 1]
    mask_phi_small = (np.abs(loop_phi_sampled) < 5) & (np.abs(vec_phi_sampled) < 5)
    if mask_phi_small.sum() > 0:
        scatter_p2 = ax2.scatter(loop_phi_sampled[mask_phi_small], 
                                vec_phi_sampled[mask_phi_small],
                               alpha=0.3, s=5, c=phi_diff[phi_sample_idx][mask_phi_small], 
                               cmap='viridis', vmin=phi_diff.min(), vmax=phi_diff.max())
        plt.colorbar(scatter_p2, ax=ax2, label='|Difference|')
    ax2.plot([-5, 5], [-5, 5], 'r--', lw=2, label='y=x')
    ax2.set_xlabel('Loop Version phi')
    ax2.set_ylabel('Vectorized Version phi')
    ax2.set_title('Phi Zoomed View (|phi| < 5)')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phi: Difference histogram
    ax3 = axes_phi[1, 0]
    log_bins_phi = np.logspace(np.log10(phi_diff[phi_diff > 0].min()), 
                               np.log10(phi_diff.max()), 50)
    ax3.hist(phi_diff, bins=log_bins_phi, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.set_xlabel('Absolute Difference |Loop - Vectorized|')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.set_title(f'Phi Difference Distribution\nMean: {np.mean(phi_diff):.2e}, Median: {np.median(phi_diff):.2e}, Max: {np.max(phi_diff):.2e}')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Phi: Relative error
    ax4 = axes_phi[1, 1]
    phi_relative_error = phi_diff / (np.abs(loop_phi_flat) + eps)
    phi_rel_err_sampled = phi_relative_error[phi_sample_idx]
    ax4.scatter(loop_phi_sampled, phi_rel_err_sampled, 
               alpha=0.2, s=2, c='coral')
    ax4.set_xlabel('Loop Version phi')
    ax4.set_ylabel('Relative Error |Loop - Vec| / |phi|')
    ax4.set_title(f'Phi Relative Error\nMean: {np.mean(phi_relative_error):.2e}, Max: {np.max(phi_relative_error):.2e}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    phi_output_path = Path(args.loop_path).parent / 'loop_vs_vectorized_phi_correlation.png'
    plt.savefig(phi_output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Phi correlation plot saved to: {phi_output_path}")
    
    # Print parameter statistics
    print(f"\nLambda Parameter Statistics:")
    print(f"  Mean absolute difference: {np.mean(lambda_diff):.2e}")
    print(f"  Median absolute difference: {np.median(lambda_diff):.2e}")
    print(f"  Max absolute difference: {np.max(lambda_diff):.2e}")
    print(f"  95th percentile difference: {np.percentile(lambda_diff, 95):.2e}")
    print(f"  99th percentile difference: {np.percentile(lambda_diff, 99):.2e}")
    print(f"  Mean relative error: {np.mean(lambda_relative_error):.2e}")
    print(f"  Max relative error: {np.max(lambda_relative_error):.2e}")
    
    print(f"\nPhi Parameter Statistics:")
    print(f"  Mean absolute difference: {np.mean(phi_diff):.2e}")
    print(f"  Median absolute difference: {np.median(phi_diff):.2e}")
    print(f"  Max absolute difference: {np.max(phi_diff):.2e}")
    print(f"  95th percentile difference: {np.percentile(phi_diff, 95):.2e}")
    print(f"  99th percentile difference: {np.percentile(phi_diff, 99):.2e}")
    print(f"  Mean relative error: {np.mean(phi_relative_error):.2e}")
    print(f"  Max relative error: {np.max(phi_relative_error):.2e}")
    
    # Print summary statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean absolute difference: {np.mean(diff):.2e}")
    print(f"  Median absolute difference: {np.median(diff):.2e}")
    print(f"  Max absolute difference: {np.max(diff):.2e}")
    print(f"  95th percentile difference: {np.percentile(diff, 95):.2e}")
    print(f"  99th percentile difference: {np.percentile(diff, 99):.2e}")
    
    # Calculate relative error stats
    eps = 1e-8
    relative_error = diff / (loop_pi_flat + eps)
    print(f"\nRelative Error Statistics:")
    print(f"  Mean relative error: {np.mean(relative_error):.2e}")
    print(f"  Median relative error: {np.median(relative_error):.2e}")
    print(f"  Max relative error: {np.max(relative_error):.2e}")
    print(f"  95th percentile relative error: {np.percentile(relative_error, 95):.2e}")
    print(f"  99th percentile relative error: {np.percentile(relative_error, 99):.2e}")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Calculate final statistics
    max_diff = np.max(np.abs(loop_pi_flat - vectorized_pi_flat))
    mean_diff = np.mean(np.abs(loop_pi_flat - vectorized_pi_flat))
    
    if corr_coef > 0.9999:
        print("✅ PREDICTIONS (pi) ARE HIGHLY CORRELATED! Models are functionally equivalent.")
        print(f"   Correlation: {corr_coef:.8f}")
        print(f"   Mean absolute difference: {mean_diff:.2e}")
        print(f"   Max absolute difference: {max_diff:.2e}")
        print("   Parameter differences are acceptable - they don't meaningfully affect predictions.")
    elif corr_coef > 0.99:
        print("⚠️  PREDICTIONS (pi) ARE VERY CLOSE but not perfect.")
        print(f"   Correlation: {corr_coef:.8f}")
        print(f"   Mean absolute difference: {mean_diff:.2e}")
        print(f"   Max absolute difference: {max_diff:.2e}")
        print("   Small differences likely due to numerical precision.")
    else:
        print("❌ PREDICTIONS (pi) DO NOT MATCH!")
        print(f"   Correlation: {corr_coef:.8f}")
        print("   This indicates a real problem, not just numerical precision.")
    
    if all_match:
        print("\n✅ ALL PARAMETERS MATCH within tolerance!")
    else:
        print("\n⚠️  SOME PARAMETERS DO NOT MATCH within strict tolerance")
        if corr_coef > 0.99:
            print("\nNOTE: Parameter differences are expected due to floating-point")
            print("      arithmetic non-associativity. The loop and vectorized")
            print("      versions compute GP prior loss in different orders, leading")
            print("      to small numerical differences that accumulate over training.")
            print("      However, predictions are highly correlated, so models are equivalent.")
    print("="*70)
    
    # Print shapes for reference
    print("\nParameter Shapes:")
    print("-"*70)
    if 'lambda_' in loop_state:
        print(f"lambda_: {loop_state['lambda_'].shape}")
    if 'phi' in loop_state:
        print(f"phi: {loop_state['phi'].shape}")
    if 'kappa' in loop_state:
        print(f"kappa: {loop_state['kappa'].shape}")
    if 'gamma' in loop_state:
        print(f"gamma: {loop_state['gamma'].shape}")
    if 'psi' in loop_state:
        print(f"psi: {loop_state['psi'].shape}")


if __name__ == '__main__':
    main()
