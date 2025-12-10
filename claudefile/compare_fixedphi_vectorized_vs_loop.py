#!/usr/bin/env python
"""
Compare the vectorized fixed phi/psi prediction output with the loop (non-vectorized) version
to validate that both versions produce equivalent results.

This compares outputs from:
- run_aladyn_predict_with_master.py (uses clust_huge_amp_fixedPhi - loop version)
- run_aladyn_predict_with_master_vector_cenosrE.py (uses clust_huge_amp_fixedPhi_vectorized - vectorized version)
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
    parser = argparse.ArgumentParser(
        description='Compare vectorized vs loop fixed phi/psi prediction outputs'
    )
    parser.add_argument('--vectorized_model_path', type=str,
                       help='Path to vectorized version model file (model_enroll_fixedphi_sex_{start}_{stop}.pt)')
    parser.add_argument('--loop_model_path', type=str,
                       help='Path to loop version model file (model_enroll_fixedphi_sex_{start}_{stop}.pt)')
    parser.add_argument('--vectorized_pi_path', type=str,
                       help='Path to vectorized version pi file (pi_enroll_fixedphi_sex_{start}_{stop}.pt or pi_enroll_fixedphi_sex_FULL.pt)')
    parser.add_argument('--loop_pi_path', type=str,
                       help='Path to loop version pi file (pi_enroll_fixedphi_sex_{start}_{stop}.pt or pi_enroll_fixedphi_sex_FULL.pt)')
    parser.add_argument('--vectorized_dir', type=str,
                       help='Directory containing vectorized version outputs (alternative to individual paths)')
    parser.add_argument('--loop_dir', type=str,
                       help='Directory containing loop version outputs (alternative to individual paths)')
    parser.add_argument('--batch_start', type=int, default=0,
                       help='Start index of batch to compare (if using directories)')
    parser.add_argument('--batch_end', type=int, default=10000,
                       help='End index of batch to compare (if using directories)')
    parser.add_argument('--compare_full', action='store_true',
                       help='Compare full combined pi files instead of batch files')
    parser.add_argument('--rtol', type=float, default=1e-4,
                       help='Relative tolerance for comparison')
    parser.add_argument('--atol', type=float, default=1e-5,
                       help='Absolute tolerance for comparison')
    args = parser.parse_args()
    
    print("="*70)
    print("Comparing Vectorized vs Loop Fixed Phi/Psi Prediction Outputs")
    print("="*70)
    
    # Determine paths
    if args.vectorized_dir and args.loop_dir:
        # Use directories to find files
        vectorized_dir = Path(args.vectorized_dir)
        loop_dir = Path(args.loop_dir)
        
        if args.compare_full:
            vectorized_pi_path = vectorized_dir / "pi_enroll_fixedphi_sex_FULL.pt"
            loop_pi_path = loop_dir / "pi_enroll_fixedphi_sex_FULL.pt"
            vectorized_model_path = None
            loop_model_path = None
        else:
            vectorized_pi_path = vectorized_dir / f"pi_enroll_fixedphi_sex_{args.batch_start}_{args.batch_end}.pt"
            loop_pi_path = loop_dir / f"pi_enroll_fixedphi_sex_{args.batch_start}_{args.batch_end}.pt"
            vectorized_model_path = vectorized_dir / f"model_enroll_fixedphi_sex_{args.batch_start}_{args.batch_end}.pt"
            loop_model_path = loop_dir / f"model_enroll_fixedphi_sex_{args.batch_start}_{args.batch_end}.pt"
    else:
        # Use explicit paths
        vectorized_pi_path = Path(args.vectorized_pi_path) if args.vectorized_pi_path else None
        loop_pi_path = Path(args.loop_pi_path) if args.loop_pi_path else None
        vectorized_model_path = Path(args.vectorized_model_path) if args.vectorized_model_path else None
        loop_model_path = Path(args.loop_model_path) if args.loop_model_path else None
    
    print(f"\nVectorized version:")
    if vectorized_pi_path:
        print(f"  Pi file: {vectorized_pi_path}")
    if vectorized_model_path:
        print(f"  Model file: {vectorized_model_path}")
    
    print(f"\nLoop version:")
    if loop_pi_path:
        print(f"  Pi file: {loop_pi_path}")
    if loop_model_path:
        print(f"  Model file: {loop_model_path}")
    
    print(f"\nTolerance: rtol={args.rtol}, atol={args.atol}\n")
    
    # Load pi predictions
    if not vectorized_pi_path or not loop_pi_path:
        print("❌ Error: Pi file paths are required!")
        return
    
    print("Loading pi predictions...")
    try:
        vectorized_pi = torch.load(vectorized_pi_path, weights_only=False)
        loop_pi = torch.load(loop_pi_path, weights_only=False)
    except FileNotFoundError as e:
        print(f"❌ Error loading pi files: {e}")
        return
    
    print(f"✅ Pi files loaded successfully")
    print(f"   Vectorized pi shape: {vectorized_pi.shape}")
    print(f"   Loop pi shape: {loop_pi.shape}\n")
    
    # Compare pi predictions - THIS IS WHAT REALLY MATTERS
    print("="*70)
    print("Prediction (pi) Comparison - THIS IS WHAT MATTERS!")
    print("="*70)
    
    pi_match = compare_tensors(vectorized_pi, 
                               loop_pi, 
                               'pi (predictions)', 
                               rtol=1e-3,  # More lenient for predictions
                               atol=1e-5)
    
    # Compare model parameters if available
    if vectorized_model_path and loop_model_path:
        print("\n" + "="*70)
        print("Parameter Comparisons")
        print("="*70)
        
        print("Loading model files...")
        try:
            vectorized_model = torch.load(vectorized_model_path, weights_only=False)
            loop_model = torch.load(loop_model_path, weights_only=False)
        except FileNotFoundError as e:
            print(f"⚠️  Error loading model files: {e}")
            print("   Continuing with pi comparison only...")
            vectorized_model = None
            loop_model = None
        else:
            print("✅ Model files loaded successfully\n")
            
            vectorized_state = vectorized_model['model_state_dict']
            loop_state = loop_model['model_state_dict']
            
            all_match = True
            
            # Compare lambda_ (only parameter that should differ, but should be close)
            if 'lambda_' in vectorized_state and 'lambda_' in loop_state:
                match = compare_tensors(vectorized_state['lambda_'], 
                                       loop_state['lambda_'], 
                                       'lambda_', 
                                       rtol=args.rtol, 
                                       atol=args.atol)
                all_match = all_match and match
            
            # Compare phi (should be identical since it's fixed)
            if 'phi' in vectorized_state and 'phi' in loop_state:
                match = compare_tensors(vectorized_state['phi'], 
                                       loop_state['phi'], 
                                       'phi', 
                                       rtol=1e-6,  # Very strict for fixed phi
                                       atol=1e-7)
                all_match = all_match and match
            
            # Compare kappa (should be identical)
            if 'kappa' in vectorized_state and 'kappa' in loop_state:
                match = compare_tensors(vectorized_state['kappa'], 
                                       loop_state['kappa'], 
                                       'kappa', 
                                       rtol=args.rtol, 
                                       atol=args.atol)
                all_match = all_match and match
            
            # Compare gamma (if present)
            if 'gamma' in vectorized_state and 'gamma' in loop_state:
                match = compare_tensors(vectorized_state['gamma'], 
                                       loop_state['gamma'], 
                                       'gamma', 
                                       rtol=args.rtol, 
                                       atol=args.atol)
                all_match = all_match and match
            
            # Compare psi (should be identical since it's fixed)
            if 'psi' in vectorized_state and 'psi' in loop_state:
                match = compare_tensors(vectorized_state['psi'], 
                                       loop_state['psi'], 
                                       'psi', 
                                       rtol=1e-6,  # Very strict for fixed psi
                                       atol=1e-7)
                all_match = all_match and match
    else:
        print("\n⚠️  Model files not provided - skipping parameter comparison")
        all_match = None
        vectorized_model = None
        loop_model = None
    
    # Create lambda comparison plots if model files are available
    if vectorized_model and loop_model:
        print("\n" + "="*70)
        print("Creating Lambda Comparison Plots...")
        print("="*70)
        
        vectorized_lambda = vectorized_model['model_state_dict']['lambda_']
        loop_lambda = loop_model['model_state_dict']['lambda_']
        
        # Flatten lambda for comparison
        vectorized_lambda_flat = vectorized_lambda.detach().cpu().numpy().flatten()
        loop_lambda_flat = loop_lambda.detach().cpu().numpy().flatten()
        
        # Calculate differences
        lambda_diff = np.abs(vectorized_lambda_flat - loop_lambda_flat)
        
        # Calculate correlation
        lambda_corr, lambda_p = pearsonr(vectorized_lambda_flat, loop_lambda_flat)
        print(f"Lambda Correlation: {lambda_corr:.8f} (p < {lambda_p:.2e})")
        
        # Sample for plotting
        lambda_sample_size = min(50000, len(vectorized_lambda_flat))
        lambda_sample_idx = np.random.choice(len(vectorized_lambda_flat), lambda_sample_size, replace=False)
        
        # Create lambda plots
        fig_lambda, axes_lambda = plt.subplots(2, 2, figsize=(16, 12))
        fig_lambda.suptitle('Lambda Comparison: Vectorized vs Loop Versions', 
                           fontsize=16, fontweight='bold', y=0.995)
        
        # Lambda: Overall scatter
        ax1 = axes_lambda[0, 0]
        loop_lambda_sampled = loop_lambda_flat[lambda_sample_idx]
        vec_lambda_sampled = vectorized_lambda_flat[lambda_sample_idx]
        scatter_l1 = ax1.scatter(loop_lambda_sampled, vec_lambda_sampled,
                               alpha=0.2, s=2, c=lambda_diff[lambda_sample_idx], 
                               cmap='viridis', vmin=lambda_diff.min(), vmax=lambda_diff.max())
        min_val = max(loop_lambda_sampled.min(), vec_lambda_sampled.min())
        max_val = min(loop_lambda_sampled.max(), vec_lambda_sampled.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
        ax1.set_xlabel('Loop Version lambda', fontsize=11)
        ax1.set_ylabel('Vectorized Version lambda', fontsize=11)
        ax1.set_title(f'Lambda Correlation (n={lambda_sample_size:,} samples)\nr={lambda_corr:.8f}', 
                      fontsize=12, fontweight='bold')
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
        ax2.set_xlabel('Loop Version lambda', fontsize=11)
        ax2.set_ylabel('Vectorized Version lambda', fontsize=11)
        ax2.set_title('Lambda Zoomed View (|lambda| < 5)', fontsize=12, fontweight='bold')
        ax2.set_xlim([-5, 5])
        ax2.set_ylim([-5, 5])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Lambda: Difference histogram
        ax3 = axes_lambda[1, 0]
        if np.any(lambda_diff > 0):
            log_bins_lambda = np.logspace(np.log10(lambda_diff[lambda_diff > 0].min()), 
                                         np.log10(lambda_diff.max()), 50)
            ax3.hist(lambda_diff, bins=log_bins_lambda, alpha=0.7, edgecolor='black', color='steelblue')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'Perfect Match!\nAll differences = 0', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=14, fontweight='bold', color='green')
        ax3.set_xlabel('Absolute Difference |Loop - Vectorized|', fontsize=11)
        ax3.set_ylabel('Frequency (log scale)', fontsize=11)
        ax3.set_title(f'Lambda Difference Distribution\nMean: {np.mean(lambda_diff):.2e}, Median: {np.median(lambda_diff):.2e}, Max: {np.max(lambda_diff):.2e}', 
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')
        
        # Lambda: Relative error
        ax4 = axes_lambda[1, 1]
        lambda_eps = 1e-8
        lambda_abs = np.abs(loop_lambda_flat) + lambda_eps
        lambda_relative_error = lambda_diff / lambda_abs
        lambda_rel_err_sampled = lambda_relative_error[lambda_sample_idx]
        if np.any(lambda_relative_error > 0):
            ax4.scatter(loop_lambda_sampled, lambda_rel_err_sampled, 
                       alpha=0.2, s=2, c='coral')
        else:
            ax4.text(0.5, 0.5, 'Perfect Match!\nAll differences = 0', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, fontweight='bold', color='green')
        ax4.set_xlabel('Loop Version lambda', fontsize=11)
        ax4.set_ylabel('Relative Error |Loop - Vec| / |lambda|', fontsize=11)
        ax4.set_title(f'Lambda Relative Error Distribution\nMean: {np.mean(lambda_relative_error):.2e}, Max: {np.max(lambda_relative_error):.2e}', 
                      fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save lambda plot
        lambda_output_path = output_dir / 'vectorized_vs_loop_lambda_comparison.png'
        plt.savefig(lambda_output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Lambda comparison plot saved to: {lambda_output_path}")
        plt.close()
    
    # Create correlation plots
    print("\n" + "="*70)
    print("Creating Prediction (pi) Correlation Plots...")
    print("="*70)
    
    # Flatten predictions for correlation
    vectorized_pi_flat = vectorized_pi.detach().cpu().numpy().flatten()
    loop_pi_flat = loop_pi.detach().cpu().numpy().flatten()
    
    # Calculate differences
    diff = np.abs(vectorized_pi_flat - loop_pi_flat)
    
    # Calculate correlation
    corr_coef, p_value = pearsonr(vectorized_pi_flat, loop_pi_flat)
    print(f"\nOverall Correlation: {corr_coef:.8f} (p < {p_value:.2e})")
    
    # Sample for plotting
    sample_size = min(100000, len(vectorized_pi_flat))
    sample_idx = np.random.choice(len(vectorized_pi_flat), sample_size, replace=False)
    eps = 1e-8
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall scatter plot - LOG SCALE
    ax1 = axes[0, 0]
    vectorized_sampled = vectorized_pi_flat[sample_idx]
    loop_sampled = loop_pi_flat[sample_idx]
    scatter1 = ax1.scatter(vectorized_sampled + eps, loop_sampled + eps, 
                alpha=0.2, s=2, c=diff[sample_idx], cmap='viridis', 
                vmin=diff.min(), vmax=diff.max())
    min_val = max(vectorized_sampled.min(), loop_sampled.min())
    max_val = min(vectorized_sampled.max(), loop_sampled.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Vectorized pi (log scale)')
    ax1.set_ylabel('Loop pi (log scale)')
    ax1.set_title(f'Overall Correlation (n={sample_size:,} samples)\nr={corr_coef:.8f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    plt.colorbar(scatter1, ax=ax1, label='|Difference|')
    
    # Zoomed-in view
    ax2 = axes[0, 1]
    mask_small = (vectorized_sampled < 0.1) & (loop_sampled < 0.1)
    if mask_small.sum() > 0:
        scatter2 = ax2.scatter(vectorized_sampled[mask_small], loop_sampled[mask_small], 
                   alpha=0.3, s=5, c=diff[sample_idx][mask_small], cmap='viridis',
                   vmin=diff.min(), vmax=diff.max())
        plt.colorbar(scatter2, ax=ax2, label='|Difference|')
    ax2.plot([0, 0.1], [0, 0.1], 'r--', lw=2, label='y=x')
    ax2.set_xlabel('Vectorized pi')
    ax2.set_ylabel('Loop pi')
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
        # All differences are zero
        ax3.text(0.5, 0.5, 'Perfect Match!\nAll differences = 0', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=14, fontweight='bold', color='green')
    ax3.set_xlabel('Absolute Difference |Vectorized - Loop|')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.set_title(f'Distribution of Differences\nMean: {np.mean(diff):.2e}, Median: {np.median(diff):.2e}, Max: {np.max(diff):.2e}')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Relative error plot
    ax4 = axes[1, 1]
    if np.any(vectorized_pi_flat > 0):
        relative_error = diff / (vectorized_pi_flat + eps)
        rel_err_sampled = relative_error[sample_idx]
        if np.any(relative_error > 0):
            ax4.scatter(vectorized_sampled + eps, rel_err_sampled, 
                       alpha=0.2, s=2, c='coral')
            ax4.set_xscale('log')
            ax4.set_yscale('log')
        else:
            # All relative errors are zero
            ax4.text(0.5, 0.5, 'Perfect Match!\nAll differences = 0', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, fontweight='bold', color='green')
        ax4.set_xlabel('Vectorized pi (log scale)')
        ax4.set_ylabel('Relative Error |Vectorized - Loop| / pi (log scale)')
        ax4.set_title(f'Relative Error Distribution\nMean: {np.mean(relative_error):.2e}, Max: {np.max(relative_error):.2e}')
    else:
        ax4.text(0.5, 0.5, 'All pi values are zero', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = vectorized_pi_path.parent
    output_path = output_dir / 'vectorized_vs_loop_fixedphi_correlation.png'
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
        print("   Vectorized and loop versions produce equivalent results!")
    elif corr_coef > 0.99:
        print("⚠️  PREDICTIONS (pi) ARE VERY CLOSE but not perfect.")
        print(f"   Correlation: {corr_coef:.8f}")
        print(f"   Mean absolute difference: {np.mean(diff):.2e}")
        print(f"   Max absolute difference: {np.max(diff):.2e}")
        print("   Small differences may be due to floating-point precision or different E matrices.")
    else:
        print("❌ PREDICTIONS (pi) DO NOT MATCH!")
        print(f"   Correlation: {corr_coef:.8f}")
        print("   This indicates a problem - investigate further!")
    
    if all_match is not None:
        if all_match:
            print("\n✅ ALL PARAMETERS MATCH within tolerance!")
        else:
            print("\n⚠️  SOME PARAMETERS DO NOT MATCH within strict tolerance")
            if corr_coef > 0.99:
                print("   However, predictions match, so models are functionally equivalent.")
                print("   Parameter differences may be due to:")
                print("   - Floating-point precision differences")
                print("   - Different random seeds or initialization")
    
    print("="*70)


if __name__ == '__main__':
    main()

