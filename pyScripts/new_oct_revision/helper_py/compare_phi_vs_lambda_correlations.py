#!/usr/bin/env python3
"""
Compare Signature Correlations: Phi vs Lambda

This script demonstrates the difference between:
1. Phi correlations: Which signatures are associated with similar diseases?
2. Lambda correlations: Which signatures co-occur in the same patients?

These measure DIFFERENT things and can give very different results!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data


def compute_phi_correlation(phi):
    """
    Compute K×K correlation matrix using Phi (signature-disease associations)
    
    Measures: Which signatures are associated with similar diseases over time?
    """
    K, D, T = phi.shape
    
    # Flatten each signature: (K, D*T)
    phi_flat = phi.reshape(K, D * T)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(phi_flat)
    
    return corr_matrix


def compute_lambda_correlation(lambda_):
    """
    Compute K×K correlation matrix using Lambda (patient-signature logits)
    
    Measures: Which signatures co-occur in the same patients over time?
    """
    N, K, T = lambda_.shape
    
    # Reshape: (K, N*T) - each row is one signature across all patients and times
    lambda_flat = lambda_.transpose(1, 0, 2).reshape(K, N * T)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(lambda_flat)
    
    return corr_matrix


def compute_theta_correlation(thetas):
    """
    Compute K×K correlation matrix using Theta (patient-signature proportions)
    
    Note: Theta correlations will be negatively biased due to softmax constraint
    (sum to 1 across signatures for each patient-time)
    
    Measures: Which signatures co-occur in the same patients over time? (normalized)
    """
    N, K, T = thetas.shape
    
    # Reshape: (K, N*T)
    theta_flat = thetas.transpose(1, 0, 2).reshape(K, N * T)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(theta_flat)
    
    return corr_matrix


def compare_correlations(model_path=None, save_plots=True):
    """
    Compare Phi vs Lambda correlations for UKB data
    """
    print("="*80)
    print("COMPARING SIGNATURE CORRELATIONS: PHI vs LAMBDA")
    print("="*80)
    
    # Load UKB data
    print("\n1. Loading UKB data...")
    Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
    
    # Convert to numpy
    if isinstance(thetas_ukb, torch.Tensor):
        thetas_ukb = thetas_ukb.numpy()
    
    N, K, T = thetas_ukb.shape
    print(f"   Data shape: {N} patients, {K} signatures, {T} timepoints")
    
    # Load model to get phi and lambda
    if model_path is None:
        # Try to find UKB model
        model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam.pt'
    
    print(f"\n2. Loading model from: {model_path}")
    try:
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Get phi and lambda from model
        if 'model_state_dict' in model_data:
            model_state = model_data['model_state_dict']
            phi = model_state.get('phi')
            lambda_ = model_state.get('lambda_')
            
            if phi is not None:
                if hasattr(phi, 'detach'):
                    phi = phi.detach().numpy()
                print(f"   Phi shape: {phi.shape}")
            else:
                print("   ⚠️  Phi not found in model")
                phi = None
            
            if lambda_ is not None:
                if hasattr(lambda_, 'detach'):
                    lambda_ = lambda_.detach().numpy()
                print(f"   Lambda shape: {lambda_.shape}")
            else:
                print("   ⚠️  Lambda not found in model")
                lambda_ = None
        else:
            print("   ⚠️  model_state_dict not found")
            phi = None
            lambda_ = None
            
    except Exception as e:
        print(f"   ⚠️  Could not load model: {e}")
        phi = None
        lambda_ = None
    
    # Compute correlations
    print("\n3. Computing correlation matrices...")
    
    # Theta correlations (always available)
    print("   Computing Theta correlations...")
    corr_theta = compute_theta_correlation(thetas_ukb)
    print(f"   Theta correlation matrix: {corr_theta.shape}")
    print(f"   Mean absolute correlation: {np.abs(corr_theta).mean():.3f}")
    print(f"   Note: Theta correlations are negatively biased due to softmax constraint")
    
    # Lambda correlations
    if lambda_ is not None:
        print("\n   Computing Lambda correlations...")
        corr_lambda = compute_lambda_correlation(lambda_)
        print(f"   Lambda correlation matrix: {corr_lambda.shape}")
        print(f"   Mean absolute correlation: {np.abs(corr_lambda).mean():.3f}")
    else:
        corr_lambda = None
        print("\n   ⚠️  Skipping Lambda correlations (not available)")
    
    # Phi correlations
    if phi is not None:
        print("\n   Computing Phi correlations...")
        corr_phi = compute_phi_correlation(phi)
        print(f"   Phi correlation matrix: {corr_phi.shape}")
        print(f"   Mean absolute correlation: {np.abs(corr_phi).mean():.3f}")
    else:
        corr_phi = None
        print("\n   ⚠️  Skipping Phi correlations (not available)")
    
    # Compare correlations
    print("\n4. Comparing correlation matrices...")
    
    if corr_lambda is not None and corr_phi is not None:
        # Compare Phi vs Lambda
        # Remove diagonal (self-correlation = 1.0)
        mask = ~np.eye(K, dtype=bool)
        phi_vals = corr_phi[mask]
        lambda_vals = corr_lambda[mask]
        
        correlation_between_methods = np.corrcoef(phi_vals, lambda_vals)[0, 1]
        print(f"   Correlation between Phi and Lambda correlation matrices: {correlation_between_methods:.3f}")
        print(f"   (Low correlation means they measure different things!)")
        
        # Compare Theta vs Lambda
        theta_vals = corr_theta[mask]
        correlation_theta_lambda = np.corrcoef(theta_vals, lambda_vals)[0, 1]
        print(f"   Correlation between Theta and Lambda correlation matrices: {correlation_theta_lambda:.3f}")
    
    # Create visualization
    print("\n5. Creating visualizations...")
    
    n_plots = sum([corr_theta is not None, corr_lambda is not None, corr_phi is not None])
    
    if n_plots == 0:
        print("   ⚠️  No correlation matrices to plot")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot Theta correlations
    if corr_theta is not None:
        ax = axes[plot_idx]
        im = ax.imshow(corr_theta, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Theta Correlations\n(Patient Co-occurrence, Normalized)', fontweight='bold')
        ax.set_xlabel('Signature Index')
        ax.set_ylabel('Signature Index')
        plt.colorbar(im, ax=ax, label='Correlation')
        plot_idx += 1
    
    # Plot Lambda correlations
    if corr_lambda is not None:
        ax = axes[plot_idx]
        im = ax.imshow(corr_lambda, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Lambda Correlations\n(Patient Co-occurrence, Logits)', fontweight='bold')
        ax.set_xlabel('Signature Index')
        ax.set_ylabel('Signature Index')
        plt.colorbar(im, ax=ax, label='Correlation')
        plot_idx += 1
    
    # Plot Phi correlations
    if corr_phi is not None:
        ax = axes[plot_idx]
        im = ax.imshow(corr_phi, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Phi Correlations\n(Biological Similarity, Disease Associations)', fontweight='bold')
        ax.set_xlabel('Signature Index')
        ax.set_ylabel('Signature Index')
        plt.colorbar(im, ax=ax, label='Correlation')
        plot_idx += 1
    
    plt.suptitle('Signature Correlation Comparison: Different Methods Measure Different Things!', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_plots:
        save_path = 'signature_correlation_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved to: {save_path}")
    
    plt.show()
    
    # Create comparison scatter plot if both available
    if corr_lambda is not None and corr_phi is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        mask = ~np.eye(K, dtype=bool)
        phi_vals = corr_phi[mask]
        lambda_vals = corr_lambda[mask]
        
        ax.scatter(phi_vals, lambda_vals, alpha=0.5, s=20)
        ax.set_xlabel('Phi Correlation (Biological Similarity)', fontsize=12)
        ax.set_ylabel('Lambda Correlation (Patient Co-occurrence)', fontsize=12)
        ax.set_title('Phi vs Lambda Correlations\n(Each point is a signature pair)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add diagonal line
        ax.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='y=x')
        ax.legend()
        
        # Add correlation text
        corr = np.corrcoef(phi_vals, lambda_vals)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            save_path = 'phi_vs_lambda_correlation_scatter.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey Differences:")
    print("1. Phi correlations: Which signatures affect similar diseases? (Biological mechanisms)")
    print("2. Lambda correlations: Which signatures co-occur in same patients? (Co-occurrence patterns)")
    print("3. Theta correlations: Same as Lambda but normalized (negatively biased due to softmax)")
    print("\nThese measure DIFFERENT things and can give very different results!")
    
    return {
        'corr_phi': corr_phi,
        'corr_lambda': corr_lambda,
        'corr_theta': corr_theta
    }


if __name__ == "__main__":
    results = compare_correlations()

