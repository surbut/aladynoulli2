"""
LAMBDA COMPARISON: IPW Recovery Analysis (90% Women Dropped)

This analysis compares lambda (individual signature loadings) across:
- Biased (no IPW) vs Biased (with IPW)

This shows how IPW correction affects lambda in the biased sample,
demonstrating the recovery effect of IPW weighting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from pathlib import Path

# Add path for utils
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred, softmax_by_k

print("="*80)
print("LAMBDA COMPARISON: IPW Recovery Analysis (Individual Level)")
print("="*80)

# Data directory - check both locations
results_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
ipw_dir = results_dir / 'ipwbatchrun113'

# Try ipwbatchrun113 first, then fall back to results/batch_*
if ipw_dir.exists():
    batch_base_dir = ipw_dir
    print(f"\nUsing data from: {ipw_dir}")
else:
    batch_base_dir = results_dir
    print(f"\nUsing data from: {results_dir}/batch_*")

# Load individual-level lambda from model files
lambda_full_list = []
lambda_biased_list = []
lambda_biased_ipw_list = []

for batch_idx in range(1, 6):  # batches 1-5
    batch_dir = batch_base_dir / f'batch_{batch_idx}'
    model_full_path = batch_dir / 'model_full.pt'
    model_biased_path = batch_dir / 'model_biased.pt'
    model_biased_ipw_path = batch_dir / 'model_biased_ipw.pt'
    
    if model_full_path.exists() and model_biased_path.exists() and model_biased_ipw_path.exists():
        # Load models
        print(f"  Loading batch {batch_idx}...")
        full_data = torch.load(model_full_path, weights_only=False, map_location='cpu')
        biased_data = torch.load(model_biased_path, weights_only=False, map_location='cpu')
        biased_ipw_data = torch.load(model_biased_ipw_path, weights_only=False, map_location='cpu')
        
        # Extract lambda
        if 'lambda' in full_data:
            lambda_full_batch = full_data['lambda'].detach()  # [N_full, K, T]
        elif 'model_state_dict' in full_data:
            lambda_full_batch = full_data['model_state_dict']['lambda_'].detach()
        else:
            print(f"    ⚠️  Batch {batch_idx}: Could not find lambda in model_full.pt")
            continue
            
        if 'lambda' in biased_data:
            lambda_biased_batch = biased_data['lambda'].detach()  # [N_biased, K, T]
        elif 'model_state_dict' in biased_data:
            lambda_biased_batch = biased_data['model_state_dict']['lambda_'].detach()
        else:
            print(f"    ⚠️  Batch {batch_idx}: Could not find lambda in model_biased.pt")
            continue
            
        if 'lambda' in biased_ipw_data:
            lambda_biased_ipw_batch = biased_ipw_data['lambda'].detach()  # [N_biased, K, T]
        elif 'model_state_dict' in biased_ipw_data:
            lambda_biased_ipw_batch = biased_ipw_data['model_state_dict']['lambda_'].detach()
        else:
            print(f"    ⚠️  Batch {batch_idx}: Could not find lambda in model_biased_ipw.pt")
            continue
        
        lambda_full_list.append(lambda_full_batch)
        lambda_biased_list.append(lambda_biased_batch)
        lambda_biased_ipw_list.append(lambda_biased_ipw_batch)
        print(f"    ✓ Batch {batch_idx}: Full {lambda_full_batch.shape}, Biased {lambda_biased_batch.shape}, Biased IPW {lambda_biased_ipw_batch.shape}")

if len(lambda_full_list) == 0:
    print("\n⚠️  No model files found. Please check the directory paths.")
    print(f"   Tried: {ipw_dir}/batch_*")
    print(f"   Tried: {results_dir}/batch_*")
else:
    # Concatenate across batches to get full individual-level data
    # Note: Since batches have different individuals, we'll compare averaged lambda per batch
    # OR we can concatenate and compare all individuals
    
    # Option 1: Compare averaged lambda (mean over individuals) per batch, then average across batches
    lambda_full_avg_per_batch = [lam.mean(dim=0).numpy() for lam in lambda_full_list]  # List of [K, T]
    lambda_biased_avg_per_batch = [lam.mean(dim=0).numpy() for lam in lambda_biased_list]
    lambda_biased_ipw_avg_per_batch = [lam.mean(dim=0).numpy() for lam in lambda_biased_ipw_list]
    
    lambda_full_avg = np.mean(lambda_full_avg_per_batch, axis=0)  # [K, T]
    lambda_biased_avg = np.mean(lambda_biased_avg_per_batch, axis=0)
    lambda_biased_ipw_avg = np.mean(lambda_biased_ipw_avg_per_batch, axis=0)
    
    # Option 2: Also compare individual-level (concatenate all batches)
    lambda_full_all = torch.cat(lambda_full_list, dim=0)  # [N_total, K, T]
    lambda_biased_all = torch.cat(lambda_biased_list, dim=0)  # [N_biased_total, K, T]
    lambda_biased_ipw_all = torch.cat(lambda_biased_ipw_list, dim=0)  # [N_biased_total, K, T]
    
    print(f"\n✓ Loaded {len(lambda_full_list)} batches")
    print(f"  Total individuals - Full: {lambda_full_all.shape[0]}, Biased: {lambda_biased_all.shape[0]}")
    print(f"  Lambda shape: {lambda_full_all.shape[1:]} (K signatures, T timepoints)")
    
    # Since we have different numbers of individuals, we'll compare:
    # 1. Averaged lambda (mean over individuals) - shape [K, T]
    # 2. Individual-level comparison using all individuals (but note different sample sizes)
    
    # ========================================================================
    # COMPARISON: Biased (no IPW) vs Biased (with IPW)
    # ========================================================================
    
    # Averaged Lambda comparison
    lambda_biased_flat_avg = lambda_biased_avg.flatten()
    lambda_biased_ipw_flat_avg = lambda_biased_ipw_avg.flatten()
    
    avg_corr = np.corrcoef(lambda_biased_flat_avg, lambda_biased_ipw_flat_avg)[0, 1]
    avg_mean_diff = np.abs(lambda_biased_flat_avg - lambda_biased_ipw_flat_avg).mean()
    
    # Individual-level Lambda comparison
    # Both biased and biased_ipw have the same individuals (same biased sample)
    lambda_biased_flat = lambda_biased_all.numpy().flatten()
    lambda_biased_ipw_flat = lambda_biased_ipw_all.numpy().flatten()
    
    # Match sample sizes (should be same, but just in case)
    min_N = min(len(lambda_biased_flat), len(lambda_biased_ipw_flat))
    lambda_biased_matched = lambda_biased_flat[:min_N]
    lambda_biased_ipw_matched = lambda_biased_ipw_flat[:min_N]
    
    individual_corr = np.corrcoef(lambda_biased_matched, lambda_biased_ipw_matched)[0, 1]
    individual_mean_diff = np.abs(lambda_biased_matched - lambda_biased_ipw_matched).mean()
    
    print(f"\nAveraged Lambda Comparison (K×T):")
    print(f"  Biased (no IPW) vs Biased (with IPW):     Correlation = {avg_corr:.6f}, Mean diff = {avg_mean_diff:.6f}")
    
    print(f"\nIndividual Lambda Comparison (N×K×T):")
    print(f"  Biased (no IPW) vs Biased (with IPW):     Correlation = {individual_corr:.6f}, Mean diff = {individual_mean_diff:.6f}")
    
    # ========================================================================
    # CREATE COMPREHENSIVE FIGURE: 2 rows x 3 columns
    # Row 1: Averaged Lambda, Row 2: Individual Lambda
    # Col 1: Scatter, Col 2: Zoomed scatter, Col 3: Histogram
    # ========================================================================
    
    fig = plt.figure(figsize=(18, 12))
    
    # ========== ROW 1: AVERAGED LAMBDA ==========
    
    # Panel 1: Averaged Lambda scatter - Biased (no IPW) vs Biased (with IPW)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(lambda_biased_flat_avg, lambda_biased_ipw_flat_avg, alpha=0.5, s=2, color='purple', edgecolors='none')
    lims = [min(lambda_biased_flat_avg.min(), lambda_biased_ipw_flat_avg.min()), 
            max(lambda_biased_flat_avg.max(), lambda_biased_ipw_flat_avg.max())]
    ax1.plot(lims, lims, 'r--', alpha=0.8, linewidth=2, label='y=x')
    ax1.set_xlabel('Biased (no IPW) Lambda (Averaged)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Biased (with IPW) Lambda (Averaged)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Averaged Lambda: Biased (no IPW) vs Biased (with IPW)\nCorrelation: {avg_corr:.6f}', 
                 fontsize=12, fontweight='bold', pad=8)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=10)
    
    # Panel 2: Averaged Lambda difference histogram
    ax2 = plt.subplot(2, 3, 2)
    diff_avg = lambda_biased_flat_avg - lambda_biased_ipw_flat_avg
    ax2.hist(diff_avg, bins=100, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax2.set_xlabel('Lambda Difference (Biased no IPW - Biased with IPW)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'Distribution of Averaged Lambda Differences\nMean: {avg_mean_diff:.6f}', 
                 fontsize=12, fontweight='bold', pad=8)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.tick_params(labelsize=10)
    
    # Panel 3: Averaged Lambda trajectories comparison (sample signatures)
    ax3 = plt.subplot(2, 3, 3)
    sample_sigs = [0, 5, 10, 15]
    for sig_idx in sample_sigs:
        if sig_idx < lambda_biased_avg.shape[0]:
            biased_traj = lambda_biased_avg[sig_idx, :]
            biased_ipw_traj = lambda_biased_ipw_avg[sig_idx, :]
            ax3.plot(biased_traj, label=f'Sig {sig_idx} (no IPW)', alpha=0.7, linewidth=1.5)
            ax3.plot(biased_ipw_traj, label=f'Sig {sig_idx} (with IPW)', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Lambda Value', fontsize=11, fontweight='bold')
    ax3.set_title('Sample Signature Trajectories\n(Averaged lambda)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=10)
    
    # ========== ROW 2: INDIVIDUAL LAMBDA ==========
    
    # Panel 4: Individual Lambda scatter - Biased (no IPW) vs Biased (with IPW)
    ax4 = plt.subplot(2, 3, 4)
    n_sample = min(50000, len(lambda_biased_matched))
    sample_idx = np.random.choice(len(lambda_biased_matched), n_sample, replace=False)
    ax4.scatter(lambda_biased_matched[sample_idx], lambda_biased_ipw_matched[sample_idx], 
                alpha=0.1, s=0.5, color='purple', edgecolors='none')
    lims_ind = [min(lambda_biased_matched.min(), lambda_biased_ipw_matched.min()), 
                max(lambda_biased_matched.max(), lambda_biased_ipw_matched.max())]
    ax4.plot(lims_ind, lims_ind, 'r--', alpha=0.8, linewidth=2, label='y=x')
    ax4.set_xlabel('Biased (no IPW) Lambda (Individual)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Biased (with IPW) Lambda (Individual)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Individual Lambda: Biased (no IPW) vs Biased (with IPW)\nCorrelation: {individual_corr:.6f}\n(n={n_sample:,} sampled)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(labelsize=10)
    
    # Panel 5: Individual Lambda difference histogram
    ax5 = plt.subplot(2, 3, 5)
    diff_ind = lambda_biased_matched - lambda_biased_ipw_matched
    ax5.hist(diff_ind, bins=100, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
    ax5.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax5.set_xlabel('Lambda Difference (Biased no IPW - Biased with IPW)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title(f'Distribution of Individual Lambda Differences\nMean: {individual_mean_diff:.6f}', 
                 fontsize=12, fontweight='bold', pad=8)
    ax5.legend(fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax5.tick_params(labelsize=10)
    
    # Panel 6: Correlation by signature
    ax6 = plt.subplot(2, 3, 6)
    sig_correlations = []
    for sig_idx in range(lambda_biased_avg.shape[0]):
        sig_biased = lambda_biased_avg[sig_idx, :].flatten()
        sig_biased_ipw = lambda_biased_ipw_avg[sig_idx, :].flatten()
        sig_correlations.append(np.corrcoef(sig_biased, sig_biased_ipw)[0, 1])
    
    ax6.bar(range(len(sig_correlations)), sig_correlations, alpha=0.7, edgecolor='black', color='purple')
    ax6.axhline(avg_corr, color='red', linestyle='--', linewidth=2, 
                label=f'Overall: {avg_corr:.4f}')
    ax6.set_xlabel('Signature', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Correlation', fontsize=11, fontweight='bold')
    ax6.set_title('Lambda Correlation by Signature\n(Averaged lambda)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.tick_params(labelsize=10)
    
    plt.suptitle('Lambda Comparison: IPW Recovery Analysis\n'
                'Biased (no IPW) vs Biased (with IPW) - Effect of IPW Correction',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = results_dir / 'ipw_recovery_lambda_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved lambda comparison figure to: {output_path}")
    plt.show()
    
    # ========================================================================
    # CREATE HEATMAPS: Mean differences by signature and time
    # ========================================================================
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Compute difference matrix [K, T]: Biased (no IPW) - Biased (with IPW)
    lambda_diff_2d = lambda_biased_avg - lambda_biased_ipw_avg
    
    vmin = -np.abs(lambda_diff_2d).max()
    vmax = np.abs(lambda_diff_2d).max()
    
    # Panel 1: Difference heatmap
    im1 = axes2[0, 0].imshow(lambda_diff_2d, aspect='auto', cmap='RdBu_r', 
                             vmin=vmin, vmax=vmax)
    axes2[0, 0].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[0, 0].set_ylabel('Signature', fontsize=11, fontweight='bold')
    axes2[0, 0].set_title('Lambda Difference: Biased (no IPW) - Biased (with IPW)\n(Averaged across individuals)', 
                          fontsize=12, fontweight='bold', pad=8)
    plt.colorbar(im1, ax=axes2[0, 0], label='Difference')
    
    # Panel 2: Absolute difference heatmap (magnitude of IPW effect)
    lambda_abs_diff_2d = np.abs(lambda_diff_2d)
    im2 = axes2[0, 1].imshow(lambda_abs_diff_2d, aspect='auto', cmap='YlOrRd')
    axes2[0, 1].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[0, 1].set_ylabel('Signature', fontsize=11, fontweight='bold')
    axes2[0, 1].set_title('Absolute Lambda Difference\n(Magnitude of IPW Effect)', 
                          fontsize=12, fontweight='bold', pad=8)
    plt.colorbar(im2, ax=axes2[0, 1], label='Absolute Difference')
    
    # Panel 3: Sample signature trajectories
    sample_sigs = [0, 5, 10, 15]
    for sig_idx in sample_sigs:
        if sig_idx < lambda_biased_avg.shape[0]:
            biased_traj = lambda_biased_avg[sig_idx, :]
            biased_ipw_traj = lambda_biased_ipw_avg[sig_idx, :]
            axes2[1, 0].plot(biased_traj, label=f'Sig {sig_idx} (no IPW)', linewidth=2, alpha=0.7)
            axes2[1, 0].plot(biased_ipw_traj, label=f'Sig {sig_idx} (with IPW)', linewidth=2, 
                            linestyle='--', alpha=0.7)
    axes2[1, 0].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[1, 0].set_ylabel('Lambda Value', fontsize=11, fontweight='bold')
    axes2[1, 0].set_title('Sample Signature Trajectories', 
                          fontsize=12, fontweight='bold', pad=8)
    axes2[1, 0].legend(fontsize=7, ncol=2)
    axes2[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Correlation by signature
    sig_correlations = []
    for sig_idx in range(lambda_biased_avg.shape[0]):
        sig_biased = lambda_biased_avg[sig_idx, :].flatten()
        sig_biased_ipw = lambda_biased_ipw_avg[sig_idx, :].flatten()
        sig_correlations.append(np.corrcoef(sig_biased, sig_biased_ipw)[0, 1])
    
    x = np.arange(len(sig_correlations))
    axes2[1, 1].bar(x, sig_correlations, alpha=0.7, edgecolor='black', color='purple')
    axes2[1, 1].axhline(avg_corr, color='red', linestyle='--', linewidth=2, 
                        label=f'Overall: {avg_corr:.4f}')
    axes2[1, 1].set_xlabel('Signature', fontsize=11, fontweight='bold')
    axes2[1, 1].set_ylabel('Correlation', fontsize=11, fontweight='bold')
    axes2[1, 1].set_title('Lambda Correlation by Signature\n(Averaged lambda)', 
                          fontsize=12, fontweight='bold', pad=8)
    axes2[1, 1].legend(fontsize=9)
    axes2[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Lambda Comparison: Heatmaps and Signature Analysis\n'
                'Biased (no IPW) vs Biased (with IPW) - Effect of IPW Correction',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save heatmap figure
    output_path2 = results_dir / 'ipw_recovery_lambda_heatmaps.pdf'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved lambda heatmap figure to: {output_path2}")
    plt.show()
    
    print(f"\n{'='*80}")
    print("✅ Lambda comparison complete!")
    print(f"{'='*80}")

