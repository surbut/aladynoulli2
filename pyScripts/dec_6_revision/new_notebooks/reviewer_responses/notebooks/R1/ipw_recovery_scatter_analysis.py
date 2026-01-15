"""
IPW Recovery Analysis: Scatter Plots, Histograms, and Heatmaps
for Phi, Pi, and Prevalence

This analysis compares:
- Full Population vs Biased (no IPW)
- Full Population vs Biased (with IPW)

Similar to the lambda comparison analysis, but for phi, pi, and prevalence
from the 90% women dropped experiment.
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
print("IPW RECOVERY ANALYSIS: Phi, Pi, and Prevalence Comparison")
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

# Load data from batches
phi_full_list = []
phi_biased_list = []
phi_biased_ipw_list = []
pi_full_list = []
pi_biased_list = []
pi_biased_ipw_list = []
prevalence_full_list = []
prevalence_biased_list = []
prevalence_biased_ipw_list = []

for batch_idx in range(1, 6):  # batches 1-5
    batch_dir = batch_base_dir / f'batch_{batch_idx}'
    
    if batch_dir.exists():
        # Load phi, pi
        phi_full_path = batch_dir / 'phi_full.npy'
        phi_biased_path = batch_dir / 'phi_biased.npy'
        phi_biased_ipw_path = batch_dir / 'phi_biased_ipw.npy'
        pi_full_path = batch_dir / 'pi_full.npy'
        pi_biased_path = batch_dir / 'pi_biased.npy'
        pi_biased_ipw_path = batch_dir / 'pi_biased_ipw.npy'
        
        if phi_full_path.exists():
            phi_full_list.append(np.load(phi_full_path))
            phi_biased_list.append(np.load(phi_biased_path))
            phi_biased_ipw_list.append(np.load(phi_biased_ipw_path))
            pi_full_list.append(np.load(pi_full_path))
            pi_biased_list.append(np.load(pi_biased_path))
            pi_biased_ipw_list.append(np.load(pi_biased_ipw_path))
            print(f"  ✓ Loaded batch {batch_idx}")
        
        # Try to load prevalence if available
        prev_full_path = batch_dir / 'prevalence_full.npy'
        if prev_full_path.exists():
            prevalence_full_list.append(np.load(prev_full_path))
            prevalence_biased_list.append(np.load(batch_dir / 'prevalence_biased.npy'))
            prevalence_biased_ipw_list.append(np.load(batch_dir / 'prevalence_biased_ipw.npy'))

if len(phi_full_list) == 0:
    print("\n⚠️  No batch data found. Please check the directory paths.")
    print(f"   Tried: {ipw_dir}")
    print(f"   Tried: {results_dir}/batch_*")
else:
    # Average across batches
    phi_full = np.mean(phi_full_list, axis=0)  # [K, D, T]
    phi_biased = np.mean(phi_biased_list, axis=0)
    phi_biased_ipw = np.mean(phi_biased_ipw_list, axis=0)
    
    pi_full = np.mean(pi_full_list, axis=0)  # [D, T]
    pi_biased = np.mean(pi_biased_list, axis=0)
    pi_biased_ipw = np.mean(pi_biased_ipw_list, axis=0)
    
    # Average phi over signatures for comparison
    phi_full_avg = phi_full.mean(axis=0)  # [D, T]
    phi_biased_avg = phi_biased.mean(axis=0)
    phi_biased_ipw_avg = phi_biased_ipw.mean(axis=0)
    
    # Load or compute prevalence
    if len(prevalence_full_list) > 0:
        prevalence_full = np.mean(prevalence_full_list, axis=0)  # [D, T]
        prevalence_biased = np.mean(prevalence_biased_list, axis=0)
        prevalence_biased_ipw = np.mean(prevalence_biased_ipw_list, axis=0)
    else:
        # If prevalence not saved, we'll skip it
        prevalence_full = None
        prevalence_biased = None
        prevalence_biased_ipw = None
        print("\n⚠️  Prevalence data not found in batches. Skipping prevalence analysis.")
    
    print(f"\n✓ Loaded and averaged {len(phi_full_list)} batches")
    print(f"  Phi shape: {phi_full.shape} (averaged to {phi_full_avg.shape})")
    print(f"  Pi shape: {pi_full.shape}")
    if prevalence_full is not None:
        print(f"  Prevalence shape: {prevalence_full.shape}")
    
    # ========================================================================
    # CREATE COMPREHENSIVE FIGURE: 3 rows x 3 columns
    # Row 1: Phi, Row 2: Pi, Row 3: Prevalence
    # Col 1: Scatter (Full vs Biased no IPW), Col 2: Scatter (Full vs Biased with IPW), Col 3: Histogram of differences
    # ========================================================================
    
    fig = plt.figure(figsize=(18, 16))
    
    # ========== ROW 1: PHI ==========
    
    # Flatten phi data for scatter plots
    phi_full_flat = phi_full_avg.flatten()
    phi_biased_flat = phi_biased_avg.flatten()
    phi_biased_ipw_flat = phi_biased_ipw_avg.flatten()
    
    # Compute correlations
    phi_corr_biased = np.corrcoef(phi_full_flat, phi_biased_flat)[0, 1]
    phi_corr_ipw = np.corrcoef(phi_full_flat, phi_biased_ipw_flat)[0, 1]
    phi_mean_diff_biased = np.abs(phi_full_flat - phi_biased_flat).mean()
    phi_mean_diff_ipw = np.abs(phi_full_flat - phi_biased_ipw_flat).mean()
    
    # Panel 1: Phi scatter - Full vs Biased (no IPW)
    ax1 = plt.subplot(3, 3, 1)
    n_sample = min(50000, len(phi_full_flat))
    sample_idx = np.random.choice(len(phi_full_flat), n_sample, replace=False)
    ax1.scatter(phi_full_flat[sample_idx], phi_biased_flat[sample_idx], 
                alpha=0.3, s=1, color='blue', edgecolors='none')
    lims = [min(phi_full_flat.min(), phi_biased_flat.min()), 
            max(phi_full_flat.max(), phi_biased_flat.max())]
    ax1.plot(lims, lims, 'r--', alpha=0.8, linewidth=2, label='y=x')
    ax1.set_xlabel('Full Population Phi', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Biased (no IPW) Phi', fontsize=11, fontweight='bold')
    ax1.set_title(f'Phi: Full vs Biased (no IPW)\nCorrelation: {phi_corr_biased:.6f} (STABLE)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=10)
    
    # Panel 2: Phi scatter - Full vs Biased (with IPW)
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(phi_full_flat[sample_idx], phi_biased_ipw_flat[sample_idx], 
                alpha=0.3, s=1, color='red', edgecolors='none')
    lims2 = [min(phi_full_flat.min(), phi_biased_ipw_flat.min()), 
             max(phi_full_flat.max(), phi_biased_ipw_flat.max())]
    ax2.plot(lims2, lims2, 'r--', alpha=0.8, linewidth=2, label='y=x')
    ax2.set_xlabel('Full Population Phi', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Biased (with IPW) Phi', fontsize=11, fontweight='bold')
    ax2.set_title(f'Phi: Full vs Biased (with IPW)\nCorrelation: {phi_corr_ipw:.6f} (STABLE)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=10)
    
    # Panel 3: Phi difference histogram
    ax3 = plt.subplot(3, 3, 3)
    phi_diff_biased = phi_full_flat - phi_biased_flat
    phi_diff_ipw = phi_full_flat - phi_biased_ipw_flat
    ax3.hist(phi_diff_biased, bins=100, alpha=0.6, label='Biased (no IPW)', 
             color='blue', edgecolor='black', linewidth=0.5)
    ax3.hist(phi_diff_ipw, bins=100, alpha=0.6, label='Biased (with IPW)', 
             color='red', edgecolor='black', linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax3.set_xlabel('Phi Difference (Full - Biased)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title(f'Distribution of Phi Differences\nMean (no IPW): {phi_mean_diff_biased:.6f}\nMean (with IPW): {phi_mean_diff_ipw:.6f}', 
                 fontsize=12, fontweight='bold', pad=8)
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.tick_params(labelsize=10)
    
    # ========== ROW 2: PI ==========
    
    # Flatten pi data
    pi_full_flat = pi_full.flatten()
    pi_biased_flat = pi_biased.flatten()
    pi_biased_ipw_flat = pi_biased_ipw.flatten()
    
    # Filter out invalid values (NaN, zero, negative)
    valid_mask_biased = ~(np.isnan(pi_full_flat) | np.isnan(pi_biased_flat) | 
                          (pi_full_flat <= 0) | (pi_biased_flat <= 0))
    valid_mask_ipw = ~(np.isnan(pi_full_flat) | np.isnan(pi_biased_ipw_flat) | 
                       (pi_full_flat <= 0) | (pi_biased_ipw_flat <= 0))
    
    # Compute correlations on valid values
    pi_corr_biased = np.corrcoef(pi_full_flat[valid_mask_biased], 
                                  pi_biased_flat[valid_mask_biased])[0, 1]
    pi_corr_ipw = np.corrcoef(pi_full_flat[valid_mask_ipw], 
                               pi_biased_ipw_flat[valid_mask_ipw])[0, 1]
    pi_mean_diff_biased = np.abs(pi_full_flat[valid_mask_biased] - 
                                  pi_biased_flat[valid_mask_biased]).mean()
    pi_mean_diff_ipw = np.abs(pi_full_flat[valid_mask_ipw] - 
                               pi_biased_ipw_flat[valid_mask_ipw]).mean()
    
    # Panel 4: Pi scatter - Full vs Biased (no IPW)
    ax4 = plt.subplot(3, 3, 4)
    n_sample_pi = min(50000, valid_mask_biased.sum())
    sample_idx_pi = np.random.choice(np.where(valid_mask_biased)[0], n_sample_pi, replace=False)
    ax4.scatter(pi_full_flat[sample_idx_pi], pi_biased_flat[sample_idx_pi], 
                alpha=0.3, s=1, color='blue', edgecolors='none')
    lims_pi = [min(pi_full_flat[valid_mask_biased].min(), pi_biased_flat[valid_mask_biased].min()),
               max(pi_full_flat[valid_mask_biased].max(), pi_biased_flat[valid_mask_biased].max())]
    ax4.plot(lims_pi, lims_pi, 'r--', alpha=0.8, linewidth=2, label='y=x')
    ax4.set_xlabel('Full Population Pi', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Biased (no IPW) Pi', fontsize=11, fontweight='bold')
    ax4.set_title(f'Pi: Full vs Biased (no IPW)\nCorrelation: {pi_corr_biased:.6f}', 
                 fontsize=12, fontweight='bold', pad=8)
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.tick_params(labelsize=10)
    
    # Panel 5: Pi scatter - Full vs Biased (with IPW)
    ax5 = plt.subplot(3, 3, 5)
    n_sample_pi2 = min(50000, valid_mask_ipw.sum())
    sample_idx_pi2 = np.random.choice(np.where(valid_mask_ipw)[0], n_sample_pi2, replace=False)
    ax5.scatter(pi_full_flat[sample_idx_pi2], pi_biased_ipw_flat[sample_idx_pi2], 
                alpha=0.3, s=1, color='red', edgecolors='none')
    lims_pi2 = [min(pi_full_flat[valid_mask_ipw].min(), pi_biased_ipw_flat[valid_mask_ipw].min()),
                max(pi_full_flat[valid_mask_ipw].max(), pi_biased_ipw_flat[valid_mask_ipw].max())]
    ax5.plot(lims_pi2, lims_pi2, 'r--', alpha=0.8, linewidth=2, label='y=x')
    ax5.set_xlabel('Full Population Pi', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Biased (with IPW) Pi', fontsize=11, fontweight='bold')
    ax5.set_title(f'Pi: Full vs Biased (with IPW)\nCorrelation: {pi_corr_ipw:.6f} (IPW RECOVERS)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax5.legend(fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.tick_params(labelsize=10)
    
    # Panel 6: Pi difference histogram
    ax6 = plt.subplot(3, 3, 6)
    pi_diff_biased = np.log10(pi_full_flat[valid_mask_biased]) - np.log10(pi_biased_flat[valid_mask_biased])
    pi_diff_ipw = np.log10(pi_full_flat[valid_mask_ipw]) - np.log10(pi_biased_ipw_flat[valid_mask_ipw])
    ax6.hist(pi_diff_biased, bins=100, alpha=0.6, label='Biased (no IPW)', 
             color='blue', edgecolor='black', linewidth=0.5)
    ax6.hist(pi_diff_ipw, bins=100, alpha=0.6, label='Biased (with IPW)', 
             color='red', edgecolor='black', linewidth=0.5)
    ax6.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax6.set_xlabel('Log10(Pi) Difference (Full - Biased)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title(f'Distribution of Pi Differences (Log Scale)\nMean (no IPW): {pi_mean_diff_biased:.6f}\nMean (with IPW): {pi_mean_diff_ipw:.6f}', 
                 fontsize=12, fontweight='bold', pad=8)
    ax6.legend(fontsize=9, framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax6.tick_params(labelsize=10)
    
    # ========== ROW 3: PREVALENCE ==========
    
    if prevalence_full is not None:
        # Flatten prevalence data
        prev_full_flat = prevalence_full.flatten()
        prev_biased_flat = prevalence_biased.flatten()
        prev_biased_ipw_flat = prevalence_biased_ipw.flatten()
        
        # Filter out invalid values
        valid_mask_prev_biased = ~(np.isnan(prev_full_flat) | np.isnan(prev_biased_flat) | 
                                    (prev_full_flat <= 0) | (prev_biased_flat <= 0))
        valid_mask_prev_ipw = ~(np.isnan(prev_full_flat) | np.isnan(prev_biased_ipw_flat) | 
                                (prev_full_flat <= 0) | (prev_biased_ipw_flat <= 0))
        
        # Compute correlations
        prev_corr_biased = np.corrcoef(prev_full_flat[valid_mask_prev_biased], 
                                        prev_biased_flat[valid_mask_prev_biased])[0, 1]
        prev_corr_ipw = np.corrcoef(prev_full_flat[valid_mask_prev_ipw], 
                                     prev_biased_ipw_flat[valid_mask_prev_ipw])[0, 1]
        prev_mean_diff_biased = np.abs(prev_full_flat[valid_mask_prev_biased] - 
                                        prev_biased_flat[valid_mask_prev_biased]).mean()
        prev_mean_diff_ipw = np.abs(prev_full_flat[valid_mask_prev_ipw] - 
                                     prev_biased_ipw_flat[valid_mask_prev_ipw]).mean()
        
        # Panel 7: Prevalence scatter - Full vs Biased (no IPW)
        ax7 = plt.subplot(3, 3, 7)
        n_sample_prev = min(50000, valid_mask_prev_biased.sum())
        sample_idx_prev = np.random.choice(np.where(valid_mask_prev_biased)[0], n_sample_prev, replace=False)
        ax7.scatter(prev_full_flat[sample_idx_prev], prev_biased_flat[sample_idx_prev], 
                    alpha=0.3, s=1, color='blue', edgecolors='none')
        lims_prev = [min(prev_full_flat[valid_mask_prev_biased].min(), prev_biased_flat[valid_mask_prev_biased].min()),
                     max(prev_full_flat[valid_mask_prev_biased].max(), prev_biased_flat[valid_mask_prev_biased].max())]
        ax7.plot(lims_prev, lims_prev, 'r--', alpha=0.8, linewidth=2, label='y=x')
        ax7.set_xlabel('Full Population Prevalence', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Biased (no IPW) Prevalence', fontsize=11, fontweight='bold')
        ax7.set_title(f'Prevalence: Full vs Biased (no IPW)\nCorrelation: {prev_corr_biased:.6f}', 
                     fontsize=12, fontweight='bold', pad=8)
        ax7.legend(fontsize=9, framealpha=0.9)
        ax7.grid(True, alpha=0.3, linestyle='--')
        ax7.set_xscale('log')
        ax7.set_yscale('log')
        ax7.tick_params(labelsize=10)
        
        # Panel 8: Prevalence scatter - Full vs Biased (with IPW)
        ax8 = plt.subplot(3, 3, 8)
        n_sample_prev2 = min(50000, valid_mask_prev_ipw.sum())
        sample_idx_prev2 = np.random.choice(np.where(valid_mask_prev_ipw)[0], n_sample_prev2, replace=False)
        ax8.scatter(prev_full_flat[sample_idx_prev2], prev_biased_ipw_flat[sample_idx_prev2], 
                    alpha=0.3, s=1, color='red', edgecolors='none')
        lims_prev2 = [min(prev_full_flat[valid_mask_prev_ipw].min(), prev_biased_ipw_flat[valid_mask_prev_ipw].min()),
                       max(prev_full_flat[valid_mask_prev_ipw].max(), prev_biased_ipw_flat[valid_mask_prev_ipw].max())]
        ax8.plot(lims_prev2, lims_prev2, 'r--', alpha=0.8, linewidth=2, label='y=x')
        ax8.set_xlabel('Full Population Prevalence', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Biased (with IPW) Prevalence', fontsize=11, fontweight='bold')
        ax8.set_title(f'Prevalence: Full vs Biased (with IPW)\nCorrelation: {prev_corr_ipw:.6f} (IPW RECOVERS)', 
                     fontsize=12, fontweight='bold', pad=8)
        ax8.legend(fontsize=9, framealpha=0.9)
        ax8.grid(True, alpha=0.3, linestyle='--')
        ax8.set_xscale('log')
        ax8.set_yscale('log')
        ax8.tick_params(labelsize=10)
        
        # Panel 9: Prevalence difference histogram
        ax9 = plt.subplot(3, 3, 9)
        prev_diff_biased = np.log10(prev_full_flat[valid_mask_prev_biased]) - np.log10(prev_biased_flat[valid_mask_prev_biased])
        prev_diff_ipw = np.log10(prev_full_flat[valid_mask_prev_ipw]) - np.log10(prev_biased_ipw_flat[valid_mask_prev_ipw])
        ax9.hist(prev_diff_biased, bins=100, alpha=0.6, label='Biased (no IPW)', 
                 color='blue', edgecolor='black', linewidth=0.5)
        ax9.hist(prev_diff_ipw, bins=100, alpha=0.6, label='Biased (with IPW)', 
                 color='red', edgecolor='black', linewidth=0.5)
        ax9.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
        ax9.set_xlabel('Log10(Prevalence) Difference (Full - Biased)', fontsize=11, fontweight='bold')
        ax9.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax9.set_title(f'Distribution of Prevalence Differences (Log Scale)\nMean (no IPW): {prev_mean_diff_biased:.6f}\nMean (with IPW): {prev_mean_diff_ipw:.6f}', 
                     fontsize=12, fontweight='bold', pad=8)
        ax9.legend(fontsize=9, framealpha=0.9)
        ax9.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax9.tick_params(labelsize=10)
    else:
        # If no prevalence data, show message
        ax7 = plt.subplot(3, 3, 7)
        ax7.text(0.5, 0.5, 'Prevalence data not available', 
                ha='center', va='center', fontsize=14, transform=ax7.transAxes)
        ax7.axis('off')
        ax8 = plt.subplot(3, 3, 8)
        ax8.text(0.5, 0.5, 'Prevalence data not available', 
                ha='center', va='center', fontsize=14, transform=ax8.transAxes)
        ax8.axis('off')
        ax9 = plt.subplot(3, 3, 9)
        ax9.text(0.5, 0.5, 'Prevalence data not available', 
                ha='center', va='center', fontsize=14, transform=ax9.transAxes)
        ax9.axis('off')
    
    plt.suptitle('IPW Recovery: Correlation Scatter Plots and Difference Distributions\n'
                'Even when average lambda and phi look similar, pi can differ - IPW corrects it',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = results_dir / 'ipw_recovery_phi_pi_prevalence_scatter.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved scatter plot figure to: {output_path}")
    plt.show()
    
    # ========================================================================
    # CREATE HEATMAPS: Mean differences for Phi, Pi, and Prevalence
    # ========================================================================
    
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 18))
    
    # Row 1: Phi heatmaps
    phi_diff_biased_2d = phi_full_avg - phi_biased_avg  # [D, T]
    phi_diff_ipw_2d = phi_full_avg - phi_biased_ipw_avg
    
    vmin_phi = -max(np.abs(phi_diff_biased_2d).max(), np.abs(phi_diff_ipw_2d).max())
    vmax_phi = max(np.abs(phi_diff_biased_2d).max(), np.abs(phi_diff_ipw_2d).max())
    
    im1 = axes2[0, 0].imshow(phi_diff_biased_2d, aspect='auto', cmap='RdBu_r', 
                             vmin=vmin_phi, vmax=vmax_phi)
    axes2[0, 0].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[0, 0].set_ylabel('Disease', fontsize=11, fontweight='bold')
    axes2[0, 0].set_title('Phi Difference: Full - Biased (no IPW)\n(Averaged over signatures)', 
                          fontsize=12, fontweight='bold', pad=8)
    plt.colorbar(im1, ax=axes2[0, 0], label='Difference')
    
    im2 = axes2[0, 1].imshow(phi_diff_ipw_2d, aspect='auto', cmap='RdBu_r', 
                             vmin=vmin_phi, vmax=vmax_phi)
    axes2[0, 1].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[0, 1].set_ylabel('Disease', fontsize=11, fontweight='bold')
    axes2[0, 1].set_title('Phi Difference: Full - Biased (with IPW)\n(Averaged over signatures)', 
                          fontsize=12, fontweight='bold', pad=8)
    plt.colorbar(im2, ax=axes2[0, 1], label='Difference')
    
    # Row 2: Pi heatmaps
    pi_diff_biased_2d = np.log10(pi_full + 1e-10) - np.log10(pi_biased + 1e-10)  # [D, T]
    pi_diff_ipw_2d = np.log10(pi_full + 1e-10) - np.log10(pi_biased_ipw + 1e-10)
    
    vmin_pi = -max(np.abs(pi_diff_biased_2d).max(), np.abs(pi_diff_ipw_2d).max())
    vmax_pi = max(np.abs(pi_diff_biased_2d).max(), np.abs(pi_diff_ipw_2d).max())
    
    im3 = axes2[1, 0].imshow(pi_diff_biased_2d, aspect='auto', cmap='RdBu_r', 
                             vmin=vmin_pi, vmax=vmax_pi)
    axes2[1, 0].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[1, 0].set_ylabel('Disease', fontsize=11, fontweight='bold')
    axes2[1, 0].set_title('Pi Difference (Log10): Full - Biased (no IPW)', 
                          fontsize=12, fontweight='bold', pad=8)
    plt.colorbar(im3, ax=axes2[1, 0], label='Log10 Difference')
    
    im4 = axes2[1, 1].imshow(pi_diff_ipw_2d, aspect='auto', cmap='RdBu_r', 
                             vmin=vmin_pi, vmax=vmax_pi)
    axes2[1, 1].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
    axes2[1, 1].set_ylabel('Disease', fontsize=11, fontweight='bold')
    axes2[1, 1].set_title('Pi Difference (Log10): Full - Biased (with IPW)\n(IPW RECOVERS)', 
                          fontsize=12, fontweight='bold', pad=8)
    plt.colorbar(im4, ax=axes2[1, 1], label='Log10 Difference')
    
    # Row 3: Prevalence heatmaps (if available)
    if prevalence_full is not None:
        prev_diff_biased_2d = np.log10(prevalence_full + 1e-10) - np.log10(prevalence_biased + 1e-10)
        prev_diff_ipw_2d = np.log10(prevalence_full + 1e-10) - np.log10(prevalence_biased_ipw + 1e-10)
        
        vmin_prev = -max(np.abs(prev_diff_biased_2d).max(), np.abs(prev_diff_ipw_2d).max())
        vmax_prev = max(np.abs(prev_diff_biased_2d).max(), np.abs(prev_diff_ipw_2d).max())
        
        im5 = axes2[2, 0].imshow(prev_diff_biased_2d, aspect='auto', cmap='RdBu_r', 
                                 vmin=vmin_prev, vmax=vmax_prev)
        axes2[2, 0].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
        axes2[2, 0].set_ylabel('Disease', fontsize=11, fontweight='bold')
        axes2[2, 0].set_title('Prevalence Difference (Log10): Full - Biased (no IPW)', 
                              fontsize=12, fontweight='bold', pad=8)
        plt.colorbar(im5, ax=axes2[2, 0], label='Log10 Difference')
        
        im6 = axes2[2, 1].imshow(prev_diff_ipw_2d, aspect='auto', cmap='RdBu_r', 
                                 vmin=vmin_prev, vmax=vmax_prev)
        axes2[2, 1].set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
        axes2[2, 1].set_ylabel('Disease', fontsize=11, fontweight='bold')
        axes2[2, 1].set_title('Prevalence Difference (Log10): Full - Biased (with IPW)\n(IPW RECOVERS)', 
                              fontsize=12, fontweight='bold', pad=8)
        plt.colorbar(im6, ax=axes2[2, 1], label='Log10 Difference')
    else:
        axes2[2, 0].text(0.5, 0.5, 'Prevalence data not available', 
                        ha='center', va='center', fontsize=14, transform=axes2[2, 0].transAxes)
        axes2[2, 0].axis('off')
        axes2[2, 1].text(0.5, 0.5, 'Prevalence data not available', 
                        ha='center', va='center', fontsize=14, transform=axes2[2, 1].transAxes)
        axes2[2, 1].axis('off')
    
    plt.suptitle('IPW Recovery: Heatmaps of Mean Differences\n'
                'Shows spatial patterns of differences across diseases and time',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save heatmap figure
    output_path2 = results_dir / 'ipw_recovery_phi_pi_prevalence_heatmaps.pdf'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap figure to: {output_path2}")
    plt.show()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nPHI:")
    print(f"  Correlation (Full vs Biased no IPW): {phi_corr_biased:.6f}")
    print(f"  Correlation (Full vs Biased with IPW): {phi_corr_ipw:.6f}")
    print(f"  Mean absolute difference (no IPW): {phi_mean_diff_biased:.6f}")
    print(f"  Mean absolute difference (with IPW): {phi_mean_diff_ipw:.6f}")
    print(f"\nPI:")
    print(f"  Correlation (Full vs Biased no IPW): {pi_corr_biased:.6f}")
    print(f"  Correlation (Full vs Biased with IPW): {pi_corr_ipw:.6f}")
    print(f"  Mean absolute difference (no IPW): {pi_mean_diff_biased:.6f}")
    print(f"  Mean absolute difference (with IPW): {pi_mean_diff_ipw:.6f}")
    if prevalence_full is not None:
        print(f"\nPREVALENCE:")
        print(f"  Correlation (Full vs Biased no IPW): {prev_corr_biased:.6f}")
        print(f"  Correlation (Full vs Biased with IPW): {prev_corr_ipw:.6f}")
        print(f"  Mean absolute difference (no IPW): {prev_mean_diff_biased:.6f}")
        print(f"  Mean absolute difference (with IPW): {prev_mean_diff_ipw:.6f}")
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}")


