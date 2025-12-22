"""
Compare pi averages from models trained with:
1. Unweighted prevalence initialization (batch_models_weighted_vec_censoredE_1218)
2. Weighted prevalence initialization (batch_models_weighted_vec_censoredE_1219)

This demonstrates the impact of using weighted vs unweighted prevalence for initialization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add path for utils
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred

print("="*80)
print("COMPARING PI: UNWEIGHTED vs WEIGHTED PREVALENCE INITIALIZATION")
print("="*80)

# Model directories
unweighted_prev_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1218/')
weighted_prev_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1219/')

# Load all batches (0-9, so 10 batches)
n_batches = 10
unweighted_pi_list = []
weighted_pi_list = []

print(f"\nLoading {n_batches} batches from each model set...")

for batch_idx in range(n_batches):
    # Unweighted prevalence models
    unweighted_paths = [
        unweighted_prev_dir / f"batch_{batch_idx:02d}_model.pt",
        unweighted_prev_dir / f"batch_{batch_idx}_model.pt",
    ]
    
    unweighted_path = None
    for p in unweighted_paths:
        if p.exists():
            unweighted_path = p
            break
    
    # Weighted prevalence models
    weighted_paths = [
        weighted_prev_dir / f"batch_{batch_idx:02d}_model.pt",
        weighted_prev_dir / f"batch_{batch_idx}_model.pt",
    ]
    
    weighted_path = None
    for p in weighted_paths:
        if p.exists():
            weighted_path = p
            break
    
    if unweighted_path and unweighted_path.exists() and weighted_path and weighted_path.exists():
        print(f"\nBatch {batch_idx}:")
        print(f"  Unweighted prev init: {unweighted_path.name}")
        print(f"  Weighted prev init: {weighted_path.name}")
        
        # Load unweighted model
        unweighted_ckpt = torch.load(unweighted_path, weights_only=False, map_location='cpu')
        if 'model_state_dict' in unweighted_ckpt:
            unweighted_lambda = unweighted_ckpt['model_state_dict']['lambda_'].detach()
            unweighted_phi = unweighted_ckpt['model_state_dict']['phi'].detach()
            unweighted_kappa = unweighted_ckpt['model_state_dict'].get('kappa', torch.tensor(1.0))
        else:
            unweighted_lambda = unweighted_ckpt['lambda_'].detach()
            unweighted_phi = unweighted_ckpt['phi'].detach()
            unweighted_kappa = unweighted_ckpt.get('kappa', torch.tensor(1.0))
        
        if torch.is_tensor(unweighted_kappa):
            unweighted_kappa = unweighted_kappa.item() if unweighted_kappa.numel() == 1 else unweighted_kappa.mean().item()
        
        # Load weighted model
        weighted_ckpt = torch.load(weighted_path, weights_only=False, map_location='cpu')
        if 'model_state_dict' in weighted_ckpt:
            weighted_lambda = weighted_ckpt['model_state_dict']['lambda_'].detach()
            weighted_phi = weighted_ckpt['model_state_dict']['phi'].detach()
            weighted_kappa = weighted_ckpt['model_state_dict'].get('kappa', torch.tensor(1.0))
        else:
            weighted_lambda = weighted_ckpt['lambda_'].detach()
            weighted_phi = weighted_ckpt['phi'].detach()
            weighted_kappa = weighted_ckpt.get('kappa', torch.tensor(1.0))
        
        if torch.is_tensor(weighted_kappa):
            weighted_kappa = weighted_kappa.item() if weighted_kappa.numel() == 1 else weighted_kappa.mean().item()
        
        # Compute pi for both
        unweighted_pi_batch = calculate_pi_pred(unweighted_lambda, unweighted_phi, unweighted_kappa)
        weighted_pi_batch = calculate_pi_pred(weighted_lambda, weighted_phi, weighted_kappa)
        
        unweighted_pi_list.append(unweighted_pi_batch)
        weighted_pi_list.append(weighted_pi_batch)
        
        print(f"    Unweighted pi shape: {unweighted_pi_batch.shape}")
        print(f"    Weighted pi shape: {weighted_pi_batch.shape}")
    else:
        print(f"\n⚠️  Batch {batch_idx}: Files not found")
        if not unweighted_path or not unweighted_path.exists():
            print(f"     Unweighted path not found")
        if not weighted_path or not weighted_path.exists():
            print(f"     Weighted path not found")

# Concatenate all batches
if len(unweighted_pi_list) > 0 and len(weighted_pi_list) > 0:
    print(f"\n{'='*80}")
    print(f"Concatenating {len(unweighted_pi_list)} batches...")
    
    unweighted_pi_all = torch.cat(unweighted_pi_list, dim=0)  # [N_total, D, T]
    weighted_pi_all = torch.cat(weighted_pi_list, dim=0)  # [N_total, D, T]
    
    print(f"Unweighted pi shape (all batches): {unweighted_pi_all.shape}")
    print(f"Weighted pi shape (all batches): {weighted_pi_all.shape}")
    
    # Average across patients: [D, T]
    unweighted_pi_avg = unweighted_pi_all.mean(dim=0)
    weighted_pi_avg = weighted_pi_all.mean(dim=0)
    
    print(f"Unweighted pi avg shape: {unweighted_pi_avg.shape}")
    print(f"Weighted pi avg shape: {weighted_pi_avg.shape}")
    
    # Calculate overall correlation
    unweighted_flat = unweighted_pi_avg.numpy().flatten()
    weighted_flat = weighted_pi_avg.numpy().flatten()
    pi_correlation = np.corrcoef(unweighted_flat, weighted_flat)[0, 1]
    pi_mean_diff = np.abs(unweighted_flat - weighted_flat).mean()
    pi_max_diff = np.abs(unweighted_flat - weighted_flat).max()
    
    print(f"\nOverall Pi Comparison (averaged across patients):")
    print(f"  Correlation: {pi_correlation:.6f}")
    print(f"  Mean absolute difference: {pi_mean_diff:.6f}")
    print(f"  Max absolute difference: {pi_max_diff:.6f}")
    
    # Plot trajectories for selected diseases
    DISEASES_TO_PLOT = [
        (112, "Myocardial Infarction"),
        (66, "Depression"),
        (16, "Breast cancer [female]"),
        (127, "Atrial fibrillation"),
        (47, "Type 2 diabetes"),
    ]
    
    # Load disease names if available
    disease_names_dict = {}
    try:
        disease_names_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv")
        if disease_names_path.exists():
            disease_df = pd.read_csv(disease_names_path)
            disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
            print(f"✓ Loaded disease names")
    except:
        pass
    
    n_diseases = len(DISEASES_TO_PLOT)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Get disease name from dict if available
        if disease_names_dict and disease_idx in disease_names_dict:
            display_name = disease_names_dict[disease_idx]
        else:
            display_name = disease_name
        
        if disease_idx < unweighted_pi_avg.shape[0] and disease_idx < weighted_pi_avg.shape[0]:
            unweighted_traj = unweighted_pi_avg[disease_idx, :].numpy()
            weighted_traj = weighted_pi_avg[disease_idx, :].numpy()
            
            # Time points (assuming starting at age 30)
            time_points = np.arange(len(unweighted_traj)) + 30
            
            ax.plot(time_points, unweighted_traj, label='Pi (Unweighted Prev Init)', 
                   linewidth=2, alpha=0.8, color='blue')
            ax.plot(time_points, weighted_traj, label='Pi (Weighted Prev Init)', 
                   linewidth=2, alpha=0.8, linestyle='--', color='red')
            
            ax.set_xlabel('Age', fontsize=11)
            ax.set_ylabel('Average Pi (Disease Hazard)', fontsize=11)
            ax.set_title(f'{display_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # Log scale for better visualization
            
            # Add difference annotation
            max_diff = np.abs(weighted_traj - unweighted_traj).max()
            mean_diff = np.abs(weighted_traj - unweighted_traj).mean()
            disease_corr = np.corrcoef(unweighted_traj, weighted_traj)[0, 1]
            ax.text(0.02, 0.98, f'Corr: {disease_corr:.4f}\nMean diff: {mean_diff:.4f}\nMax diff: {max_diff:.4f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Disease {disease_idx}\nnot found', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{disease_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')
    
    # Remove extra subplot
    if len(DISEASES_TO_PLOT) < len(axes):
        axes[len(DISEASES_TO_PLOT)].axis('off')
    
    plt.suptitle(f'Pi Comparison: Unweighted vs Weighted Prevalence Initialization\n(All {len(unweighted_pi_list)} batches, N={unweighted_pi_all.shape[0]:,} patients)\nOverall Correlation: {pi_correlation:.4f}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'pi_comparison_unweighted_vs_weighted_prev_init.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved trajectory plot to: {plot_path}")
    plt.show()
    
    # Summary scatter plot: all diseases, all time points
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.scatter(unweighted_flat, weighted_flat, alpha=0.3, s=1)
    ax.plot([unweighted_flat.min(), unweighted_flat.max()], 
           [unweighted_flat.min(), unweighted_flat.max()], 'r--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Pi Average (Unweighted Prev Init)', fontsize=12)
    ax.set_ylabel('Pi Average (Weighted Prev Init)', fontsize=12)
    ax.set_title(f'Pi Comparison: All Diseases, All Time Points\nCorrelation: {pi_correlation:.4f}\n(N={unweighted_pi_all.shape[0]:,} patients)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    scatter_path = output_dir / 'pi_comparison_unweighted_vs_weighted_prev_init_scatter.pdf'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plot to: {scatter_path}")
    plt.show()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"✓ Compared pi from models with unweighted vs weighted prevalence initialization")
    print(f"✓ Overall correlation: {pi_correlation:.6f}")
    print(f"✓ Mean absolute difference: {pi_mean_diff:.6f}")
    print(f"\nKey Insight:")
    if pi_correlation > 0.99:
        print(f"  Very high correlation ({pi_correlation:.4f}) suggests that:")
        print(f"  - Both initialization methods lead to similar pi predictions")
        print(f"  - The model adapts through lambda/pi regardless of initialization")
        print(f"  - However, small differences exist (mean diff: {pi_mean_diff:.6f})")
    else:
        print(f"  Correlation ({pi_correlation:.4f}) shows meaningful differences:")
        print(f"  - Weighted prevalence initialization leads to different pi predictions")
        print(f"  - This demonstrates the impact of initialization on final predictions")
    
else:
    print("\n⚠️  No batches loaded successfully. Check file paths and naming conventions.")

