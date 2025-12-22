"""
Compare:
1. Weighted vs Unweighted Pi (from models)
2. Weighted vs Unweighted Prevalence

This shows the impact of IPW on both model predictions (pi) and observed population patterns (prevalence).
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
print("COMPARING WEIGHTED vs UNWEIGHTED: PI AND PREVALENCE")
print("="*80)

# Data directories
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
weighted_model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1219/')
unweighted_pi_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/')

# Load prevalences
print("\n1. Loading prevalences...")
weighted_prevalence_path = data_dir / 'prevalence_t_weighted_corrected.pt'
unweighted_prevalence_path = data_dir / 'prevalence_t_corrected.pt'  # or prevalence_t_unweighted_corrected.pt

if weighted_prevalence_path.exists():
    prevalence_t_weighted = torch.load(str(weighted_prevalence_path), weights_only=False)
    if torch.is_tensor(prevalence_t_weighted):
        prevalence_t_weighted = prevalence_t_weighted.numpy()
    print(f"   ✓ Loaded weighted prevalence: {prevalence_t_weighted.shape}")
else:
    raise FileNotFoundError(f"Weighted prevalence not found: {weighted_prevalence_path}")

if unweighted_prevalence_path.exists():
    prevalence_t_unweighted = torch.load(str(unweighted_prevalence_path), weights_only=False)
    if torch.is_tensor(prevalence_t_unweighted):
        prevalence_t_unweighted = prevalence_t_unweighted.numpy()
    print(f"   ✓ Loaded unweighted prevalence: {prevalence_t_unweighted.shape}")
else:
    # Try alternative name
    alt_path = data_dir / 'prevalence_t_unweighted_corrected.pt'
    if alt_path.exists():
        prevalence_t_unweighted = torch.load(str(alt_path), weights_only=False)
        if torch.is_tensor(prevalence_t_unweighted):
            prevalence_t_unweighted = prevalence_t_unweighted.numpy()
        print(f"   ✓ Loaded unweighted prevalence: {prevalence_t_unweighted.shape}")
    else:
        raise FileNotFoundError(f"Unweighted prevalence not found: {unweighted_prevalence_path} or {alt_path}")

# Load weighted pi from models (batch_models_weighted_vec_censoredE_1219)
print("\n2. Loading weighted pi from models...")
n_batches = 10
weighted_pi_list = []

for batch_idx in range(n_batches):
    weighted_paths = [
        weighted_model_dir / f"batch_{batch_idx:02d}_model.pt",
        weighted_model_dir / f"batch_{batch_idx}_model.pt",
    ]
    
    weighted_path = None
    for p in weighted_paths:
        if p.exists():
            weighted_path = p
            break
    
    if weighted_path and weighted_path.exists():
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
        
        weighted_pi_batch = calculate_pi_pred(weighted_lambda, weighted_phi, weighted_kappa)
        weighted_pi_list.append(weighted_pi_batch)
        
        if batch_idx == 0:
            print(f"   Batch {batch_idx}: pi shape {weighted_pi_batch.shape}")

if len(weighted_pi_list) > 0:
    weighted_pi_all = torch.cat(weighted_pi_list, dim=0)  # [N_total, D, T]
    weighted_pi_avg = weighted_pi_all.mean(dim=0)  # [D, T]
    print(f"   ✓ Loaded weighted pi: {weighted_pi_all.shape} -> avg: {weighted_pi_avg.shape}")
else:
    raise FileNotFoundError(f"No weighted pi models found in {weighted_model_dir}")

# Load unweighted pi from enrollment predictions
print("\n3. Loading unweighted pi from enrollment predictions...")
unweighted_pi_path = unweighted_pi_dir / 'pi_enroll_fixedphi_sex_FULL.pt'

if unweighted_pi_path.exists():
    unweighted_pi_all = torch.load(str(unweighted_pi_path), weights_only=False)
    if torch.is_tensor(unweighted_pi_all):
        unweighted_pi_all = unweighted_pi_all
    print(f"   ✓ Loaded unweighted pi: {unweighted_pi_all.shape}")
    
    # Average across patients: [D, T]
    unweighted_pi_avg = unweighted_pi_all.mean(dim=0)  # [D, T]
    print(f"   ✓ Averaged unweighted pi: {unweighted_pi_avg.shape}")
else:
    raise FileNotFoundError(f"Unweighted pi not found: {unweighted_pi_path}")

# Ensure same shape
min_D = min(weighted_pi_avg.shape[0], unweighted_pi_avg.shape[0], 
            prevalence_t_weighted.shape[0], prevalence_t_unweighted.shape[0])
min_T = min(weighted_pi_avg.shape[1], unweighted_pi_avg.shape[1],
            prevalence_t_weighted.shape[1], prevalence_t_unweighted.shape[1])

weighted_pi_avg = weighted_pi_avg[:min_D, :min_T]
unweighted_pi_avg = unweighted_pi_avg[:min_D, :min_T]
prevalence_t_weighted = prevalence_t_weighted[:min_D, :min_T]
prevalence_t_unweighted = prevalence_t_unweighted[:min_D, :min_T]

print(f"\n   Using D={min_D}, T={min_T} for comparison")

# Calculate correlations
print("\n4. Calculating correlations...")

# Pi comparison
pi_weighted_flat = weighted_pi_avg.numpy().flatten()
pi_unweighted_flat = unweighted_pi_avg.numpy().flatten()
pi_correlation = np.corrcoef(pi_weighted_flat, pi_unweighted_flat)[0, 1]
pi_mean_diff = np.abs(pi_weighted_flat - pi_unweighted_flat).mean()

# Prevalence comparison
prev_weighted_flat = prevalence_t_weighted.flatten()
prev_unweighted_flat = prevalence_t_unweighted.flatten()
prev_correlation = np.corrcoef(prev_weighted_flat, prev_unweighted_flat)[0, 1]
prev_mean_diff = np.abs(prev_weighted_flat - prev_unweighted_flat).mean()

print(f"   Pi correlation (weighted vs unweighted): {pi_correlation:.6f}")
print(f"   Pi mean absolute difference: {pi_mean_diff:.6f}")
print(f"   Prevalence correlation (weighted vs unweighted): {prev_correlation:.6f}")
print(f"   Prevalence mean absolute difference: {prev_mean_diff:.6f}")

# Plot comparison for selected diseases
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

# Create figure with 2 columns: Pi comparison and Prevalence comparison
fig, axes = plt.subplots(len(DISEASES_TO_PLOT), 2, figsize=(16, 3*len(DISEASES_TO_PLOT)))
if len(DISEASES_TO_PLOT) == 1:
    axes = axes.reshape(1, -1)

time_points = np.arange(min_T) + 30

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if disease_idx >= min_D:
        continue
    
    # Get disease name from dict if available
    if disease_names_dict and disease_idx in disease_names_dict:
        display_name = disease_names_dict[disease_idx]
    else:
        display_name = disease_name
    
    # ===== LEFT COLUMN: Pi Comparison =====
    ax1 = axes[idx, 0]
    
    weighted_pi_traj = weighted_pi_avg[disease_idx, :].numpy()
    unweighted_pi_traj = unweighted_pi_avg[disease_idx, :].numpy()
    
    ax1.plot(time_points, unweighted_pi_traj, label='Unweighted Pi', 
            linewidth=2, alpha=0.8, color='blue')
    ax1.plot(time_points, weighted_pi_traj, label='Weighted Pi (IPW)', 
            linewidth=2, alpha=0.8, linestyle='--', color='red')
    
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Average Pi (Disease Hazard)', fontsize=11)
    ax1.set_title(f'{display_name}\nPi: Weighted vs Unweighted', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add correlation annotation
    disease_pi_corr = np.corrcoef(unweighted_pi_traj, weighted_pi_traj)[0, 1]
    disease_pi_diff = np.abs(weighted_pi_traj - unweighted_pi_traj).mean()
    ax1.text(0.02, 0.98, f'Corr: {disease_pi_corr:.4f}\nMean diff: {disease_pi_diff:.4f}', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== RIGHT COLUMN: Prevalence Comparison =====
    ax2 = axes[idx, 1]
    
    weighted_prev_traj = prevalence_t_weighted[disease_idx, :]
    unweighted_prev_traj = prevalence_t_unweighted[disease_idx, :]
    
    ax2.plot(time_points, unweighted_prev_traj, label='Unweighted Prevalence', 
            linewidth=2, alpha=0.8, color='blue')
    ax2.plot(time_points, weighted_prev_traj, label='Weighted Prevalence (IPW)', 
            linewidth=2, alpha=0.8, linestyle='--', color='red')
    
    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel('Prevalence', fontsize=11)
    ax2.set_title(f'{display_name}\nPrevalence: Weighted vs Unweighted', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add correlation annotation
    disease_prev_corr = np.corrcoef(unweighted_prev_traj, weighted_prev_traj)[0, 1]
    disease_prev_diff = np.abs(weighted_prev_traj - unweighted_prev_traj).mean()
    ax2.text(0.02, 0.98, f'Corr: {disease_prev_corr:.4f}\nMean diff: {disease_prev_diff:.4f}', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Weighted vs Unweighted: Pi and Prevalence Comparison\nPi Correlation: {pi_correlation:.4f} | Prevalence Correlation: {prev_correlation:.4f}', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / 'weighted_vs_unweighted_pi_and_prevalence.pdf'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison plot to: {plot_path}")
plt.show()

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pi scatter
ax1 = axes[0]
ax1.scatter(pi_unweighted_flat, pi_weighted_flat, alpha=0.3, s=1)
ax1.plot([pi_unweighted_flat.min(), pi_unweighted_flat.max()], 
        [pi_unweighted_flat.min(), pi_unweighted_flat.max()], 'r--', alpha=0.7, linewidth=2)
ax1.set_xlabel('Unweighted Pi Average', fontsize=12)
ax1.set_ylabel('Weighted Pi Average (IPW)', fontsize=12)
ax1.set_title(f'Pi Comparison: All Diseases, All Time Points\nCorrelation: {pi_correlation:.4f}', 
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Prevalence scatter
ax2 = axes[1]
ax2.scatter(prev_unweighted_flat, prev_weighted_flat, alpha=0.3, s=1)
ax2.plot([prev_unweighted_flat.min(), prev_unweighted_flat.max()], 
        [prev_unweighted_flat.min(), prev_unweighted_flat.max()], 'r--', alpha=0.7, linewidth=2)
ax2.set_xlabel('Unweighted Prevalence', fontsize=12)
ax2.set_ylabel('Weighted Prevalence (IPW)', fontsize=12)
ax2.set_title(f'Prevalence Comparison: All Diseases, All Time Points\nCorrelation: {prev_correlation:.4f}', 
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.tight_layout()

scatter_path = output_dir / 'weighted_vs_unweighted_pi_and_prevalence_scatter.pdf'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved scatter plots to: {scatter_path}")
plt.show()

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"✓ Compared weighted vs unweighted pi (from models)")
print(f"✓ Compared weighted vs unweighted prevalence (from data)")
print(f"\nPi Correlation: {pi_correlation:.6f}")
print(f"Prevalence Correlation: {prev_correlation:.6f}")
print(f"\nThis shows how IPW affects both:")
print(f"  1. Model predictions (pi) - how the model adapts to reweighted population")
print(f"  2. Observed patterns (prevalence) - how the population demographics change with IPW")

