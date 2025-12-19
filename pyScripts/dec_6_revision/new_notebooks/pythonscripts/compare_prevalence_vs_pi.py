"""
Compare prevalence vs pi for both unweighted and weighted models.

This script:
1. Loads unweighted prevalence (prevalence_t_corrected.pt)
2. Loads weighted prevalence (prevalence_t_weighted_corrected.pt)
3. Loads/computes unweighted pi (from unweighted models)
4. Loads/computes weighted pi (from weighted models)
5. Creates side-by-side comparison plots showing:
   - Unweighted prevalence vs unweighted pi
   - Weighted prevalence vs weighted pi

This demonstrates that each model produces pi that matches its corresponding prevalence.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add paths
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred

print("="*80)
print("COMPARING PREVALENCE VS PI: UNWEIGHTED AND WEIGHTED")
print("="*80)

# Configuration
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
output_dir.mkdir(parents=True, exist_ok=True)

# Load prevalences
print("\n1. Loading prevalences...")
unweighted_prevalence_path = data_dir / 'prevalence_t_corrected.pt'
weighted_prevalence_path = data_dir / 'prevalence_t_weighted_corrected.pt'

if unweighted_prevalence_path.exists():
    prevalence_t_unweighted = torch.load(str(unweighted_prevalence_path), weights_only=False)
    if torch.is_tensor(prevalence_t_unweighted):
        prevalence_t_unweighted = prevalence_t_unweighted.numpy()
    print(f"   ✓ Loaded unweighted prevalence: {prevalence_t_unweighted.shape}")
else:
    raise FileNotFoundError(f"Unweighted prevalence not found: {unweighted_prevalence_path}")

if weighted_prevalence_path.exists():
    prevalence_t_weighted = torch.load(str(weighted_prevalence_path), weights_only=False)
    if torch.is_tensor(prevalence_t_weighted):
        prevalence_t_weighted = prevalence_t_weighted.numpy()
    print(f"   ✓ Loaded weighted prevalence: {prevalence_t_weighted.shape}")
else:
    raise FileNotFoundError(f"Weighted prevalence not found: {weighted_prevalence_path}")

# Load unweighted pi (from unweighted models)
print("\n2. Loading unweighted pi from models...")
unweighted_model_dir = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/")
n_batches = 10
unweighted_pi_list = []

for batch_idx in range(n_batches):
    unweighted_path = unweighted_model_dir / f"enrollment_model_W0.0001_batch_{batch_idx*10000}_{(batch_idx+1)*10000}.pt"
    
    if unweighted_path.exists():
        print(f"   Loading batch {batch_idx+1}/{n_batches}: {unweighted_path.name}")
        unweighted_ckpt = torch.load(str(unweighted_path), weights_only=False, map_location='cpu')
        
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
        
        unweighted_pi_batch = calculate_pi_pred(unweighted_lambda, unweighted_phi, unweighted_kappa)
        unweighted_pi_list.append(unweighted_pi_batch)
    else:
        print(f"   ⚠️  Batch {batch_idx+1} not found: {unweighted_path}")

if len(unweighted_pi_list) > 0:
    unweighted_pi_all = torch.cat(unweighted_pi_list, dim=0)
    unweighted_pi_avg = unweighted_pi_all.mean(dim=0)  # [D, T]
    print(f"   ✓ Computed unweighted pi average: {unweighted_pi_avg.shape}")
else:
    raise RuntimeError("No unweighted models found!")

# Load weighted pi (from weighted models)
print("\n3. Loading weighted pi from models...")
weighted_model_dir = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1218/")
weighted_pi_list = []

for batch_idx in range(n_batches):
    weighted_paths = [
        weighted_model_dir / f"batch_0{batch_idx:02d}_model.pt",
        weighted_model_dir / f"batch_0{batch_idx}_model.pt",
        weighted_model_dir / f"enrollment_model_W0.0001_batch_{batch_idx*10000}_{(batch_idx+1)*10000}.pt",
    ]
    
    weighted_path = None
    for p in weighted_paths:
        if p.exists():
            weighted_path = p
            break
    
    if weighted_path and weighted_path.exists():
        print(f"   Loading batch {batch_idx+1}/{n_batches}: {weighted_path.name}")
        weighted_ckpt = torch.load(str(weighted_path), weights_only=False, map_location='cpu')
        
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
    else:
        print(f"   ⚠️  Batch {batch_idx+1} not found")

if len(weighted_pi_list) > 0:
    weighted_pi_all = torch.cat(weighted_pi_list, dim=0)
    weighted_pi_avg = weighted_pi_all.mean(dim=0)  # [D, T]
    print(f"   ✓ Computed weighted pi average: {weighted_pi_avg.shape}")
else:
    raise RuntimeError("No weighted models found!")

# Compare
print("\n" + "="*80)
print("COMPARISON STATISTICS")
print("="*80)

# Unweighted comparison
unweighted_pi_flat = unweighted_pi_avg.numpy().flatten()
unweighted_prev_flat = prevalence_t_unweighted.flatten()

valid_unweighted = ~(np.isnan(unweighted_pi_flat) | np.isnan(unweighted_prev_flat) | 
                     np.isinf(unweighted_pi_flat) | np.isinf(unweighted_prev_flat))
unweighted_pi_valid = unweighted_pi_flat[valid_unweighted]
unweighted_prev_valid = unweighted_prev_flat[valid_unweighted]

unweighted_corr = np.corrcoef(unweighted_pi_valid, unweighted_prev_valid)[0, 1]
unweighted_mean_diff = np.abs(unweighted_pi_valid - unweighted_prev_valid).mean()

print(f"\nUnweighted Model:")
print(f"  Correlation (prevalence vs pi): {unweighted_corr:.6f}")
print(f"  Mean absolute difference: {unweighted_mean_diff:.6f}")

# Weighted comparison
weighted_pi_flat = weighted_pi_avg.numpy().flatten()
weighted_prev_flat = prevalence_t_weighted.flatten()

valid_weighted = ~(np.isnan(weighted_pi_flat) | np.isnan(weighted_prev_flat) | 
                   np.isinf(weighted_pi_flat) | np.isinf(weighted_prev_flat))
weighted_pi_valid = weighted_pi_flat[valid_weighted]
weighted_prev_valid = weighted_prev_flat[valid_weighted]

weighted_corr = np.corrcoef(weighted_pi_valid, weighted_prev_valid)[0, 1]
weighted_mean_diff = np.abs(weighted_pi_valid - weighted_prev_valid).mean()

print(f"\nWeighted Model:")
print(f"  Correlation (prevalence vs pi): {weighted_corr:.6f}")
print(f"  Mean absolute difference: {weighted_mean_diff:.6f}")

# Load disease names
disease_names_dict = {}
try:
    disease_names_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv")
    if disease_names_path.exists():
        disease_df = pd.read_csv(disease_names_path)
        disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
except:
    pass

DISEASES_TO_PLOT = [
    (112, "Myocardial Infarction"),
    (66, "Depression"),
    (16, "Breast cancer [female]"),
    (127, "Atrial fibrillation"),
    (47, "Type 2 diabetes"),
]

# Create comparison plots
print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)

# 1. Side-by-side scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Unweighted scatter
ax1 = axes[0]
ax1.scatter(unweighted_prev_valid, unweighted_pi_valid, alpha=0.3, s=1)
ax1.plot([unweighted_prev_valid.min(), unweighted_prev_valid.max()], 
         [unweighted_prev_valid.min(), unweighted_prev_valid.max()], 
         'r--', alpha=0.7, linewidth=2)
ax1.set_xlabel('Unweighted Prevalence', fontsize=12)
ax1.set_ylabel('Unweighted Pi (from model)', fontsize=12)
ax1.set_title(f'Unweighted Model\nCorrelation: {unweighted_corr:.4f}\nMean diff: {unweighted_mean_diff:.6f}', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Weighted scatter
ax2 = axes[1]
ax2.scatter(weighted_prev_valid, weighted_pi_valid, alpha=0.3, s=1)
ax2.plot([weighted_prev_valid.min(), weighted_prev_valid.max()], 
         [weighted_prev_valid.min(), weighted_prev_valid.max()], 
         'r--', alpha=0.7, linewidth=2)
ax2.set_xlabel('Weighted Prevalence (IPW)', fontsize=12)
ax2.set_ylabel('Weighted Pi (from model)', fontsize=12)
ax2.set_title(f'Weighted Model (IPW)\nCorrelation: {weighted_corr:.4f}\nMean diff: {weighted_mean_diff:.6f}', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.suptitle('Prevalence vs Pi: Model Alignment\n(Each model produces pi that matches its corresponding prevalence)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

scatter_path = output_dir / 'prevalence_vs_pi_comparison_scatter.pdf'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved scatter plot to: {scatter_path}")
plt.close()

# 2. Trajectory comparison for selected diseases
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if idx >= len(axes):
        break
    
    ax = axes[idx]
    
    if disease_names_dict and disease_idx in disease_names_dict:
        display_name = disease_names_dict[disease_idx]
    else:
        display_name = disease_name
    
    if disease_idx < unweighted_pi_avg.shape[0] and disease_idx < weighted_pi_avg.shape[0]:
        unweighted_pi_traj = unweighted_pi_avg[disease_idx, :].numpy()
        unweighted_prev_traj = prevalence_t_unweighted[disease_idx, :]
        weighted_pi_traj = weighted_pi_avg[disease_idx, :].numpy()
        weighted_prev_traj = prevalence_t_weighted[disease_idx, :]
        
        time_points = np.arange(len(unweighted_pi_traj)) + 30
        
        # Unweighted
        ax.plot(time_points, unweighted_prev_traj, label='Unweighted Prevalence', 
               linewidth=2, alpha=0.8, color='blue', linestyle='-')
        ax.plot(time_points, unweighted_pi_traj, label='Unweighted Pi', 
               linewidth=2, alpha=0.8, color='blue', linestyle='--')
        
        # Weighted
        ax.plot(time_points, weighted_prev_traj, label='Weighted Prevalence (IPW)', 
               linewidth=2, alpha=0.8, color='red', linestyle='-')
        ax.plot(time_points, weighted_pi_traj, label='Weighted Pi (IPW)', 
               linewidth=2, alpha=0.8, color='red', linestyle='--')
        
        ax.set_xlabel('Age', fontsize=11)
        ax.set_ylabel('Prevalence / Pi', fontsize=11)
        ax.set_title(f'{display_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add correlations
        unweighted_disease_corr = np.corrcoef(unweighted_prev_traj, unweighted_pi_traj)[0, 1]
        weighted_disease_corr = np.corrcoef(weighted_prev_traj, weighted_pi_traj)[0, 1]
        ax.text(0.02, 0.98, f'Unw: {unweighted_disease_corr:.3f}\nWtd: {weighted_disease_corr:.3f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, f'Disease {disease_idx}\nnot found', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(f'{disease_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')

# Remove extra subplot
if len(DISEASES_TO_PLOT) < len(axes):
    axes[len(DISEASES_TO_PLOT)].axis('off')

plt.suptitle(f'Prevalence vs Pi Trajectories: Unweighted vs Weighted Models\n(All {n_batches} batches, N={unweighted_pi_all.shape[0]:,} patients)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

trajectory_path = output_dir / 'prevalence_vs_pi_comparison_trajectories.pdf'
plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved trajectory plot to: {trajectory_path}")
plt.close()

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"✓ Unweighted model: Correlation = {unweighted_corr:.6f}")
print(f"✓ Weighted model: Correlation = {weighted_corr:.6f}")
print(f"\nKey Insight:")
print(f"  Each model produces pi that closely matches its corresponding prevalence.")
print(f"  This demonstrates that the model adapts correctly to the population it's trained on.")

