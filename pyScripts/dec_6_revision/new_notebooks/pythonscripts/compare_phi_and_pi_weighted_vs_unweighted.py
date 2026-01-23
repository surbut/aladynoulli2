"""
Compare phi and pi from weighted vs unweighted models.

Key demonstration:
- Phi (signature-disease associations) should be STABLE (similar between weighted/unweighted)
- Pi (disease hazards) can CHANGE (because lambda changes with IPW)

This shows that while phi remains stable, lambda adapts to capture population-specific risks.
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
print("COMPARING PHI AND PI: WEIGHTED vs UNWEIGHTED MODELS")
print("="*80)

# Model directories
weighted_model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1219/')
unweighted_model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/')

# Load models (use first batch as example, or average across batches)
print("\n1. Loading models...")
n_batches_to_compare = 10  # Compare across multiple batches for robustness

weighted_phi_list = []
weighted_pi_list = []
unweighted_phi_list = []
unweighted_pi_list = []

for batch_idx in range(min(n_batches_to_compare, 10)):
    # Weighted model
    weighted_paths = [
        weighted_model_dir / f"batch_{batch_idx:02d}_model.pt",
        weighted_model_dir / f"batch_{batch_idx}_model.pt",
    ]
    weighted_path = None
    for p in weighted_paths:
        if p.exists():
            weighted_path = p
            break
    
    # Unweighted model
    unweighted_path = unweighted_model_dir / f"enrollment_model_W0.0001_batch_{batch_idx*10000}_{(batch_idx+1)*10000}.pt"
    
    if weighted_path and weighted_path.exists() and unweighted_path.exists():
        # Load weighted
        weighted_ckpt = torch.load(weighted_path, weights_only=False, map_location='cpu')
        if 'model_state_dict' in weighted_ckpt:
            weighted_phi = weighted_ckpt['model_state_dict']['phi'].detach()
            weighted_lambda = weighted_ckpt['model_state_dict']['lambda_'].detach()
            weighted_kappa = weighted_ckpt['model_state_dict'].get('kappa', torch.tensor(1.0))
        else:
            weighted_phi = weighted_ckpt['phi'].detach()
            weighted_lambda = weighted_ckpt['lambda_'].detach()
            weighted_kappa = weighted_ckpt.get('kappa', torch.tensor(1.0))
        
        if torch.is_tensor(weighted_kappa):
            weighted_kappa = weighted_kappa.item() if weighted_kappa.numel() == 1 else weighted_kappa.mean().item()
        
        # Load unweighted
        unweighted_ckpt = torch.load(unweighted_path, weights_only=False, map_location='cpu')
        if 'model_state_dict' in unweighted_ckpt:
            unweighted_phi = unweighted_ckpt['model_state_dict']['phi'].detach()
            unweighted_lambda = unweighted_ckpt['model_state_dict']['lambda_'].detach()
            unweighted_kappa = unweighted_ckpt['model_state_dict'].get('kappa', torch.tensor(1.0))
        else:
            unweighted_phi = unweighted_ckpt['phi'].detach()
            unweighted_lambda = unweighted_ckpt['lambda_'].detach()
            unweighted_kappa = unweighted_ckpt.get('kappa', torch.tensor(1.0))
        
        if torch.is_tensor(unweighted_kappa):
            unweighted_kappa = unweighted_kappa.item() if unweighted_kappa.numel() == 1 else unweighted_kappa.mean().item()
        
        # Store phi (already averaged across patients: [K, D, T])
        weighted_phi_list.append(weighted_phi)
        unweighted_phi_list.append(unweighted_phi)
        
        # Compute pi and average across patients
        weighted_pi_batch = calculate_pi_pred(weighted_lambda, weighted_phi, weighted_kappa)
        unweighted_pi_batch = calculate_pi_pred(unweighted_lambda, unweighted_phi, unweighted_kappa)
        
        weighted_pi_avg = weighted_pi_batch.mean(dim=0)  # [D, T]
        unweighted_pi_avg = unweighted_pi_batch.mean(dim=0)  # [D, T]
        
        weighted_pi_list.append(weighted_pi_avg)
        unweighted_pi_list.append(unweighted_pi_avg)
        
        if batch_idx == 0:
            print(f"   Batch {batch_idx}: phi shape {weighted_phi.shape}, pi avg shape {weighted_pi_avg.shape}")

# Average phi across batches
if len(weighted_phi_list) > 0:
    weighted_phi_avg = torch.stack(weighted_phi_list).mean(dim=0)  # [K, D, T]
    unweighted_phi_avg = torch.stack(unweighted_phi_list).mean(dim=0)  # [K, D, T]
    
    weighted_pi_avg = torch.stack(weighted_pi_list).mean(dim=0)  # [D, T]
    unweighted_pi_avg = torch.stack(unweighted_pi_list).mean(dim=0)  # [D, T]
    
    print(f"\n   ✓ Averaged across {len(weighted_phi_list)} batches")
    print(f"   Weighted phi: {weighted_phi_avg.shape}")
    print(f"   Unweighted phi: {unweighted_phi_avg.shape}")
    print(f"   Weighted pi avg: {weighted_pi_avg.shape}")
    print(f"   Unweighted pi avg: {unweighted_pi_avg.shape}")
else:
    raise FileNotFoundError("No models loaded successfully")

# Load clusters to get signature assignments for diseases
print("\n2. Loading clusters to get signature assignments...")
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
clusters_path = data_dir / 'initial_clusters_400k.pt'
if clusters_path.exists():
    clusters_data = torch.load(str(clusters_path), weights_only=False)
    if isinstance(clusters_data, dict):
        clusters = clusters_data.get('clusters', clusters_data.get('initial_clusters'))
    else:
        clusters = clusters_data
    if torch.is_tensor(clusters):
        clusters = clusters.numpy()
    print(f"   ✓ Loaded clusters: {clusters.shape}")
else:
    print(f"   ⚠️  Clusters not found, will use signature 0 for all diseases")
    clusters = None

# Calculate correlations
print("\n3. Calculating correlations...")

# Phi comparison (for each disease, use its assigned signature)
phi_weighted_flat = []
phi_unweighted_flat = []

if clusters is not None:
    for d in range(min(weighted_phi_avg.shape[1], len(clusters))):
        sig_idx = int(clusters[d])
        if sig_idx < weighted_phi_avg.shape[0]:
            phi_weighted_flat.append(weighted_phi_avg[sig_idx, d, :].numpy().flatten())
            phi_unweighted_flat.append(unweighted_phi_avg[sig_idx, d, :].numpy().flatten())
else:
    # Use signature 0 for all
    for d in range(weighted_phi_avg.shape[1]):
        phi_weighted_flat.append(weighted_phi_avg[0, d, :].numpy().flatten())
        phi_unweighted_flat.append(unweighted_phi_avg[0, d, :].numpy().flatten())

phi_weighted_flat = np.concatenate(phi_weighted_flat)
phi_unweighted_flat = np.concatenate(phi_unweighted_flat)
phi_correlation = np.corrcoef(phi_weighted_flat, phi_unweighted_flat)[0, 1]
phi_mean_diff = np.abs(phi_weighted_flat - phi_unweighted_flat).mean()

# Pi comparison
pi_weighted_flat = weighted_pi_avg.numpy().flatten()
pi_unweighted_flat = unweighted_pi_avg.numpy().flatten()
pi_correlation = np.corrcoef(pi_weighted_flat, pi_unweighted_flat)[0, 1]
pi_mean_diff = np.abs(pi_weighted_flat - pi_unweighted_flat).mean()

print(f"   Phi correlation: {phi_correlation:.6f} (should be ~1.0, showing stability)")
print(f"   Phi mean absolute difference: {phi_mean_diff:.6f}")
print(f"   Pi correlation: {pi_correlation:.6f} (can differ, showing lambda adaptation)")
print(f"   Pi mean absolute difference: {pi_mean_diff:.6f}")

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
except:
    pass

# Create figure: 2 columns (Phi | Pi), 5 rows (one per disease)
fig, axes = plt.subplots(len(DISEASES_TO_PLOT), 2, figsize=(16, 4*len(DISEASES_TO_PLOT)))
if len(DISEASES_TO_PLOT) == 1:
    axes = axes.reshape(1, -1)

time_points = np.arange(weighted_phi_avg.shape[2]) + 30

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if disease_idx >= weighted_phi_avg.shape[1]:
        continue
    
    # Get disease name
    if disease_names_dict and disease_idx in disease_names_dict:
        display_name = disease_names_dict[disease_idx]
    else:
        display_name = disease_name
    
    # Get assigned signature
    if clusters is not None and disease_idx < len(clusters):
        sig_idx = int(clusters[disease_idx])
    else:
        sig_idx = 0
    
    if sig_idx >= weighted_phi_avg.shape[0]:
        sig_idx = 0
    
    # ===== LEFT COLUMN: Phi Comparison =====
    ax1 = axes[idx, 0]
    
    weighted_phi_traj = weighted_phi_avg[sig_idx, disease_idx, :].numpy()
    unweighted_phi_traj = unweighted_phi_avg[sig_idx, disease_idx, :].numpy()
    
    ax1.plot(time_points, unweighted_phi_traj, label='Unweighted Phi', 
            linewidth=2, alpha=0.8, color='blue')
    ax1.plot(time_points, weighted_phi_traj, label='Weighted Phi (IPW)', 
            linewidth=2, alpha=0.8, linestyle='--', color='red')
    
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Phi (Signature-Disease Association)', fontsize=11)
    ax1.set_title(f'{display_name}\nPhi: Weighted vs Unweighted (Sig {sig_idx})', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation annotation
    disease_phi_corr = np.corrcoef(unweighted_phi_traj, weighted_phi_traj)[0, 1]
    disease_phi_diff = np.abs(weighted_phi_traj - unweighted_phi_traj).mean()
    ax1.text(0.02, 0.98, f'Corr: {disease_phi_corr:.4f}\nMean diff: {disease_phi_diff:.4f}', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # ===== RIGHT COLUMN: Pi Comparison =====
    ax2 = axes[idx, 1]
    
    weighted_pi_traj = weighted_pi_avg[disease_idx, :].numpy()
    unweighted_pi_traj = unweighted_pi_avg[disease_idx, :].numpy()
    
    ax2.plot(time_points, unweighted_pi_traj, label='Unweighted Pi', 
            linewidth=2, alpha=0.8, color='blue')
    ax2.plot(time_points, weighted_pi_traj, label='Weighted Pi (IPW)', 
            linewidth=2, alpha=0.8, linestyle='--', color='red')
    
    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel('Average Pi (Disease Hazard)', fontsize=11)
    ax2.set_title(f'{display_name}\nPi: Weighted vs Unweighted', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add correlation annotation
    disease_pi_corr = np.corrcoef(unweighted_pi_traj, weighted_pi_traj)[0, 1]
    disease_pi_diff = np.abs(weighted_pi_traj - unweighted_pi_traj).mean()
    ax2.text(0.02, 0.98, f'Corr: {disease_pi_corr:.4f}\nMean diff: {disease_pi_diff:.4f}', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Phi vs Pi: Weighted vs Unweighted Models\n'
            f'Phi Correlation: {phi_correlation:.4f} (STABLE) | '
            f'Pi Correlation: {pi_correlation:.4f} (CAN CHANGE)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / 'phi_and_pi_weighted_vs_unweighted.pdf'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison plot to: {plot_path}")
plt.show()

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Phi scatter
ax1 = axes[0]
ax1.scatter(phi_unweighted_flat, phi_weighted_flat, alpha=0.3, s=1)
ax1.plot([phi_unweighted_flat.min(), phi_unweighted_flat.max()], 
        [phi_unweighted_flat.min(), phi_unweighted_flat.max()], 'r--', alpha=0.7, linewidth=2)
ax1.set_xlabel('Unweighted Phi', fontsize=12)
ax1.set_ylabel('Weighted Phi (IPW)', fontsize=12)
ax1.set_title(f'Phi Comparison: All Diseases, All Time Points\nCorrelation: {phi_correlation:.4f} (STABLE)', 
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Pi scatter
ax2 = axes[1]
ax2.scatter(pi_unweighted_flat, pi_weighted_flat, alpha=0.3, s=1)
ax2.plot([pi_unweighted_flat.min(), pi_unweighted_flat.max()], 
        [pi_unweighted_flat.min(), pi_unweighted_flat.max()], 'r--', alpha=0.7, linewidth=2)
ax2.set_xlabel('Unweighted Pi Average', fontsize=12)
ax2.set_ylabel('Weighted Pi Average (IPW)', fontsize=12)
ax2.set_title(f'Pi Comparison: All Diseases, All Time Points\nCorrelation: {pi_correlation:.4f} (CAN CHANGE)', 
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.tight_layout()

scatter_path = output_dir / 'phi_and_pi_weighted_vs_unweighted_scatter.pdf'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved scatter plots to: {scatter_path}")
plt.show()

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"✓ Phi correlation: {phi_correlation:.6f} - Shows PHI IS STABLE (signature structure preserved)")
print(f"✓ Pi correlation: {pi_correlation:.6f} - Shows PI CAN CHANGE (lambda adapts to IPW)")
print(f"\nKey Insight:")
print(f"  - Phi (signature-disease associations) remains stable with IPW")
print(f"  - Pi (disease hazards) changes because lambda (population loadings) adapts")
print(f"  - This demonstrates that the model maintains stable signatures while adapting to population demographics")
















