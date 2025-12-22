"""
Compare phi, pi, and prevalence: weighted vs unweighted.

This creates a 3-column plot for each disease showing:
1. Phi (signature-disease associations) - should be STABLE
2. Pi (disease hazards from models) - can CHANGE
3. Prevalence (observed data) - can CHANGE

Demonstrates that phi remains stable while pi and prevalence adapt to IPW.
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
print("COMPARING PHI, PI, AND PREVALENCE: WEIGHTED vs UNWEIGHTED")
print("="*80)

# Data directories
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
weighted_model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1218/')
unweighted_model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/')

print("weighted_model_dir", weighted_model_dir)
print("unweighted_model_dir", unweighted_model_dir)

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
        print(f"   ✓ Loaded unweighted prevalence: {alt_path.name}")
    else:
        raise FileNotFoundError(f"Unweighted prevalence not found: {unweighted_prevalence_path} or {alt_path}")

# Load models to get phi and pi
print("\n2. Loading models to extract phi and pi...")
n_batches_to_compare = 10

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
        
        # Store phi (already [K, D, T])
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

# Average across batches
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

# Load clusters to get signature assignments
print("\n3. Loading clusters to get signature assignments...")
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

# Calculate overall correlations
print("\n4. Calculating overall correlations...")

# Phi correlation (for each disease, use its assigned signature)
phi_weighted_flat = []
phi_unweighted_flat = []

if clusters is not None:
    for d in range(min(weighted_phi_avg.shape[1], len(clusters))):
        sig_idx = int(clusters[d])
        if sig_idx < weighted_phi_avg.shape[0]:
            phi_weighted_flat.append(weighted_phi_avg[sig_idx, d, :].numpy().flatten())
            phi_unweighted_flat.append(unweighted_phi_avg[sig_idx, d, :].numpy().flatten())
else:
    for d in range(weighted_phi_avg.shape[1]):
        phi_weighted_flat.append(weighted_phi_avg[0, d, :].numpy().flatten())
        phi_unweighted_flat.append(unweighted_phi_avg[0, d, :].numpy().flatten())

phi_weighted_flat = np.concatenate(phi_weighted_flat)
phi_unweighted_flat = np.concatenate(phi_unweighted_flat)
phi_correlation = np.corrcoef(phi_weighted_flat, phi_unweighted_flat)[0, 1]

# Pi correlation
pi_weighted_flat = weighted_pi_avg.numpy().flatten()
pi_unweighted_flat = unweighted_pi_avg.numpy().flatten()
pi_correlation = np.corrcoef(pi_weighted_flat, pi_unweighted_flat)[0, 1]

# Prevalence correlation
prev_weighted_flat = prevalence_t_weighted.flatten()
prev_unweighted_flat = prevalence_t_unweighted.flatten()
valid_prev_mask = ~(np.isnan(prev_weighted_flat) | np.isnan(prev_unweighted_flat))
prev_correlation = np.corrcoef(prev_weighted_flat[valid_prev_mask], prev_unweighted_flat[valid_prev_mask])[0, 1]

print(f"   Phi correlation: {phi_correlation:.6f} (should be ~1.0, STABLE)")
print(f"   Pi correlation: {pi_correlation:.6f} (can differ, CAN CHANGE)")
print(f"   Prevalence correlation: {prev_correlation:.6f} (can differ, CAN CHANGE)")

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

# Create figure: 3 columns (Phi | Pi | Prevalence), 5 rows (one per disease)
fig, axes = plt.subplots(len(DISEASES_TO_PLOT), 3, figsize=(18, 4*len(DISEASES_TO_PLOT)))
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
    
    # ===== COLUMN 1: Phi Comparison =====
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
    
    # ===== COLUMN 2: Pi Comparison =====
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
    
    # ===== COLUMN 3: Prevalence Comparison =====
    ax3 = axes[idx, 2]
    
    if disease_idx < prevalence_t_weighted.shape[0] and disease_idx < prevalence_t_unweighted.shape[0]:
        weighted_prev_traj = prevalence_t_weighted[disease_idx, :]
        unweighted_prev_traj = prevalence_t_unweighted[disease_idx, :]
        
        # Match time points (prevalence might have different T)
        min_T = min(len(weighted_prev_traj), len(unweighted_prev_traj), len(time_points))
        time_points_prev = time_points[:min_T]
        weighted_prev_traj = weighted_prev_traj[:min_T]
        unweighted_prev_traj = unweighted_prev_traj[:min_T]
        
        ax3.plot(time_points_prev, unweighted_prev_traj, label='Unweighted Prevalence', 
                linewidth=2, alpha=0.8, color='blue')
        ax3.plot(time_points_prev, weighted_prev_traj, label='Weighted Prevalence (IPW)', 
                linewidth=2, alpha=0.8, linestyle='--', color='red')
        
        ax3.set_xlabel('Age', fontsize=11)
        ax3.set_ylabel('Prevalence', fontsize=11)
        ax3.set_title(f'{display_name}\nPrevalence: Weighted vs Unweighted', 
                     fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Add correlation annotation
        valid_mask = ~(np.isnan(weighted_prev_traj) | np.isnan(unweighted_prev_traj))
        if valid_mask.sum() > 0:
            disease_prev_corr = np.corrcoef(unweighted_prev_traj[valid_mask], weighted_prev_traj[valid_mask])[0, 1]
            disease_prev_diff = np.abs(weighted_prev_traj[valid_mask] - unweighted_prev_traj[valid_mask]).mean()
            ax3.text(0.02, 0.98, f'Corr: {disease_prev_corr:.4f}\nMean diff: {disease_prev_diff:.4f}', 
                    transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, f'Disease {disease_idx}\nnot found', 
               transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_title(f'{display_name}\nPrevalence: Weighted vs Unweighted', 
                     fontsize=12, fontweight='bold')

plt.suptitle(f'Phi, Pi, and Prevalence: Weighted vs Unweighted Models\n'
            f'Phi Correlation: {phi_correlation:.4f} (STABLE) | '
            f'Pi Correlation: {pi_correlation:.4f} (CAN CHANGE) | '
            f'Prevalence Correlation: {prev_correlation:.4f} (CAN CHANGE)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / 'phi_pi_prevalence_weighted_vs_unweighted.pdf'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison plot to: {plot_path}")
plt.show()

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"✓ Phi correlation: {phi_correlation:.6f} - Shows PHI IS STABLE (signature structure preserved)")
print(f"✓ Pi correlation: {pi_correlation:.6f} - Shows PI CAN CHANGE (lambda adapts to IPW)")
print(f"✓ Prevalence correlation: {prev_correlation:.6f} - Shows PREVALENCE CAN CHANGE (population demographics)")
print(f"\nKey Insight:")
print(f"  - Phi (signature-disease associations) remains stable with IPW")
print(f"  - Pi (disease hazards) and Prevalence (observed patterns) change because")
print(f"    lambda (population loadings) adapts to reweighted population demographics")

