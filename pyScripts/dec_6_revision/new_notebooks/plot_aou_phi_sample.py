#!/usr/bin/env python3
"""
Plot sample phi values from AOU batches.
Shows phi for a specific signature-disease pair with:
- Mean phi across all batches
- Individual lines for each batch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy.special import expit as sigmoid

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Load clusters and disease names
clusters_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt'
clusters_ckpt = torch.load(clusters_path, map_location='cpu', weights_only=False)
clusters = clusters_ckpt['clusters']
if torch.is_tensor(clusters):
    clusters = clusters.numpy()

disease_names = clusters_ckpt['disease_names']
if isinstance(disease_names, (list, tuple)):
    disease_names = list(disease_names)
elif hasattr(disease_names, 'values'):
    disease_names = disease_names.values.tolist()

K = int(clusters.max() + 1)
D = len(clusters)
print(f"Number of signatures: {K}")
print(f"Number of diseases: {D}")

# Find breast lump/mass diseases
print("\nSearching for breast lump/mass diseases...")
breast_diseases = []
for d in range(D):
    if d < len(disease_names):
        name = str(disease_names[d]).lower()
        if ('breast' in name or 'mammary' in name) and ('lump' in name or 'mass' in name):
            sig = int(clusters[d])
            breast_diseases.append((d, sig, disease_names[d]))
            print(f"  Disease {d}: '{disease_names[d]}' -> Signature {sig}")

if len(breast_diseases) == 0:
    print("  No exact match found, searching for breast-related diseases...")
    for d in range(D):
        if d < len(disease_names):
            name = str(disease_names[d]).lower()
            if 'breast' in name or 'mammary' in name:
                sig = int(clusters[d])
                breast_diseases.append((d, sig, disease_names[d]))
                print(f"  Disease {d}: '{disease_names[d]}' -> Signature {sig}")

# Load phi from all batches
batch_pattern = '/Users/sarahurbut/Library/CloudStorage/Dropbox/aou_batches/aou_model_batch_*_*_*.pt'
batch_files = sorted(glob.glob(batch_pattern))
print(f"\nFound {len(batch_files)} batch files")

all_phis = []
for batch_file in batch_files:
    print(f"  Loading {Path(batch_file).name}...")
    ckpt = torch.load(batch_file, map_location='cpu', weights_only=False)
    
    # Extract phi
    if 'model_state_dict' in ckpt and 'phi' in ckpt['model_state_dict']:
        phi = ckpt['model_state_dict']['phi']
    elif 'phi' in ckpt:
        phi = ckpt['phi']
    else:
        print(f"    Warning: No phi found in {Path(batch_file).name}")
        continue
    
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    
    all_phis.append(phi)
    print(f"    Phi shape: {phi.shape}")

if len(all_phis) == 0:
    raise ValueError("No phi arrays loaded!")

# Stack all phis: (n_batches, K, D, T)
phi_stack = np.stack(all_phis, axis=0)
n_batches, K, D, T = phi_stack.shape
ages = np.arange(30, 30 + T)

print(f"\nStacked phi shape: {phi_stack.shape}")
print(f"Number of batches: {n_batches}")

# Compute mean across batches
phi_mean = np.mean(phi_stack, axis=0)  # (K, D, T)

# Plot breast lump/mass diseases (or fallback to sample diseases)
if len(breast_diseases) > 0:
    # Use breast diseases, focusing on signature 0 if available
    diseases_to_plot = []
    for d, sig, name in breast_diseases:
        if sig == 0:  # Prioritize signature 0
            diseases_to_plot.insert(0, (d, sig, name))
        else:
            diseases_to_plot.append((d, sig, name))
    
    # Also add other signature 0 diseases for comparison
    print("\nFinding other Signature 0 diseases for comparison...")
    sig0_diseases = []
    for d in range(D):
        if int(clusters[d]) == 0 and d < len(disease_names):
            name = disease_names[d]
            # Skip if already in breast_diseases
            if not any(bd[0] == d for bd in breast_diseases):
                sig0_diseases.append((d, 0, name))
    
    # Add a few other sig 0 diseases for comparison
    if len(sig0_diseases) > 0:
        print(f"  Found {len(sig0_diseases)} other Signature 0 diseases")
        # Add top 3-4 other sig 0 diseases
        diseases_to_plot.extend(sig0_diseases[:4])
    
    # Limit to first 6 for readability
    diseases_to_plot = diseases_to_plot[:6]
    print(f"\nPlotting {len(diseases_to_plot)} diseases (breast + other Sig 0):")
    for d, sig, name in diseases_to_plot:
        print(f"  Disease {d}: '{name}' -> Signature {sig}")
else:
    # Fallback to sample diseases
    print("\nNo breast diseases found, using sample diseases...")
    diseases_to_plot = [(d, int(clusters[d]), f'Disease {d}') for d in [0, 10, 50, 100, 200] if d < D]

n_plots = len(diseases_to_plot)
# Create 2 rows per disease: phi on top, probability below
fig, axes = plt.subplots(2*n_plots, 1, figsize=(14, 3*2*n_plots), sharex=True)

for idx, (disease_idx, sig_idx, disease_name) in enumerate(diseases_to_plot):
    # Phi plot (top)
    ax_phi = axes[2*idx]
    
    # Calculate std across batches at each age
    phi_std_values = phi_stack[:, sig_idx, disease_idx, :].std(axis=0)
    phi_mean_values = phi_mean[sig_idx, disease_idx, :]
    
    # Plot individual batch lines (light, transparent)
    for batch_idx in range(n_batches):
        phi_batch = phi_stack[batch_idx, sig_idx, disease_idx, :]
        ax_phi.plot(ages, phi_batch, linewidth=0.8, alpha=0.3, color='gray')
    
    # Plot mean line (bold, prominent)
    ax_phi.plot(ages, phi_mean_values, linewidth=3, color='darkblue', 
                label=f'Mean (n={n_batches} batches)', zorder=10)
    
    # Plot ±1 SD bands to show variability by age
    ax_phi.fill_between(ages, 
                        phi_mean_values - phi_std_values,
                        phi_mean_values + phi_std_values,
                        alpha=0.2, color='blue', label='±1 SD', zorder=5)
    
    ax_phi.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax_phi.set_ylabel(f'φ (Sig {sig_idx})', fontsize=11)
    
    # Add max std to title to show variability
    max_std = phi_std_values.max()
    ax_phi.set_title(f'{disease_name} (Disease {disease_idx}, Signature {sig_idx}) - Max std: {max_std:.4f}', 
                     fontsize=12, fontweight='bold')
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(fontsize=10)
    
    # Probability plot (bottom)
    ax_prob = axes[2*idx + 1]
    
    # Convert phi to probability using sigmoid
    prob_stack = sigmoid(phi_stack[:, sig_idx, disease_idx, :])  # (n_batches, T)
    prob_mean = sigmoid(phi_mean_values)
    prob_std_values = prob_stack.std(axis=0)
    
    # Plot individual batch probability lines
    for batch_idx in range(n_batches):
        prob_batch = prob_stack[batch_idx, :]
        ax_prob.plot(ages, prob_batch, linewidth=0.8, alpha=0.3, color='gray')
    
    # Plot mean probability line
    ax_prob.plot(ages, prob_mean, linewidth=3, color='darkred', 
                 label=f'Mean (n={n_batches} batches)', zorder=10)
    
    # Plot ±1 SD bands
    ax_prob.fill_between(ages,
                         prob_mean - prob_std_values,
                         prob_mean + prob_std_values,
                         alpha=0.2, color='red', label='±1 SD', zorder=5)
    
    ax_prob.set_ylabel('Probability', fontsize=11)
    ax_prob.set_title(f'{disease_name} - Probability (sigmoid(φ))', fontsize=12, fontweight='bold')
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(fontsize=10)
    ax_prob.set_ylim([0, None])  # Probability should be >= 0

axes[-1].set_xlabel('Age (yr)', fontsize=12)
title_suffix = "Breast Lump/Mass Diseases" if len(breast_diseases) > 0 else "Sample Diseases"
plt.suptitle(f'AOU Phi Trajectories: {title_suffix} - Mean and Individual Batches (n={n_batches})', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

if len(breast_diseases) > 0:
    output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/aou_analysis/aou_phi_breast_lump_mass.pdf'
else:
    output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/aou_analysis/aou_phi_sample_with_batches.pdf'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")
plt.show()

# Print statistics
print(f"\n{'='*60}")
print("PHI STATISTICS")
print(f"{'='*60}")
print(f"Phi shape: {phi_mean.shape}")
print(f"Phi range: [{phi_mean.min():.4f}, {phi_mean.max():.4f}]")
print(f"Phi mean: {phi_mean.mean():.4f}")
print(f"Phi std (across batches): [{phi_stack.std(axis=0).min():.4f}, {phi_stack.std(axis=0).max():.4f}]")

# Show signature-disease assignments for plotted diseases
print(f"\nPlotted diseases - phi and probability statistics:")
for d, sig, name in diseases_to_plot:
    phi_mean_val = phi_mean[sig, d, :].mean()
    phi_max = phi_mean[sig, d, :].max()
    phi_min = phi_mean[sig, d, :].min()
    phi_std_overall = phi_stack[:, sig, d, :].std(axis=0).mean()
    phi_std_max = phi_stack[:, sig, d, :].std(axis=0).max()
    
    # Probability statistics
    prob_mean_vals = sigmoid(phi_mean[sig, d, :])
    prob_max = prob_mean_vals.max()
    prob_min = prob_mean_vals.min()
    prob_mean_overall = prob_mean_vals.mean()
    prob_stack = sigmoid(phi_stack[:, sig, d, :])
    prob_std_overall = prob_stack.std(axis=0).mean()
    prob_std_max = prob_stack.std(axis=0).max()
    
    # Calculate std by age ranges
    phi_std_by_age = phi_stack[:, sig, d, :].std(axis=0)
    prob_std_by_age = prob_stack.std(axis=0)
    age_ranges = [(30, 40), (40, 50), (50, 60), (60, 70), (70, 81)]
    age_indices = [(int(a-30), int(b-30)) for a, b in age_ranges]
    
    print(f"  {name} (D{d}, Sig{sig}):")
    print(f"    φ: mean={phi_mean_val:.4f}, min={phi_min:.4f}, max={phi_max:.4f}")
    print(f"    φ std: overall={phi_std_overall:.4f}, max={phi_std_max:.4f}")
    print(f"    Probability: mean={prob_mean_overall:.6f}, min={prob_min:.6f}, max={prob_max:.6f}")
    print(f"    Prob std: overall={prob_std_overall:.6f}, max={prob_std_max:.6f}")
    print(f"    Std by age range:")
    for (age_start, age_end), (idx_start, idx_end) in zip(age_ranges, age_indices):
        if idx_end <= len(phi_std_by_age):
            phi_std_in_range = phi_std_by_age[idx_start:idx_end].mean()
            prob_std_in_range = prob_std_by_age[idx_start:idx_end].mean()
            print(f"      Age {age_start}-{age_end}: φ std={phi_std_in_range:.4f}, prob std={prob_std_in_range:.6f}")

