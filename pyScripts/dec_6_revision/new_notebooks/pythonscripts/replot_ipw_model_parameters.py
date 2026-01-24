"""
Replot IPW model parameter comparison plots using saved aggregated parameters.
This avoids retraining the models - just loads saved results and replots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd

print("="*80)
print("REPLOTTING IPW MODEL PARAMETER COMPARISONS (from saved aggregated results)")
print("="*80)

# Data directories
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')

# Load disease names if available
disease_names_dict = {}
try:
    disease_names_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv")
    if not disease_names_path.exists():
        disease_names_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv")
    
    if disease_names_path.exists():
        disease_df = pd.read_csv(disease_names_path)
        if 'x' in disease_df.columns:
            first_col = disease_df.columns[0]
            disease_indices = disease_df[first_col].astype(int) - 1
            disease_names_dict = dict(zip(disease_indices, disease_df['x']))
        elif 'index' in disease_df.columns and 'name' in disease_df.columns:
            disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
        print(f"✓ Loaded {len(disease_names_dict)} disease names")
except Exception as e:
    print(f"⚠ Could not load disease names: {e}")

# Define diseases to plot (same as demonstrate_ipw_correction.py)
DISEASES_TO_PLOT = [
    (21, "Prostate cancer [male]"),
    (112, "Myocardial Infarction"),
    (256, "Disease 256 [sex-biased]"),
    (260, "Disease 260 [sex-biased]"),
    (265, "Disease 265 [sex-biased]"),
    (270, "Disease 270 [sex-biased]"),
    (275, "Disease 275 [sex-biased]"),
    (280, "Disease 280 [sex-biased]"),
]

# Load aggregated parameters
print("\n1. Loading aggregated parameters from saved files...")
phi_full = np.load(output_dir / 'aggregated_phi_full.npy')
phi_biased = np.load(output_dir / 'aggregated_phi_biased.npy')
phi_biased_ipw = np.load(output_dir / 'aggregated_phi_biased_ipw.npy')

lambda_full = np.load(output_dir / 'aggregated_lambda_full.npy')
lambda_biased = np.load(output_dir / 'aggregated_lambda_biased.npy')
lambda_biased_ipw = np.load(output_dir / 'aggregated_lambda_biased_ipw.npy')

pi_full = np.load(output_dir / 'aggregated_pi_full.npy')
pi_biased = np.load(output_dir / 'aggregated_pi_biased.npy')
pi_biased_ipw = np.load(output_dir / 'aggregated_pi_biased_ipw.npy')

print(f"   ✓ Loaded phi: {phi_full.shape}")
print(f"   ✓ Loaded lambda: {lambda_full.shape}")
print(f"   ✓ Loaded pi: {pi_full.shape}")

# ============================================================================
# PLOT MODEL PARAMETER COMPARISONS
# ============================================================================
print("\n2. Creating model parameter comparison plots...")
print(f"   Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Plot phi, lambda, pi comparisons for selected diseases
fig_params, axes_params = plt.subplots(len(DISEASES_TO_PLOT), 3, figsize=(18, 4*len(DISEASES_TO_PLOT)))
if len(DISEASES_TO_PLOT) == 1:
    axes_params = axes_params.reshape(1, -1)

time_points = np.arange(phi_full.shape[2]) + 30

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if disease_idx >= phi_full.shape[1]:
        continue
    
    display_name = disease_names_dict.get(disease_idx, disease_name) if disease_names_dict else disease_name
    
    # Column 1: Phi comparison
    ax1 = axes_params[idx, 0]
    phi_full_disease = phi_full[:, disease_idx, :].mean(axis=0)  # Average over signatures
    phi_biased_disease = phi_biased[:, disease_idx, :].mean(axis=0)
    phi_biased_ipw_disease = phi_biased_ipw[:, disease_idx, :].mean(axis=0)
    
    ax1.plot(time_points, phi_full_disease, label='Full Population', linewidth=2, color='black', linestyle='-')
    ax1.plot(time_points, phi_biased_disease, label='Biased (no IPW)', linewidth=2, linestyle='-', color='blue')
    ax1.plot(time_points, phi_biased_ipw_disease, label='Biased (with IPW)', linewidth=2, linestyle='--', color='red')
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Average Phi', fontsize=11)
    ax1.set_title(f'{display_name}\nPhi Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Column 2: Lambda comparison (for top signature for this disease)
    # Find the signature with highest average phi for this disease (using full population phi)
    phi_full_disease_all_sigs = phi_full[:, disease_idx, :]  # [K, T]
    top_sig_idx = np.argmax(phi_full_disease_all_sigs.mean(axis=1))  # Average over time, then argmax over signatures
    
    # Extract lambda for the top signature (lambda is already [K, T] after averaging over individuals)
    lambda_full_disease = lambda_full[top_sig_idx, :]  # [T] - lambda for top signature
    lambda_biased_disease = lambda_biased[top_sig_idx, :]
    lambda_biased_ipw_disease = lambda_biased_ipw[top_sig_idx, :]
    
    ax2 = axes_params[idx, 1]
    ax2.plot(time_points, lambda_full_disease, label='Full Population', linewidth=2, color='black', linestyle='-')
    ax2.plot(time_points, lambda_biased_disease, label='Biased (no IPW)', linewidth=2, linestyle='-', color='blue')
    ax2.plot(time_points, lambda_biased_ipw_disease, label='Biased (with IPW)', linewidth=2, linestyle='--', color='red')
    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel(f'Lambda (Sig {top_sig_idx})', fontsize=11)
    ax2.set_title(f'{display_name}\nLambda Comparison (Top Signature)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Column 3: Pi comparison
    ax3 = axes_params[idx, 2]
    pi_full_disease = pi_full[disease_idx, :]
    pi_biased_disease = pi_biased[disease_idx, :]
    pi_biased_ipw_disease = pi_biased_ipw[disease_idx, :]
    
    ax3.plot(time_points, pi_full_disease, label='Full Population', linewidth=2, color='black', linestyle='-')
    ax3.plot(time_points, pi_biased_disease, label='Biased (no IPW)', linewidth=2, linestyle='-', color='blue')
    ax3.plot(time_points, pi_biased_ipw_disease, label='Biased (with IPW)', linewidth=2, linestyle='--', color='red')
    ax3.set_xlabel('Age', fontsize=11)
    ax3.set_ylabel('Average Pi', fontsize=11)
    ax3.set_title(f'{display_name}\nPi Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

plt.suptitle('Model Parameter Comparison: Full Population vs Biased Sample (with/without IPW)\nSame Prevalence Initialization', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save parameter comparison plot
params_plot_path = output_dir / 'ipw_correction_model_parameters.pdf'
plt.savefig(params_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved model parameter comparison plot to: {params_plot_path}")

print(f"\n{'='*80}")
print(f"COMPLETE - Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print(f"✓ Replotted model parameter comparisons using saved aggregated results")
print(f"✓ Lambda now shows top signature for each disease (instead of averaged)")
print(f"✓ Saved: {params_plot_path}")









