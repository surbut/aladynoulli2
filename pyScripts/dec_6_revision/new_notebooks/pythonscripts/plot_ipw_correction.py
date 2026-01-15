"""
Quick replot script for IPW correction demonstration.
Generates ipw_correction_demonstration.pdf with updated line styles.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from scipy.ndimage import gaussian_filter1d
import pandas as pd

print("="*80)
print("GENERATING IPW CORRECTION DEMONSTRATION PLOT")
print("="*80)

# Data directories
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')

# Load disease names if available
disease_names_dict = {}
try:
    disease_names_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv")
    if not disease_names_path.exists():
        disease_names_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv")
    
    if disease_names_path.exists():
        disease_df = pd.read_csv(disease_names_path)
        if 'index' in disease_df.columns and 'name' in disease_df.columns:
            disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
        elif 'x' in disease_df.columns:
            first_col = disease_df.columns[0]
            disease_indices = disease_df[first_col].astype(int) - 1
            disease_names_dict = dict(zip(disease_indices, disease_df['x']))
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

# ============================================================================
# PLOT PREVALENCE COMPARISON (ipw_correction_demonstration.pdf)
# ============================================================================
print(f"\n3. Computing prevalence for demonstration plot...")
print(f"   Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def compute_smoothed_prevalence_at_risk(Y, E_corrected, weights=None, window_size=5, smooth_on_logit=True):
    """Compute smoothed prevalence with at-risk filtering."""
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    
    is_weighted = weights is not None
    if weights is not None:
        weights_norm = weights / weights.sum() * N
    
    for d in range(D):
        for t in range(T):
            at_risk_mask = (E_corrected[:, d] >= t)
            
            if at_risk_mask.sum() == 0:
                prevalence_t[d, t] = 0.0
                continue
            
            Y_at_risk = Y[at_risk_mask, d, t]
            
            if is_weighted:
                weights_at_risk = weights_norm[at_risk_mask]
                numerator = np.sum(weights_at_risk * Y_at_risk)
                denominator = np.sum(weights_at_risk)
                if denominator > 0:
                    prevalence_t[d, t] = numerator / denominator
                else:
                    prevalence_t[d, t] = 0.0
            else:
                prevalence_t[d, t] = Y_at_risk.mean()
    
    # Smooth
    for d in range(D):
        if smooth_on_logit:
            prev_d = prevalence_t[d, :]
            prev_d_clipped = np.clip(prev_d, 1e-6, 1 - 1e-6)
            logit_prev = np.log(prev_d_clipped / (1 - prev_d_clipped))
            logit_prev_smooth = gaussian_filter1d(logit_prev, sigma=window_size/3)
            prevalence_t[d, :] = 1 / (1 + np.exp(-logit_prev_smooth))
        else:
            prevalence_t[d, :] = gaussian_filter1d(prevalence_t[d, :], sigma=window_size/3)
    
    return prevalence_t

# Load full 400K data
print("   Loading full 400K data...")
n_patients = 400000
Y = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
E_corrected = torch.load(str(data_dir / 'E_matrix_corrected.pt'), weights_only=False)

if torch.is_tensor(Y):
    Y = Y.numpy()
if torch.is_tensor(E_corrected):
    E_corrected = E_corrected.numpy()

Y = Y[:n_patients]
E_corrected = E_corrected[:n_patients]

# Load patient IDs and covariates to identify women
pids_csv_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/processed_ids.csv')
pids_df = pd.read_csv(pids_csv_path)
pids = pids_df['eid'].values[:n_patients]

covariates_path = data_dir / 'baselinagefamh_withpcs.csv'
cov_df = pd.read_csv(covariates_path)
sex_col = 'sex'
cov_df = cov_df[['identifier', sex_col]].dropna(subset=['identifier'])
cov_df = cov_df.drop_duplicates(subset=['identifier'])
cov_map = cov_df.set_index('identifier')

# Identify women
is_female = np.zeros(n_patients, dtype=bool)
for i, pid in enumerate(pids):
    if pid in cov_map.index:
        sex_val = cov_map.at[pid, sex_col]
        if sex_val == 0 or sex_val == 'Female' or str(sex_val).lower() == 'female':
            is_female[i] = True

print(f"   Full dataset: {Y.shape}, Women: {is_female.sum():,} ({100*is_female.sum()/n_patients:.1f}%)")

# Compute full population prevalence
print("   Computing full population prevalence...")
prevalence_full = compute_smoothed_prevalence_at_risk(
    Y, E_corrected, weights=None, window_size=5, smooth_on_logit=True
)

# Drop 90% of women (same logic as demonstrate_ipw_correction.py)
print("   Dropping 90% of women...")
np.random.seed(42)  # Same seed for reproducibility
female_indices = np.where(is_female)[0]
n_females_to_keep = int(len(female_indices) * 0.1)  # Keep only 10% = drop 90%
females_to_keep = np.random.choice(female_indices, size=n_females_to_keep, replace=False)
female_mask = np.zeros(n_patients, dtype=bool)
female_mask[females_to_keep] = True
male_mask = ~is_female

remaining_mask = male_mask | female_mask
Y_dropped = Y[remaining_mask]
E_dropped = E_corrected[remaining_mask]
is_female_dropped = is_female[remaining_mask]

print(f"   After drop: {remaining_mask.sum():,} patients ({is_female_dropped.sum():,} women)")

# Prevalence without IPW
print("   Computing prevalence without IPW...")
prevalence_biased = compute_smoothed_prevalence_at_risk(
    Y_dropped, E_dropped, weights=None, window_size=5, smooth_on_logit=True
)

# Compute IPW weights
n_women_full = is_female.sum()
n_men_full = (~is_female).sum()
n_women_dropped = is_female_dropped.sum()
n_men_dropped = (~is_female_dropped).sum()

prop_women_full = n_women_full / n_patients
prop_men_full = n_men_full / n_patients
prop_women_dropped = n_women_dropped / remaining_mask.sum()
prop_men_dropped = n_men_dropped / remaining_mask.sum()

ipw_weights = np.ones(remaining_mask.sum())
ipw_weights[is_female_dropped] = prop_women_full / (prop_women_dropped + 1e-10)
ipw_weights[~is_female_dropped] = prop_men_full / (prop_men_dropped + 1e-10)
ipw_weights = ipw_weights / ipw_weights.mean()

# Prevalence with IPW
print("   Computing prevalence with IPW...")
prevalence_biased_ipw = compute_smoothed_prevalence_at_risk(
    Y_dropped, E_dropped, weights=ipw_weights, window_size=5, smooth_on_logit=True
)

print(f"   ✓ Computed all prevalence curves")

print(f"\n4. Creating prevalence demonstration plot...")
print(f"   Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

fig, axes = plt.subplots(len(DISEASES_TO_PLOT), 1, figsize=(12, 4*len(DISEASES_TO_PLOT)))
if len(DISEASES_TO_PLOT) == 1:
    axes = [axes]

time_points = np.arange(prevalence_full.shape[1]) + 30

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if disease_idx >= prevalence_full.shape[0]:
        continue
    
    ax = axes[idx]
    
    if disease_names_dict and disease_idx in disease_names_dict:
        display_name = disease_names_dict[disease_idx]
    else:
        display_name = disease_name
    
    # Get trajectories
    full_traj = prevalence_full[disease_idx, :]
    no_adj_traj = prevalence_biased[disease_idx, :]
    ipw_traj = prevalence_biased_ipw[disease_idx, :]
    
    # Plot
    ax.plot(time_points, full_traj, label='Full Population (Baseline)', 
           linewidth=3, alpha=0.9, color='black', linestyle='-')
    ax.plot(time_points, no_adj_traj, label='90% Women Dropped (No Adjustment)', 
           linewidth=2, alpha=0.8, color='blue', linestyle='-')
    ax.plot(time_points, ipw_traj, label='90% Women Dropped (With IPW Reweighting)', 
           linewidth=2, alpha=0.8, color='red', linestyle='--')
    
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Prevalence', fontsize=12)
    ax.set_title(f'{display_name}\nDemonstrating IPW Correction for Selection Bias', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add annotations
    # Calculate how much prevalence dropped and recovered
    mid_age_idx = len(time_points) // 2
    drop_pct = 100 * (1 - no_adj_traj[mid_age_idx] / (full_traj[mid_age_idx] + 1e-10))
    recovery_pct = 100 * (ipw_traj[mid_age_idx] / (full_traj[mid_age_idx] + 1e-10))
    
    ax.text(0.02, 0.98, 
           f'At age {int(time_points[mid_age_idx])}:\n'
           f'Drop: {drop_pct:.1f}%\n'
           f'IPW Recovery: {recovery_pct:.1f}% of baseline', 
           transform=ax.transAxes, verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('IPW Correction Demonstration: Dropping 90% of Women\nShows how IPW recovers full population patterns', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save prevalence demonstration plot
demo_plot_path = output_dir / 'ipw_correction_demonstration.pdf'
plt.savefig(demo_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved prevalence demonstration plot to: {demo_plot_path}")

print(f"\n{'='*80}")
print(f"COMPLETE - Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print(f"✓ Generated prevalence demonstration plot")
print(f"✓ Saved: {demo_plot_path}")

