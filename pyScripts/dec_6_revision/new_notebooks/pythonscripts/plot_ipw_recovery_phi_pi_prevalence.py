"""
Plot IPW Recovery: Phi, Pi, and Prevalence comparison

This script creates the IPW recovery plot showing:
- Phi: Full Population vs 90% female-reduced (no IPW) vs 90% female-reduced (with IPW)
- Pi: Full Population vs 90% female-reduced (no IPW) vs 90% female-reduced (with IPW)  
- Prevalence: Full Population vs 90% female-reduced (no IPW) vs 90% female-reduced (with IPW)

Note: Uses linear scale (no log scale) for Pi and Prevalence columns.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import sys

# Add path for utils
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred, softmax_by_k

print("="*80)
print("CREATING IPW RECOVERY PLOT: Phi, Pi, and Prevalence")
print("="*80)

# Data directories
results_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')

# Try ipwbatchrun113 first, then fall back to results/batch_*
ipw_dir = results_dir / 'ipwbatchrun113'
if ipw_dir.exists():
    batch_base_dir = ipw_dir
    print(f"\nUsing data from: {ipw_dir}")
else:
    batch_base_dir = results_dir
    print(f"\nUsing data from: {results_dir}/batch_*")

# Load disease names
disease_names_dict = {}
try:
    disease_names_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv')
    if disease_names_path.exists():
        disease_names_df = pd.read_csv(disease_names_path)
        # CSV format: first column is index (1-indexed), second column 'x' is name
        first_col = disease_names_df.columns[0]
        disease_names_dict = dict(zip(disease_names_df[first_col].astype(int) - 1, disease_names_df['x']))
        print(f"✓ Loaded {len(disease_names_dict)} disease names")
except Exception as e:
    print(f"⚠️  Could not load disease names: {e}")

# Define diseases to plot
DISEASES_TO_PLOT = [
    (21, "Prostate cancer [male]"),
    (281, "Postmenopausal bleeding [female]"),
    (17, "Breast cancer")
]

# Helper function to compute Pi with at-risk filtering
def compute_pi_at_risk(pi_pred, E_corrected):
    """
    Compute average Pi filtering by at-risk status.
    
    Args:
        pi_pred: Individual Pi predictions [N, D, T]
        E_corrected: Corrected event times [N, D]
    
    Returns:
        pi_avg: Average Pi [D, T], computed only over at-risk individuals
    """
    if torch.is_tensor(pi_pred):
        pi_pred = pi_pred.numpy()
    if torch.is_tensor(E_corrected):
        E_corrected = E_corrected.numpy()
    
    N, D, T = pi_pred.shape
    pi_avg = np.zeros((D, T))
    
    for d in range(D):
        for t in range(T):
            at_risk_mask = (E_corrected[:, d] >= t)
            if at_risk_mask.sum() > 0:
                pi_avg[d, t] = pi_pred[at_risk_mask, d, t].mean()
            else:
                pi_avg[d, t] = 0.0
    
    return pi_avg

# Load E_corrected and processed_ids for at-risk filtering
print("\n0. Loading E_corrected and processed_ids for at-risk filtering...")
E_corrected_full = torch.load(str(data_dir / 'E_matrix_corrected.pt'), weights_only=False)
if torch.is_tensor(E_corrected_full):
    E_corrected_full = E_corrected_full.numpy()

# Load processed_ids to match model pids
pids_csv_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/processed_ids.csv')
pids_df = pd.read_csv(pids_csv_path)
processed_ids = pids_df['eid'].values

print(f"✓ Loaded E_corrected: {E_corrected_full.shape}")
print(f"✓ Loaded processed_ids: {len(processed_ids):,}")

# Create mapping from pid to index for fast lookup
pid_to_idx = {pid: idx for idx, pid in enumerate(processed_ids)}

# Load and recompute phi/pi across batches 1-5 with at-risk filtering
print("\n1. Loading models and recomputing phi/pi from batches 1-5 (with at-risk filtering)...")
phi_full_list = []
phi_biased_list = []
phi_biased_ipw_list = []
pi_full_list = []
pi_biased_list = []
pi_biased_ipw_list = []

N_train_per_batch = 20000
for batch_idx in range(1, 6):  # batches 1-5
    batch_dir = batch_base_dir / f'batch_{batch_idx}'
    
    if batch_dir.exists():
        # Load phi (still from saved files)
        phi_full_list.append(np.load(batch_dir / 'phi_full.npy'))
        phi_biased_list.append(np.load(batch_dir / 'phi_biased.npy'))
        phi_biased_ipw_list.append(np.load(batch_dir / 'phi_biased_ipw.npy'))
        
        # Load models to recompute Pi with at-risk filtering
        model_full_path = batch_dir / 'model_full.pt'
        model_biased_path = batch_dir / 'model_biased.pt'
        model_biased_ipw_path = batch_dir / 'model_biased_ipw.pt'
        
        if model_full_path.exists() and model_biased_path.exists() and model_biased_ipw_path.exists():
            # Load models
            ckpt_full = torch.load(model_full_path, weights_only=False, map_location='cpu')
            ckpt_biased = torch.load(model_biased_path, weights_only=False, map_location='cpu')
            ckpt_biased_ipw = torch.load(model_biased_ipw_path, weights_only=False, map_location='cpu')
            
            # Extract parameters
            if 'lambda' in ckpt_full:
                lambda_full = ckpt_full['lambda']
                phi_full_model = ckpt_full['phi']
                kappa_full = ckpt_full.get('kappa', torch.tensor(1.0))
                pids_full = ckpt_full.get('pids', None)
            else:
                lambda_full = ckpt_full['model_state_dict']['lambda_']
                phi_full_model = ckpt_full['model_state_dict']['phi']
                kappa_full = ckpt_full['model_state_dict'].get('kappa', torch.tensor(1.0))
                pids_full = None
            
            if torch.is_tensor(kappa_full):
                kappa_full = kappa_full.item() if kappa_full.numel() == 1 else kappa_full.mean().item()
            
            # Similar for biased models
            if 'lambda' in ckpt_biased:
                lambda_biased = ckpt_biased['lambda']
                phi_biased_model = ckpt_biased['phi']
                kappa_biased = ckpt_biased.get('kappa', torch.tensor(1.0))
                pids_biased = ckpt_biased.get('pids', None)
            else:
                lambda_biased = ckpt_biased['model_state_dict']['lambda_']
                phi_biased_model = ckpt_biased['model_state_dict']['phi']
                kappa_biased = ckpt_biased['model_state_dict'].get('kappa', torch.tensor(1.0))
                pids_biased = None
            
            if torch.is_tensor(kappa_biased):
                kappa_biased = kappa_biased.item() if kappa_biased.numel() == 1 else kappa_biased.mean().item()
            
            if 'lambda' in ckpt_biased_ipw:
                lambda_biased_ipw = ckpt_biased_ipw['lambda']
                phi_biased_ipw_model = ckpt_biased_ipw['phi']
                kappa_biased_ipw = ckpt_biased_ipw.get('kappa', torch.tensor(1.0))
                pids_biased_ipw = ckpt_biased_ipw.get('pids', None)
            else:
                lambda_biased_ipw = ckpt_biased_ipw['model_state_dict']['lambda_']
                phi_biased_ipw_model = ckpt_biased_ipw['model_state_dict']['phi']
                kappa_biased_ipw = ckpt_biased_ipw['model_state_dict'].get('kappa', torch.tensor(1.0))
                pids_biased_ipw = None
            
            if torch.is_tensor(kappa_biased_ipw):
                kappa_biased_ipw = kappa_biased_ipw.item() if kappa_biased_ipw.numel() == 1 else kappa_biased_ipw.mean().item()
            
            # Compute individual Pi predictions
            pi_full_batch = calculate_pi_pred(lambda_full, phi_full_model, kappa_full)  # [N, D, T]
            pi_biased_batch = calculate_pi_pred(lambda_biased, phi_biased_model, kappa_biased)  # [N_biased, D, T]
            pi_biased_ipw_batch = calculate_pi_pred(lambda_biased_ipw, phi_biased_ipw_model, kappa_biased_ipw)  # [N_biased, D, T]
            
            # Get E_corrected for full model (all patients in batch)
            batch_start = (batch_idx - 1) * N_train_per_batch
            batch_end = batch_idx * N_train_per_batch
            E_batch_full = E_corrected_full[batch_start:batch_end]
            
            # Compute Pi with at-risk filtering for full model
            pi_full_at_risk = compute_pi_at_risk(pi_full_batch, E_batch_full)
            
            # For biased models, match by pids to get correct E_corrected subset
            if pids_biased is not None:
                # Convert pids to numpy array if needed
                if torch.is_tensor(pids_biased):
                    pids_biased_np = pids_biased.numpy()
                else:
                    pids_biased_np = np.array(pids_biased)
                
                # Match pids to indices
                indices_biased = np.array([pid_to_idx.get(pid, -1) for pid in pids_biased_np])
                valid_mask = indices_biased >= 0
                if valid_mask.sum() == len(pids_biased_np):
                    E_batch_biased = E_corrected_full[indices_biased]
                    pi_biased_at_risk = compute_pi_at_risk(pi_biased_batch, E_batch_biased)
                else:
                    print(f"  ⚠️  Warning: Could not match all pids for biased model in batch {batch_idx}")
                    # Fallback: use full batch (approximation)
                    E_batch_biased = E_batch_full[:len(pi_biased_batch)]
                    pi_biased_at_risk = compute_pi_at_risk(pi_biased_batch, E_batch_biased)
            else:
                # Fallback: use full batch (approximation)
                E_batch_biased = E_batch_full[:len(pi_biased_batch)]
                pi_biased_at_risk = compute_pi_at_risk(pi_biased_batch, E_batch_biased)
            
            # For biased IPW model, match by pids
            if pids_biased_ipw is not None:
                # Convert pids to numpy array if needed
                if torch.is_tensor(pids_biased_ipw):
                    pids_biased_ipw_np = pids_biased_ipw.numpy()
                else:
                    pids_biased_ipw_np = np.array(pids_biased_ipw)
                
                # Match pids to indices
                indices_biased_ipw = np.array([pid_to_idx.get(pid, -1) for pid in pids_biased_ipw_np])
                valid_mask_ipw = indices_biased_ipw >= 0
                if valid_mask_ipw.sum() == len(pids_biased_ipw_np):
                    E_batch_biased_ipw = E_corrected_full[indices_biased_ipw]
                    pi_biased_ipw_at_risk = compute_pi_at_risk(pi_biased_ipw_batch, E_batch_biased_ipw)
                else:
                    print(f"  ⚠️  Warning: Could not match all pids for biased IPW model in batch {batch_idx}")
                    # Fallback: use full batch (approximation)
                    E_batch_biased_ipw = E_batch_full[:len(pi_biased_ipw_batch)]
                    pi_biased_ipw_at_risk = compute_pi_at_risk(pi_biased_ipw_batch, E_batch_biased_ipw)
            else:
                # Fallback: use full batch (approximation)
                E_batch_biased_ipw = E_batch_full[:len(pi_biased_ipw_batch)]
                pi_biased_ipw_at_risk = compute_pi_at_risk(pi_biased_ipw_batch, E_batch_biased_ipw)
            
            pi_full_list.append(pi_full_at_risk)
            pi_biased_list.append(pi_biased_at_risk)
            pi_biased_ipw_list.append(pi_biased_ipw_at_risk)
        else:
            # Fallback: use pre-computed files (no at-risk filtering)
            print(f"  ⚠️  Models not found for batch {batch_idx}, using pre-computed pi files")
            pi_full_list.append(np.load(batch_dir / 'pi_full.npy'))
            pi_biased_list.append(np.load(batch_dir / 'pi_biased.npy'))
            pi_biased_ipw_list.append(np.load(batch_dir / 'pi_biased_ipw.npy'))

# Average across batches
phi_full = np.mean(phi_full_list, axis=0)  # [K, D, T]
phi_biased = np.mean(phi_biased_list, axis=0)
phi_biased_ipw = np.mean(phi_biased_ipw_list, axis=0)

pi_full = np.mean(pi_full_list, axis=0)  # [D, T]
pi_biased = np.mean(pi_biased_list, axis=0)
pi_biased_ipw = np.mean(pi_biased_ipw_list, axis=0)

phi_full_avg = phi_full.mean(axis=0)  # [D, T]
phi_biased_avg = phi_biased.mean(axis=0)
phi_biased_ipw_avg = phi_biased_ipw.mean(axis=0)

print(f"✓ Loaded and averaged {len(phi_full_list)} batches")

# Prevalence computation function (same as in demonstrate_ipw_correction.py)
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

# Load full 400K data and recompute prevalence
print("\n2. Loading full 400K data and recomputing prevalence...")
n_patients = 400000
Y = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
E_corrected = torch.load(str(data_dir / 'E_matrix_corrected.pt'), weights_only=False)

if torch.is_tensor(Y):
    Y = Y.numpy()
if torch.is_tensor(E_corrected):
    E_corrected = E_corrected.numpy()

Y = Y[:n_patients]
E_corrected = E_corrected[:n_patients]

# Load patient IDs and covariates
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

print(f"✓ Loaded full 400K data: {Y.shape}")
print(f"  Women: {is_female.sum():,} ({100*is_female.sum()/n_patients:.1f}%)")

# Full population prevalence (baseline)
print("  Computing full population prevalence...")
prevalence_full = compute_smoothed_prevalence_at_risk(
    Y, E_corrected, weights=None, window_size=5, smooth_on_logit=True
)

# Drop 90% of women (same logic as demonstrate_ipw_correction.py)
print("  Dropping 90% of women...")
np.random.seed(42)  # Same seed as in demonstrate script
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

print(f"  After drop: {remaining_mask.sum():,} patients ({is_female_dropped.sum():,} women)")

# Prevalence without IPW
print("  Computing prevalence without IPW...")
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
print("  Computing prevalence with IPW...")
prevalence_biased_ipw = compute_smoothed_prevalence_at_risk(
    Y_dropped, E_dropped, weights=ipw_weights, window_size=5, smooth_on_logit=True
)

print(f"✓ Recomputed prevalence from full 400K dataset")

# Create 3-column plot
print("\n3. Creating 3-column plot (Phi, Pi, Prevalence)...")
time_points = np.arange(phi_full_avg.shape[1]) + 30

fig, axes = plt.subplots(len(DISEASES_TO_PLOT), 3, figsize=(18, 5*len(DISEASES_TO_PLOT)))
if len(DISEASES_TO_PLOT) == 1:
    axes = axes.reshape(1, -1)

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if disease_idx >= phi_full_avg.shape[0]:
        continue
    
    display_name = disease_names_dict.get(disease_idx, disease_name) if disease_names_dict else disease_name
    
    # Column 1: Phi comparison
    ax1 = axes[idx, 0]
    phi_full_traj = phi_full_avg[disease_idx, :]
    phi_biased_traj = phi_biased_avg[disease_idx, :]
    phi_biased_ipw_traj = phi_biased_ipw_avg[disease_idx, :]
    
    ax1.plot(time_points, phi_full_traj, label='Full Population', 
            linewidth=2, color='black', linestyle='-')
    ax1.plot(time_points, phi_biased_traj, label='90% female-reduced', 
            linewidth=2, color='blue', linestyle='--')
    ax1.plot(time_points, phi_biased_ipw_traj, label='90% female-reduced + IPW', 
            linewidth=2, color='red', linestyle=':')
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Average Phi (across signatures)', fontsize=11)
    ax1.set_title(f'{display_name}\nPhi: Stable with Same Init', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Column 2: Pi comparison
    ax2 = axes[idx, 1]
    pi_full_traj = pi_full[disease_idx, :]
    pi_biased_traj = pi_biased[disease_idx, :]
    pi_biased_ipw_traj = pi_biased_ipw[disease_idx, :]
    
    ax2.plot(time_points, pi_full_traj, label='Full Population', 
            linewidth=2, color='black', linestyle='-')
    ax2.plot(time_points, pi_biased_traj, label='90% female-reduced', 
            linewidth=2, color='blue', linestyle='--')
    ax2.plot(time_points, pi_biased_ipw_traj, label='90% female-reduced + IPW', 
            linewidth=2, color='red', linestyle=':')
    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel('Average Pi (Disease Hazard)', fontsize=11)
    ax2.set_title(f'{display_name}\nPi: IPW Recovers', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to match Phi and better visualize changes
    
    # Column 3: Prevalence comparison (from full 400K)
    ax3 = axes[idx, 2]
    prev_full_traj = prevalence_full[disease_idx, :]
    prev_biased_traj = prevalence_biased[disease_idx, :]
    prev_biased_ipw_traj = prevalence_biased_ipw[disease_idx, :]
    
    ax3.plot(time_points, prev_full_traj, label='Full Population', 
            linewidth=2, color='black', linestyle='-')
    ax3.plot(time_points, prev_biased_traj, label='90% female-reduced', 
            linewidth=2, color='blue', linestyle='--')
    ax3.plot(time_points, prev_biased_ipw_traj, label='90% female-reduced + IPW', 
            linewidth=2, color='red', linestyle=':')
    ax3.set_xlabel('Age', fontsize=11)
    ax3.set_ylabel('Prevalence', fontsize=11)
    ax3.set_title(f'{display_name}\nEmpirical Incidence: IPW Recovers', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale to match Phi and better visualize changes

plt.suptitle('IPW Recovery: Full Population vs 90% Female-Reduced Sample (with/without IPW)\n'
            'Phi/Pi: Pooled across 5 batches (20K each) | Prevalence: Full 400K dataset', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
output_path = results_dir / 'ipw_recovery_phi_pi_prevalence.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")
plt.show()

print(f"\n{'='*80}")
print("IPW RECOVERY PLOT COMPLETE")
print("="*80)

