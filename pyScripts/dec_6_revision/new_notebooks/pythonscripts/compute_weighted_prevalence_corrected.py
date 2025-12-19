"""
Compute weighted and unweighted prevalence with corrected E matrix.

This script:
1. Loads Y, E_corrected for first 100K patients (10 batches)
2. Loads IPW weights and matches to patient IDs
3. Computes weighted prevalence with corrected E (at-risk filtering + IPW weighting)
4. Computes unweighted prevalence with corrected E (at-risk filtering only)
5. Generates comparison plots

Output:
- prevalence_t_weighted_corrected.pt
- prevalence_t_unweighted_corrected.pt
- Comparison plots
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import sys

# Add path for weightedprev
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts_forPublish')
from weightedprev import match_weights_to_ids

print("="*80)
print("COMPUTING WEIGHTED AND UNWEIGHTED PREVALENCE WITH CORRECTED E")
print("="*80)

# Data directory
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')

# Load full Y and E_corrected
print("\n1. Loading Y and E_corrected...")
Y_full = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
E_corrected_full = torch.load(str(data_dir / 'E_matrix_corrected.pt'), weights_only=False)

print(f"   Y_full shape: {Y_full.shape}")
print(f"   E_corrected_full shape: {E_corrected_full.shape}")

# Use first 400K patients to match processed_ids
n_patients = 400000  # Match processed_ids length
Y = Y_full[:n_patients]
E_corrected = E_corrected_full[:n_patients]

print(f"\n2. Using first {n_patients:,} patients (matching processed_ids):")
print(f"   Y shape: {Y.shape}")
print(f"   E_corrected shape: {E_corrected.shape}")

# Load patient IDs
print("\n3. Loading patient IDs...")
try:
    # Try CSV file first (as used in training notebook)
    csv_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/processed_ids.csv')
    if csv_path.exists():
        pids_df = pd.read_csv(csv_path)
        processed_ids = pids_df['eid'].values
        print(f"   ✓ Loaded processed_ids from CSV: {len(processed_ids):,}")
    else:
        # Try processed_ids.npy
        processed_ids_path = data_dir / 'processed_ids.npy'
        if processed_ids_path.exists():
            processed_ids = np.load(processed_ids_path)
            print(f"   ✓ Loaded processed_ids: {len(processed_ids):,}")
        else:
            # Try model_essentials
            essentials_path = data_dir / 'model_essentials.pt'
            if essentials_path.exists():
                essentials = torch.load(essentials_path, weights_only=False)
                if 'pids' in essentials:
                    processed_ids = essentials['pids'].numpy() if torch.is_tensor(essentials['pids']) else essentials['pids']
                    print(f"   ✓ Loaded pids from model_essentials: {len(processed_ids):,}")
                else:
                    raise FileNotFoundError("No pids found in model_essentials")
            else:
                raise FileNotFoundError("No patient ID file found")
except Exception as e:
    print(f"   ⚠️  Error loading patient IDs: {e}")
    print("   Will try to load weights assuming same order...")
    processed_ids = None

# Use first 400K processed_ids (matching n_patients)
if processed_ids is not None:
    if len(processed_ids) >= n_patients:
        processed_ids_subset = processed_ids[:n_patients]
        print(f"   Using first {n_patients:,} patient IDs (matching Y size)")
    else:
        print(f"   ⚠️  Warning: Only {len(processed_ids):,} IDs available, but need {n_patients:,}")
        print(f"   Using all available IDs: {len(processed_ids):,}")
        processed_ids_subset = processed_ids
        # Update n_patients to match
        n_patients = len(processed_ids)
        Y = Y[:n_patients]
        E_corrected = E_corrected[:n_patients]
        print(f"   Adjusted Y and E_corrected to {n_patients:,} patients")
else:
    processed_ids_subset = None

# Load IPW weights
print("\n4. Loading IPW weights...")
weights_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/UKBWeights-main/UKBSelectionWeights.csv")
weights_df = pd.read_csv(weights_path, sep='\s+', engine='python')
print(f"   ✓ Loaded weights: {len(weights_df):,} rows")
print(f"   Columns: {weights_df.columns.tolist()}")

# Match weights to patient IDs
if processed_ids_subset is not None:
    matched_weights, match_mask = match_weights_to_ids(weights_df, processed_ids_subset)
    print(f"\n5. Matched weights:")
    print(f"   Matched: {match_mask.sum():,} / {len(processed_ids_subset):,} ({100*match_mask.sum()/len(processed_ids_subset):.1f}%)")
    print(f"   Weight stats (matched): mean={matched_weights[match_mask].mean():.3f}, std={matched_weights[match_mask].std():.3f}")
    
    # Ensure matched_weights has the same length as Y (n_patients)
    if len(matched_weights) != n_patients:
        if len(matched_weights) < n_patients:
            # Pad with 1.0 for unmatched
            padded_weights = np.ones(n_patients)
            padded_weights[:len(matched_weights)] = matched_weights
            matched_weights = padded_weights
            # Update match_mask
            padded_mask = np.zeros(n_patients, dtype=bool)
            padded_mask[:len(match_mask)] = match_mask
            match_mask = padded_mask
            print(f"   Padded weights to {n_patients:,} (unmatched set to 1.0)")
        else:
            # Truncate if somehow longer
            matched_weights = matched_weights[:n_patients]
            match_mask = match_mask[:n_patients]
            print(f"   Truncated weights to {n_patients:,}")
    
    # For unmatched patients, set weight to 1.0 (unweighted)
    if match_mask.sum() < n_patients:
        num_unmatched = (match_mask == False).sum()
        matched_weights[~match_mask] = 1.0
        print(f"   Setting {num_unmatched:,} unmatched patients to weight=1.0")
else:
    # Fallback: assume same order
    print("   ⚠️  No patient IDs available, assuming same order...")
    if len(weights_df) >= n_patients:
        matched_weights = weights_df['LassoWeight'].values[:n_patients]
    else:
        # If not enough weights, pad with 1.0
        matched_weights = np.ones(n_patients)
        matched_weights[:len(weights_df)] = weights_df['LassoWeight'].values
        print(f"   ⚠️  Only {len(weights_df):,} weights available, padding with 1.0")
    match_mask = np.ones(n_patients, dtype=bool)
    print(f"   Using weights for {n_patients:,} patients")

# Convert to numpy
if torch.is_tensor(Y):
    Y = Y.numpy()
if torch.is_tensor(E_corrected):
    E_corrected = E_corrected.numpy()

N, D, T = Y.shape

print(f"\n6. Computing prevalence with corrected E...")
print(f"   N={N:,} patients, D={D} diseases, T={T} timepoints")

def compute_smoothed_prevalence_at_risk(Y, E_corrected, weights=None, window_size=5, smooth_on_logit=True):
    """
    Compute smoothed prevalence with proper at-risk filtering.
    If weights provided, computes weighted prevalence.
    
    Parameters:
    -----------
    Y : np.ndarray (N × D × T)
    E_corrected : np.ndarray (N × D) - corrected event/censor times
    weights : np.ndarray (N,), optional - IPW weights. If None, computes unweighted.
    window_size : int - Gaussian smoothing window size
    smooth_on_logit : bool - Smooth on logit scale
    """
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    
    # Convert timepoints to ages (assuming timepoint 0 = age 30)
    timepoint_ages = np.arange(T) + 30
    
    is_weighted = weights is not None
    print(f"\n  Computing {'weighted' if is_weighted else 'unweighted'} prevalence for {D} diseases, {T} timepoints...")
    print(f"  Using at-risk filtering with corrected E")
    
    # Normalize weights if provided
    if weights is not None:
        weights_norm = weights / weights.sum() * N
    
    for d in range(D):
        if d % 50 == 0:
            print(f"    Processing disease {d}/{D}...")
        
        for t in range(T):
            age_t = timepoint_ages[t]
            
            # Only include people who are still at risk at timepoint t
            # This matches the verified code in R3_Verify_Corrected_Data.ipynb and with_bigdata.ipynb
            at_risk_mask = (E_corrected[:, d] >= t)
            
            if at_risk_mask.sum() > 0:
                Y_at_risk = Y[at_risk_mask, d, t]
                
                if weights is not None:
                    # Weighted prevalence: sum(Y * weights) / sum(weights)
                    weights_at_risk = weights_norm[at_risk_mask]
                    weighted_sum = np.sum(Y_at_risk * weights_at_risk)
                    weights_sum = np.sum(weights_at_risk)
                    
                    if weights_sum > 0:
                        prevalence_t[d, t] = weighted_sum / weights_sum
                    else:
                        prevalence_t[d, t] = np.nan
                else:
                    # Unweighted prevalence: simple mean
                    prevalence_t[d, t] = Y_at_risk.mean()
            else:
                prevalence_t[d, t] = np.nan
        
        # Smooth as before
        if smooth_on_logit:
            epsilon = 1e-8
            # Handle NaN values
            valid_mask = ~np.isnan(prevalence_t[d, :])
            if valid_mask.sum() > 0:
                logit_prev = np.full(T, np.nan)
                logit_prev[valid_mask] = np.log(
                    (prevalence_t[d, valid_mask] + epsilon) / 
                    (1 - prevalence_t[d, valid_mask] + epsilon)
                )
                # Smooth only valid values
                smoothed_logit = gaussian_filter1d(
                    np.nan_to_num(logit_prev, nan=0), 
                    sigma=window_size
                )
                # Restore NaN where original was NaN
                smoothed_logit[~valid_mask] = np.nan
                prevalence_t[d, :] = 1 / (1 + np.exp(-smoothed_logit))
        else:
            prevalence_t[d, :] = gaussian_filter1d(
                np.nan_to_num(prevalence_t[d, :], nan=0), 
                sigma=window_size
            )
    
    return prevalence_t

# Compute weighted prevalence
print("\n" + "="*80)
print("COMPUTING WEIGHTED PREVALENCE")
print("="*80)
prevalence_t_weighted = compute_smoothed_prevalence_at_risk(
    Y=Y,
    E_corrected=E_corrected,
    weights=matched_weights,
    window_size=5,
    smooth_on_logit=True
)

print(f"\n✓ Computed weighted prevalence: {prevalence_t_weighted.shape}")
print(f"  Range: [{np.nanmin(prevalence_t_weighted):.6f}, {np.nanmax(prevalence_t_weighted):.6f}]")

# Compute unweighted prevalence
print("\n" + "="*80)
print("COMPUTING UNWEIGHTED PREVALENCE")
print("="*80)
prevalence_t_unweighted = compute_smoothed_prevalence_at_risk(
    Y=Y,
    E_corrected=E_corrected,
    weights=None,  # Unweighted
    window_size=5,
    smooth_on_logit=True
)

print(f"\n✓ Computed unweighted prevalence: {prevalence_t_unweighted.shape}")
print(f"  Range: [{np.nanmin(prevalence_t_unweighted):.6f}, {np.nanmax(prevalence_t_unweighted):.6f}]")

# Save results
output_dir = data_dir
weighted_path = output_dir / 'prevalence_t_weighted_corrected.pt'
unweighted_path = output_dir / 'prevalence_t_unweighted_corrected.pt'

torch.save(torch.tensor(prevalence_t_weighted), str(weighted_path))
torch.save(torch.tensor(prevalence_t_unweighted), str(unweighted_path))

print(f"\n✓ Saved weighted prevalence to: {weighted_path}")
print(f"✓ Saved unweighted prevalence to: {unweighted_path}")

# Compare
print("\n" + "="*80)
print("COMPARISON STATISTICS")
print("="*80)
diff = prevalence_t_weighted - prevalence_t_unweighted
mean_diff = np.nanmean(np.abs(diff))
max_diff = np.nanmax(np.abs(diff))

# Flatten for correlation (excluding NaN)
valid_mask = ~(np.isnan(prevalence_t_weighted) | np.isnan(prevalence_t_unweighted))
weighted_flat = prevalence_t_weighted[valid_mask]
unweighted_flat = prevalence_t_unweighted[valid_mask]

correlation = np.corrcoef(weighted_flat, unweighted_flat)[0, 1]

print(f"   Mean absolute difference: {mean_diff:.6f}")
print(f"   Max absolute difference: {max_diff:.6f}")
print(f"   Correlation: {correlation:.6f}")

# Load disease names if available
disease_names_dict = {}
try:
    disease_names_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv')
    if disease_names_path.exists():
        disease_df = pd.read_csv(disease_names_path)
        disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
        print(f"\n✓ Loaded disease names")
except:
    pass

# Select diseases for plotting
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

# Time points (assuming starting at age 30)
time_points = np.arange(T) + 30

# Plot 1: Side-by-side trajectories for selected diseases
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
    
    if disease_idx < prevalence_t_weighted.shape[0]:
        weighted_traj = prevalence_t_weighted[disease_idx, :]
        unweighted_traj = prevalence_t_unweighted[disease_idx, :]
        
        # Plot both trajectories
        ax.plot(time_points, unweighted_traj, label='Unweighted', linewidth=2, alpha=0.8, color='blue')
        ax.plot(time_points, weighted_traj, label='Weighted (IPW)', linewidth=2, alpha=0.8, 
               linestyle='--', color='red')
        
        ax.set_xlabel('Age', fontsize=11)
        ax.set_ylabel('Prevalence', fontsize=11)
        ax.set_title(f'{display_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        
        # Add difference annotation
        valid_mask = ~(np.isnan(weighted_traj) | np.isnan(unweighted_traj))
        if valid_mask.sum() > 0:
            max_diff = np.abs(weighted_traj[valid_mask] - unweighted_traj[valid_mask]).max()
            mean_diff = np.abs(weighted_traj[valid_mask] - unweighted_traj[valid_mask]).mean()
            ax.text(0.02, 0.98, f'Max diff: {max_diff:.4f}\nMean diff: {mean_diff:.4f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, f'Disease {disease_idx}\nnot found', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(f'{disease_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')

# Remove extra subplot
if len(DISEASES_TO_PLOT) < len(axes):
    axes[len(DISEASES_TO_PLOT)].axis('off')

    plt.suptitle(f'Prevalence Trajectories: Weighted vs Unweighted\n(N={n_patients:,} patients, corrected E, all data)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
plot_path = output_dir / 'prevalence_weighted_vs_unweighted_comparison.pdf'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison plot to: {plot_path}")
plt.show()

# Plot 2: Scatter plot of all diseases×timepoints
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(unweighted_flat, weighted_flat, alpha=0.3, s=1)
ax.plot([unweighted_flat.min(), unweighted_flat.max()], 
       [unweighted_flat.min(), unweighted_flat.max()], 'r--', alpha=0.7, linewidth=2)

ax.set_xlabel('Unweighted Prevalence', fontsize=12)
ax.set_ylabel('Weighted (IPW) Prevalence', fontsize=12)
ax.set_title(f'Prevalence Comparison: All Diseases, All Time Points\nCorrelation: {correlation:.4f}\n(N={n_patients:,} patients, corrected E, all data)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()

# Save scatter plot
scatter_path = output_dir / 'prevalence_weighted_vs_unweighted_scatter.pdf'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved scatter plot to: {scatter_path}")
plt.show()

# Plot 3: Difference heatmap (sample diseases)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean difference heatmap (all diseases)
ax = axes[0]
im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', 
               vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
ax.set_xlabel('Time (Age)', fontsize=11)
ax.set_ylabel('Disease', fontsize=11)
ax.set_title('Prevalence Difference: Weighted - Unweighted\n(All Diseases)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Difference')

# Difference heatmap for sample diseases
ax = axes[1]
sample_disease_indices = [d[0] for d in DISEASES_TO_PLOT if d[0] < diff.shape[0]]
if len(sample_disease_indices) > 0:
    sample_diff = diff[sample_disease_indices, :]
    im = ax.imshow(sample_diff, aspect='auto', cmap='RdBu_r',
                   vmin=-np.nanmax(np.abs(sample_diff)), vmax=np.nanmax(np.abs(sample_diff)))
    ax.set_xlabel('Time (Age)', fontsize=11)
    ax.set_ylabel('Disease', fontsize=11)
    ax.set_yticks(range(len(sample_disease_indices)))
    ax.set_yticklabels([f'Disease {d}' for d in sample_disease_indices])
    ax.set_title('Prevalence Difference: Weighted - Unweighted\n(Sample Diseases)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Difference')

plt.tight_layout()

# Save heatmap
heatmap_path = output_dir / 'prevalence_weighted_vs_unweighted_heatmap.pdf'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved heatmap to: {heatmap_path}")
plt.show()

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
print(f"\nSummary:")
print(f"  - Weighted prevalence saved to: {weighted_path}")
print(f"  - Unweighted prevalence saved to: {unweighted_path}")
print(f"  - Comparison plots saved to:")
print(f"    * {plot_path}")
print(f"    * {scatter_path}")
print(f"    * {heatmap_path}")
print(f"\n  Correlation: {correlation:.6f}")
print(f"  Mean absolute difference: {mean_diff:.6f}")

