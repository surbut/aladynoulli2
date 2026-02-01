#!/usr/bin/env python3
"""
Export calibration data from Figure 4 to CSV for figure editor.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("Loading data...")

# Load pre-computed pi (full 400k dataset)
pi_full = torch.load("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_vectorized/pi_enroll_fixedphi_sex_FULL.pt")[:50000]
print(f"✓ Loaded pre-computed pi: {pi_full.shape}")

# Load Y (full dataset)
Y_full = torch.load("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt", 
                    map_location='cpu', weights_only=False)[:50000]
print(f"✓ Loaded Y: {Y_full.shape}")

# Load corrected E matrix (full dataset)
E_corrected_full = torch.load("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_matrix_corrected.pt", 
                              map_location='cpu', weights_only=False)[:50000]
print(f"✓ Loaded E_corrected: {E_corrected_full.shape}")

# Convert to numpy
pi_np = pi_full.detach().numpy()
Y_np = Y_full.detach().numpy()
if torch.is_tensor(E_corrected_full):
    E_corrected_np = E_corrected_full.detach().numpy()
else:
    E_corrected_np = E_corrected_full

N, D, T = Y_np.shape
print(f"\nDataset dimensions: {N} patients × {D} diseases × {T} timepoints")

# Create at_risk mask using corrected E matrix
print("\nCreating at-risk mask...")
at_risk = np.zeros((N, D, T), dtype=bool)
for n in range(N):
    for d in range(D):
        at_risk[n, d, :] = (E_corrected_np[n, d] >= np.arange(T))

print("✓ At-risk mask created")

# Collect all predictions and observations (at-risk only)
print("\nCollecting predictions and observations...")
all_pred = []
all_obs = []

for t in range(T):
    mask_t = at_risk[:,:,t]
    if mask_t.sum() > 0:
        all_pred.extend(pi_np[:,:,t][mask_t])
        all_obs.extend(Y_np[:,:,t][mask_t])

all_pred = np.array(all_pred)
all_obs = np.array(all_obs)

print(f"\n✓ Collected {len(all_pred):,} predictions/observations")

# Create bins in log space (same as in notebook)
n_bins = 50
min_bin_count = 10000

bin_edges = np.logspace(np.log10(max(1e-7, min(all_pred))), 
                      np.log10(max(all_pred)), 
                      n_bins + 1)

# Calculate statistics for each bin
bin_data = []

for i in range(n_bins):
    mask = (all_pred >= bin_edges[i]) & (all_pred < bin_edges[i + 1])
    count = np.sum(mask)
    if count >= min_bin_count:
        bin_mean = np.mean(all_pred[mask])
        obs_mean = np.mean(all_obs[mask])
        bin_data.append({
            'bin_index': i,
            'bin_lower': bin_edges[i],
            'bin_upper': bin_edges[i + 1],
            'predicted_mean': bin_mean,
            'observed_mean': obs_mean,
            'count': count,
            'difference': obs_mean - bin_mean,
            'ratio': obs_mean / bin_mean if bin_mean > 0 else np.nan
        })

# Create DataFrame
df = pd.DataFrame(bin_data)

# Add summary statistics
mse = np.mean((df['predicted_mean'] - df['observed_mean'])**2)
mean_pred = np.mean(all_pred)
mean_obs = np.mean(all_obs)

print("\n" + "="*70)
print("CALIBRATION DATA SUMMARY")
print("="*70)
print(f"MSE: {mse:.2e}")
print(f"Mean Predicted (overall): {mean_pred:.2e}")
print(f"Mean Observed (overall): {mean_obs:.2e}")
print(f"Total observations: {len(all_pred):,}")
print(f"Bins with ≥{min_bin_count:,} observations: {len(df)}")
print("="*70)

print("\nCalibration bin data:")
print(df.to_string(index=False))

# Save to CSV
output_dir = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig5")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "calibration_data_full_400k.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved calibration data to: {output_path}")

# Also save summary statistics
summary_df = pd.DataFrame([{
    'metric': 'MSE',
    'value': mse
}, {
    'metric': 'mean_predicted_overall',
    'value': mean_pred
}, {
    'metric': 'mean_observed_overall', 
    'value': mean_obs
}, {
    'metric': 'total_observations',
    'value': len(all_pred)
}, {
    'metric': 'total_binned_observations',
    'value': df['count'].sum()
}, {
    'metric': 'n_bins_used',
    'value': len(df)
}])

summary_path = output_dir / "calibration_summary_full_400k.csv"
summary_df.to_csv(summary_path, index=False)
print(f"✓ Saved summary statistics to: {summary_path}")

print("\n✅ DONE!")
