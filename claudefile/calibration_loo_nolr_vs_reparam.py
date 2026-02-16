#!/usr/bin/env python3
"""
Calibration comparison: LOO nolr vs reparam predictions.
Loads LOO pi tensors for both models and creates side-by-side calibration plots.
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Load data
# ============================================================================
N_eval = 50000
batch_size = 10000
n_batches = 5

LOO_NOLR_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_loo/'
LOO_REPARAM_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_loo/'

def load_batches(config_dir):
    pi_batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        f = Path(config_dir) / f'pi_enroll_fixedphi_sex_{start}_{end}.pt'
        pi = torch.load(f, map_location='cpu', weights_only=False)
        pi_batches.append(pi)
        print(f"  Loaded {f.name}: {pi.shape}")
    return torch.cat(pi_batches, dim=0)

print("Loading LOO nolr predictions...")
pi_nolr = load_batches(LOO_NOLR_DIR)
print(f"LOO nolr pi: {pi_nolr.shape}")

print("\nLoading LOO reparam predictions...")
pi_reparam = load_batches(LOO_REPARAM_DIR)
print(f"LOO reparam pi: {pi_reparam.shape}")

# Load Y and E_corrected
data_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
Y_full = torch.load(data_dir + 'Y_tensor.pt', map_location='cpu', weights_only=False)[:N_eval]
E_corrected_full = torch.load(data_dir + 'E_matrix_corrected.pt', map_location='cpu', weights_only=False)[:N_eval]
print(f"\nY: {Y_full.shape}, E_corrected: {E_corrected_full.shape}")

# Convert to numpy
pi_nolr_np = pi_nolr.detach().numpy()
pi_reparam_np = pi_reparam.detach().numpy()
Y_np = Y_full.detach().numpy()
E_np = E_corrected_full.detach().numpy() if torch.is_tensor(E_corrected_full) else np.array(E_corrected_full)

N, D, T = Y_np.shape
print(f"Dataset: {N} patients x {D} diseases x {T} timepoints")

# ============================================================================
# Create at-risk mask
# ============================================================================
print("\nCreating at-risk mask...")
at_risk = np.zeros((N, D, T), dtype=bool)
for n in range(N):
    if n % 10000 == 0:
        print(f"  Processing patient {n}/{N}...")
    for d in range(D):
        at_risk[n, d, :] = (E_np[n, d] >= np.arange(T))
print("At-risk mask created")

# ============================================================================
# Collect predictions and observations
# ============================================================================
def collect_pred_obs(pi_np, Y_np, at_risk, T):
    all_pred, all_obs = [], []
    for t in range(T):
        mask_t = at_risk[:, :, t]
        if mask_t.sum() > 0:
            all_pred.extend(pi_np[:, :, t][mask_t])
            all_obs.extend(Y_np[:, :, t][mask_t])
    return np.array(all_pred), np.array(all_obs)

print("\nCollecting nolr predictions...")
pred_nolr, obs_nolr = collect_pred_obs(pi_nolr_np, Y_np, at_risk, T)
print(f"  {len(pred_nolr):,} at-risk observations, mean pred={pred_nolr.mean():.2e}, mean obs={obs_nolr.mean():.2e}")

print("Collecting reparam predictions...")
pred_reparam, obs_reparam = collect_pred_obs(pi_reparam_np, Y_np, at_risk, T)
print(f"  {len(pred_reparam):,} at-risk observations, mean pred={pred_reparam.mean():.2e}, mean obs={obs_reparam.mean():.2e}")

# ============================================================================
# Calibration binning
# ============================================================================
def compute_calibration_bins(all_pred, all_obs, n_bins=50, min_bin_count=10000):
    bin_edges = np.logspace(np.log10(max(1e-7, all_pred.min())),
                            np.log10(all_pred.max()), n_bins + 1)
    bin_means, obs_means, counts = [], [], []
    for i in range(n_bins):
        mask = (all_pred >= bin_edges[i]) & (all_pred < bin_edges[i + 1])
        if np.sum(mask) >= min_bin_count:
            bin_means.append(np.mean(all_pred[mask]))
            obs_means.append(np.mean(all_obs[mask]))
            counts.append(np.sum(mask))
    return np.array(bin_means), np.array(obs_means), np.array(counts)

print("\nComputing calibration bins...")
bm_nolr, om_nolr, ct_nolr = compute_calibration_bins(pred_nolr, obs_nolr)
bm_reparam, om_reparam, ct_reparam = compute_calibration_bins(pred_reparam, obs_reparam)
print(f"  nolr: {len(bm_nolr)} bins, reparam: {len(bm_reparam)} bins")

# MSE
mse_nolr = np.mean((bm_nolr - om_nolr) ** 2)
mse_reparam = np.mean((bm_reparam - om_reparam) ** 2)
print(f"  Calibration MSE: nolr={mse_nolr:.2e}, reparam={mse_reparam:.2e}")

# ============================================================================
# Plot: side by side
# ============================================================================
print("\nCreating calibration plots...")
fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=300)

for ax, bm, om, ct, pred, obs, label, color, mse in [
    (axes[0], bm_nolr, om_nolr, ct_nolr, pred_nolr, obs_nolr, 'NOLR (centered)', '#1f77b4', mse_nolr),
    (axes[1], bm_reparam, om_reparam, ct_reparam, pred_reparam, obs_reparam, 'REPARAM (non-centered)', '#d62728', mse_reparam),
]:
    ax.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5, label='Perfect calibration', linewidth=2)
    ax.plot(bm, om, 'o-', color=color, markersize=10, linewidth=2.5, label='Observed rates', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')

    for x, y, c in zip(bm, om, ct):
        ax.annotate(f'n={c:,}', (x, y), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=8)

    stats_text = (f'Calibration MSE: {mse:.2e}\n'
                  f'Mean Predicted: {pred.mean():.2e}\n'
                  f'Mean Observed: {obs.mean():.2e}\n'
                  f'N total: {ct.sum():,}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel('Predicted Event Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Observed Event Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'Calibration: {label} (LOO)\n50k patients, at-risk only',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()

save_path = str(Path(__file__).parent / 'calibration_loo_nolr_vs_reparam.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"\nSaved: {save_path}")

# ============================================================================
# Plot: overlay on single axes
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 10), dpi=300)
ax2.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5, label='Perfect calibration', linewidth=2)
ax2.plot(bm_nolr, om_nolr, 'o-', color='#1f77b4', markersize=8, linewidth=2, label=f'NOLR (MSE={mse_nolr:.2e})', alpha=0.8)
ax2.plot(bm_reparam, om_reparam, 's-', color='#d62728', markersize=8, linewidth=2, label=f'REPARAM (MSE={mse_reparam:.2e})', alpha=0.8)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which='both', linestyle='--', alpha=0.3)
ax2.set_xlabel('Predicted Event Rate', fontsize=14, fontweight='bold')
ax2.set_ylabel('Observed Event Rate', fontsize=14, fontweight='bold')
ax2.set_title('Calibration Comparison: NOLR vs REPARAM (LOO)\n50k patients, at-risk only',
              fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='lower right', fontsize=13)
plt.tight_layout()

save_path2 = str(Path(__file__).parent / 'calibration_loo_overlay.pdf')
plt.savefig(save_path2, format='pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {save_path2}")

print("\nDone.")
