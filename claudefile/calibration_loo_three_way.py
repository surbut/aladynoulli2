#!/usr/bin/env python3
"""
Calibration comparison: LOO nolr vs reparam v1 vs nokappa predictions.
3-panel side-by-side + overlay plot.
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Config
# ============================================================================
N_eval = 50000
batch_size = 10000
n_batches = 5

LOO_NOLR_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_loo/'
LOO_REPARAM_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_loo/'
LOO_NOKAPPA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nokappa_loo/'
data_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'

# ============================================================================
# Load data
# ============================================================================
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

print("\nLoading LOO reparam v1 predictions...")
pi_reparam = load_batches(LOO_REPARAM_DIR)

print("\nLoading LOO nokappa predictions...")
pi_nokappa = load_batches(LOO_NOKAPPA_DIR)

Y_full = torch.load(data_dir + 'Y_tensor.pt', map_location='cpu', weights_only=False)[:N_eval]
E_full = torch.load(data_dir + 'E_matrix_corrected.pt', map_location='cpu', weights_only=False)[:N_eval]
print(f"\nY: {Y_full.shape}, E: {E_full.shape}")

pi_nolr_np = pi_nolr.detach().numpy()
pi_reparam_np = pi_reparam.detach().numpy()
pi_nokappa_np = pi_nokappa.detach().numpy()
Y_np = Y_full.detach().numpy()
E_np = E_full.detach().numpy() if torch.is_tensor(E_full) else np.array(E_full)

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
print(f"At-risk mask: {at_risk.sum():,} observations")

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

print("\nCollecting predictions...")
pred_nolr, obs_nolr = collect_pred_obs(pi_nolr_np, Y_np, at_risk, T)
print(f"  nolr: {len(pred_nolr):,} obs, mean pred={pred_nolr.mean():.2e}, mean obs={obs_nolr.mean():.2e}")

pred_reparam, obs_reparam = collect_pred_obs(pi_reparam_np, Y_np, at_risk, T)
print(f"  reparam: {len(pred_reparam):,} obs, mean pred={pred_reparam.mean():.2e}, mean obs={obs_reparam.mean():.2e}")

pred_nokappa, obs_nokappa = collect_pred_obs(pi_nokappa_np, Y_np, at_risk, T)
print(f"  nokappa: {len(pred_nokappa):,} obs, mean pred={pred_nokappa.mean():.2e}, mean obs={obs_nokappa.mean():.2e}")

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
bm_nokappa, om_nokappa, ct_nokappa = compute_calibration_bins(pred_nokappa, obs_nokappa)

mse_nolr = np.mean((bm_nolr - om_nolr) ** 2)
mse_reparam = np.mean((bm_reparam - om_reparam) ** 2)
mse_nokappa = np.mean((bm_nokappa - om_nokappa) ** 2)
print(f"  Calibration MSE: nolr={mse_nolr:.2e}, reparam={mse_reparam:.2e}, nokappa={mse_nokappa:.2e}")

# ============================================================================
# Plot 1: 3-panel side by side
# ============================================================================
print("\nCreating 3-panel calibration plot...")
fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=300)

panels = [
    (axes[0], bm_nolr, om_nolr, ct_nolr, pred_nolr, obs_nolr,
     'Centered (nolr)', '#1f77b4', mse_nolr),
    (axes[1], bm_reparam, om_reparam, ct_reparam, pred_reparam, obs_reparam,
     'Reparameterized (kappa=4.5)', '#d62728', mse_reparam),
    (axes[2], bm_nokappa, om_nokappa, ct_nokappa, pred_nokappa, obs_nokappa,
     'Nokappa (kappa=1)', '#2ca02c', mse_nokappa),
]

for ax, bm, om, ct, pred, obs, label, color, mse in panels:
    ax.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5,
            label='Perfect calibration', linewidth=2)
    ax.plot(bm, om, 'o-', color=color, markersize=10, linewidth=2.5,
            label='Observed rates', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pred_obs_ratio = pred.mean() / obs.mean()
    stats_text = (f'Calibration MSE: {mse:.2e}\n'
                  f'Mean Predicted: {pred.mean():.4e}\n'
                  f'Mean Observed: {obs.mean():.4e}\n'
                  f'Pred/Obs ratio: {pred_obs_ratio:.3f}\n'
                  f'N obs: {ct.sum():,}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel('Predicted Event Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Observed Event Rate', fontsize=14, fontweight='bold')
    ax.set_title(label, fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12)

fig.suptitle('Calibration: Centered vs Reparameterized vs Nokappa (50k patients, at-risk only)',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

save_path = str(Path(__file__).parent / 'calibration_loo_three_way.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")

# ============================================================================
# Plot 2: 3-way overlay
# ============================================================================
print("\nCreating overlay plot...")
fig2, ax2 = plt.subplots(figsize=(12, 10), dpi=300)
ax2.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5,
         label='Perfect calibration', linewidth=2)
ax2.plot(bm_nolr, om_nolr, 'o-', color='#1f77b4', markersize=8, linewidth=2,
         label=f'Centered (MSE={mse_nolr:.2e})', alpha=0.8)
ax2.plot(bm_reparam, om_reparam, 's-', color='#d62728', markersize=8, linewidth=2,
         label=f'Reparam v1 (MSE={mse_reparam:.2e})', alpha=0.8)
ax2.plot(bm_nokappa, om_nokappa, '^-', color='#2ca02c', markersize=8, linewidth=2,
         label=f'Nokappa (MSE={mse_nokappa:.2e})', alpha=0.8)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which='both', linestyle='--', alpha=0.3)
ax2.set_xlabel('Predicted Event Rate', fontsize=14, fontweight='bold')
ax2.set_ylabel('Observed Event Rate', fontsize=14, fontweight='bold')
ax2.set_title('Calibration: 3-Way LOO Comparison\n50k patients, at-risk only',
              fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='lower right', fontsize=13)
plt.tight_layout()

save_path2 = str(Path(__file__).parent / 'calibration_loo_three_way_overlay.pdf')
plt.savefig(save_path2, format='pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {save_path2}")

print("\nDone.")
