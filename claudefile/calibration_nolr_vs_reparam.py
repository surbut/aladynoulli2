"""
Side-by-side calibration comparison: nolr vs reparam
Uses first 50k patients (reparam only has 5 batches)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────
N_USE = 50000
DROPBOX = "/Users/sarahurbut/Library/CloudStorage/Dropbox"
DATA = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running"

# Load pi for both models (concat 5 batches for reparam)
print("Loading nolr pi...")
pi_nolr = torch.load(f"{DROPBOX}/enrollment_predictions_fixedphi_fixedgk_vectorized/pi_enroll_fixedphi_sex_FULL.pt",
                      map_location='cpu', weights_only=False)[:N_USE]

print("Loading reparam pi (5 batches)...")
reparam_dir = f"{DROPBOX}/enrollment_predictions_fixedphi_fixedgk_reparam_vectorized"
reparam_batches = []
for start in range(0, N_USE, 10000):
    end = start + 10000
    batch = torch.load(f"{reparam_dir}/pi_enroll_fixedphi_sex_{start}_{end}.pt",
                       map_location='cpu', weights_only=False)
    reparam_batches.append(batch)
pi_reparam = torch.cat(reparam_batches, dim=0)

print(f"nolr pi shape: {pi_nolr.shape}")
print(f"reparam pi shape: {pi_reparam.shape}")

# Load Y and E
Y = torch.load(f"{DATA}/Y_tensor.pt", map_location='cpu', weights_only=False)[:N_USE].numpy()
E = torch.load(f"{DATA}/E_matrix_corrected.pt", map_location='cpu', weights_only=False)[:N_USE]
if torch.is_tensor(E):
    E = E.numpy()

N, D, T = Y.shape
print(f"Dimensions: {N} patients x {D} diseases x {T} timepoints")

# ── Build at-risk mask ───────────────────────────────────────────────────────
print("Building at-risk mask...")
time_grid = np.arange(T)[None, None, :]  # [1, 1, T]
at_risk = (E[:, :, None] >= time_grid)  # [N, D, T] broadcasting
print(f"At-risk observations: {at_risk.sum():,}")

# ── Collect predictions/observations ─────────────────────────────────────────
def collect_calibration_data(pi, Y, at_risk):
    pi_np = pi.detach().numpy()
    preds = pi_np[at_risk]
    obs = Y[at_risk]
    return preds, obs

print("Collecting calibration data...")
pred_nolr, obs_nolr = collect_calibration_data(pi_nolr, Y, at_risk)
pred_reparam, obs_reparam = collect_calibration_data(pi_reparam, Y, at_risk)

print(f"  nolr:    {len(pred_nolr):,} obs, mean pred={pred_nolr.mean():.4e}, mean obs={obs_nolr.mean():.4e}")
print(f"  reparam: {len(pred_reparam):,} obs, mean pred={pred_reparam.mean():.4e}, mean obs={obs_reparam.mean():.4e}")

# ── Bin and compute calibration ──────────────────────────────────────────────
def compute_calibration_bins(preds, obs, n_bins=50, min_count=5000):
    bin_edges = np.logspace(np.log10(max(1e-7, preds.min())),
                            np.log10(preds.max()), n_bins + 1)
    bin_means, obs_means, counts = [], [], []
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if mask.sum() >= min_count:
            bin_means.append(preds[mask].mean())
            obs_means.append(obs[mask].mean())
            counts.append(mask.sum())
    return np.array(bin_means), np.array(obs_means), np.array(counts)

bm_nolr, om_nolr, cn_nolr = compute_calibration_bins(pred_nolr, obs_nolr)
bm_reparam, om_reparam, cn_reparam = compute_calibration_bins(pred_reparam, obs_reparam)

# ── Compute summary stats ────────────────────────────────────────────────────
def calibration_mse(bin_means, obs_means):
    return np.mean((bin_means - obs_means)**2)

mse_nolr = calibration_mse(bm_nolr, om_nolr)
mse_reparam = calibration_mse(bm_reparam, om_reparam)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=150)

for ax, bm, om, cn, label, mse, pred, obs, color in [
    (axes[0], bm_nolr, om_nolr, cn_nolr, 'Centered (nolr)', mse_nolr, pred_nolr, obs_nolr, '#1f77b4'),
    (axes[1], bm_reparam, om_reparam, cn_reparam, 'Reparameterized', mse_reparam, pred_reparam, obs_reparam, '#d62728'),
]:
    ax.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5, linewidth=2, label='Perfect calibration')
    ax.plot(bm, om, 'o-', color=color, markersize=8, linewidth=2.5, label='Observed rates', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted Event Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('Observed Event Rate', fontsize=13, fontweight='bold')
    ax.set_title(label, fontsize=15, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    stats = (f'Calibration MSE: {mse:.2e}\n'
             f'Mean Predicted: {pred.mean():.4e}\n'
             f'Mean Observed: {obs.mean():.4e}\n'
             f'Pred/Obs ratio: {pred.mean()/obs.mean():.3f}\n'
             f'N obs: {len(pred):,}')
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10)
    ax.legend(loc='lower right', fontsize=11)

fig.suptitle('Calibration: Centered vs Reparameterized (50k patients, at-risk only)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

save_path = "/Users/sarahurbut/aladynoulli2/claudefile/calibration_nolr_vs_reparam.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to {save_path}")
plt.close()

# Also print a quick summary
print("\n" + "="*60)
print("CALIBRATION SUMMARY")
print("="*60)
print(f"{'Metric':<25} {'Centered':>12} {'Reparam':>12}")
print("-"*60)
print(f"{'Calibration MSE':<25} {mse_nolr:>12.2e} {mse_reparam:>12.2e}")
print(f"{'Mean predicted':<25} {pred_nolr.mean():>12.4e} {pred_reparam.mean():>12.4e}")
print(f"{'Mean observed':<25} {obs_nolr.mean():>12.4e} {obs_reparam.mean():>12.4e}")
print(f"{'Pred/Obs ratio':<25} {pred_nolr.mean()/obs_nolr.mean():>12.3f} {pred_reparam.mean()/obs_reparam.mean():>12.3f}")
print("="*60)
