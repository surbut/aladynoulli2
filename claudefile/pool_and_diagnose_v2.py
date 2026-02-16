#!/usr/bin/env python
"""
Pool phi, psi, kappa, gamma from v2 reparam training batches.
Also plots sample loss curves and prints stability diagnostics.

Usage:
    MPLBACKEND=Agg python claudefile/pool_and_diagnose_v2.py
"""

import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

V2_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2'
OUT_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
PATTERN = str(Path(V2_DIR) / 'enrollment_model_REPARAM_W0.0001_batch_*_*.pt')

# ── Load all checkpoints ──
files = sorted(glob.glob(PATTERN))
print(f"Found {len(files)} v2 checkpoint files")
if len(files) == 0:
    raise FileNotFoundError(f"No files matching {PATTERN}")

all_phi, all_psi, all_kappa, all_gamma, all_losses = [], [], [], [], []
batch_labels = []

def _extract(ckpt, name):
    if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
        return ckpt['model_state_dict'][name]
    return ckpt.get(name)

for fp in files:
    ckpt = torch.load(fp, weights_only=False)
    phi = _extract(ckpt, 'phi')
    gamma = _extract(ckpt, 'gamma')
    if phi is None or gamma is None:
        print(f"  Skip {Path(fp).name}: missing phi or gamma")
        continue

    phi = phi.detach().cpu().numpy() if torch.is_tensor(phi) else np.array(phi)
    gamma = gamma.detach().cpu().numpy() if torch.is_tensor(gamma) else np.array(gamma)

    psi = _extract(ckpt, 'psi')
    if psi is not None:
        psi = psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.array(psi)

    kappa = _extract(ckpt, 'kappa')
    k = kappa.item() if torch.is_tensor(kappa) else float(kappa) if kappa is not None else 1.0

    losses = ckpt.get('losses', [])

    all_phi.append(phi)
    all_psi.append(psi)
    all_kappa.append(k)
    all_gamma.append(gamma)
    all_losses.append(losses)
    batch_labels.append(Path(fp).stem.replace('enrollment_model_REPARAM_W0.0001_', ''))

n = len(all_phi)
print(f"Loaded {n} batches")

# ── Pool parameters ──
phi_pooled = np.mean(np.stack(all_phi), axis=0)
kappa_pooled = float(np.mean(all_kappa))
gamma_pooled = np.mean(np.stack(all_gamma), axis=0)

psi_valid = [p for p in all_psi if p is not None]
psi_pooled = np.mean(np.stack(psi_valid), axis=0) if psi_valid else None

print(f"\n{'='*60}")
print(f"POOLED v2 PARAMS ({n} batches)")
print(f"{'='*60}")
print(f"  phi:   {phi_pooled.shape}")
print(f"  kappa: {kappa_pooled:.4f}")
print(f"  gamma: {gamma_pooled.shape}, mean|gamma|={np.abs(gamma_pooled).mean():.4f}")
if psi_pooled is not None:
    print(f"  psi:   {psi_pooled.shape}")

# ── Stability diagnostics ──
print(f"\n{'='*60}")
print("STABILITY DIAGNOSTICS")
print(f"{'='*60}")

kappas = np.array(all_kappa)
print(f"  kappa: mean={kappas.mean():.4f}, std={kappas.std():.4f}, "
      f"range=[{kappas.min():.4f}, {kappas.max():.4f}]")

gamma_mags = np.array([np.abs(g).mean() for g in all_gamma])
print(f"  |gamma|: mean={gamma_mags.mean():.4f}, std={gamma_mags.std():.4f}, "
      f"range=[{gamma_mags.min():.4f}, {gamma_mags.max():.4f}]")

final_losses = np.array([L[-1] for L in all_losses if len(L) > 0])
if len(final_losses) > 0:
    print(f"  final loss: mean={final_losses.mean():.2f}, std={final_losses.std():.2f}, "
          f"range=[{final_losses.min():.2f}, {final_losses.max():.2f}]")

# Cross-batch phi correlation
if n >= 2:
    phi_flat = [p.flatten() for p in all_phi]
    corrs = []
    for i in range(min(n, 10)):
        for j in range(i+1, min(n, 10)):
            corrs.append(np.corrcoef(phi_flat[i], phi_flat[j])[0, 1])
    print(f"  phi cross-batch corr: mean={np.mean(corrs):.4f} (first 10 batches)")

# ── Save pooled params ──
out_path = Path(OUT_DIR) / 'pooled_phi_kappa_gamma_reparam_v2.pt'
save_dict = {
    'phi': phi_pooled,
    'kappa': kappa_pooled,
    'gamma': gamma_pooled,
    'n_batches': n,
    'model_type': 'reparam_v2',
}
if psi_pooled is not None:
    save_dict['psi'] = psi_pooled
torch.save(save_dict, out_path)
print(f"\nSaved pooled params to: {out_path}")

# ── Plot sample loss curves ──
sample_indices = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else list(range(n))
sample_indices = sorted(set(sample_indices))

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
for idx in sample_indices:
    if len(all_losses[idx]) > 0:
        ax.plot(all_losses[idx], linewidth=1.2, alpha=0.8, label=batch_labels[idx])

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Training Loss', fontsize=13)
ax.set_title(f'Reparam v2 Training Loss — {len(sample_indices)} sample batches out of {n}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()

plot_path = str(Path(__file__).parent / 'v2_sample_loss_curves.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved loss plot to: {plot_path}")

# ── Plot kappa and |gamma| across batches ──
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

ax1.bar(range(n), kappas, color='steelblue', alpha=0.7)
ax1.axhline(kappa_pooled, color='red', linestyle='--', label=f'pooled={kappa_pooled:.3f}')
ax1.set_xlabel('Batch', fontsize=12)
ax1.set_ylabel('kappa', fontsize=12)
ax1.set_title('kappa across batches', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(range(n), gamma_mags, color='darkorange', alpha=0.7)
ax2.axhline(gamma_mags.mean(), color='red', linestyle='--', label=f'mean={gamma_mags.mean():.4f}')
ax2.set_xlabel('Batch', fontsize=12)
ax2.set_ylabel('mean |gamma|', fontsize=12)
ax2.set_title('mean |gamma| across batches', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path2 = str(Path(__file__).parent / 'v2_kappa_gamma_stability.png')
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
print(f"Saved stability plot to: {plot_path2}")

print("\nDone.")
