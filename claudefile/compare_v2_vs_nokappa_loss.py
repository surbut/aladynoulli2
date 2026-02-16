#!/usr/bin/env python
"""
Compare training loss curves: v2 (with kappa) vs v2-nokappa (kappa=1 fixed).

Loads one batch checkpoint from each and plots loss curves side by side.
Note: absolute loss values aren't directly comparable (kappa changes the scale),
but convergence speed and relative shape are informative.

Usage:
    MPLBACKEND=Agg python claudefile/compare_v2_vs_nokappa_loss.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

V2_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2'
NOKAPPA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa'

# Load v2 (with kappa) batch 0
v2_path = Path(V2_DIR) / 'enrollment_model_REPARAM_W0.0001_batch_0_10000.pt'
v2_ckpt = torch.load(v2_path, weights_only=False)
v2_losses = v2_ckpt.get('losses', [])
v2_kappa = v2_ckpt['model_state_dict']['kappa'].item() if 'kappa' in v2_ckpt['model_state_dict'] else v2_ckpt.get('kappa', torch.ones(1)).item()
v2_gamma_mag = v2_ckpt['model_state_dict']['gamma'].abs().mean().item()

# Load nokappa batch 0
nk_path = Path(NOKAPPA_DIR) / 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_0_10000.pt'
nk_ckpt = torch.load(nk_path, weights_only=False)
nk_losses = nk_ckpt.get('losses', [])
nk_kappa = 1.0
nk_gamma_mag = nk_ckpt['model_state_dict']['gamma'].abs().mean().item()

print(f"v2 (kappa={v2_kappa:.3f}): {len(v2_losses)} epochs, final loss={v2_losses[-1]:.4f}")
print(f"nokappa (kappa=1.0):       {len(nk_losses)} epochs, final loss={nk_losses[-1]:.4f}")
print(f"v2 |gamma|={v2_gamma_mag:.4f}, nokappa |gamma|={nk_gamma_mag:.4f}")

# Compare phi
v2_phi = v2_ckpt['phi'] if 'phi' in v2_ckpt else v2_ckpt['model_state_dict'].get('phi')
nk_phi = nk_ckpt['phi'] if 'phi' in nk_ckpt else nk_ckpt['model_state_dict'].get('phi')
if v2_phi is not None and nk_phi is not None:
    v2_phi_np = v2_phi.detach().cpu().numpy() if torch.is_tensor(v2_phi) else np.array(v2_phi)
    nk_phi_np = nk_phi.detach().cpu().numpy() if torch.is_tensor(nk_phi) else np.array(nk_phi)
    print(f"\nphi range: v2=[{v2_phi_np.min():.2f}, {v2_phi_np.max():.2f}], "
          f"nokappa=[{nk_phi_np.min():.2f}, {nk_phi_np.max():.2f}]")
    print(f"sigmoid(phi) range: v2=[{1/(1+np.exp(-v2_phi_np.min())):.4f}, {1/(1+np.exp(-v2_phi_np.max())):.4f}], "
          f"nokappa=[{1/(1+np.exp(-nk_phi_np.min())):.4f}, {1/(1+np.exp(-nk_phi_np.max())):.4f}]")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

# Left: loss curves
ax = axes[0]
ax.plot(v2_losses, linewidth=1.5, alpha=0.8, color='steelblue',
        label=f'v2 (kappa={v2_kappa:.2f}), final={v2_losses[-1]:.2f}')
ax.plot(nk_losses, linewidth=1.5, alpha=0.8, color='darkorange',
        label=f'nokappa (kappa=1.0), final={nk_losses[-1]:.2f}')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Training Loss', fontsize=13)
ax.set_title('Training Loss: v2 vs No-Kappa', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: normalized loss (each divided by initial)
ax2 = axes[1]
v2_norm = np.array(v2_losses) / v2_losses[0]
nk_norm = np.array(nk_losses) / nk_losses[0]
ax2.plot(v2_norm, linewidth=1.5, alpha=0.8, color='steelblue',
         label=f'v2 (kappa={v2_kappa:.2f})')
ax2.plot(nk_norm, linewidth=1.5, alpha=0.8, color='darkorange',
         label='nokappa (kappa=1.0)')
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('Loss / Initial Loss', fontsize=13)
ax2.set_title('Normalized Loss (relative convergence)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = str(Path(__file__).parent / 'loss_v2_vs_nokappa.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
