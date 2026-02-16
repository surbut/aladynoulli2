#!/usr/bin/env python
"""
Plot training loss convergence: v1 reparam vs v2 reparam vs v2 nokappa.
Loads losses from checkpoints (batch 0 for each).

Usage:
    MPLBACKEND=Agg python claudefile/plot_loss_v1_v2_nokappa.py
"""
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# v1 reparam (200 epochs, vanilla Adam) — losses in log file, not checkpoint
v1_log = '/Users/sarahurbut/aladynoulli2/claudefile/logs/batch_0_10000_reparam.log'
# v2 reparam (500 epochs, cosine + clip, kappa learned)
v2_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2/enrollment_model_REPARAM_W0.0001_batch_0_10000.pt'
# v2 nokappa (500 epochs, cosine + clip, kappa=1 fixed)
nk_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa/enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_0_10000.pt'

losses = {}
colors = {}

# Parse v1 losses from log
try:
    with open(v1_log) as f:
        v1_losses = [float(m.group(1)) for line in f for m in [re.search(r'^Loss:\s+([\d.]+)', line)] if m]
    if v1_losses:
        losses['v1 reparam (kappa=4.5, 200ep)'] = v1_losses
        colors['v1 reparam (kappa=4.5, 200ep)'] = 'steelblue'
        print(f"  v1 reparam: {len(v1_losses)} epochs, final={v1_losses[-1]:.2f}")
except FileNotFoundError:
    print("  v1 log not found")

# Load v2 and nokappa from checkpoints
for name, path, color in [
    ('v2 reparam (kappa=5.3, 500ep)', v2_path, 'darkorange'),
    ('v2 nokappa (kappa=1.0, 500ep)', nk_path, 'seagreen'),
]:
    try:
        ckpt = torch.load(path, weights_only=False)
        L = ckpt.get('losses', [])
        if len(L) == 0:
            print(f"  {name}: no losses in checkpoint")
            continue
        losses[name] = L
        colors[name] = color

        state = ckpt.get('model_state_dict', {})
        if 'kappa' in state:
            k = state['kappa'].item() if torch.is_tensor(state['kappa']) else float(state['kappa'])
        elif 'kappa' in ckpt:
            kk = ckpt['kappa']
            k = kk.item() if torch.is_tensor(kk) else float(kk)
        else:
            k = 1.0
        gamma_mag = state['gamma'].abs().mean().item() if 'gamma' in state else 0

        print(f"  {name}: {len(L)} epochs, final={L[-1]:.2f}, kappa={k:.3f}, |gamma|={gamma_mag:.4f}")
    except FileNotFoundError:
        print(f"  {name}: file not found, skipping")

if len(losses) < 2:
    print("Need at least 2 runs to compare")
    exit(1)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

# Left: absolute loss
for name, L in losses.items():
    ax1.plot(L, linewidth=1.8, alpha=0.85, color=colors[name],
             label=f"{name} (final={L[-1]:.1f})")
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('Training Loss', fontsize=13)
ax1.set_title('Training Loss Convergence (batch 0)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: normalized by initial loss
for name, L in losses.items():
    norm = np.array(L) / L[0]
    ax2.plot(norm, linewidth=1.8, alpha=0.85, color=colors[name], label=name)
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('Loss / Initial Loss', fontsize=13)
ax2.set_title('Normalized Loss (relative convergence)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = str(Path(__file__).parent / 'loss_v1_v2_nokappa.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
