#!/usr/bin/env python
"""
Plot training loss convergence: reparam v1 (200 epochs, vanilla Adam)
vs reparam v2 (500 epochs, cosine annealing + grad clipping).

Usage:
    MPLBACKEND=Agg python claudefile/plot_v1_vs_v2_training.py
"""
import re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load v1 losses from training log ──
v1_log = Path(__file__).parent / 'logs' / 'batch_0_10000_reparam.log'
v1_losses = []
with open(v1_log) as f:
    for line in f:
        m = re.match(r'^Loss:\s+([\d.]+)', line)
        if m:
            v1_losses.append(float(m.group(1)))
print(f"v1 losses: {len(v1_losses)} epochs, final={v1_losses[-1]:.4f}")

# ── Load v2 losses from checkpoint ──
v2_ckpt = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/'
               'censor_e_batchrun_vectorized_REPARAM_v2/'
               'enrollment_model_REPARAM_W0.0001_batch_0_10000.pt')
ckpt = torch.load(v2_ckpt, map_location='cpu', weights_only=False)
v2_losses = ckpt['losses']
print(f"v2 losses: {len(v2_losses)} epochs, final={v2_losses[-1]:.4f}")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

# Left panel: full view (log scale)
ax = axes[0]
ax.plot(range(len(v1_losses)), v1_losses, color='#d62728', linewidth=1.2,
        alpha=0.8, label=f'v1: vanilla Adam (200 ep, final={v1_losses[-1]:.1f})')
ax.plot(range(len(v2_losses)), v2_losses, color='#1f77b4', linewidth=1.2,
        alpha=0.8, label=f'v2: cosine+clip (500 ep, final={v2_losses[-1]:.1f})')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Training Loss', fontsize=13)
ax.set_title('Loss Convergence: v1 vs v2 (log scale)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=v2_losses[-1], color='#1f77b4', linestyle='--', alpha=0.3)

# Right panel: zoomed into converged region (linear scale)
ax2 = axes[1]
# Start from epoch 50 for v1 (after the wild early phase)
v1_start = 50
ax2.plot(range(v1_start, len(v1_losses)), v1_losses[v1_start:],
         color='#d62728', linewidth=1.2, alpha=0.8,
         label=f'v1 (ep 50-199)')
ax2.plot(range(v1_start, len(v2_losses)), v2_losses[v1_start:],
         color='#1f77b4', linewidth=1.2, alpha=0.8,
         label=f'v2 (ep 50-499)')
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('Training Loss', fontsize=13)
ax2.set_title('Zoomed: epoch 50+ (linear scale)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=v2_losses[-1], color='#1f77b4', linestyle='--', alpha=0.3,
            label=f'v2 final: {v2_losses[-1]:.1f}')

# Add annotation for the gap at epoch 200
v1_final = v1_losses[-1]
v2_at_200 = v2_losses[199] if len(v2_losses) > 199 else v2_losses[-1]
ax2.annotate(f'v1 stops: {v1_final:.1f}\nv2 at 200: {v2_at_200:.1f}\n'
             f'gap: {v1_final - v2_at_200:.1f}',
             xy=(200, v1_final), fontsize=10,
             xytext=(250, v1_final + 1),
             arrowprops=dict(arrowstyle='->', color='gray'),
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Summary text
fig.text(0.5, 0.01,
         f'Batch 0-10k  |  v1: 200ep vanilla Adam → {v1_final:.1f}  |  '
         f'v2: 500ep cosine+clip → {v2_losses[-1]:.1f}  |  '
         f'improvement: {v1_final - v2_losses[-1]:.1f} ({(v1_final - v2_losses[-1])/v1_final*100:.1f}%)',
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_path = str(Path(__file__).parent / 'loss_convergence_v1_vs_v2.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {out_path}")
