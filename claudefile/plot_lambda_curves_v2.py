#!/usr/bin/env python
"""
Plot sample lambda curves from v2 reparam training.
Shows lambda = mean_lambda(G@gamma) + delta decomposition.

Usage:
    MPLBACKEND=Agg python claudefile/plot_lambda_curves_v2.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

V2_CKPT = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2/enrollment_model_REPARAM_W0.0001_batch_0_10000.pt'
DATA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'

# ── Load checkpoint ──
print("Loading v2 checkpoint...")
ckpt = torch.load(V2_CKPT, weights_only=False)
state = ckpt['model_state_dict']

delta = state['delta'].detach().cpu()      # (N, K+1, T)
gamma = state['gamma'].detach().cpu()      # (P, K)
sig_refs = state['signature_refs'].detach().cpu()  # (K, T) or (K,)
G = ckpt['G']  # (N, P) numpy or tensor
if not torch.is_tensor(G):
    G = torch.tensor(G, dtype=torch.float32)

N, K_total, T = delta.shape
K = gamma.shape[1]
P = gamma.shape[0]
print(f"N={N}, K={K}, T={T}, P={P}")

# Also load Y to find people with actual disease events
Y = ckpt['Y']
if not torch.is_tensor(Y):
    Y = torch.tensor(Y, dtype=torch.float32)

# ── Reconstruct mean_lambda and lambda ──
# mean_lambda[n, k, t] = sig_refs[k] + G[n] @ gamma[:, k]  (broadcast over t)
mean_lambda = torch.zeros(N, K_total, T)
for k in range(K):
    genetic_component = G @ gamma[:, k]  # (N,)
    if sig_refs.dim() == 1:
        mean_lambda[:, k, :] = sig_refs[k] + genetic_component.unsqueeze(1)
    else:
        mean_lambda[:, k, :] = sig_refs[k].unsqueeze(0) + genetic_component.unsqueeze(1)

lam = mean_lambda + delta  # (N, K_total, T)

# ── Pick interesting individuals (ones with disease events) ──
# Find people with at least a few events
n_events_per_person = Y.sum(dim=(1, 2))  # (N,)
# Pick from different event-count quartiles
sorted_idx = torch.argsort(n_events_per_person)
quartiles = [int(0.25 * N), int(0.5 * N), int(0.75 * N), int(0.95 * N)]
sample_people = [sorted_idx[q].item() for q in quartiles]
print(f"Sample individuals: {sample_people}")
print(f"  Event counts: {[n_events_per_person[i].item() for i in sample_people]}")

# Pick top 4 clusters by mean absolute lambda for these people
top_clusters = []
for person in sample_people:
    lam_abs = lam[person, :K, :].abs().mean(dim=1)  # (K,)
    top_k = torch.argsort(lam_abs, descending=True)[:4].tolist()
    top_clusters.append(top_k)

time_axis = np.arange(T)

# ── Plot 1: Lambda curves for 4 individuals ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
axes = axes.flatten()

for idx, (person, top_k) in enumerate(zip(sample_people, top_clusters)):
    ax = axes[idx]
    n_ev = int(n_events_per_person[person].item())

    for k in top_k:
        l = lam[person, k, :].numpy()
        ml = mean_lambda[person, k, :].numpy()
        d = delta[person, k, :].numpy()
        ax.plot(time_axis, l, linewidth=1.5, label=f'k={k} (lambda)', alpha=0.9)
        ax.plot(time_axis, ml, '--', linewidth=1, alpha=0.4, color='gray')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('lambda')
    ax.set_title(f'Individual {person} ({n_ev} events) — top 4 clusters')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

fig.suptitle('Reparam v2: Lambda trajectories (solid) vs G@gamma baseline (dashed gray)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = str(Path(__file__).parent / 'v2_lambda_curves.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved: {out1}")

# ── Plot 2: Decomposition for 1 individual ──
person = sample_people[-1]  # high-event person
top_k = top_clusters[-1][:3]

fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), dpi=150, sharex=True)

for row, k in enumerate(top_k):
    ax = axes2[row]
    l = lam[person, k, :].numpy()
    ml = mean_lambda[person, k, :].numpy()
    d = delta[person, k, :].numpy()

    ax.fill_between(time_axis, 0, ml, alpha=0.15, color='blue', label='G@gamma (genetic baseline)')
    ax.fill_between(time_axis, ml, l, alpha=0.15, color='red', label='delta (individual deviation)')
    ax.plot(time_axis, l, 'k-', linewidth=1.5, label='lambda (total)')
    ax.plot(time_axis, ml, 'b--', linewidth=1, alpha=0.6)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel(f'Cluster {k}')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

axes2[-1].set_xlabel('Time (years)')
n_ev = int(n_events_per_person[person].item())
fig2.suptitle(f'Lambda decomposition — Individual {person} ({n_ev} events)\n'
              f'lambda = signature_ref + G@gamma + delta',
              fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
out2 = str(Path(__file__).parent / 'v2_lambda_decomposition.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

# ── Print summary stats ──
print(f"\n{'='*60}")
print("Lambda component magnitudes (all individuals, all clusters)")
print(f"{'='*60}")
mean_lam_mag = mean_lambda[:, :K, :].abs().mean().item()
delta_mag = delta[:, :K, :].abs().mean().item()
lam_mag = lam[:, :K, :].abs().mean().item()
print(f"  |mean_lambda| (sig_ref + G@gamma): {mean_lam_mag:.4f}")
print(f"  |delta|:                            {delta_mag:.4f}")
print(f"  |lambda| (total):                   {lam_mag:.4f}")
print(f"  delta / lambda ratio:               {delta_mag / lam_mag:.2%}")
