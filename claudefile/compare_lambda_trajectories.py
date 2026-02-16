#!/usr/bin/env python
"""
Compare lambda trajectories from grid search batch 0 checkpoints vs nolr batch 0.
Shows that nokappa models produce reasonable lambda curves.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))
from clust_huge_amp_vectorized_reparam import *

# --- Load nolr batch 0 ---
nolr_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt'
nolr_ckpt = torch.load(nolr_path, weights_only=False)
# nolr stores lambda directly (centered parameterization)
nolr_state = nolr_ckpt['model_state_dict']
# lambda_ is the raw parameter in centered model
nolr_lambda = nolr_state['lambda_'].detach().numpy()  # (N, K, T)
print(f"nolr lambda shape: {nolr_lambda.shape}")

# --- Load nokappa checkpoints for the 3 W values ---
grid_dir = Path('claudefile/grid_results')
configs = {
    'W=1e-5': 'nok_lr01_300_w1em5',
    'W=1e-4': 'nok_lr01_300_w1em4',
    'W=5e-4': 'nok_lr01_300_w5em4',
}

nokappa_lambdas = {}
for label, cfg_name in configs.items():
    ckpt_path = grid_dir / cfg_name / 'checkpoint.pt'
    ckpt = torch.load(ckpt_path, weights_only=False)
    state = ckpt['model_state_dict']

    # NCP: lambda = mean_lambda(gamma) + delta
    # mean_lambda = signature_refs[k] + genetic_scale * G @ gamma[:, k]
    # We need to reconstruct. But the checkpoint has delta and gamma.
    delta = state['delta'].detach().numpy()  # (N, K, T)
    gamma = state['gamma'].detach().numpy()  # (P, K)

    # Load G_with_sex for batch 0
    if 'G' not in dir():
        data_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
        import pandas as pd
        G_raw = torch.load(data_dir + 'G_matrix.pt', weights_only=False)
        G_batch = G_raw[:10000]
        fh = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
        if 'Sex' in fh.columns:
            sex = fh['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
        else:
            sex = fh['sex'].values
        sex_batch = sex[:10000]
        pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
        pcs = fh.iloc[:10000][pc_cols].values
        G_with_sex = np.column_stack([G_batch, sex_batch, pcs])

        refs = torch.load(data_dir + 'reference_trajectories.pt', weights_only=False)
        sig_refs = refs['signature_refs'].numpy()  # (K+1, T)
        del refs, G_raw

    # Reconstruct lambda = sig_refs[k] + G @ gamma[:, k] + delta[:, k, :]
    N, K, T = delta.shape
    lam = np.zeros_like(delta)
    Ggamma = G_with_sex @ gamma  # (N, K)
    for k in range(K):
        lam[:, k, :] = sig_refs[k, :T] + Ggamma[:, k:k+1] + delta[:, k, :]

    nokappa_lambdas[label] = lam
    print(f"{label} lambda shape: {lam.shape}, mean|delta|={np.abs(delta).mean():.4f}, mean|Gg|={np.abs(Ggamma).mean():.4f}")

# --- Plot comparison ---
# Pick 6 random patients, show their lambda for signature 0 (healthy) and a few disease signatures
np.random.seed(42)
sample_patients = np.random.choice(min(nolr_lambda.shape[0], 10000), 6, replace=False)
sigs_to_show = [0, 1, 2, 5]  # healthy + a few disease signatures
T = min(nolr_lambda.shape[2], nokappa_lambdas['W=1e-4'].shape[2])

fig, axes = plt.subplots(len(sample_patients), len(sigs_to_show),
                          figsize=(4*len(sigs_to_show), 3*len(sample_patients)))

colors = {'nolr': 'black', 'W=1e-5': '#1f77b4', 'W=1e-4': '#ff7f0e', 'W=5e-4': '#2ca02c'}

for i, pat in enumerate(sample_patients):
    for j, k in enumerate(sigs_to_show):
        ax = axes[i, j]

        # nolr
        ax.plot(range(T), nolr_lambda[pat, k, :T], color=colors['nolr'],
                linewidth=2, label='nolr', alpha=0.8)

        # nokappa configs
        for label, lam in nokappa_lambdas.items():
            ax.plot(range(T), lam[pat, k, :T], color=colors[label],
                    linewidth=1.5, label=label, alpha=0.8, linestyle='--')

        if i == 0:
            ax.set_title(f'Signature {k}', fontsize=11)
        if j == 0:
            ax.set_ylabel(f'Patient {pat}', fontsize=9)
        if i == len(sample_patients) - 1:
            ax.set_xlabel('Time')
        ax.tick_params(labelsize=8)

# Single legend at bottom
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Lambda trajectories: nolr vs nokappa (3 W values)\nBatch 0, 6 random patients',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('claudefile/lambda_trajectory_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to claudefile/lambda_trajectory_comparison.png")

# --- Summary stats ---
print(f"\n{'='*60}")
print(f"Lambda summary statistics (batch 0)")
print(f"{'='*60}")
print(f"{'Config':<12} {'mean':<10} {'std':<10} {'min':<10} {'max':<10}")
print(f"{'nolr':<12} {nolr_lambda[:,:,:T].mean():<10.4f} {nolr_lambda[:,:,:T].std():<10.4f} "
      f"{nolr_lambda[:,:,:T].min():<10.4f} {nolr_lambda[:,:,:T].max():<10.4f}")
for label, lam in nokappa_lambdas.items():
    print(f"{label:<12} {lam.mean():<10.4f} {lam.std():<10.4f} "
          f"{lam.min():<10.4f} {lam.max():<10.4f}")

# Correlation of lambda across all patients/sigs/timepoints
nolr_flat = nolr_lambda[:,:,:T].flatten()
for label, lam in nokappa_lambdas.items():
    nok_flat = lam.flatten()
    r = np.corrcoef(nolr_flat, nok_flat)[0, 1]
    print(f"corr(nolr, {label}): {r:.4f}")
