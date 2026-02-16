#!/usr/bin/env python3
"""
Plot λ for nolr vs reparam — v3 style, same patient (e.g. patient 0).
Compares lambda from censor_e_batchrun_vectorized_nolr and censor_e_batchrun_vectorized_REPARAM.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
NOLR_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr'
REPARAM_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM'
DISEASE_NAMES_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv'


def compute_mean_lambda_numpy(G, gamma, signature_refs, K, T, healthy_ref=-5.0, genetic_scale=1):
    """mean(γ) = r_k(t) + G @ gamma[:, k]. Returns (N, K_total, T)."""
    N, P = G.shape
    K_total = K + 1
    r = np.array(signature_refs) if hasattr(signature_refs, '__len__') else np.array(signature_refs)
    mean_lam = np.zeros((N, K_total, T), dtype=np.float32)
    for k in range(K):
        r_k = np.atleast_1d(r[k]).reshape(-1)
        if r_k.size == 1:
            r_k = np.full(T, float(r_k.flat[0]))
        else:
            r_k = np.asarray(r_k, dtype=np.float32)[:T]
            if r_k.shape[0] < T:
                r_k = np.pad(r_k, (0, T - r_k.shape[0]), constant_values=(r_k[-1] if r_k.size else 0))
        mean_lam[:, k, :] = (genetic_scale * (G @ gamma[:, k]))[:, np.newaxis] + r_k
    mean_lam[:, K, :] = healthy_ref
    return mean_lam


def _extract(ckpt, name):
    if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
        return ckpt['model_state_dict'][name]
    return ckpt.get(name)


def main(patient_idx=0, batch=0, output_path=None):
    batch_size = 10000
    start = batch * batch_size
    stop = start + batch_size

    # Load data
    print("Loading data...")
    Y = torch.load(DATA_DIR + 'Y_tensor.pt', weights_only=False)
    G = torch.load(DATA_DIR + 'G_matrix.pt', weights_only=False)
    refs = torch.load(DATA_DIR + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    fh = pd.read_csv(COV_PATH)
    initial_clusters = torch.load(DATA_DIR + 'initial_clusters_400k.pt', weights_only=False)
    if torch.is_tensor(initial_clusters):
        initial_clusters = initial_clusters.numpy()
    disease_names = pd.read_csv(DISEASE_NAMES_PATH)['x'].tolist()

    Y_batch = Y[start:stop].numpy() if torch.is_tensor(Y) else Y[start:stop]
    G_batch = G[start:stop].numpy() if torch.is_tensor(G) else G[start:stop]
    sub = fh.iloc[start:stop].reset_index(drop=True)
    sex = sub['sex'].values if 'sex' in sub.columns else sub['Sex'].map({'Female': 0, 'Male': 1}).values
    pcs = sub[[f'f.22009.0.{i}' for i in range(1, 11)]].values
    G_with_sex = np.column_stack([G_batch, sex, pcs])
    G_centered = G_with_sex - G_with_sex.mean(axis=0, keepdims=True)
    G_std = G_centered.std(axis=0, keepdims=True)
    G_std = np.where(G_std < 1e-8, np.ones_like(G_std), G_std)
    G_with_sex = G_centered / G_std

    # Load checkpoints
    nolr_file = Path(NOLR_DIR) / f'enrollment_model_VECTORIZED_W0.0001_nolr_batch_{start}_{stop}.pt'
    reparam_file = Path(REPARAM_DIR) / f'enrollment_model_REPARAM_W0.0001_batch_{start}_{stop}.pt'
    if not nolr_file.exists() or not reparam_file.exists():
        raise FileNotFoundError(f"Need {nolr_file} and {reparam_file}")

    print(f"Loading nolr: {nolr_file.name}")
    ckpt_nolr = torch.load(nolr_file, map_location='cpu', weights_only=False)
    print(f"Loading reparam: {reparam_file.name}")
    ckpt_reparam = torch.load(reparam_file, map_location='cpu', weights_only=False)

    lam_nolr = _extract(ckpt_nolr, 'lambda_')
    lam_nolr = lam_nolr.numpy() if torch.is_tensor(lam_nolr) else np.array(lam_nolr)

    delta = _extract(ckpt_reparam, 'delta')
    gamma = _extract(ckpt_reparam, 'gamma')
    delta = delta.numpy() if torch.is_tensor(delta) else np.array(delta)
    gamma = gamma.numpy() if torch.is_tensor(gamma) else np.array(gamma)
    sig_refs = signature_refs.numpy() if torch.is_tensor(signature_refs) else np.array(signature_refs)
    K = lam_nolr.shape[1] - 1
    T = lam_nolr.shape[2]
    mean_lam = compute_mean_lambda_numpy(G_with_sex, gamma, sig_refs, K, T)
    lam_reparam = mean_lam + delta

    # θ = softmax(λ) for signature loadings (like v3)
    def softmax_np(x, axis=1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    theta_nolr = softmax_np(lam_nolr[patient_idx])
    theta_reparam = softmax_np(lam_reparam[patient_idx])

    patient_Y = Y_batch[patient_idx]
    diagnosis_times = {}
    for d in range(patient_Y.shape[0]):
        ev = np.where(patient_Y[d, :] == 1)[0]
        if len(ev) > 0:
            diagnosis_times[d] = ev.tolist()

    ages = np.arange(30, 30 + T)
    K_total = K + 1
    colors = plt.cm.tab20(np.linspace(0, 1, K_total))

    # v3-style: Panel 1 = θ (nolr vs reparam), Panel 2 = disease timeline
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax1, ax2 = axes

    # Panel 1: θ (signature loadings) — nolr solid, reparam dashed
    for k in range(K_total):
        lab = f'Sig{k}' if k < K else 'Health'
        ax1.plot(ages, theta_nolr[k], '-', color=colors[k], lw=2, alpha=0.9, label=f'{lab} (nolr)')
        ax1.plot(ages, theta_reparam[k], '--', color=colors[k], lw=1.5, alpha=0.7, label=f'{lab} (reparam)')

    for d, times in diagnosis_times.items():
        for t in times:
            if t < T:
                ax1.axvline(30 + t, color='gray', ls=':', alpha=0.25, lw=0.8)
                break
    ax1.set_ylabel('Signature Loading (θ)')
    ax1.set_title(f'Patient {patient_idx}: θ from NOLR (solid) vs REPARAM (dashed) — do λ look similar?')
    ax1.legend(loc='upper left', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([30, 81])
    ax1.set_ylim([0, max(theta_nolr.max(), theta_reparam.max()) * 1.08])

    # Panel 2: disease timeline (same as v3)
    if diagnosis_times:
        diag_order = sorted([(d, times[0]) for d, times in diagnosis_times.items()], key=lambda x: x[1])
        max_show = min(30, len(diag_order))
        for i, (d, t_d) in enumerate(diag_order[:max_show]):
            if t_d >= T:
                continue
            sig = int(initial_clusters[d]) if d < len(initial_clusters) else 0
            color = colors[sig % K_total]
            y_pos = max_show - i - 1
            ax2.plot([30, 30 + t_d], [y_pos, y_pos], color=color, lw=1, alpha=0.3)
            ax2.scatter(30 + t_d, y_pos, s=90, color=color, alpha=0.85, zorder=10, edgecolors='black', lw=1.2)
            ax2.text(29.5, y_pos, f'{i+1}', fontsize=8, va='center', ha='right')
        ax2.set_yticks(range(max_show))
        ax2.set_yticklabels([])
        ax2.set_ylim([-0.5, max_show - 0.5])
    ax2.set_ylabel('Disease (chronological)')
    ax2.set_xlabel('Age (years)')
    ax2.set_title('Disease Timeline')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([30, 81])

    fig.suptitle(f'Patient {patient_idx}: NOLR vs REPARAM λ (batch {batch})', fontsize=14)
    plt.tight_layout()
    out = output_path or str(Path(__file__).parent / f'lambda_nolr_vs_reparam_patient{patient_idx}.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()
    return fig


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patient', type=int, default=0)
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--out', type=str, default='')
    args = p.parse_args()
    main(patient_idx=args.patient, batch=args.batch, output_path=args.out or None)
