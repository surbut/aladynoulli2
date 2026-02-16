#!/usr/bin/env python3
"""
Compare lambda trajectories across three W values (nokappa grid search) + nolr baseline.
Plots sample patients with many diagnoses, one column per W value.

Usage:
    python claudefile/compare_lambda_three_W.py [--n_patients 4 --n_sigs 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def _extract(ckpt, name):
    if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
        return ckpt['model_state_dict'][name]
    return ckpt.get(name)


def to_numpy(x):
    if x is None:
        return None
    return x.numpy() if torch.is_tensor(x) else np.array(x)


def compute_mean_lambda(G, gamma, signature_refs, K, T, healthy_ref=-5.0):
    N = G.shape[0]
    K_total = K + 1
    mean_lam = np.zeros((N, K_total, T), dtype=np.float32)
    for k in range(K):
        r_k = np.atleast_1d(signature_refs[k]).reshape(-1).astype(np.float32)
        if r_k.size == 1:
            r_k = np.full(T, float(r_k[0]))
        else:
            r_k = r_k[:T]
            if r_k.shape[0] < T:
                r_k = np.pad(r_k, (0, T - r_k.shape[0]), constant_values=r_k[-1])
        mean_lam[:, k, :] = (G @ gamma[:, k])[:, np.newaxis] + r_k
    mean_lam[:, K, :] = healthy_ref
    return mean_lam


def load_nokappa_checkpoint(ckpt_path, G, sig_refs, K, T):
    """Load a nokappa (reparam) checkpoint → lambda = mean(gamma) + delta."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    delta = to_numpy(_extract(ckpt, 'delta'))
    gamma = to_numpy(_extract(ckpt, 'gamma'))
    mean_lam = compute_mean_lambda(G, gamma, sig_refs, K, T)
    lam = mean_lam + delta
    return lam, mean_lam, delta, gamma


def load_nolr_checkpoint(ckpt_path):
    """Load a nolr (centered) checkpoint → lambda directly."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    lam = to_numpy(_extract(ckpt, 'lambda_'))
    return lam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patients', type=int, default=4)
    parser.add_argument('--n_sigs', type=int, default=5)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    args = parser.parse_args()

    grid_dir = Path(__file__).parent / 'grid_results'
    nolr_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/enrollment_model_VECTORIZED_W0.0001_nolr_batch_0_10000.pt'

    w_configs = [
        ('W=1e-5', grid_dir / 'nok_lr01_300_w1em5' / 'checkpoint.pt'),
        ('W=1e-4', grid_dir / 'nok_lr01_300_w1em4' / 'checkpoint.pt'),
        ('W=5e-4', grid_dir / 'nok_lr01_300_w5em4' / 'checkpoint.pt'),
    ]

    # Check files exist
    for name, path in w_configs:
        if not path.exists():
            print(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)

    print("Loading data...")
    Y = torch.load(args.data_dir + 'Y_tensor.pt', weights_only=False)
    G = torch.load(args.data_dir + 'G_matrix.pt', weights_only=False)
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    sig_refs = to_numpy(refs['signature_refs'])
    del refs

    # Batch 0
    Y_batch = to_numpy(Y[:10000])
    G_batch = to_numpy(G[:10000])
    fh = pd.read_csv(args.covariates_path)
    sub = fh.iloc[:10000].reset_index(drop=True)
    sex = sub['sex'].values if 'sex' in sub.columns else sub['Sex'].map({'Female': 0, 'Male': 1}).values
    pcs = sub[[f'f.22009.0.{i}' for i in range(1, 11)]].values
    G_full = np.column_stack([G_batch, sex, pcs])
    G_full = (G_full - G_full.mean(axis=0, keepdims=True))
    G_std = G_full.std(axis=0, keepdims=True)
    G_std = np.where(G_std < 1e-8, 1.0, G_std)
    G_full = G_full / G_std

    # Load nolr for shape reference
    print("Loading nolr baseline...")
    lam_nolr = load_nolr_checkpoint(nolr_path)
    N, K_total, T = lam_nolr.shape
    K = K_total - 1

    # Load nokappa checkpoints
    lambdas = {}
    gammas = {}
    for name, path in w_configs:
        print(f"Loading {name}...")
        lam, mean_lam, delta, gamma = load_nokappa_checkpoint(path, G_full, sig_refs, K, T)
        lambdas[name] = lam
        gammas[name] = gamma

    # Select patients with many diagnoses
    n_diag = Y_batch.sum(axis=(1, 2))
    top_idx = np.argsort(n_diag)[-args.n_patients:][::-1]

    sigs_to_plot = min(args.n_sigs, K_total)
    t = np.arange(T)
    cols = ['NOLR (baseline)'] + [name for name, _ in w_configs]

    # --- Plot 1: Lambda curves side by side ---
    fig, axes = plt.subplots(args.n_patients, len(cols),
                             figsize=(4 * len(cols), 3 * args.n_patients),
                             sharex=True, sharey='row')
    if args.n_patients == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Lambda curves: NOLR vs Nokappa at three W values (batch 0)', fontsize=13, y=1.01)

    all_lambdas = [lam_nolr] + [lambdas[name] for name, _ in w_configs]

    for i in range(args.n_patients):
        idx = top_idx[i]
        n_d = int(n_diag[idx])
        for j, (col_name, lam) in enumerate(zip(cols, all_lambdas)):
            ax = axes[i, j]
            for k in range(sigs_to_plot):
                lab = f'sig{k}' if k < K else 'health'
                ax.plot(t, lam[idx, k], '-', alpha=0.8, linewidth=1.2, label=lab)
            if i == 0:
                ax.set_title(col_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f'Pt {idx}\n({n_d} dx)', fontsize=9)
            if i == 0 and j == len(cols) - 1:
                ax.legend(loc='upper right', fontsize=7, ncol=2)

    for ax in axes[-1]:
        ax.set_xlabel('Time step')
    plt.tight_layout()
    out1 = str(Path(__file__).parent / 'lambda_three_W_comparison.pdf')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out1}')
    plt.close()

    # --- Plot 2: Summary stats ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    # Mean |lambda| distribution
    ax = axes2[0]
    for col_name, lam in zip(cols, all_lambdas):
        vals = np.abs(lam).mean(axis=(1, 2))
        ax.hist(vals, bins=50, alpha=0.5, label=col_name, density=True)
    ax.set_xlabel('Mean |λ| per patient')
    ax.set_title('λ magnitude distribution')
    ax.legend(fontsize=8)

    # Mean |gamma| per W
    ax = axes2[1]
    w_names = [name for name, _ in w_configs]
    gamma_mags = [np.abs(gammas[name]).mean() for name in w_names]
    ax.bar(w_names, gamma_mags, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Mean |γ|')
    ax.set_title('Genetic effect magnitude')

    # Lambda range (spread across signatures)
    ax = axes2[2]
    for col_name, lam in zip(cols, all_lambdas):
        lam_range = lam[:, :K, :].max(axis=1).mean(axis=1) - lam[:, :K, :].min(axis=1).mean(axis=1)
        ax.hist(lam_range, bins=50, alpha=0.5, label=col_name, density=True)
    ax.set_xlabel('λ range (max sig - min sig)')
    ax.set_title('Signature differentiation')
    ax.legend(fontsize=8)

    plt.suptitle('Summary: NOLR vs Nokappa at three W values', fontsize=12)
    plt.tight_layout()
    out2 = str(Path(__file__).parent / 'lambda_three_W_summary.pdf')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f'Saved: {out2}')
    plt.close()

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Config':<20} {'mean|λ|':>10} {'std|λ|':>10} {'λ range':>10} {'mean|γ|':>10}")
    print("-" * 70)
    for col_name, lam in zip(cols, all_lambdas):
        mag = np.abs(lam).mean(axis=(1, 2))
        lam_range = lam[:, :K, :].max(axis=1).mean(axis=1) - lam[:, :K, :].min(axis=1).mean(axis=1)
        gamma_str = ''
        if col_name in gammas:
            gamma_str = f'{np.abs(gammas[col_name]).mean():.4f}'
        else:
            gamma_str = 'n/a'
        print(f'{col_name:<20} {mag.mean():>10.4f} {mag.std():>10.4f} {lam_range.mean():>10.4f} {gamma_str:>10}')
    print("=" * 70)


if __name__ == '__main__':
    main()
