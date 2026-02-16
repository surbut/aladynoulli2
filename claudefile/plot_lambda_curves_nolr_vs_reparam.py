#!/usr/bin/env python3
"""
Plot lambda curves for sample patients with many diagnoses: nolr vs reparam.

Uses lambda from training batch checkpoints directly (like Figure3 Individual Trajectories).
No fitting—just load and plot.

Usage:
    python claudefile/plot_lambda_curves_nolr_vs_reparam.py [--batch 0 --n_patients 4]
"""

import argparse
import glob
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


def compute_mean_lambda_numpy(G, gamma, signature_refs, K, T, healthy_ref=-5.0, genetic_scale=1):
    """mean(γ) = r_k(t) + G @ gamma[:, k]. r_k may be (T,) or scalar. Returns (N, K_total, T)."""
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
        # (N, 1) + (1, T) -> (N, T)
        mean_lam[:, k, :] = (genetic_scale * (G @ gamma[:, k]))[:, np.newaxis] + r_k
    mean_lam[:, K, :] = healthy_ref
    return mean_lam


def main():
    parser = argparse.ArgumentParser(description='Plot lambda curves from checkpoints (no fitting)')
    parser.add_argument('--batch', type=int, default=0, help='Batch index (0–39)')
    parser.add_argument('--n_patients', type=int, default=4, help='Number of patients to plot')
    parser.add_argument('--n_sigs', type=int, default=5, help='Number of signature clusters to plot')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--nolr_train_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr')
    parser.add_argument('--reparam_train_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    parser.add_argument('--out', type=str, default='', help='Output path for plot')
    args = parser.parse_args()

    batch_size = 10000
    start = args.batch * batch_size
    stop = start + batch_size

    print("Loading data...")
    Y = torch.load(args.data_dir + 'Y_tensor.pt', weights_only=False)
    G = torch.load(args.data_dir + 'G_matrix.pt', weights_only=False)
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    fh = pd.read_csv(args.covariates_path)

    Y_batch = Y[start:stop].numpy() if torch.is_tensor(Y) else Y[start:stop]
    G_batch = G[start:stop].numpy() if torch.is_tensor(G) else G[start:stop]
    sub = fh.iloc[start:stop].reset_index(drop=True)
    sex = sub['sex'].values if 'sex' in sub.columns else sub['Sex'].map({'Female': 0, 'Male': 1}).values
    pcs = sub[[f'f.22009.0.{i}' for i in range(1, 11)]].values
    G_with_sex = np.column_stack([G_batch, sex, pcs])
    if G_with_sex.shape[0] > 1:
        G_centered = G_with_sex - G_with_sex.mean(axis=0, keepdims=True)
        G_std = G_centered.std(axis=0, keepdims=True)
        G_std = np.where(G_std < 1e-8, np.ones_like(G_std), G_std)
        G_with_sex = G_centered / G_std

    # Load batch checkpoints (same logic as Figure3 Individual Trajectories)
    nolr_files = sorted(glob.glob(str(Path(args.nolr_train_dir) / 'enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt')))
    reparam_files = sorted(glob.glob(str(Path(args.reparam_train_dir) / 'enrollment_model_REPARAM_W0.0001_batch_*_*.pt')))

    nolr_batch_file = [nolr_files[args.batch]] if args.batch < len(nolr_files) else []
    reparam_batch_file = [reparam_files[args.batch]] if args.batch < len(reparam_files) else []

    if not nolr_batch_file or not reparam_batch_file:
        print(f"ERROR: Need nolr and reparam checkpoints for batch {args.batch}")
        print(f"  Nolr files: {len(nolr_files)}, reparam files: {len(reparam_files)}")
        sys.exit(1)

    print(f"Loading nolr: {Path(nolr_batch_file[0]).name}")
    ckpt_nolr = torch.load(nolr_batch_file[0], map_location='cpu', weights_only=False)
    lam_nolr = _extract(ckpt_nolr, 'lambda_')
    lam_nolr = lam_nolr.numpy() if torch.is_tensor(lam_nolr) else np.array(lam_nolr)

    print(f"Loading reparam: {Path(reparam_batch_file[0]).name}")
    ckpt_reparam = torch.load(reparam_batch_file[0], map_location='cpu', weights_only=False)
    delta = _extract(ckpt_reparam, 'delta')
    gamma = _extract(ckpt_reparam, 'gamma')
    delta = delta.numpy() if torch.is_tensor(delta) else np.array(delta)
    gamma = gamma.numpy() if torch.is_tensor(gamma) else np.array(gamma)
    sig_refs = signature_refs.numpy() if torch.is_tensor(signature_refs) else np.array(signature_refs)
    K = lam_nolr.shape[1] - 1  # K disease + 1 healthy
    T = lam_nolr.shape[2]
    mean_lam = compute_mean_lambda_numpy(G_with_sex, gamma, sig_refs, K, T)
    lam_reparam = mean_lam + delta
    delta_reparam = delta

    # Select patients with many diagnoses
    n_diag = Y_batch.sum(axis=(1, 2))
    top_idx = np.argsort(n_diag)[-args.n_patients:][::-1]

    N, K_total, T = lam_nolr.shape
    sigs_to_plot = min(args.n_sigs, K_total)
    t = np.arange(T)

    # Side-by-side: nolr | reparam, one row per patient
    fig, axes = plt.subplots(args.n_patients, 2, figsize=(14, 3 * args.n_patients), sharex='col', sharey='row')
    if args.n_patients == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'λ curves (batch {args.batch}, patients with most diagnoses)', fontsize=12)

    for i in range(args.n_patients):
        idx = top_idx[i]
        n_d = int(n_diag[idx])
        ax_n, ax_r = axes[i, 0], axes[i, 1]

        for k in range(sigs_to_plot):
            lab = f'sig{k}' if k < K_total - 1 else 'health'
            ax_n.plot(t, lam_nolr[idx, k], '-', alpha=0.8, linewidth=1, label=lab)
            ax_r.plot(t, lam_reparam[idx, k], '-', alpha=0.8, linewidth=1, label=lab)

        ax_n.set_title(f'Patient {idx} (n_diag={n_d}) — NOLR')
        ax_r.set_title(f'Patient {idx} (n_diag={n_d}) — REPARAM')
        ax_n.set_ylabel('λ')
        ax_n.legend(loc='upper right', fontsize=8, ncol=2)
        ax_r.legend(loc='upper right', fontsize=8, ncol=2)
        ax_n.grid(True, alpha=0.3)
        ax_r.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel('Time step')
    plt.tight_layout()

    out_path = args.out or str(Path(__file__).parent / 'lambda_curves_nolr_vs_reparam.pdf')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    # Delta decomposition for reparam
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, idx in enumerate(top_idx[:2]):
        ax = axes2[i]
        k = 0
        ax.plot(t, lam_reparam[idx, k], 'b-', lw=2, label='λ (reparam) = mean(γ) + δ')
        ax.plot(t, mean_lam[idx, k], 'g--', lw=1.5, label='mean(γ)')
        ax.plot(t, delta_reparam[idx, k], 'r-', lw=1, alpha=0.8, label='δ (residual)')
        ax.set_ylabel('λ / δ')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'Patient {idx}: λ decomposition (sig0)')
        ax.grid(True, alpha=0.3)
    axes2[1].set_xlabel('Time step')
    fig2.suptitle('Reparam: δ is the residual', fontsize=11)
    plt.tight_layout()
    out_path2 = str(Path(out_path).with_suffix('')) + '_delta_decomp.pdf'
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path2}')
    plt.close()

    # Compare full lambdas and kappa (kappa scales π, not λ—but models may partition scale differently)
    kappa_nolr = _extract(ckpt_nolr, 'kappa')
    kappa_reparam = _extract(ckpt_reparam, 'kappa')
    k_n = kappa_nolr.item() if torch.is_tensor(kappa_nolr) else float(kappa_nolr) if kappa_nolr is not None else np.nan
    k_r = kappa_reparam.item() if torch.is_tensor(kappa_reparam) else float(kappa_reparam) if kappa_reparam is not None else np.nan

    delta_mag = np.abs(delta_reparam).mean(axis=(1, 2))
    lam_nolr_mag = np.abs(lam_nolr).mean(axis=(1, 2))
    lam_reparam_mag = np.abs(lam_reparam).mean(axis=(1, 2))
    print(f'\n|λ| comparison (mean over k,t)—kappa scales π, not λ:')
    print(f'  mean |λ_nolr|:   {lam_nolr_mag.mean():.4f}')
    print(f'  mean |λ_reparam|: {lam_reparam_mag.mean():.4f}')
    print(f'  ratio (reparam/nolr): {(lam_reparam_mag / (lam_nolr_mag + 1e-8)).mean():.4f}')
    print(f'\n  mean |δ|:        {delta_mag.mean():.4f}  (δ/λ_nolr: {(delta_mag / (lam_nolr_mag + 1e-8)).mean():.4f})')
    print(f'\n  kappa nolr:      {k_n:.4f}')
    print(f'  kappa reparam:   {k_r:.4f}')
    print(f'  (kappa scales π; if reparam λ≈nolr λ, higher kappa → higher π)')


if __name__ == '__main__':
    main()
