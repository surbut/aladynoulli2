#!/usr/bin/env python3
"""
Health Signature Loading Experiment

Uses master phi/psi, pooled kappa/gamma (from nolr fixedgk workflow) to show how
theta_health (or AEX_health) varies across individuals with different disease patterns.

Selects ~10 individuals: CVD-like, cancer-like, scattered diagnoses, few diagnoses,
and high health loading. Plots disease patterns + AEX_health for each.

Data sources (configurable):
- Lambda: from fixedgk prediction checkpoints (enrollment_predictions_fixedphi_fixedgk_nolr_vectorized)
  or from nolr training batches
- Y: disease status from data_for_running/Y_tensor.pt
- Disease names: from model_essentials or disease_names.csv
- Phi: from exported parameters or master checkpoint (for disease-signature mapping)

Usage:
    python health_signature_loading_experiment.py [--data_dir PATH] [--output_dir PATH]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.special import softmax
import argparse
import pandas as pd

# Health signature index (0-indexed: 20 = 21st signature)
HEALTH_SIG_IDX = 20


def load_lambda_from_fixedgk_checkpoints(pred_dir, max_batches=2):
    """Load lambda from fixedgk prediction model checkpoints."""
    pred_dir = Path(pred_dir)
    pattern = 'model_enroll_fixedphi_sex_*_*.pt'
    files = sorted(pred_dir.glob(pattern),
                   key=lambda f: int(f.stem.split('_')[-2]) if f.stem.split('_')[-2].isdigit() else 0)
    if not files:
        return None, None
    files = files[:max_batches]
    lambdas = []
    for fp in files:
        ck = torch.load(fp, map_location='cpu', weights_only=False)
        lam = ck['model_state_dict']['lambda_'].detach().cpu().numpy()
        lambdas.append(lam)
    return np.concatenate(lambdas, axis=0), files


def load_lambda_from_nolr_training(training_dir, max_batches=2):
    """Load lambda from nolr training batch checkpoints (fallback)."""
    training_dir = Path(training_dir)
    pattern = 'enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt'
    files = sorted(training_dir.glob(pattern),
                   key=lambda f: int(f.stem.split('_')[-2]) if f.stem.split('_')[-2].isdigit() else 0)
    if not files:
        return None, None
    files = files[:max_batches]
    lambdas = []
    for fp in files:
        ck = torch.load(fp, map_location='cpu', weights_only=False)
        lam = ck['model_state_dict']['lambda_'].detach().cpu().numpy()
        lambdas.append(lam)
    return np.concatenate(lambdas, axis=0), files


def compute_aex(theta):
    """AEX = time-averaged theta (trapezoidal approx). theta: (N, K, T)."""
    T = theta.shape[2]
    if T <= 1:
        return theta.mean(axis=2)
    return np.trapz(theta, dx=1.0, axis=2) / (T - 1)  # normalized


def compute_aex_simple(theta):
    """Simple mean over time."""
    return theta.mean(axis=2)


def load_Y(data_dir, n_max=None):
    """Load disease status Y from data_dir."""
    data_dir = Path(data_dir)
    y_path = data_dir / 'Y_tensor.pt'
    if not y_path.exists():
        return None
    Y = torch.load(y_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(Y):
        Y = Y.numpy()
    if n_max is not None:
        Y = Y[:n_max]
    return Y


def load_disease_names(data_dir, export_dir=None):
    """Load disease names from model_essentials or exported CSV."""
    data_dir = Path(data_dir)
    # Try model_essentials first
    ess_path = data_dir / 'model_essentials.pt'
    if ess_path.exists():
        ess = torch.load(ess_path, map_location='cpu', weights_only=False)
        if 'disease_names' in ess:
            names = ess['disease_names']
            if hasattr(names, 'columns') and 'x' in names.columns:
                return names['x'].dropna().astype(str).tolist()
            if isinstance(names, (list, np.ndarray)):
                return [str(n) for n in names if pd.notna(n)]
    # Try disease_names.csv
    for base in [data_dir, Path(__file__).parent.parent / 'pyScripts' / 'csv', export_dir]:
        if base is None:
            continue
        csv_path = Path(base) / 'disease_names.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = 'x' if 'x' in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()
    return None


def load_phi_for_mapping(export_dir=None, master_path=None):
    """Load phi to map diseases to signatures (argmax of mean phi over time)."""
    if export_dir:
        phi_path = Path(export_dir) / 'phi_master_pooled.npy'
        if phi_path.exists():
            phi = np.load(phi_path)
            # phi: (K, D, T); disease d -> sig = argmax_k mean_t phi[k,d,t]
            phi_mean = phi.mean(axis=2)  # (K, D)
            return np.argmax(phi_mean, axis=0)  # (D,) disease -> signature
    if master_path and Path(master_path).exists():
        ck = torch.load(master_path, map_location='cpu', weights_only=False)
        phi = ck.get('model_state_dict', ck).get('phi', ck.get('phi'))
        if phi is not None:
            if torch.is_tensor(phi):
                phi = phi.numpy()
            phi_mean = phi.mean(axis=2)
            return np.argmax(phi_mean, axis=0)
    return None


def select_diverse_individuals(theta, Y, disease_to_sig, disease_names, n_per=2):
    """
    Select individuals with diverse patterns:
    - Few diagnoses (low sum Y)
    - CVD-like (diagnoses in CVD signature diseases, e.g. MI ~112)
    - Cancer-like (diagnoses in cancer signature diseases)
    - Scattered (many diseases across different signatures)
    - High AEX_health
    """
    aex = compute_aex_simple(theta)  # (N, K+1)
    aex_health = aex[:, HEALTH_SIG_IDX]

    n_per = min(n_per, 3)
    selected = []
    labels = []

    N = theta.shape[0]
    if Y is not None:
        D, T = Y.shape[1], Y.shape[2]
        total_dx = Y.sum(axis=(1, 2))
        n_diseases_per_person = (Y.sum(axis=2) > 0).sum(axis=1)

        # Few diagnoses
        few_mask = (total_dx >= 1) & (total_dx <= 3)
        if few_mask.any():
            idx = np.where(few_mask)[0]
            idx = idx[np.argsort(-aex_health[idx])[:n_per]]
            for i in idx:
                selected.append(i)
                labels.append(f'Few dx (n={int(total_dx[i])}, AEX_health={aex_health[i]:.3f})')

        # High AEX_health (and at least one diagnosis so not "no data")
        high_health = (aex_health >= np.percentile(aex_health, 90)) & (total_dx >= 1)
        if high_health.any():
            idx = np.where(high_health)[0]
            idx = idx[np.argsort(-aex_health[idx])[:n_per]]
            for i in idx:
                if i not in selected:
                    selected.append(i)
                    labels.append(f'High health loading (AEX={aex_health[i]:.3f})')

        # CVD-like: MI (112) or similar
        cvd_diseases = [112, 113, 114, 115, 116]  # ASCVD
        cvd_diseases = [d for d in cvd_diseases if d < D]
        if cvd_diseases:
            has_cvd = (Y[:, cvd_diseases, :].sum(axis=(1, 2)) > 0)
            if has_cvd.any():
                idx = np.where(has_cvd)[0]
                idx = idx[np.argsort(-total_dx[idx])[:n_per]]
                for i in idx:
                    if i not in selected:
                        selected.append(i)
                        labels.append(f'CVD-like (AEX_health={aex_health[i]:.3f})')

        # Cancer-like: common cancer indices
        cancer_diseases = [21, 174, 175, 176]  # prostate, breast, lung, colorectal
        cancer_diseases = [d for d in cancer_diseases if d < D]
        if cancer_diseases:
            has_cancer = (Y[:, cancer_diseases, :].sum(axis=(1, 2)) > 0)
            if has_cancer.any():
                idx = np.where(has_cancer)[0]
                idx = idx[np.argsort(-total_dx[idx])[:n_per]]
                for i in idx:
                    if i not in selected:
                        selected.append(i)
                        labels.append(f'Cancer-like (AEX_health={aex_health[i]:.3f})')

        # Scattered: many different diseases
        scattered = (n_diseases_per_person >= 5) & (total_dx >= 8)
        if scattered.any():
            idx = np.where(scattered)[0]
            idx = idx[np.argsort(-n_diseases_per_person[idx])[:n_per]]
            for i in idx:
                if i not in selected:
                    selected.append(i)
                    labels.append(f'Scattered (n_dx={int(n_diseases_per_person[i])}, AEX={aex_health[i]:.3f})')

    # If no Y or not enough selected, use AEX_health spread
    if len(selected) < 6:
        for pct in [5, 25, 50, 75, 95]:
            target = np.percentile(aex_health, pct)
            i = np.argmin(np.abs(aex_health - target))
            if i not in selected:
                selected.append(i)
                labels.append(f'P{pct} AEX_health={aex_health[i]:.3f}')

    return selected[:10], labels[:10]


def plot_individual(Y_i, theta_i, ages, disease_names, ax_heat, ax_theta, title):
    """Plot one individual: disease heatmap + theta trajectory (especially health)."""
    D, T = Y_i.shape
    K = theta_i.shape[0]
    if ages is None:
        ages = np.arange(30, 30 + T)

    # Disease heatmap (diagnoses only): rows=diseases, cols=age
    ax_heat.imshow(Y_i, aspect='auto', cmap='Greys', vmin=0, vmax=1)
    ax_heat.set_ylabel('Diseases')
    ax_heat.set_xlabel('Age')
    ax_heat.set_title(f'{title}\nDisease pattern (1=diagnosed)')
    ax_heat.set_yticks([])

    # Theta trajectories (signature 20 = health highlighted)
    for k in range(K):
        alpha = 0.8 if k == HEALTH_SIG_IDX else 0.3
        lw = 2 if k == HEALTH_SIG_IDX else 0.8
        color = 'green' if k == HEALTH_SIG_IDX else None
        ax_theta.plot(ages, theta_i[k, :], alpha=alpha, lw=lw, color=color,
                     label=f'Sig{k}' if k == HEALTH_SIG_IDX else None)
    ax_theta.axhline(y=1/(K+1), color='gray', ls='--', alpha=0.5)
    ax_theta.set_xlabel('Age')
    ax_theta.set_ylabel(r'$\theta_k(t)$')
    ax_theta.set_title(f'Signature loadings (green=health Sig{HEALTH_SIG_IDX})')
    ax_theta.legend()
    ax_theta.set_ylim(0, 1)


def main():
    parser = argparse.ArgumentParser(description='Health signature loading experiment')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--pred_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_vectorized/')
    parser.add_argument('--training_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='Exported parameters dir (phi_master_pooled.npy)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_batches', type=int, default=2,
                        help='Max batches to load for lambda')
    args = parser.parse_args()

    out_dir = Path(args.output_dir or Path(__file__).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('='*80)
    print('HEALTH SIGNATURE LOADING EXPERIMENT')
    print('='*80)

    # Load lambda
    lambda_full, src_files = load_lambda_from_fixedgk_checkpoints(args.pred_dir, args.max_batches)
    if lambda_full is None:
        print('Fixedgk checkpoints not found, trying nolr training batches...')
        lambda_full, src_files = load_lambda_from_nolr_training(args.training_dir, args.max_batches)
    if lambda_full is None:
        print('ERROR: Could not load lambda. Check --pred_dir or --training_dir.')
        return 1

    print(f'Loaded lambda: {lambda_full.shape} from {len(src_files)} files')
    theta = softmax(lambda_full, axis=1)
    print(f'Theta shape: {theta.shape}, Health sig index: {HEALTH_SIG_IDX}')

    # Load Y
    n_max = lambda_full.shape[0]
    Y = load_Y(args.data_dir, n_max)
    if Y is not None:
        print(f'Loaded Y: {Y.shape}')
    else:
        print('Y not found; will select individuals by AEX_health percentiles only.')

    disease_names = load_disease_names(args.data_dir, args.export_dir)
    disease_to_sig = load_phi_for_mapping(args.export_dir)

    selected, labels = select_diverse_individuals(theta, Y, disease_to_sig, disease_names, n_per=2)
    print(f'\nSelected {len(selected)} individuals:')
    for i, lab in zip(selected, labels):
        print(f'  {i}: {lab}')

    T = theta.shape[2]
    ages = np.arange(30, 30 + T)

    # Plot grid: one row per individual
    n_show = min(6, len(selected))
    fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for row, (idx, lab) in enumerate(zip(selected[:n_show], labels[:n_show])):
        ax_heat = axes[row, 0]
        ax_theta = axes[row, 1]
        Y_i = Y[idx] if Y is not None else np.zeros((theta.shape[1] - 1, T))  # D ≈ K for display
        if Y is not None:
            Y_i = Y[idx]  # (D, T)
        else:
            Y_i = np.zeros((max(1, theta.shape[1]), T))
        plot_individual(Y_i, theta[idx], ages, disease_names, ax_heat, ax_theta, lab)
    plt.tight_layout()
    out_path = out_dir / 'health_signature_loading_experiment.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_path}')

    # Summary table
    aex = compute_aex_simple(theta)
    summary = []
    for idx, lab in zip(selected[:n_show], labels[:n_show]):
        row = {'idx': idx, 'label': lab, 'AEX_health': aex[idx, HEALTH_SIG_IDX]}
        if Y is not None:
            row['n_diagnoses'] = int(Y[idx].sum())
        summary.append(row)
    df = pd.DataFrame(summary)
    csv_path = out_dir / 'health_signature_loading_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved summary: {csv_path}')

    return 0


if __name__ == '__main__':
    exit(main())
