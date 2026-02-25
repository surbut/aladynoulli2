#!/usr/bin/env python3
"""
Holdout evaluation for slope model: LOO across batches.

For each held-out batch b:
  1. Pool gamma_level, gamma_slope, psi from all OTHER slope checkpoints
  2. Fit delta on the held-out batch (using the slope model, delta-only optimization)
  3. Compute holdout NLL

Also compares against the no-slope baseline (pooled gamma from base nokappa v3).

Usage:
    python slope_holdout_evaluation.py [--n_batches 40]
"""

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from clust_huge_amp_vectorized_reparam_slope import AladynSurvivalReparamWithSlope, subset_data

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
SLOPE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3/')
BASE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/')
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_feb18/')

BATCH_SIZE = 10000
W = 1e-4
HOLDOUT_EPOCHS = 200  # delta-only fitting on holdout


def load_data():
    """Load full dataset."""
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)
    E = torch.load(DATA_DIR / 'E_matrix_corrected.pt', weights_only=False)
    G_prs = torch.load(DATA_DIR / 'G_matrix.pt', weights_only=False)
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    refs = torch.load(DATA_DIR / 'reference_trajectories.pt', weights_only=False)
    prevalence_t = torch.load(DATA_DIR / 'prevalence_t_corrected.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    disease_names = essentials['disease_names']

    fh = pd.read_csv(COV_PATH)
    sex = fh['sex'].values
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh[pc_cols].values
    G_full = np.column_stack([G_prs, sex, pcs])

    return Y, E, G_full, prevalence_t, signature_refs, disease_names


def load_slope_checkpoints():
    """Load all slope checkpoints."""
    files = sorted(glob.glob(str(SLOPE_CKPT_DIR / 'slope_model_batch_*.pt')))
    checkpoints = {}
    for f in files:
        ck = torch.load(f, weights_only=False)
        batch_idx = ck['config']['batch']
        checkpoints[batch_idx] = {
            'gamma_level': ck['gamma_level'].detach().cpu().numpy() if isinstance(ck['gamma_level'], torch.Tensor) else ck['gamma_level'],
            'gamma_slope': ck['gamma_slope'].detach().cpu().numpy() if isinstance(ck['gamma_slope'], torch.Tensor) else ck['gamma_slope'],
            'psi': ck['psi'].detach().cpu().numpy() if isinstance(ck['psi'], torch.Tensor) else ck['psi'],
            'gamma_health': ck.get('gamma_health'),
        }
    print(f'Loaded {len(checkpoints)} slope checkpoints: batches {sorted(checkpoints.keys())}')
    return checkpoints


def load_base_checkpoints():
    """Load gamma from all base nokappa v3 checkpoints (no-slope baseline)."""
    files = sorted(glob.glob(str(BASE_CKPT_DIR / '*.pt')))
    gammas, psis = [], []
    for f in files:
        ck = torch.load(f, weights_only=False)
        g = ck['gamma']
        p = ck['psi']
        gammas.append(g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else np.array(g))
        psis.append(p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p))
    return np.stack(gammas), np.stack(psis)


def pool_loo(all_params, exclude_idx):
    """Pool parameters excluding one batch (LOO)."""
    included = {k: v for k, v in all_params.items() if k != exclude_idx}
    if not included:
        raise ValueError(f'No batches to pool after excluding {exclude_idx}')

    gamma_levels = np.stack([v['gamma_level'] for v in included.values()])
    gamma_slopes = np.stack([v['gamma_slope'] for v in included.values()])
    psis = np.stack([v['psi'] for v in included.values()])

    return (
        torch.tensor(gamma_levels.mean(axis=0), dtype=torch.float32),
        torch.tensor(gamma_slopes.mean(axis=0), dtype=torch.float32),
        torch.tensor(psis.mean(axis=0), dtype=torch.float32),
    )


def fit_delta_holdout(Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
                      gamma_level, gamma_slope, psi, n_epochs=200):
    """
    Fit delta-only on holdout batch with fixed population params.
    Returns final holdout NLL (per-patient-disease-time).
    """
    N, D, T = Y_batch.shape
    K = len(signature_refs)
    P = G_batch.shape[1]

    model = AladynSurvivalReparamWithSlope(
        N=N, D=D, T=T, K=K, P=P,
        G=G_batch, Y=Y_batch, R=signature_refs, W=W,
        prevalence_t=prevalence_t,
        init_sd_scaler=1.0, genetic_scale=1.0,
        signature_references=signature_refs,
        healthy_reference=-5.0,
        pretrained_gamma=gamma_level,
        pretrained_psi=psi,
    )

    # Overwrite gamma_slope from pooled
    model.gamma_slope.data.copy_(gamma_slope)

    # Freeze everything except delta
    model.gamma_level.requires_grad_(False)
    model.gamma_slope.requires_grad_(False)
    model.psi.requires_grad_(False)
    model.epsilon.requires_grad_(False)

    optimizer = torch.optim.Adam([model.delta], lr=0.1)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, nll, gp = model.compute_loss(E_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, final_nll, _ = model.compute_loss(E_batch)

    final_nll_val = final_nll.item()
    del model, optimizer
    gc.collect()
    return final_nll_val


def fit_delta_noslope(Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
                      gamma, psi, n_epochs=200):
    """
    Fit delta-only on holdout batch with the NO-SLOPE model.
    Uses the base AladynSurvivalReparamWithSlope but with gamma_slope=0.
    """
    N, D, T = Y_batch.shape
    K = len(signature_refs)
    P = G_batch.shape[1]

    model = AladynSurvivalReparamWithSlope(
        N=N, D=D, T=T, K=K, P=P,
        G=G_batch, Y=Y_batch, R=signature_refs, W=W,
        prevalence_t=prevalence_t,
        init_sd_scaler=1.0, genetic_scale=1.0,
        signature_references=signature_refs,
        healthy_reference=-5.0,
        pretrained_gamma=gamma,
        pretrained_psi=psi,
    )

    # gamma_slope stays at zero (initialized as zeros)
    model.gamma_level.requires_grad_(False)
    model.gamma_slope.requires_grad_(False)
    model.psi.requires_grad_(False)
    model.epsilon.requires_grad_(False)

    optimizer = torch.optim.Adam([model.delta], lr=0.1)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, nll, gp = model.compute_loss(E_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, final_nll, _ = model.compute_loss(E_batch)

    final_nll_val = final_nll.item()
    del model, optimizer
    gc.collect()
    return final_nll_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_batches', type=int, default=40)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('SLOPE MODEL HOLDOUT EVALUATION (LOO)')
    print('=' * 60)

    print('\nLoading data...')
    Y, E, G_full, prevalence_t, signature_refs, disease_names = load_data()

    print('\nLoading slope checkpoints...')
    slope_ckpts = load_slope_checkpoints()
    available_batches = sorted(slope_ckpts.keys())
    n_eval = min(args.n_batches, len(available_batches))

    print('\nLoading base (no-slope) checkpoints...')
    base_gammas, base_psis = load_base_checkpoints()

    results = []
    print(f'\nRunning LOO evaluation on {n_eval} batches...\n')

    for holdout_idx in available_batches[:n_eval]:
        print(f'--- Holdout batch {holdout_idx} ---')
        start = holdout_idx * BATCH_SIZE
        stop = start + BATCH_SIZE

        Y_batch, E_batch, G_batch, _ = subset_data(Y, E, G_full, start, stop)

        # Slope model: pool from all other slope checkpoints
        gamma_level_loo, gamma_slope_loo, psi_loo = pool_loo(slope_ckpts, holdout_idx)

        t0 = time.time()
        nll_slope = fit_delta_holdout(
            Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
            gamma_level_loo, gamma_slope_loo, psi_loo,
            n_epochs=HOLDOUT_EPOCHS,
        )
        t_slope = time.time() - t0

        # No-slope baseline: pool base gamma excluding this batch
        mask = np.ones(len(base_gammas), dtype=bool)
        if holdout_idx < len(base_gammas):
            mask[holdout_idx] = False
        base_gamma_loo = torch.tensor(base_gammas[mask].mean(axis=0), dtype=torch.float32)
        base_psi_loo = torch.tensor(base_psis[mask].mean(axis=0), dtype=torch.float32)

        t0 = time.time()
        nll_noslope = fit_delta_noslope(
            Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
            base_gamma_loo, base_psi_loo,
            n_epochs=HOLDOUT_EPOCHS,
        )
        t_noslope = time.time() - t0

        delta_nll = nll_slope - nll_noslope
        print(f'  Slope NLL: {nll_slope:.4f}, No-slope NLL: {nll_noslope:.4f}, '
              f'Delta: {delta_nll:.4f} ({"slope better" if delta_nll < 0 else "no-slope better"})')
        print(f'  Time: slope {t_slope:.0f}s, no-slope {t_noslope:.0f}s')

        results.append({
            'batch': holdout_idx,
            'nll_slope': nll_slope,
            'nll_noslope': nll_noslope,
            'delta_nll': delta_nll,
            'slope_better': delta_nll < 0,
        })

        del Y_batch, E_batch, G_batch
        gc.collect()

    df = pd.DataFrame(results)
    save_path = RESULTS_DIR / 'slope_vs_noslope_holdout_nll.csv'
    df.to_csv(save_path, index=False)
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(df.to_string(index=False))
    print(f'\nMean slope NLL:    {df["nll_slope"].mean():.4f}')
    print(f'Mean no-slope NLL: {df["nll_noslope"].mean():.4f}')
    print(f'Mean delta:        {df["delta_nll"].mean():.4f}')
    print(f'Slope wins:        {df["slope_better"].sum()}/{len(df)} batches')
    print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()
