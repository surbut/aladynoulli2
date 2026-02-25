#!/usr/bin/env python3
"""
Holdout AUC evaluation: slope model vs no-slope baseline.

Splits 40 batches into TRAIN (for pooling population params) and TEST (held-out).

No-slope baseline: uses the PROVEN prediction pathway (AladynSurvivalFixedPhi...Reparam)
  with baked phi = logit_prev + psi + epsilon pooled from base checkpoints.
  Identical to the Feb 18 evaluation pipeline.

Slope model: uses AladynSurvivalReparamWithSlope (only model with gamma_slope support)
  with pooled gamma_level, gamma_slope, psi, epsilon all frozen; only delta optimized.

Usage:
    python slope_holdout_auc.py                       # default: test=0-9, train=10-39 (two-phase slope)
    python slope_holdout_auc.py --single_phase         # one-phase slope, train=10-15, test=0-10
    python slope_holdout_auc.py --single_phase --single_phase_wide   # 1-phase slope, train=10-39 (30 batches), test=0-9 → holdout_auc_slope_1phase_pool30_vs_noslope.csv
    python slope_holdout_auc.py --use_loo_noslope      # load no-slope pi from enrollment_predictions_nokappa_v3_loo_all40 (no re-fit)
    python slope_holdout_auc.py --test_end 5          # test=0-4, train=5-39
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
from nokappa_prediction_utils import fit_and_extract_pi  # proven no-slope pathway (baked phi)

sys.path.insert(0, '/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
)

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
SLOPE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3/')
SLOPE_CKPT_DIR_1PHASE = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3_single_phase/')
BASE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/')
NOSLOPE_PI_LOO_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_nokappa_v3_loo_all40/')  # precomputed no-slope pi from run_loo_prediction_all40
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
PCE_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv'
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc/')

BATCH_SIZE = 10000
W = 1e-4
HOLDOUT_EPOCHS = 200


def load_data():
    """Load full dataset + covariates needed for AUC evaluation."""
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)
    E = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)
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

    pce_df = pd.read_csv(PCE_PATH)
    if 'Sex' not in pce_df.columns and 'sex' in pce_df.columns:
        pce_df['Sex'] = pce_df['sex'].map({0: 'Female', 1: 'Male'}).fillna('Unknown')
    if 'sex' not in pce_df.columns and 'Sex' in pce_df.columns:
        pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).fillna(-1)
    if 'age' not in pce_df.columns and 'Age' in pce_df.columns:
        pce_df['age'] = pce_df['Age']

    return Y, E, G_full, prevalence_t, signature_refs, disease_names, pce_df


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# ──────────────────────────────────────────────────────────────────────
# No-slope baseline: pool baked phi + gamma from base checkpoints,
# use the proven AladynSurvivalFixedPhiFixedGammaFixedKappaReparam
# ──────────────────────────────────────────────────────────────────────

def load_and_pool_base_params(train_indices):
    """Load base checkpoints and pool phi (baked), gamma, psi, kappa."""
    files = sorted(glob.glob(str(BASE_CKPT_DIR / '*.pt')))
    phis, gammas, psis = [], [], []
    for b in train_indices:
        if b >= len(files):
            continue
        ck = torch.load(files[b], weights_only=False)
        phis.append(_to_numpy(ck['phi']))       # already baked: logit_prev + psi + epsilon
        gammas.append(_to_numpy(ck['gamma']))
        psis.append(_to_numpy(ck['psi']))

    phi_pool = torch.tensor(np.mean(phis, axis=0), dtype=torch.float32)
    gamma_pool = torch.tensor(np.mean(gammas, axis=0), dtype=torch.float32)
    psi_pool = torch.tensor(np.mean(psis, axis=0), dtype=torch.float32)
    kappa = torch.tensor(1.0, dtype=torch.float32)

    print(f'  Base: pooled {len(phis)} checkpoints, '
          f'phi {phi_pool.shape}, gamma {gamma_pool.shape}')
    return phi_pool, gamma_pool, psi_pool, kappa


# ──────────────────────────────────────────────────────────────────────
# Slope model: pool gamma_level, gamma_slope, psi, epsilon from slope
# checkpoints. Uses AladynSurvivalReparamWithSlope (only model with
# gamma_slope). Epsilon frozen = same as baking phi.
# ──────────────────────────────────────────────────────────────────────

def load_and_pool_slope_params(train_indices, slope_ckpt_dir=None):
    """Load slope checkpoints and pool all population params."""
    ckpt_dir = slope_ckpt_dir if slope_ckpt_dir is not None else SLOPE_CKPT_DIR
    gamma_levels, gamma_slopes, psis, epsilons, gamma_healths = [], [], [], [], []
    for b in train_indices:
        start = b * BATCH_SIZE
        stop = start + BATCH_SIZE
        f = ckpt_dir / f'slope_model_batch_{start}_{stop}.pt'
        if not f.exists():
            print(f'  WARNING: missing slope checkpoint for batch {b}')
            continue
        ck = torch.load(f, weights_only=False)
        gamma_levels.append(_to_numpy(ck['gamma_level']))
        gamma_slopes.append(_to_numpy(ck['gamma_slope']))
        psis.append(_to_numpy(ck['psi']))
        epsilons.append(_to_numpy(ck['epsilon']))
        gh = ck.get('gamma_health')
        if gh is not None:
            gamma_healths.append(_to_numpy(gh))

    gamma_level = torch.tensor(np.mean(gamma_levels, axis=0), dtype=torch.float32)
    gamma_slope = torch.tensor(np.mean(gamma_slopes, axis=0), dtype=torch.float32)
    psi = torch.tensor(np.mean(psis, axis=0), dtype=torch.float32)
    epsilon = torch.tensor(np.mean(epsilons, axis=0), dtype=torch.float32)
    gamma_health = None
    if gamma_healths:
        gamma_health = torch.tensor(np.mean(gamma_healths, axis=0), dtype=torch.float32)

    print(f'  Slope: pooled {len(gamma_levels)} checkpoints, '
          f'|gamma_level|={torch.abs(gamma_level).mean():.4f}, '
          f'|gamma_slope|={torch.abs(gamma_slope).mean():.4f}, '
          f'|epsilon|={torch.abs(epsilon).mean():.4f}')
    return gamma_level, gamma_slope, psi, epsilon, gamma_health


def fit_slope_delta_and_extract_pi(Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
                                    gamma_level, gamma_slope, psi, epsilon,
                                    gamma_health=None, pretrained_delta=None, n_epochs=200):
    """Fit delta-only on held-out batch using slope model, return pi.
    Optional pretrained_delta warm-starts from no-slope checkpoint (e.g. for LOO)."""
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
        pretrained_epsilon=epsilon,
        gamma_health=gamma_health,
        pretrained_delta=pretrained_delta,
    )
    model.gamma_slope.data.copy_(gamma_slope)

    model.gamma_level.requires_grad_(False)
    model.gamma_slope.requires_grad_(False)
    model.psi.requires_grad_(False)
    model.epsilon.requires_grad_(False)

    optimizer = torch.optim.Adam([model.delta], lr=0.1)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = model.compute_loss(E_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pi, _, _ = model.forward()
        final_nll = model.compute_nll_only(E_batch)

    pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    pi = torch.clamp(pi, 1e-8, 1 - 1e-8)
    nll_val = final_nll.item()

    del model, optimizer
    gc.collect()
    return pi, nll_val


def run_auc_evaluation(pi, Y_test, E_test, disease_names, pce_df, label, n_bootstraps=100):
    """Run static and dynamic AUC evaluation, return results DataFrame."""
    all_results = []
    for horizon_name, eval_fn, kwargs in [
        ('static_10yr', evaluate_major_diseases_wsex_with_bootstrap_from_pi,
         dict(follow_up_duration_years=10)),
        ('dynamic_10yr', evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
         dict(follow_up_duration_years=10)),
    ]:
        print(f'\n  --- {label}: {horizon_name} ---')
        results = eval_fn(
            pi, Y_test, E_test, disease_names, pce_df,
            n_bootstraps=n_bootstraps, **kwargs,
        )
        for disease, metrics in results.items():
            row = {'model': label, 'horizon': horizon_name, 'disease': disease}
            if isinstance(metrics, dict):
                row.update(metrics)
            all_results.append(row)
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_start', type=int, default=0)
    parser.add_argument('--test_end', type=int, default=10)
    parser.add_argument('--train_start', type=int, default=10)
    parser.add_argument('--train_end', type=int, default=40)
    parser.add_argument('--single_phase', action='store_true',
                        help='Use one-phase slope checkpoints (single-phase training); defaults train 10-15, test 0-10')
    parser.add_argument('--single_phase_wide', action='store_true',
                        help='With --single_phase: use train 10-39 (30 batches) instead of 10-14; output suffix _1phase_pool30')
    parser.add_argument('--use_loo_noslope', action='store_true',
                        help='Load no-slope pi from enrollment_predictions_nokappa_v3_loo_all40 instead of re-fitting (same as nokappa_v3_auc_evaluation)')
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--holdout_epochs', type=int, default=HOLDOUT_EPOCHS)
    args = parser.parse_args()

    if args.single_phase:
        if getattr(args, 'single_phase_wide', False):
            # 1-phase with pool-30: train 10-39, same test 0-9
            if args.train_end == 40 and args.train_start == 10:
                pass  # keep train_end=40 -> batches 10..39
            slope_ckpt_dir = SLOPE_CKPT_DIR_1PHASE
            slope_label = 'slope_1phase_pool30'
            out_suffix = '_1phase_pool30'
        else:
            if args.train_end == 40 and args.train_start == 10:
                args.train_end = 15  # one-phase only has batches 10-14
            slope_ckpt_dir = SLOPE_CKPT_DIR_1PHASE
            slope_label = 'slope_1phase'
            out_suffix = '_1phase'
    else:
        slope_ckpt_dir = SLOPE_CKPT_DIR
        slope_label = 'slope'
        out_suffix = ''

    use_loo_noslope = getattr(args, 'use_loo_noslope', False)

    test_batches = list(range(args.test_start, args.test_end))
    train_batches = list(range(args.train_start, args.train_end))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('SLOPE vs NO-SLOPE: HOLDOUT AUC EVALUATION' + (' (ONE-PHASE SLOPE)' if args.single_phase else ''))
    print(f'  Train batches: {train_batches[0]}-{train_batches[-1]} ({len(train_batches)} batches)')
    print(f'  Test batches:  {test_batches[0]}-{test_batches[-1]} ({len(test_batches)} batches = {len(test_batches) * BATCH_SIZE} patients)')
    print(f'  Delta epochs:  {args.holdout_epochs}')
    print(f'  No-slope: ' + ('LOAD from ' + str(NOSLOPE_PI_LOO_DIR.name) if use_loo_noslope else 'proven pathway (baked phi from base checkpoints)'))
    print(f'  Slope:    {slope_label} from {slope_ckpt_dir.name}')
    print('=' * 70)

    # --- Load data ---
    print('\n[1/5] Loading data...')
    Y, E, G_full, prevalence_t, signature_refs, disease_names, pce_df = load_data()
    print(f'  Y: {Y.shape}, G: {G_full.shape}')

    # --- Load and pool params ---
    print('\n[2/5] Loading and pooling parameters...')
    phi_base, gamma_base, psi_base, kappa_base = load_and_pool_base_params(train_batches)
    gamma_level_s, gamma_slope_s, psi_s, eps_s, gh_s = load_and_pool_slope_params(train_batches, slope_ckpt_dir=slope_ckpt_dir)

    # --- Fit delta on each held-out batch, collect pi ---
    print(f'\n[3/5] Fitting delta on {len(test_batches)} held-out batches...')
    pi_slope_list, pi_noslope_list = [], []
    nll_records = []

    for b in test_batches:
        start = b * BATCH_SIZE
        stop = start + BATCH_SIZE
        print(f'\n  Batch {b} (patients {start}-{stop}):')

        Y_batch, E_batch, G_batch, _ = subset_data(Y, E, G_full, start, stop)

        # --- No-slope: proven pathway with baked phi ---
        print(f'    No-slope (baked phi)...')
        t0 = time.time()
        pi_n, loss_n = fit_and_extract_pi(
            Y_batch, E_batch, G_batch,
            phi_base, psi_base, kappa_base, gamma_base,
            signature_refs, prevalence_t, disease_names,
            num_epochs=args.holdout_epochs, learning_rate=0.1,
        )
        t_n = time.time() - t0

        # --- Slope: AladynReparamWithSlope ---
        print(f'    Slope...')
        t0 = time.time()
        pi_s, nll_s = fit_slope_delta_and_extract_pi(
            Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
            gamma_level_s, gamma_slope_s, psi_s, eps_s,
            gamma_health=gh_s,
            n_epochs=args.holdout_epochs,
        )
        t_s = time.time() - t0

        pi_noslope_list.append(pi_n)
        pi_slope_list.append(pi_s)

        print(f'    {slope_label} NLL={nll_s:.4f}, No-slope loss={loss_n:.4f}')
        print(f'    Time: slope {t_s:.0f}s, no-slope {t_n:.0f}s')

        nll_records.append({
            'batch': b, f'nll_{slope_label}': nll_s, 'nll_noslope': loss_n,
            'delta_nll': nll_s - loss_n,
        })

        del Y_batch, E_batch, G_batch, pi_s, pi_n
        gc.collect()

    # --- Concatenate pi and subset evaluation data ---
    pi_slope_all = torch.cat(pi_slope_list, dim=0)
    pi_noslope_all = torch.cat(pi_noslope_list, dim=0)
    del pi_slope_list, pi_noslope_list
    gc.collect()

    n_test = len(test_batches) * BATCH_SIZE
    Y_test = Y[:n_test]
    E_test = E[:n_test]
    pce_test = pce_df.iloc[:n_test].reset_index(drop=True)

    print(f'\n  Concatenated pi: slope {pi_slope_all.shape}, noslope {pi_noslope_all.shape}')
    print(f'  Eval data: Y_test {Y_test.shape}, E_test {E_test.shape}, pce_test {len(pce_test)}')

    # --- Save NLL results ---
    nll_df = pd.DataFrame(nll_records)
    nll_path = RESULTS_DIR / f'holdout_nll_slope{out_suffix}_vs_noslope.csv'
    nll_df.to_csv(nll_path, index=False)
    print(f'\n  NLL saved: {nll_path}')

    # --- Save pi tensors ---
    torch.save(pi_slope_all, RESULTS_DIR / f'pi_slope_holdout{out_suffix}.pt')
    if out_suffix == '_1phase_pool30':
        torch.save(pi_noslope_all, RESULTS_DIR / 'pi_noslope_holdout_1phase_pool30.pt')
    elif out_suffix:
        torch.save(pi_noslope_all, RESULTS_DIR / 'pi_noslope_holdout_1phase_run.pt')  # pool-5 no-slope
    else:
        torch.save(pi_noslope_all, RESULTS_DIR / 'pi_noslope_holdout.pt')
    print(f'  Saved pi: pi_slope_holdout{out_suffix}.pt' + (', pi_noslope_holdout_1phase_pool30.pt' if out_suffix == '_1phase_pool30' else (', pi_noslope_holdout_1phase_run.pt' if out_suffix else ', pi_noslope_holdout.pt')))

    # --- AUC evaluation ---
    print(f'\n[4/5] Running AUC evaluation ({args.n_bootstraps} bootstraps)...')
    df_slope = run_auc_evaluation(
        pi_slope_all, Y_test, E_test, disease_names, pce_test, slope_label,
        n_bootstraps=args.n_bootstraps,
    )
    df_noslope = run_auc_evaluation(
        pi_noslope_all, Y_test, E_test, disease_names, pce_test, 'noslope',
        n_bootstraps=args.n_bootstraps,
    )

    combined = pd.concat([df_slope, df_noslope], ignore_index=True)
    auc_path = RESULTS_DIR / f'holdout_auc_slope{out_suffix}_vs_noslope.csv'
    combined.to_csv(auc_path, index=False)
    print(f'\n  AUC results saved: {auc_path}')

    # --- Summary ---
    print(f'\n[5/5] Summary')
    print('=' * 70)
    print('HOLDOUT AUC: SLOPE vs NO-SLOPE')
    print('=' * 70)

    for horizon in combined['horizon'].unique():
        h = combined[combined['horizon'] == horizon]
        slope_rows = h[h['model'] == 'slope'].set_index('disease')
        noslope_rows = h[h['model'] == 'noslope'].set_index('disease')

        common = slope_rows.index.intersection(noslope_rows.index)
        print(f'\n  {horizon}:')
        print(f'  {"Disease":<25} {slope_label + " AUC":>14} {"NoSlope AUC":>12} {"Diff":>8}')
        print(f'  {"-"*60}')

        auc_col = 'auc' if 'auc' in slope_rows.columns else 'AUC'
        if auc_col not in slope_rows.columns:
            for c in slope_rows.columns:
                if 'auc' in c.lower():
                    auc_col = c
                    break

        for d in common:
            s_auc = slope_rows.loc[d, auc_col] if auc_col in slope_rows.columns else np.nan
            n_auc = noslope_rows.loc[d, auc_col] if auc_col in noslope_rows.columns else np.nan
            if pd.notna(s_auc) and pd.notna(n_auc):
                diff = s_auc - n_auc
                print(f'  {d:<25} {s_auc:>12.4f} {n_auc:>12.4f} {diff:>+8.4f}')

    print(f'\nAll results in: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
