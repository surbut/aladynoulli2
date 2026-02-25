#!/usr/bin/env python3
"""
Compare LOO vs holdout pool composition: slope vs no-slope (first 100k).

For both models we form:
- 10 "LOO" pools: for leave-out batch i in 0..9, pool from the other 39 batches.
- 1 holdout pool: batches 10-39 (no test batch in the pool).

We report the same metrics for slope (gamma_level, gamma_slope) and no-slope (gamma).
If slope's pooled params change MORE when we switch from LOO to holdout (or vary more
across the 10 LOO folds), that supports "slope is more cohort-sensitive than no-slope."

Run from claudefile (env with numpy, torch). Requires slope and base (no-slope) checkpoints.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slope_holdout_auc import load_and_pool_slope_params, BASE_CKPT_DIR

# Use same default as slope_holdout_auc (override with env SLOPE_CKPT_DIR if needed)
SLOPE_CKPT_DIR = Path(__file__).resolve().parent / 'slope_model_nokappa_v3_single_phase'
if not SLOPE_CKPT_DIR.exists():
    SLOPE_CKPT_DIR = Path(os.environ.get('SLOPE_CKPT_DIR', '/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3_single_phase'))
RESULTS_DIR = Path(__file__).resolve().parent / 'results_holdout_auc'


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def load_noslope_gammas_same_order_as_loo(ckpt_dir):
    """Load no-slope gamma from all 40 batches in the same order as run_loo_prediction_all40.
    Returns list of 40 numpy arrays (one per batch). Ensures batch index b = files[b].
    """
    pattern = str(Path(ckpt_dir) / 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_*_*.pt')
    files = sorted(glob.glob(pattern), key=lambda f: int(Path(f).stem.split('_')[-2]))
    if len(files) < 40:
        raise FileNotFoundError(f'Expected 40 no-slope checkpoints, got {len(files)} in {ckpt_dir}')
    gammas = []
    for fp in files[:40]:
        ck = torch.load(fp, weights_only=False)
        g = ck.get('gamma') or (ck.get('model_state_dict') or {}).get('gamma')
        if g is None:
            raise KeyError(f'No gamma in {Path(fp).name}')
        gammas.append(_to_numpy(g))
    return gammas


def main():
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    n_folds = 10   # batches 0-9 (first 100k)
    n_batches = 40

    # ----- Slope: 10 LOO pools + 1 holdout pool -----
    print('Loading 10 LOO pooled slope params (leave-out batch 0..9)...')
    loo_gl_list, loo_gs_list = [], []
    for leave_out in range(n_folds):
        train_indices = [j for j in range(n_batches) if j != leave_out]
        gl, gs, _, _, _ = load_and_pool_slope_params(
            train_indices, slope_ckpt_dir=SLOPE_CKPT_DIR)
        loo_gl_list.append(_to_numpy(gl))
        loo_gs_list.append(_to_numpy(gs))

    print('Loading slope holdout pool (batches 10-39)...')
    holdout_gl, holdout_gs, _, _, _ = load_and_pool_slope_params(
        list(range(10, n_batches)), slope_ckpt_dir=SLOPE_CKPT_DIR)
    holdout_gl = _to_numpy(holdout_gl)
    holdout_gs = _to_numpy(holdout_gs)

    loo_gl = np.stack(loo_gl_list, axis=0)
    loo_gs = np.stack(loo_gs_list, axis=0)
    mean_loo_gl = np.mean(loo_gl, axis=0)
    mean_loo_gs = np.mean(loo_gs, axis=0)
    diff_gl = mean_loo_gl - holdout_gl
    diff_gs = mean_loo_gs - holdout_gs
    std_loo_gl = np.std(loo_gl, axis=0, ddof=1)
    std_loo_gs = np.std(loo_gs, axis=0, ddof=1)

    # ----- No-slope: same 10 LOO pools + 1 holdout pool (same file order as run_loo_prediction_all40) -----
    print('\nLoading no-slope gammas (same pattern + sort as run_loo_prediction_all40)...')
    all_noslope_g = load_noslope_gammas_same_order_as_loo(BASE_CKPT_DIR)
    print('Forming 10 LOO pools (leave-out 0..9) and 1 holdout pool (10-39)...')
    loo_g_list = []
    for leave_out in range(n_folds):
        indices = [j for j in range(n_batches) if j != leave_out]
        loo_g_list.append(np.mean(np.stack([all_noslope_g[j] for j in indices]), axis=0))
    holdout_g = np.mean(np.stack([all_noslope_g[j] for j in range(10, n_batches)]), axis=0)

    loo_g = np.stack(loo_g_list, axis=0)
    mean_loo_g = np.mean(loo_g, axis=0)
    diff_g = mean_loo_g - holdout_g
    std_loo_g = np.std(loo_g, axis=0, ddof=1)

    # ----- Report: side-by-side slope vs no-slope -----
    print('\n' + '=' * 60)
    print('Pool composition: LOO for batch i = pool from {0..39} \\ {i}; holdout = 10-39')
    print('=' * 60)
    print('\n--- Mean |LOO - holdout| (how much pool changes when we exclude test batches) ---')
    slope_mean_diff = (np.abs(diff_gl).mean() + np.abs(diff_gs).mean()) / 2
    noslope_mean_diff = np.abs(diff_g).mean()
    print(f'  Slope:   gamma_level mean |diff| = {np.abs(diff_gl).mean():.6f},  gamma_slope = {np.abs(diff_gs).mean():.6f}')
    print(f'  No-slope: gamma mean |diff| = {noslope_mean_diff:.6f}')
    print('\n--- Std of pooled params across 10 LOO folds (sensitivity to who is in the pool) ---')
    print(f'  Slope:   gamma_level mean std = {std_loo_gl.mean():.6f},  gamma_slope mean std = {std_loo_gs.mean():.6f}')
    print(f'  No-slope: gamma mean std = {std_loo_g.mean():.6f}')
    print('\n  Interpretation: If slope numbers are larger, slope params are more cohort-sensitive.')
    print('  (Same pool design for both; only the model differs.)')

    # Save summary
    out = results_dir / 'loo_vs_holdout_pool_comparison.txt'
    with open(out, 'w') as f:
        f.write('LOO vs Holdout pool comparison: slope vs no-slope\n')
        f.write('LOO for batch i = pool from {0..39} \\ {i}; holdout = 10-39.\n\n')
        f.write('Slope:  mean |mean_loo - holdout|  gamma_level = %.6f  gamma_slope = %.6f\n' % (np.abs(diff_gl).mean(), np.abs(diff_gs).mean()))
        f.write('Slope:  mean std across 10 LOO folds  gamma_level = %.6f  gamma_slope = %.6f\n' % (std_loo_gl.mean(), std_loo_gs.mean()))
        f.write('No-slope:  mean |mean_loo - holdout| gamma = %.6f\n' % noslope_mean_diff)
        f.write('No-slope:  mean std across 10 LOO folds gamma = %.6f\n' % std_loo_g.mean())
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
