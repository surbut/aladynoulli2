#!/usr/bin/env python3
"""
Pool 40 LOO pi batches into a single tensor, then compute AUCs:
  1. Static 10-year
  2. Dynamic 10-year
  3. 1-year at enrollment (offset 0)
  4. Rolling 1-year at enrollment + 0..9 years

Usage:
    python pool_and_evaluate_nokappa_v3.py                   # full run
    python pool_and_evaluate_nokappa_v3.py --pool-only        # just assemble pi
    python pool_and_evaluate_nokappa_v3.py --eval-only        # skip assembly, compute AUCs
    python pool_and_evaluate_nokappa_v3.py --n_patients 10000 # quick test on 10k
"""

import argparse
import sys
import os
import gc
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PI_BATCH_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_nokappa_v3_loo_all40/'
PI_FULL_PATH = os.path.join(PI_BATCH_DIR, 'pi_enroll_fixedphi_sex_FULL.pt')
DATA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
RESULTS_DIR = '/Users/sarahurbut/aladynoulli2/claudefile/results_nokappa_v3/'

# Import AUC evaluation utilities
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Pool batch pi files into single tensor
# ═══════════════════════════════════════════════════════════════════════════
def pool_pi_batches(batch_dir, output_path, n_batches=40, batch_size=10000):
    """Concatenate 40 batch pi tensors into one [400k, 348, 52] tensor."""
    if os.path.exists(output_path):
        print(f'FULL pi already exists: {output_path}')
        pi = torch.load(output_path, weights_only=False)
        print(f'  Shape: {pi.shape}')
        return pi

    print(f'Pooling {n_batches} batch pi files from {batch_dir}...')
    pi_list = []
    for i in range(n_batches):
        start = i * batch_size
        stop = (i + 1) * batch_size
        fp = os.path.join(batch_dir, f'pi_enroll_fixedphi_sex_{start}_{stop}.pt')
        if not os.path.exists(fp):
            raise FileNotFoundError(f'Missing batch file: {fp}')
        pi_batch = torch.load(fp, weights_only=False)
        print(f'  Batch {i+1}/{n_batches}: {Path(fp).name} → {pi_batch.shape}')
        pi_list.append(pi_batch)

    pi_full = torch.cat(pi_list, dim=0)
    print(f'\nConcatenated: {pi_full.shape}')

    print(f'Saving to {output_path}...')
    torch.save(pi_full, output_path)
    print(f'Saved.')

    del pi_list
    gc.collect()
    return pi_full


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Compute AUCs
# ═══════════════════════════════════════════════════════════════════════════
def compute_aucs(pi_full, n_patients=None, n_bootstraps=100):
    """Compute static 10yr, dynamic 10yr, 1yr at baseline, and rolling 1yr AUCs."""
    from fig5utils import (
        evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
        evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    )
    from evaluatetdccode import evaluate_major_diseases_rolling_1year_roc_curves

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load shared data
    print('Loading shared data...')
    Y_full = torch.load(DATA_DIR + 'Y_tensor.pt', weights_only=False)
    E_full = torch.load(DATA_DIR + 'E_enrollment_full.pt', weights_only=False)
    essentials = torch.load(DATA_DIR + 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')

    # Subset to requested size
    N = n_patients or 400000
    N = min(N, pi_full.shape[0], Y_full.shape[0])
    print(f'Using first {N} patients')
    pi = pi_full[:N]
    Y = Y_full[:N]
    E = E_full[:N]
    pce_df = pce_df_full.iloc[:N].reset_index(drop=True)
    print(f'  pi: {pi.shape}, Y: {Y.shape}, E: {E.shape}, pce_df: {len(pce_df)}')

    # ── 2a: Static 10-year ──────────────────────────────────────────────
    static10_path = results_dir / 'static_10yr_results.csv'
    if static10_path.exists():
        print(f'\nStatic 10yr already exists: {static10_path}')
    else:
        print(f'\n{"="*70}')
        print('Computing STATIC 10-YEAR AUCs...')
        print(f'{"="*70}')
        results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi, Y_100k=Y, E_100k=E,
            disease_names=disease_names, pce_df=pce_df,
            n_bootstraps=n_bootstraps, follow_up_duration_years=10)
        df = _results_to_df(results)
        df.to_csv(static10_path)
        print(f'Saved: {static10_path}')

    # ── 2b: Dynamic 10-year ─────────────────────────────────────────────
    dynamic10_path = results_dir / 'dynamic_10yr_results.csv'
    if dynamic10_path.exists():
        print(f'\nDynamic 10yr already exists: {dynamic10_path}')
    else:
        print(f'\n{"="*70}')
        print('Computing DYNAMIC 10-YEAR AUCs...')
        print(f'{"="*70}')
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi, Y_100k=Y, E_100k=E,
            disease_names=disease_names, pce_df=pce_df,
            n_bootstraps=n_bootstraps, follow_up_duration_years=10)
        df = _results_to_df(results)
        df.to_csv(dynamic10_path)
        print(f'Saved: {dynamic10_path}')

    # ── 2c: 1-year at enrollment (static) ───────────────────────────────
    static1yr_path = results_dir / 'static_1yr_results.csv'
    if static1yr_path.exists():
        print(f'\nStatic 1yr already exists: {static1yr_path}')
    else:
        print(f'\n{"="*70}')
        print('Computing STATIC 1-YEAR AUC (at enrollment)...')
        print(f'{"="*70}')
        results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi, Y_100k=Y, E_100k=E,
            disease_names=disease_names, pce_df=pce_df,
            n_bootstraps=n_bootstraps, follow_up_duration_years=1)
        df = _results_to_df(results)
        df.to_csv(static1yr_path)
        print(f'Saved: {static1yr_path}')

    # ── 2d: Rolling 1-year (enrollment + 0..9) ─────────────────────────
    rolling_path = results_dir / 'rolling_1yr_results.csv'
    if rolling_path.exists():
        print(f'\nRolling 1yr already exists: {rolling_path}')
    else:
        print(f'\n{"="*70}')
        print('Computing ROLLING 1-YEAR AUCs (enrollment + 0..9)...')
        print(f'{"="*70}')
        rolling_results = evaluate_major_diseases_rolling_1year_roc_curves(
            pi=pi, Y_full=Y, E_full=E,
            disease_names=disease_names, pce_df=pce_df,
            max_offset=9)
        # rolling_results is a dict of offset -> dict of disease -> results
        rows = []
        for offset, diseases in rolling_results.items():
            for disease, res in diseases.items():
                auc_val = res['auc'] if isinstance(res, dict) else res
                rows.append({
                    'Offset': offset,
                    'Disease': disease,
                    'AUC': auc_val,
                })
        df = pd.DataFrame(rows)
        df.to_csv(rolling_path, index=False)
        print(f'Saved: {rolling_path}')

    print(f'\n{"="*70}')
    print('ALL AUC COMPUTATIONS COMPLETE')
    print(f'Results in: {results_dir}')
    print(f'{"="*70}')


def _results_to_df(results):
    """Convert evaluation results dict to a sorted DataFrame."""
    df = pd.DataFrame({
        'Disease': list(results.keys()),
        'AUC': [r['auc'] for r in results.values()],
        'CI_lower': [r.get('ci_lower', np.nan) for r in results.values()],
        'CI_upper': [r.get('ci_upper', np.nan) for r in results.values()],
        'N_Events': [r.get('n_events', np.nan) for r in results.values()],
        'Event_Rate': [r.get('event_rate', np.nan) for r in results.values()],
    })
    return df.set_index('Disease').sort_values('AUC', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Pool nokappa v3 LOO pis and compute AUCs')
    parser.add_argument('--pool-only', action='store_true', help='Only assemble the FULL pi tensor')
    parser.add_argument('--eval-only', action='store_true', help='Skip pooling, just compute AUCs')
    parser.add_argument('--n_patients', type=int, default=None, help='Number of patients (default: all 400k)')
    parser.add_argument('--n_bootstraps', type=int, default=100, help='Bootstrap iterations for CIs')
    args = parser.parse_args()

    if not args.eval_only:
        print('='*70)
        print('STEP 1: Pool LOO pi batches')
        print('='*70)
        pi_full = pool_pi_batches(PI_BATCH_DIR, PI_FULL_PATH)
    else:
        print('Loading pre-assembled pi...')
        pi_full = torch.load(PI_FULL_PATH, weights_only=False)
        print(f'  Shape: {pi_full.shape}')

    if args.pool_only:
        print('\n--pool-only: done.')
        return

    print()
    print('='*70)
    print('STEP 2: Compute AUCs')
    print('='*70)
    compute_aucs(pi_full, n_patients=args.n_patients, n_bootstraps=args.n_bootstraps)


if __name__ == '__main__':
    main()
