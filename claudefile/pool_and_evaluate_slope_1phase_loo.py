#!/usr/bin/env python3
"""
Pool 40 slope 1-phase LOO pi batches and compute AUCs (same metrics as no-slope LOO).

Prerequisite: Run run_loo_slope_1phase_all40.py so enrollment_predictions_slope_1phase_loo_all40/
  contains pi_enroll_fixedphi_sex_{start}_{stop}.pt for each batch.

Then this script:
  1. Concatenates 40 pi files → pi_enroll_fixedphi_sex_FULL.pt
  2. Computes: static 10yr, dynamic 10yr, static 1yr, rolling 1yr AUCs

Usage:
    python pool_and_evaluate_slope_1phase_loo.py
    python pool_and_evaluate_slope_1phase_loo.py --pool-only
    python pool_and_evaluate_slope_1phase_loo.py --eval-only --n_bootstraps 100
"""

import argparse
import gc
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

PI_BATCH_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_slope_1phase_loo_all40/'
PI_FULL_PATH = os.path.join(PI_BATCH_DIR, 'pi_enroll_fixedphi_sex_FULL.pt')
NOSLOPE_PI_FULL_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_nokappa_v3_loo_all40/pi_enroll_fixedphi_sex_FULL.pt'
DATA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_slope_1phase_loo/')

sys.path.insert(0, '/Users/sarahurbut/aladynoulli2/pyScripts/')
sys.path.insert(0, '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')


def pool_pi_batches(batch_dir, output_path, n_batches=40, batch_size=10000):
    """Concatenate 40 batch pi tensors into one [400k, 348, 52] tensor."""
    if os.path.exists(output_path):
        print(f'FULL pi already exists: {output_path}')
        pi = torch.load(output_path, weights_only=False)
        print(f'  Shape: {pi.shape}')
        return pi

    print(f'Pooling {n_batches} slope LOO batch pi files from {batch_dir}...')
    pi_list = []
    for i in range(n_batches):
        start = i * batch_size
        stop = (i + 1) * batch_size
        fp = os.path.join(batch_dir, f'pi_enroll_fixedphi_sex_{start}_{stop}.pt')
        if not os.path.exists(fp):
            raise FileNotFoundError(f'Missing batch file: {fp}. Run run_loo_slope_1phase_all40.py first.')
        pi_batch = torch.load(fp, weights_only=False)
        print(f'  Batch {i+1}/{n_batches}: {Path(fp).name} → {pi_batch.shape}')
        pi_list.append(pi_batch)

    pi_full = torch.cat(pi_list, dim=0)
    print(f'\nConcatenated: {pi_full.shape}')
    print(f'Saving to {output_path}...')
    torch.save(pi_full, output_path)
    print('Saved.')
    del pi_list
    gc.collect()
    return pi_full


def _results_to_df(results):
    df = pd.DataFrame({
        'Disease': list(results.keys()),
        'AUC': [r['auc'] for r in results.values()],
        'CI_lower': [r.get('ci_lower', np.nan) for r in results.values()],
        'CI_upper': [r.get('ci_upper', np.nan) for r in results.values()],
        'N_Events': [r.get('n_events', np.nan) for r in results.values()],
        'Event_Rate': [r.get('event_rate', np.nan) for r in results.values()],
    })
    return df.set_index('Disease').sort_values('AUC', ascending=False)


def compute_aucs(pi_full, n_patients=None, n_bootstraps=100, args=None):
    from fig5utils import (
        evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
        evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    )
    from evaluatetdccode import evaluate_major_diseases_rolling_1year_roc_curves

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print('Loading shared data...')
    Y_full = torch.load(os.path.join(DATA_DIR, 'Y_tensor.pt'), weights_only=False)
    E_full = torch.load(os.path.join(DATA_DIR, 'E_enrollment_full.pt'), weights_only=False)
    essentials = torch.load(os.path.join(DATA_DIR, 'model_essentials.pt'), weights_only=False)
    disease_names = essentials['disease_names']
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')

    N = n_patients or 400000
    N = min(N, pi_full.shape[0], Y_full.shape[0])
    print(f'Using first {N} patients')
    pi = pi_full[:N]
    Y = Y_full[:N]
    E = E_full[:N]
    pce_df = pce_df_full.iloc[:N].reset_index(drop=True)
    print(f'  pi: {pi.shape}, Y: {Y.shape}, E: {E.shape}')

    # Static 10yr: score = 1-year risk at enrollment (pi at t_enroll), outcome = event in next 10 years
    # Same as slope_holdout_auc: follow_up_duration_years=10 (not 1-year outcome)
    suffix = f'{N}' if N != 400000 else '400k'  # e.g. 100000 -> "100000", full -> "400k"
    static10_path = results_dir / f'static_10yr_results_{suffix}.csv'
    force_recompute = getattr(args, 'force_recompute', False) if args else False
    if static10_path.exists() and not force_recompute:
        print(f'\nStatic 10yr already exists: {static10_path}')
    else:
        # Sanity check: run same evaluation on no-slope pi; ASCVD should be ~0.75-0.77
        if os.path.exists(NOSLOPE_PI_FULL_PATH):
            pi_noslope = torch.load(NOSLOPE_PI_FULL_PATH, weights_only=False)[:N]
            print('\nSanity check: static 10yr on NO-SLOPE pi (expect ASCVD ~0.75-0.77)...')
            res_noslope = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
                pi=pi_noslope, Y_100k=Y, E_100k=E, disease_names=disease_names, pce_df=pce_df,
                n_bootstraps=min(10, n_bootstraps), follow_up_duration_years=10)
            auc_noslope_ascvd = res_noslope.get('ASCVD', {}).get('auc', np.nan)
            print(f'  No-slope ASCVD AUC = {auc_noslope_ascvd:.3f}')
            if not (0.72 <= auc_noslope_ascvd <= 0.80):
                print('  WARNING: No-slope ASCVD outside expected [0.72, 0.80]. Check data/code.')
            del pi_noslope, res_noslope
            gc.collect()
        print('\nComputing STATIC 10-YEAR AUCs (score=1yr risk at enrollment, outcome=10yr; follow_up_duration_years=10)...')
        results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi, Y_100k=Y, E_100k=E, disease_names=disease_names, pce_df=pce_df,
            n_bootstraps=n_bootstraps, follow_up_duration_years=10)
        df10 = _results_to_df(results)
        ascvd_auc = df10.loc['ASCVD', 'AUC'] if 'ASCVD' in df10.index else np.nan
        if ascvd_auc > 0.82:
            print(f'  WARNING: Slope ASCVD static 10yr = {ascvd_auc:.3f} (no-slope ~0.76). If unexpected, check pi alignment.')
        df10.to_csv(static10_path)
        print(f'Saved: {static10_path}')

    # Dynamic 10yr
    dynamic10_path = results_dir / f'dynamic_10yr_results_{suffix}.csv'
    if dynamic10_path.exists():
        print(f'\nDynamic 10yr already exists: {dynamic10_path}')
    else:
        print('\nComputing DYNAMIC 10-YEAR AUCs...')
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi, Y_100k=Y, E_100k=E, disease_names=disease_names, pce_df=pce_df,
            n_bootstraps=n_bootstraps, follow_up_duration_years=10)
        _results_to_df(results).to_csv(dynamic10_path)
        print(f'Saved: {dynamic10_path}')

    # Static 1yr
    static1yr_path = results_dir / f'static_1yr_results_{suffix}.csv'
    if static1yr_path.exists():
        print(f'\nStatic 1yr already exists: {static1yr_path}')
    else:
        print('\nComputing STATIC 1-YEAR AUC (at enrollment)...')
        results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi, Y_100k=Y, E_100k=E, disease_names=disease_names, pce_df=pce_df,
            n_bootstraps=n_bootstraps, follow_up_duration_years=1)
        _results_to_df(results).to_csv(static1yr_path)
        print(f'Saved: {static1yr_path}')

    # Rolling 1yr
    rolling_path = results_dir / f'rolling_1yr_results_{suffix}.csv'
    if rolling_path.exists():
        print(f'\nRolling 1yr already exists: {rolling_path}')
    else:
        print('\nComputing ROLLING 1-YEAR AUCs (enrollment + 0..9)...')
        rolling_results = evaluate_major_diseases_rolling_1year_roc_curves(
            pi=pi, Y_full=Y, E_full=E, disease_names=disease_names, pce_df=pce_df, max_offset=9)
        rows = []
        for offset, diseases in rolling_results.items():
            for disease, res in diseases.items():
                auc_val = res['auc'] if isinstance(res, dict) else res
                rows.append({'Offset': offset, 'Disease': disease, 'AUC': auc_val})
        pd.DataFrame(rows).to_csv(rolling_path, index=False)
        print(f'Saved: {rolling_path}')

    print(f'\n{"="*70}')
    print('SLOPE 1-PHASE LOO AUC COMPLETE')
    print(f'Results: {results_dir}')
    print(f'{"="*70}')


def main():
    parser = argparse.ArgumentParser(description='Pool slope 1-phase LOO pi batches and compute AUCs')
    parser.add_argument('--pool-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--n_patients', type=int, default=None)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--force-recompute', action='store_true', help='Recompute static 10yr even if CSV exists (and run no-slope sanity check)')
    args = parser.parse_args()

    if not args.eval_only:
        print('='*70)
        print('STEP 1: Pool slope LOO pi batches')
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
    compute_aucs(pi_full, n_patients=args.n_patients, n_bootstraps=args.n_bootstraps, args=args)


if __name__ == '__main__':
    main()
