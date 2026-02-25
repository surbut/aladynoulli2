#!/usr/bin/env python3
"""
1-year-at-enrollment AUC evaluation on saved holdout pi tensors.

Run AFTER slope_holdout_auc.py finishes (uses its saved pi tensors).

Usage:
    python slope_holdout_auc_1yr.py
    python slope_holdout_auc_1yr.py --single_phase        # pool-5 (1-phase) pi
    python slope_holdout_auc_1yr.py --single_phase_wide   # pool-30 (1-phase) pi
    python slope_holdout_auc_1yr.py --n_bootstraps 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
PCE_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv'
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc/')

sys.path.insert(0, '/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year,
)

N_TEST = 100000  # 10 batches x 10k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--single_phase', action='store_true',
                        help='Use one-phase slope pi (pi_slope_holdout_1phase.pt, pool-5)')
    parser.add_argument('--single_phase_wide', action='store_true',
                        help='Use 1-phase pool-30 pi (pi_slope_holdout_1phase_pool30.pt)')
    args = parser.parse_args()

    if args.single_phase_wide:
        pi_slope_path = RESULTS_DIR / 'pi_slope_holdout_1phase_pool30.pt'
        pi_noslope_path = RESULTS_DIR / 'pi_noslope_holdout_1phase_pool30.pt'
        slope_label = 'slope_1phase_pool30'
        out_suffix = '_1phase_pool30'
    elif args.single_phase:
        pi_slope_path = RESULTS_DIR / 'pi_slope_holdout_1phase.pt'
        pi_noslope_path = RESULTS_DIR / 'pi_noslope_holdout_1phase_run.pt'
        slope_label = 'slope_1phase'
        out_suffix = '_1phase'
    else:
        pi_slope_path = RESULTS_DIR / 'pi_slope_holdout.pt'
        pi_noslope_path = RESULTS_DIR / 'pi_noslope_holdout.pt'
        slope_label = 'slope'
        out_suffix = ''

    print('=' * 70)
    print('1-YEAR AT ENROLLMENT AUC: ' + slope_label.upper() + ' vs NO-SLOPE (from saved pi)')
    print('=' * 70)

    # Load saved pi tensors (paths set above)
    pi_slope = torch.load(pi_slope_path, weights_only=False)
    pi_noslope = torch.load(pi_noslope_path, weights_only=False)
    print(f'Loaded pi: {slope_label} {pi_slope.shape}, noslope {pi_noslope.shape}')

    # Load evaluation data
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)[:N_TEST]
    E = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)[:N_TEST]
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']

    pce_df = pd.read_csv(PCE_PATH).iloc[:N_TEST].reset_index(drop=True)
    if 'Sex' not in pce_df.columns and 'sex' in pce_df.columns:
        pce_df['Sex'] = pce_df['sex'].map({0: 'Female', 1: 'Male'}).fillna('Unknown')
    if 'sex' not in pce_df.columns and 'Sex' in pce_df.columns:
        pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).fillna(-1)
    if 'age' not in pce_df.columns and 'Age' in pce_df.columns:
        pce_df['age'] = pce_df['Age']

    print(f'Y: {Y.shape}, E: {E.shape}, pce_df: {len(pce_df)}')

    all_results = []

    # --- Static 1yr (1-year score at enrollment, 1-year outcome) ---
    for label, pi in [(slope_label, pi_slope), ('noslope', pi_noslope)]:
        print(f'\n--- {label}: static_1yr ---')
        results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi, Y, E, disease_names, pce_df,
            n_bootstraps=args.n_bootstraps, follow_up_duration_years=1,
        )
        for disease, metrics in results.items():
            row = {'model': label, 'horizon': 'static_1yr', 'disease': disease}
            if isinstance(metrics, dict):
                row.update(metrics)
            all_results.append(row)

    # --- Dynamic 1yr (1-year rolling risk) ---
    for label, pi in [(slope_label, pi_slope), ('noslope', pi_noslope)]:
        print(f'\n--- {label}: dynamic_1yr ---')
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year(
            pi, Y, E, disease_names, pce_df,
            n_bootstraps=args.n_bootstraps, follow_up_duration_years=1,
        )
        for disease, metrics in results.items():
            row = {'model': label, 'horizon': 'dynamic_1yr', 'disease': disease}
            if isinstance(metrics, dict):
                row.update(metrics)
            all_results.append(row)

    combined = pd.DataFrame(all_results)
    save_path = RESULTS_DIR / f'holdout_auc_1yr_slope{out_suffix}_vs_noslope.csv'
    combined.to_csv(save_path, index=False)
    print(f'\nSaved: {save_path}')

    # --- Summary ---
    print('\n' + '=' * 70)
    print(f'1-YEAR AUC: {slope_label.upper()} vs NO-SLOPE')
    print('=' * 70)

    for horizon in combined['horizon'].unique():
        h = combined[combined['horizon'] == horizon]
        slope_rows = h[h['model'] == slope_label].set_index('disease')
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

    print(f'\nDone. Results in: {save_path}')


if __name__ == '__main__':
    main()
