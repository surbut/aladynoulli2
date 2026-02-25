#!/usr/bin/env python3
"""
Re-run no-slope LOO static 10-year AUC evaluation (same as nokappa_v3_auc_evaluation.ipynb).
Overwrites results_feb18/static_10yr_results_400k.csv so no-slope and slope LOO use the same code path.

Usage:
    python rerun_noslope_static10yr_feb18.py
    python rerun_noslope_static10yr_feb18.py --n_bootstraps 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/Users/sarahurbut/aladynoulli2/pyScripts/')
sys.path.insert(0, '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
PI_PATH = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_nokappa_v3_loo_all40/pi_enroll_fixedphi_sex_FULL.pt')
PCE_PATH = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_feb18')
N_FULL = 400_000


def main():
    parser = argparse.ArgumentParser(description='Re-run no-slope LOO static 10yr → results_feb18')
    parser.add_argument('--n_bootstraps', type=int, default=100)
    args = parser.parse_args()

    print('Loading no-slope LOO pi and data...')
    pi_full = torch.load(PI_PATH, weights_only=False)[:N_FULL]
    Y_full = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)[:N_FULL]
    E_full = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)[:N_FULL]
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    pce_df_full = pd.read_csv(PCE_PATH).iloc[:N_FULL].reset_index(drop=True)

    if 'prevent_impute' not in pce_df_full.columns and 'prevent_base_ascvd_risk' in pce_df_full.columns:
        pce_df_full['prevent_impute'] = pce_df_full['prevent_base_ascvd_risk'].fillna(
            pce_df_full['prevent_base_ascvd_risk'].mean())

    print(f'  pi: {pi_full.shape}, Y: {Y_full.shape}, E: {E_full.shape}, pce: {len(pce_df_full)} rows')
    print('Computing static 10-year AUCs (follow_up_duration_years=10)...')

    results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_full, Y_100k=Y_full, E_100k=E_full,
        disease_names=disease_names, pce_df=pce_df_full,
        n_bootstraps=args.n_bootstraps, follow_up_duration_years=10)

    static10_df = pd.DataFrame({
        'Disease': list(results.keys()),
        'AUC': [r['auc'] for r in results.values()],
        'CI_lower': [r.get('ci_lower', np.nan) for r in results.values()],
        'CI_upper': [r.get('ci_upper', np.nan) for r in results.values()],
        'N_Events': [r.get('n_events', np.nan) for r in results.values()],
        'Event_Rate': [r.get('event_rate', np.nan) for r in results.values()],
    }).set_index('Disease').sort_values('AUC', ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'static_10yr_results_400k.csv'
    static10_df.to_csv(out_path)
    print(f'Saved: {out_path}')
    print('Top 5:', static10_df.head().to_string())


if __name__ == '__main__':
    main()
