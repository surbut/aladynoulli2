#!/usr/bin/env python3
"""
Sanity check: slope LOO 100k vs slope holdout 100k — same test set, same evaluation.

Pool 5/10/30 gave slope ≈ no-slope (~0.76). LOO gave slope ~0.85. So something is off.
This script:
  1. Loads slope LOO pi (first 100k from FULL.pt) and slope holdout pi (100k).
  2. Verifies same shape, same Y/E (first 100k).
  3. Reports per-disease correlation(pi_LOO, pi_holdout) and mean risk.
  4. Recomputes static_10yr AUC for both on the same 100k with same n_bootstraps.

If LOO and holdout use the same (Y, E) and same eval function but LOO AUC >> holdout AUC,
then the pi values must differ. This script quantifies how much and confirms no eval bug.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results_holdout_auc'
DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
PCE_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv'

# Paths (adjust if your FULL.pt or holdout pi live elsewhere)
PI_LOO_FULL = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_slope_1phase_loo_all40/pi_enroll_fixedphi_sex_FULL.pt')
PI_HOLDOUT_POOL30 = RESULTS_DIR / 'pi_slope_holdout_1phase_pool30.pt'
PI_HOLDOUT_POOL5 = RESULTS_DIR / 'pi_slope_holdout.pt'

N_100K = 100_000


def main():
    # fig5utils lives in pyScripts (same as slope_holdout_auc.py)
    sys.path.insert(0, str(SCRIPT_DIR.parent / 'pyScripts'))
    from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi

    print('=' * 60)
    print('Sanity check: Slope LOO 100k vs Slope Holdout 100k')
    print('=' * 60)

    # Load pi
    if not PI_LOO_FULL.exists():
        print(f'Missing LOO FULL: {PI_LOO_FULL}')
        return
    pi_loo_full = torch.load(PI_LOO_FULL, weights_only=False)
    pi_loo = pi_loo_full[:N_100K]
    print(f'Slope LOO pi (first 100k): {pi_loo.shape}')

    pi_holdout_path = PI_HOLDOUT_POOL30 if PI_HOLDOUT_POOL30.exists() else PI_HOLDOUT_POOL5
    if not pi_holdout_path.exists():
        print(f'Missing holdout pi: {PI_HOLDOUT_POOL30} or {PI_HOLDOUT_POOL5}')
        return
    pi_holdout = torch.load(pi_holdout_path, weights_only=False)
    print(f'Slope holdout pi: {pi_holdout.shape} (from {pi_holdout_path.name})')

    if pi_holdout.shape[0] != N_100K or pi_loo.shape[0] != N_100K:
        print(f'ERROR: shape mismatch. LOO {pi_loo.shape[0]}, holdout {pi_holdout.shape[0]}, expected {N_100K}')
        return
    if pi_loo.shape != pi_holdout.shape:
        print(f'WARNING: pi shapes differ {pi_loo.shape} vs {pi_holdout.shape}')

    # Load Y, E (same for both)
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)[:N_100K]
    E = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)[:N_100K]
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    pce_df = pd.read_csv(PCE_PATH).iloc[:N_100K].reset_index(drop=True)
    print(f'Y: {Y.shape}, E: {E.shape}')

    # Per-disease: correlation(pi_LOO, pi_holdout) and mean risk
    # pi is (N, D, T); static 10yr uses 1-year score at enrollment -> 10yr outcome; fig5utils uses specific indices
    # Simplified: use first time point risk for each disease (d) as proxy: pi[:, d, 0] or time-avg
    D = pi_loo.shape[1]
    print('\n--- Per-disease: corr(pi_LOO, pi_holdout) and mean risk (first time) ---')
    cors = []
    for d in range(min(20, D)):  # first 20 diseases
        n = disease_names[d] if d < len(disease_names) else f'D{d}'
        p_loo = pi_loo[:, d, 0].numpy().ravel()
        p_ho = pi_holdout[:, d, 0].numpy().ravel()
        r = np.corrcoef(p_loo, p_ho)[0, 1] if np.std(p_loo) > 0 and np.std(p_ho) > 0 else np.nan
        cors.append(r)
        print(f'  {n}: corr={r:.4f}, mean_LOO={p_loo.mean():.4f}, mean_holdout={p_ho.mean():.4f}')
    print(f'  Mean correlation (first 20): {np.nanmean(cors):.4f}')

    # Recompute AUC for both with same settings
    print('\n--- Recomputing static_10yr AUC on same 100k (same eval function) ---')
    res_loo = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_loo, Y_100k=Y, E_100k=E, disease_names=disease_names, pce_df=pce_df,
        n_bootstraps=50, follow_up_duration_years=10)
    res_holdout = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_holdout, Y_100k=Y, E_100k=E, disease_names=disease_names, pce_df=pce_df,
        n_bootstraps=50, follow_up_duration_years=10)

    print('  Disease          LOO_AUC   Holdout_AUC   Diff')
    for d in ['ASCVD', 'Heart_Failure', 'Atrial_Fib', 'COPD', 'Diabetes']:
        a_loo = res_loo.get(d, {}).get('auc', np.nan)
        a_ho = res_holdout.get(d, {}).get('auc', np.nan)
        print(f'  {d:<18} {a_loo:.4f}   {a_ho:.4f}        {a_loo - a_ho:+.4f}')
    ascvd_loo = res_loo.get('ASCVD', {}).get('auc', np.nan)
    ascvd_ho = res_holdout.get('ASCVD', {}).get('auc', np.nan)
    print(f'\n  → Same data, same eval: LOO ASCVD={ascvd_loo:.4f}, Holdout ASCVD={ascvd_ho:.4f}')
    if ascvd_loo > ascvd_ho + 0.05:
        print('  → LOO is still much higher → not an evaluation bug; pi really differs.')
    print('=' * 60)


if __name__ == '__main__':
    main()
