#!/usr/bin/env python3
"""
Evaluate AUC for reparam v2 LOO predictions.
3 metrics: static 10yr, dynamic 10yr, dynamic 1yr.

Usage:
    PYTHONUNBUFFERED=1 python claudefile/evaluate_v2_auc.py --n_bootstraps 100
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
)
from evaluatetdccode import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
)

V2_LOO_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_v2_loo/'


def pool_batches(config_dir, batch_size=10000, n_batches=5):
    pi_batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        f = Path(config_dir) / f'pi_enroll_fixedphi_sex_{start}_{end}.pt'
        if not f.exists():
            print(f"  Missing: {f.name}")
            continue
        pi = torch.load(f, map_location='cpu', weights_only=False)
        pi_batches.append(pi)
        print(f"  Loaded {f.name}: {pi.shape}")
    return torch.cat(pi_batches, dim=0)


def evaluate_and_collect(pi, Y, E, disease_names, pce_df, n_bootstraps, horizon):
    if horizon == 'static_10yr':
        results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi, Y_100k=Y, E_100k=E, disease_names=disease_names,
            pce_df=pce_df, n_bootstraps=n_bootstraps, follow_up_duration_years=10)
    elif horizon == 'dynamic_10yr':
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi, Y_100k=Y, E_100k=E, disease_names=disease_names,
            pce_df=pce_df, n_bootstraps=n_bootstraps, follow_up_duration_years=10)
    elif horizon == 'dynamic_1yr':
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
            pi=pi, Y_100k=Y, E_100k=E, disease_names=disease_names,
            pce_df=pce_df, n_bootstraps=n_bootstraps)
    rows = []
    for disease, metrics in results.items():
        rows.append({
            'disease': disease,
            'auc': metrics.get('auc', np.nan),
            'ci_lower': metrics.get('ci_lower', np.nan),
            'ci_upper': metrics.get('ci_upper', np.nan),
            'n_events': metrics.get('n_events', 0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--n_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    N_eval = args.batch_size * args.n_batches

    print("=" * 80)
    print(f"REPARAM v2 AUC EVALUATION — {N_eval // 1000}K patients, {args.n_bootstraps} bootstraps")
    print("=" * 80)

    # Load data
    data_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
    Y = torch.load(data_dir + 'Y_tensor.pt', weights_only=False)[:N_eval]
    E = torch.load(data_dir + 'E_enrollment_full.pt', weights_only=False)[:N_eval]
    essentials = torch.load(data_dir + 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    pce_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    pce_df = pce_df.iloc[:N_eval].reset_index(drop=True)

    if 'Sex' not in pce_df.columns and 'sex' in pce_df.columns:
        pce_df['Sex'] = pce_df['sex'].map({0: 'Female', 1: 'Male'}).fillna('Unknown')
    if 'sex' not in pce_df.columns and 'Sex' in pce_df.columns:
        pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).fillna(-1)
    if 'age' not in pce_df.columns and 'Age' in pce_df.columns:
        pce_df['age'] = pce_df['Age']

    print(f"Y: {Y.shape}, E: {E.shape}, pce_df: {len(pce_df)}")

    # Load v2 LOO predictions
    print("\nLoading v2 LOO predictions...")
    pi_v2 = pool_batches(V2_LOO_DIR, args.batch_size, args.n_batches)
    print(f"v2 pi: {pi_v2.shape}")

    all_results = []

    for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING {horizon.upper()}")
        print(f"{'=' * 80}")
        df = evaluate_and_collect(pi_v2, Y, E, disease_names, pce_df, args.n_bootstraps, horizon)
        df['horizon'] = horizon

        # Print results
        print(f"\n{'DISEASE':<25} {'AUC (95% CI)':<30} {'N_EVENTS':>10}")
        print("-" * 70)
        for _, row in df.iterrows():
            ci = f"{row['auc']:.3f} ({row['ci_lower']:.3f}-{row['ci_upper']:.3f})"
            print(f"{row['disease']:<25} {ci:<30} {int(row['n_events']):>10}")

        valid = df.dropna(subset=['auc'])
        print(f"\nMean AUC: {valid['auc'].mean():.4f} ({len(valid)} diseases)")

        all_results.append(df)

    # Save
    combined = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(args.output_dir, 'reparam_v2_auc_LOO.csv')
    combined.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
        h = combined[combined['horizon'] == horizon]
        print(f"  {horizon:<15}: mean AUC = {h['auc'].mean():.4f}")


if __name__ == '__main__':
    main()
