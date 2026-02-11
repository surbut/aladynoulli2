#!/usr/bin/env python3
"""
Compare nolr vs reparam AUC on first 5 batches (50K patients).
Evaluates static 10-year and dynamic 1-year AUC (matching paper metrics).

Usage:
    python compare_nolr_vs_reparam_5batches_auc.py --n_bootstraps 100
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi
from evaluatetdccode import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
)

NOLR_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_vectorized/'
REPARAM_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_vectorized/'


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


def print_comparison(merged, horizon, nolr_col='nolr_auc', reparam_col='reparam_auc'):
    print(f"\n{'='*95}")
    print(f"{horizon.upper()} AUC COMPARISON")
    print(f"{'='*95}")
    print(f"{'DISEASE':<25} {'NOLR AUC (95% CI)':<30} {'REPARAM AUC (95% CI)':<30} {'DELTA':>8}")
    print("-" * 95)
    for _, row in merged.iterrows():
        na = row.get(nolr_col, np.nan)
        ra = row.get(reparam_col, np.nan)
        nc = f"{na:.3f} ({row.get('nolr_ci_lower', np.nan):.3f}-{row.get('nolr_ci_upper', np.nan):.3f})"
        rc = f"{ra:.3f} ({row.get('reparam_ci_lower', np.nan):.3f}-{row.get('reparam_ci_upper', np.nan):.3f})"
        d = ra - na if not (np.isnan(na) or np.isnan(ra)) else np.nan
        ds = f"{d:+.3f}" if not np.isnan(d) else "N/A"
        print(f"{row['disease']:<25} {nc:<30} {rc:<30} {ds:>8}")

    valid = merged.dropna(subset=[nolr_col, reparam_col])
    if len(valid) > 0:
        nm = valid[nolr_col].mean()
        rm = valid[reparam_col].mean()
        nw = (valid[nolr_col] > valid[reparam_col]).sum()
        rw = (valid[reparam_col] > valid[nolr_col]).sum()
        print(f"\nMean AUC -- nolr: {nm:.4f}, reparam: {rm:.4f}, delta: {rm-nm:+.4f}")
        print(f"Nolr wins: {nw}, Reparam wins: {rw}, Tied: {len(valid)-nw-rw}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--n_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    N_eval = args.batch_size * args.n_batches

    print("=" * 80)
    print(f"NOLR vs REPARAM AUC COMPARISON - {N_eval//1000}K patients, {args.n_bootstraps} bootstraps")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
    Y = torch.load(data_dir + 'Y_tensor.pt', weights_only=False)[:N_eval]
    E = torch.load(data_dir + 'E_enrollment_full.pt', weights_only=False)[:N_eval]
    essentials = torch.load(data_dir + 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    pce_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    pce_df = pce_df.iloc[:N_eval].reset_index(drop=True)

    # Ensure sex/age columns
    if 'Sex' not in pce_df.columns and 'sex' in pce_df.columns:
        pce_df['Sex'] = pce_df['sex'].map({0: 'Female', 1: 'Male'}).fillna('Unknown')
    if 'sex' not in pce_df.columns and 'Sex' in pce_df.columns:
        pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).fillna(-1)
    if 'age' not in pce_df.columns and 'Age' in pce_df.columns:
        pce_df['age'] = pce_df['Age']

    print(f"Y: {Y.shape}, E: {E.shape}, pce_df: {len(pce_df)}")

    # Pool pi from batches
    print("\nLoading nolr predictions...")
    pi_nolr = pool_batches(NOLR_DIR, args.batch_size, args.n_batches)
    print(f"Nolr pi: {pi_nolr.shape}")

    print("\nLoading reparam predictions...")
    pi_reparam = pool_batches(REPARAM_DIR, args.batch_size, args.n_batches)
    print(f"Reparam pi: {pi_reparam.shape}")

    all_results = []

    # Static 10-year AUC
    for horizon in ['static_10yr', 'dynamic_1yr']:
        print(f"\n{'='*80}")
        print(f"EVALUATING {horizon.upper()}: NOLR")
        print(f"{'='*80}")
        df_nolr = evaluate_and_collect(pi_nolr, Y, E, disease_names, pce_df, args.n_bootstraps, horizon)

        print(f"\n{'='*80}")
        print(f"EVALUATING {horizon.upper()}: REPARAM")
        print(f"{'='*80}")
        df_reparam = evaluate_and_collect(pi_reparam, Y, E, disease_names, pce_df, args.n_bootstraps, horizon)

        # Merge
        df_nolr = df_nolr.rename(columns={c: f'nolr_{c}' for c in df_nolr.columns if c != 'disease'})
        df_reparam = df_reparam.rename(columns={c: f'reparam_{c}' for c in df_reparam.columns if c != 'disease'})
        merged = df_nolr.merge(df_reparam, on='disease', how='outer')
        merged['horizon'] = horizon

        print_comparison(merged, horizon)
        all_results.append(merged)

    # Save
    combined = pd.concat(all_results, ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'nolr_vs_reparam_5batches_auc.csv')
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
