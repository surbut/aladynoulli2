#!/usr/bin/env python3
"""
Compare LOO nolr vs reparam AUC, and compare against non-LOO results.
Reuses evaluation functions from the main comparison script.

Usage:
    python compare_loo_auc.py --n_bootstraps 100
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
)
from evaluatetdccode import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
)

LOO_NOLR_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_loo/'
LOO_REPARAM_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_loo/'
ORIG_CSV = str(Path(__file__).parent / 'nolr_vs_reparam_5batches_auc.csv')


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


def print_comparison(merged, horizon, suffix=''):
    print(f"\n{'=' * 100}")
    print(f"{horizon.upper()} AUC — LOO{suffix}")
    print(f"{'=' * 100}")
    print(f"{'DISEASE':<25} {'NOLR_LOO (95% CI)':<30} {'REPARAM_LOO (95% CI)':<30} {'DELTA':>8}")
    print("-" * 100)
    for _, row in merged.iterrows():
        na = row.get('nolr_auc', np.nan)
        ra = row.get('reparam_auc', np.nan)
        nc = f"{na:.3f} ({row.get('nolr_ci_lower', np.nan):.3f}-{row.get('nolr_ci_upper', np.nan):.3f})"
        rc = f"{ra:.3f} ({row.get('reparam_ci_lower', np.nan):.3f}-{row.get('reparam_ci_upper', np.nan):.3f})"
        d = ra - na if not (np.isnan(na) or np.isnan(ra)) else np.nan
        ds = f"{d:+.3f}" if not np.isnan(d) else "N/A"
        print(f"{row['disease']:<25} {nc:<30} {rc:<30} {ds:>8}")

    valid = merged.dropna(subset=['nolr_auc', 'reparam_auc'])
    if len(valid) > 0:
        nm = valid['nolr_auc'].mean()
        rm = valid['reparam_auc'].mean()
        nw = (valid['nolr_auc'] > valid['reparam_auc']).sum()
        rw = (valid['reparam_auc'] > valid['nolr_auc']).sum()
        print(f"\nMean AUC — nolr_loo: {nm:.4f}, reparam_loo: {rm:.4f}, delta: {rm - nm:+.4f}")
        print(f"Nolr wins: {nw}, Reparam wins: {rw}, Tied: {len(valid) - nw - rw}")


def compare_with_original(loo_df, horizon):
    """Compare LOO results with original (non-LOO) results."""
    if not os.path.exists(ORIG_CSV):
        print("  (No original results to compare against)")
        return

    orig = pd.read_csv(ORIG_CSV)
    orig_h = orig[orig['horizon'] == horizon].copy()

    loo_h = loo_df[loo_df['horizon'] == horizon].copy()

    merged = loo_h.merge(orig_h[['disease', 'nolr_auc', 'reparam_auc']],
                         on='disease', suffixes=('_loo', '_orig'))

    if len(merged) == 0:
        return

    print(f"\n  LOO vs ORIGINAL comparison ({horizon}):")
    print(f"  {'DISEASE':<25} {'NOLR orig→LOO':<25} {'REPARAM orig→LOO':<25}")
    print(f"  {'-' * 75}")
    for _, r in merged.iterrows():
        nd = r['nolr_auc_loo'] - r['nolr_auc_orig']
        rd = r['reparam_auc_loo'] - r['reparam_auc_orig']
        print(f"  {r['disease']:<25} {r['nolr_auc_orig']:.3f}→{r['nolr_auc_loo']:.3f} ({nd:+.3f})   "
              f"{r['reparam_auc_orig']:.3f}→{r['reparam_auc_loo']:.3f} ({rd:+.3f})")

    nolr_mean_delta = (merged['nolr_auc_loo'] - merged['nolr_auc_orig']).mean()
    reparam_mean_delta = (merged['reparam_auc_loo'] - merged['reparam_auc_orig']).mean()
    print(f"\n  Mean AUC change (LOO - original): nolr {nolr_mean_delta:+.4f}, reparam {reparam_mean_delta:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--n_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    N_eval = args.batch_size * args.n_batches

    print("=" * 80)
    print(f"LOO NOLR vs REPARAM AUC COMPARISON — {N_eval // 1000}K patients, {args.n_bootstraps} bootstraps")
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

    # Load LOO pi
    print("\nLoading LOO nolr predictions...")
    pi_nolr = pool_batches(LOO_NOLR_DIR, args.batch_size, args.n_batches)
    print(f"LOO nolr pi: {pi_nolr.shape}")

    print("\nLoading LOO reparam predictions...")
    pi_reparam = pool_batches(LOO_REPARAM_DIR, args.batch_size, args.n_batches)
    print(f"LOO reparam pi: {pi_reparam.shape}")

    all_results = []

    for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING {horizon.upper()}: LOO NOLR")
        print(f"{'=' * 80}")
        df_nolr = evaluate_and_collect(pi_nolr, Y, E, disease_names, pce_df, args.n_bootstraps, horizon)

        print(f"\n{'=' * 80}")
        print(f"EVALUATING {horizon.upper()}: LOO REPARAM")
        print(f"{'=' * 80}")
        df_reparam = evaluate_and_collect(pi_reparam, Y, E, disease_names, pce_df, args.n_bootstraps, horizon)

        df_nolr = df_nolr.rename(columns={c: f'nolr_{c}' for c in df_nolr.columns if c != 'disease'})
        df_reparam = df_reparam.rename(columns={c: f'reparam_{c}' for c in df_reparam.columns if c != 'disease'})
        merged = df_nolr.merge(df_reparam, on='disease', how='outer')
        merged['horizon'] = horizon

        print_comparison(merged, horizon)
        all_results.append(merged)

    # Save LOO results
    combined = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(args.output_dir, 'nolr_vs_reparam_5batches_auc_LOO.csv')
    combined.to_csv(out_path, index=False)
    print(f"\nSaved LOO results to: {out_path}")

    # Compare with original
    print(f"\n{'=' * 80}")
    print("LOO vs ORIGINAL COMPARISON")
    print(f"{'=' * 80}")
    for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
        compare_with_original(combined, horizon)


if __name__ == '__main__':
    main()
