#!/usr/bin/env python3
"""
Compare nokappa LOO AUC vs nolr and v1 reparam LOO AUC.

Usage:
    python compare_nokappa_auc.py --n_bootstraps 100
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

NOKAPPA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nokappa_loo/'
OLD_CSV = str(Path(__file__).parent / 'nolr_vs_reparam_5batches_auc_LOO.csv')


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

    print("=" * 110)
    print(f"NOKAPPA vs NOLR vs V1-REPARAM LOO AUC -- {N_eval // 1000}K patients, {args.n_bootstraps} bootstraps")
    print("=" * 110)

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

    # Load nokappa LOO pi
    print("\nLoading nokappa LOO predictions...")
    pi_nokappa = pool_batches(NOKAPPA_DIR, args.batch_size, args.n_batches)
    print(f"Nokappa pi: {pi_nokappa.shape}")

    # Load old results for comparison
    if os.path.exists(OLD_CSV):
        old = pd.read_csv(OLD_CSV)
        print(f"Loaded old LOO results: {OLD_CSV}")
    else:
        old = None
        print("No old results found -- will only show nokappa")

    # Evaluate nokappa
    nokappa_results = []
    for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING {horizon.upper()}: NOKAPPA LOO")
        print(f"{'=' * 80}")
        df_nk = evaluate_and_collect(pi_nokappa, Y, E, disease_names, pce_df,
                                     args.n_bootstraps, horizon)
        df_nk['horizon'] = horizon
        nokappa_results.append(df_nk)

    nokappa_df = pd.concat(nokappa_results, ignore_index=True)

    # Save nokappa results
    nk_csv = os.path.join(args.output_dir, 'nokappa_auc_LOO.csv')
    nokappa_df.to_csv(nk_csv, index=False)
    print(f"\nSaved nokappa results: {nk_csv}")

    # 3-way comparison
    print(f"\n{'=' * 110}")
    print("3-WAY COMPARISON: NOLR vs V1-REPARAM vs NOKAPPA (LOO)")
    print(f"{'=' * 110}")

    for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
        nk_h = nokappa_df[nokappa_df['horizon'] == horizon].set_index('disease')

        if old is not None:
            old_h = old[old['horizon'] == horizon].set_index('disease')
            diseases = sorted(set(nk_h.index) & set(old_h.index))
        else:
            diseases = sorted(nk_h.index)

        print(f"\n{'=' * 110}")
        print(f"  {horizon.upper()}")
        print(f"{'=' * 110}")
        header = f"{'DISEASE':<25} {'NOLR LOO':>10} {'V1 REPARAM':>12} {'NOKAPPA':>12} {'NK-nolr':>10} {'NK-v1':>10}"
        print(header)
        print("-" * 110)

        nolr_aucs, v1_aucs, nk_aucs = [], [], []
        nk_wins_nolr, nk_wins_v1 = 0, 0

        for d in diseases:
            nk_a = nk_h.loc[d, 'auc']
            nk_aucs.append(nk_a)

            if old is not None and d in old_h.index:
                nolr = old_h.loc[d, 'nolr_auc']
                v1 = old_h.loc[d, 'reparam_auc']
                nolr_aucs.append(nolr)
                v1_aucs.append(v1)

                delta_nolr = nk_a - nolr
                delta_v1 = nk_a - v1
                if nk_a > nolr:
                    nk_wins_nolr += 1
                if nk_a > v1:
                    nk_wins_v1 += 1

                fn = "+" if delta_nolr > 0 else "-"
                fv = "+" if delta_v1 > 0 else "-"
                print(f"{d:<25} {nolr:>10.3f} {v1:>12.3f} {nk_a:>12.3f}  {fn}{abs(delta_nolr):.3f}      {fv}{abs(delta_v1):.3f}")
            else:
                print(f"{d:<25} {'N/A':>10} {'N/A':>12} {nk_a:>12.3f}")

        n = len(diseases)
        mnk = np.mean(nk_aucs)
        if nolr_aucs and v1_aucs:
            mn = np.mean(nolr_aucs)
            m1 = np.mean(v1_aucs)
            print(f"\n  Mean AUC:     nolr={mn:.4f}  v1={m1:.4f}  nokappa={mnk:.4f}   nk-nolr={mnk-mn:+.4f}  nk-v1={mnk-m1:+.4f}")
            print(f"  NK wins:                                       {nk_wins_nolr}/{n} vs nolr  {nk_wins_v1}/{n} vs v1")
        else:
            print(f"\n  Mean nokappa AUC: {mnk:.4f}")


if __name__ == '__main__':
    main()
