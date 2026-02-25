#!/usr/bin/env python3
"""
Run slope LOO AUC evaluation restricted to first 100k (same test set as holdout).

Uses existing slope LOO FULL.pt; no new predictions. Calls pool_and_evaluate_slope_1phase_loo
with --eval-only --n_patients 100000 to compute static 10yr, dynamic 10yr, static 1yr, rolling 1yr
on the same 100k as holdout (batches 0-9). Results go to results_slope_1phase_loo/*_100k.csv.

Purpose: Compare slope LOO (pool-39) vs slope holdout (pool-5) on the same 100k. If LOO-on-100k
AUC ≈ holdout (~0.76), the gap to LOO-on-400k (~0.85) is test-set size. If LOO-on-100k ≈ 0.85,
the gap is pooling (5 vs 39).

Usage:
    python run_slope_loo_100k_eval.py
    python run_slope_loo_100k_eval.py --n_bootstraps 100 --force-recompute
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Evaluate slope LOO on first 100k (same as holdout test set)')
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--force-recompute', action='store_true', help='Recompute even if CSVs exist')
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / 'pool_and_evaluate_slope_1phase_loo.py'),
        '--eval-only',
        '--n_patients', '100000',
        '--n_bootstraps', str(args.n_bootstraps),
    ]
    if args.force_recompute:
        cmd.append('--force-recompute')

    print('Running: slope LOO AUC on first 100k (batches 0-9, same as holdout)')
    print(' ', ' '.join(cmd))
    print()
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        sys.exit(result.returncode)
    print()
    print('Done. Results in results_slope_1phase_loo/*_100k.csv')
    print('Compare to holdout: results_holdout_auc/holdout_auc_slope_1phase_vs_noslope.csv (static_10yr, static_1yr)')


if __name__ == '__main__':
    main()
