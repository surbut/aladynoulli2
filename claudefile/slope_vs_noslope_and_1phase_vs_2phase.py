#!/usr/bin/env python3
"""
1) Establish slope > no-slope in the two 1:1 comparisons (holdout and 1phase).
2) Compare 1-phase slope vs 2-phase slope on the same 100k test set.

Uses existing AUC CSVs in results_holdout_auc/.
"""
import pandas as pd
from pathlib import Path

RESULTS = Path('results_holdout_auc')


def load_auc_wide(csv_path, model_slope, horizon):
    """Load AUC for one model and horizon; return series index by disease."""
    df = pd.read_csv(csv_path)
    s = df[(df['model'] == model_slope) & (df['horizon'] == horizon)].set_index('disease')['auc']
    return s


def load_auc_with_ci(csv_path, model_slope, horizon):
    """Load AUC and CI columns for slope model."""
    df = pd.read_csv(csv_path)
    sub = df[(df['model'] == model_slope) & (df['horizon'] == horizon)].set_index('disease')
    return sub[['auc', 'ci_lower', 'ci_upper']]


def main():
    print('=' * 70)
    print('1) SLOPE vs NO-SLOPE (two 1:1 comparisons on 100k test 0-9)')
    print('=' * 70)

    # --- Holdout (2-phase): train 10-39, test 0-9 ---
    h1 = pd.read_csv(RESULTS / 'holdout_auc_1yr_slope_vs_noslope.csv')
    h10 = pd.read_csv(RESULTS / 'holdout_auc_slope_vs_noslope.csv')
    for horizon, h in [('static_1yr', h1), ('static_10yr', h10)]:
        s_slope = h[(h['model'] == 'slope') & (h['horizon'] == horizon)].set_index('disease')['auc']
        s_noslope = h[(h['model'] == 'noslope') & (h['horizon'] == horizon)].set_index('disease')['auc']
        common = s_slope.index.intersection(s_noslope.index)
        s_slope, s_noslope = s_slope.reindex(common).dropna(), s_noslope.reindex(common).dropna()
        common = s_slope.index.intersection(s_noslope.index)
        diff = s_slope - s_noslope
        n_wins = (diff > 0).sum()
        n_diseases = len(common)
        mean_diff = diff.mean()
        print(f'\nHoldout (2-phase) {horizon}:')
        print(f'  Diseases: {n_diseases}  |  Slope wins: {n_wins}/{n_diseases}  |  Mean ΔAUC (slope−noslope): {mean_diff:.4f}')
        if (diff <= 0).any():
            losses = diff[diff <= 0].sort_values()
            print(f'  Diseases where no-slope ≥ slope: {list(losses.index)}')

    # --- 1-phase: train 10-14, test 0-9 ---
    p1 = pd.read_csv(RESULTS / 'holdout_auc_1yr_slope_1phase_vs_noslope.csv')
    p10 = pd.read_csv(RESULTS / 'holdout_auc_slope_1phase_vs_noslope.csv')
    for horizon, h in [('static_1yr', p1), ('static_10yr', p10)]:
        s_slope = h[(h['model'] == 'slope_1phase') & (h['horizon'] == horizon)].set_index('disease')['auc']
        s_noslope = h[(h['model'] == 'noslope') & (h['horizon'] == horizon)].set_index('disease')['auc']
        common = s_slope.index.intersection(s_noslope.index)
        s_slope, s_noslope = s_slope.reindex(common).dropna(), s_noslope.reindex(common).dropna()
        common = s_slope.index.intersection(s_noslope.index)
        diff = s_slope - s_noslope
        n_wins = (diff > 0).sum()
        n_diseases = len(common)
        mean_diff = diff.mean()
        print(f'\n1-phase {horizon}:')
        print(f'  Diseases: {n_diseases}  |  Slope wins: {n_wins}/{n_diseases}  |  Mean ΔAUC (slope−noslope): {mean_diff:.4f}')
        if (diff <= 0).any():
            losses = diff[diff <= 0].sort_values()
            print(f'  Diseases where no-slope ≥ slope: {list(losses.index)}')

    print('\n' + '=' * 70)
    print('2) 1-PHASE SLOPE vs 2-PHASE SLOPE (same 100k test 0-9)')
    print('=' * 70)

    # Both evaluated on same 100k; 2-phase from holdout, 1-phase from 1phase run
    slope2_1yr = load_auc_wide(RESULTS / 'holdout_auc_1yr_slope_vs_noslope.csv', 'slope', 'static_1yr')
    slope2_10yr = load_auc_wide(RESULTS / 'holdout_auc_slope_vs_noslope.csv', 'slope', 'static_10yr')
    slope1_1yr = load_auc_wide(RESULTS / 'holdout_auc_1yr_slope_1phase_vs_noslope.csv', 'slope_1phase', 'static_1yr')
    slope1_10yr = load_auc_wide(RESULTS / 'holdout_auc_slope_1phase_vs_noslope.csv', 'slope_1phase', 'static_10yr')

    for label, s2, s1 in [('static_1yr', slope2_1yr, slope1_1yr), ('static_10yr', slope2_10yr, slope1_10yr)]:
        common = s2.index.intersection(s1.index).dropna(how='all')
        a2 = s2.reindex(common).dropna()
        a1 = s1.reindex(common).dropna()
        common = a2.index.intersection(a1.index)
        a2, a1 = a2.loc[common], a1.loc[common]
        diff = a1 - a2  # positive = 1-phase better
        r = a2.corr(a1)
        mean_diff = diff.mean()
        n_1phase_wins = (diff > 0).sum()
        n = len(common)
        print(f'\n{label}:  n={n} diseases')
        print(f'  Correlation (2-phase vs 1-phase AUC): r = {r:.4f}')
        print(f'  Mean ΔAUC (1-phase − 2-phase): {mean_diff:.4f}')
        print(f'  1-phase wins: {n_1phase_wins}/{n}  |  2-phase wins: {(diff < 0).sum()}/{n}  |  ties: {(diff == 0).sum()}')
        if abs(mean_diff) >= 0.001:
            better = '1-phase' if mean_diff > 0 else '2-phase'
            print(f'  On average {better} slope is slightly better on this 100k test.')

    # Save 1-phase vs 2-phase comparison table
    common = slope2_1yr.index.intersection(slope1_1yr.index).intersection(slope2_10yr.index).intersection(slope1_10yr.index)
    common = sorted(common)
    df = pd.DataFrame({
        'disease': common,
        'slope_2phase_1yr': [slope2_1yr.loc[d] for d in common],
        'slope_1phase_1yr': [slope1_1yr.loc[d] for d in common],
        'slope_2phase_10yr': [slope2_10yr.loc[d] for d in common],
        'slope_1phase_10yr': [slope1_10yr.loc[d] for d in common],
    })
    df['diff_1yr'] = df['slope_1phase_1yr'] - df['slope_2phase_1yr']
    df['diff_10yr'] = df['slope_1phase_10yr'] - df['slope_2phase_10yr']
    out_path = RESULTS / 'slope_1phase_vs_2phase_100k.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
