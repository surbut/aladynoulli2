#!/usr/bin/env python3
"""Compare 1-year static AUC for no-slope across: Holdout (100k), 1phase-run (100k), LOO (400k)."""
import pandas as pd
from pathlib import Path

RESULTS = Path('results_holdout_auc')
FEB18 = Path('results_feb18')

# Holdout no-slope 1yr (100k)
h = pd.read_csv(RESULTS / 'holdout_auc_1yr_slope_vs_noslope.csv')
holdout = h[(h['model'] == 'noslope') & (h['horizon'] == 'static_1yr')].set_index('disease')['auc']

# 1phase-run no-slope 1yr (100k)
h1 = pd.read_csv(RESULTS / 'holdout_auc_1yr_slope_1phase_vs_noslope.csv')
onephase = h1[(h1['model'] == 'noslope') & (h1['horizon'] == 'static_1yr')].set_index('disease')['auc']

# LOO no-slope 1yr (400k) from results_feb18
summary = pd.read_csv(FEB18 / 'summary_all_metrics.csv')
loo = summary.set_index('Disease')['Static_1yr_400k']

# Align disease names (same in all)
common = holdout.index.intersection(onephase.index).intersection(loo.index)
common = sorted(common)

df = pd.DataFrame({
    'Disease': common,
    'noslope_holdout_100k': [holdout[d] for d in common],
    'noslope_1phase_100k': [onephase[d] for d in common],
    'noslope_LOO_400k': [loo[d] for d in common],
})

# Summary stats (holdout vs 1phase are same N; LOO is different N so just report)
r_hold_1ph = df['noslope_holdout_100k'].corr(df['noslope_1phase_100k'])
r_hold_loo = df['noslope_holdout_100k'].corr(df['noslope_LOO_400k'])
r_1ph_loo = df['noslope_1phase_100k'].corr(df['noslope_LOO_400k'])
mean_abs_hold_1ph = (df['noslope_holdout_100k'] - df['noslope_1phase_100k']).abs().mean()
mean_abs_hold_loo = (df['noslope_holdout_100k'] - df['noslope_LOO_400k']).abs().mean()
mean_abs_1ph_loo = (df['noslope_1phase_100k'] - df['noslope_LOO_400k']).abs().mean()

out_path = RESULTS / 'compare_1yr_noslope_three_approaches.csv'
df.to_csv(out_path, index=False)
print('Saved:', out_path)
print()
print('1-YEAR NO-SLOPE AUC: Three approaches')
print('  Holdout 100k:  pool 10-39, test 0-9')
print('  1phase  100k:  pool 10-14, test 0-9')
print('  LOO     400k:  pool 39 per batch, all 40 batches')
print()
print(df.to_string(index=False))
print()
print('Correlation (across diseases):')
print('  Holdout vs 1phase:  r = {:.4f}  (both 100k)'.format(r_hold_1ph))
print('  Holdout vs LOO:     r = {:.4f}  (100k vs 400k)'.format(r_hold_loo))
print('  1phase vs LOO:      r = {:.4f}  (100k vs 400k)'.format(r_1ph_loo))
print()
print('Mean |AUC difference|:')
print('  Holdout vs 1phase:  {:.4f}'.format(mean_abs_hold_1ph))
print('  Holdout vs LOO:     {:.4f}'.format(mean_abs_hold_loo))
print('  1phase vs LOO:      {:.4f}'.format(mean_abs_1ph_loo))
