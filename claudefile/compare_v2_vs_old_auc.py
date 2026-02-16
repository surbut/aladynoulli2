#!/usr/bin/env python3
"""Compare v2 reparam LOO AUC vs nolr and v1 reparam LOO AUC."""
import pandas as pd
import numpy as np

old = pd.read_csv('/Users/sarahurbut/aladynoulli2/claudefile/nolr_vs_reparam_5batches_auc_LOO.csv')
v2 = pd.read_csv('/Users/sarahurbut/aladynoulli2/claudefile/reparam_v2_auc_LOO.csv')

for horizon in ['static_10yr', 'dynamic_10yr', 'dynamic_1yr']:
    old_h = old[old['horizon'] == horizon].set_index('disease')
    v2_h = v2[v2['horizon'] == horizon].set_index('disease')
    diseases = sorted(set(old_h.index) & set(v2_h.index))

    print(f"\n{'='*105}")
    print(f"  {horizon.upper()}")
    print(f"{'='*105}")
    header = f"{'DISEASE':<25} {'NOLR LOO':>10} {'v1 REPARAM':>12} {'v2 REPARAM':>12} {'v2-nolr':>10} {'v2-v1':>10}"
    print(header)
    print("-" * 105)

    nolr_aucs, v1_aucs, v2_aucs = [], [], []
    v2_wins_nolr, v2_wins_v1 = 0, 0

    for d in diseases:
        nolr = old_h.loc[d, 'nolr_auc']
        v1 = old_h.loc[d, 'reparam_auc']
        v2a = v2_h.loc[d, 'auc']

        delta_nolr = v2a - nolr
        delta_v1 = v2a - v1

        nolr_aucs.append(nolr)
        v1_aucs.append(v1)
        v2_aucs.append(v2a)
        if v2a > nolr:
            v2_wins_nolr += 1
        if v2a > v1:
            v2_wins_v1 += 1

        fn = "+" if delta_nolr > 0 else "-"
        fv = "+" if delta_v1 > 0 else "-"
        print(f"{d:<25} {nolr:>10.3f} {v1:>12.3f} {v2a:>12.3f}  {fn}{abs(delta_nolr):.3f}      {fv}{abs(delta_v1):.3f}")

    n = len(diseases)
    mn = np.mean(nolr_aucs)
    m1 = np.mean(v1_aucs)
    m2 = np.mean(v2_aucs)
    print(f"\n  Mean AUC:            {mn:.4f}       {m1:.4f}       {m2:.4f}   {m2-mn:+.4f}     {m2-m1:+.4f}")
    print(f"  v2 wins:                                                   {v2_wins_nolr}/{n} vs nolr  {v2_wins_v1}/{n} vs v1")
