"""
Compare 10-year static vs dynamic performance across diseases
"""

import pandas as pd
import numpy as np

# Load the comparison file
df = pd.read_csv("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/time_horizons/pooled_retrospective/comparison_all_horizons.csv", index_col=0)
df = df.reset_index()
df = df.rename(columns={'index': 'Disease'})

# Compare static vs dynamic 10yr
df['static_better'] = df['static_10yr_AUC'] > df['10yr_AUC']
df['difference'] = df['static_10yr_AUC'] - df['10yr_AUC']

print("="*80)
print("10-YEAR STATIC vs DYNAMIC COMPARISON")
print("="*80)
print(f"\nTotal diseases: {len(df)}")
print(f"Diseases where static is better: {df['static_better'].sum()} ({100*df['static_better'].sum()/len(df):.1f}%)")
print(f"Diseases where dynamic is better: {(~df['static_better']).sum()} ({100*(~df['static_better']).sum()/len(df):.1f}%)")

print(f"\nMean difference (static - dynamic): {df['difference'].mean():.4f}")
print(f"Median difference: {df['difference'].median():.4f}")

print("\n" + "="*80)
print("DISEASES WHERE STATIC IS BETTER:")
print("="*80)
static_better = df[df['static_better']].sort_values('difference', ascending=False)
for _, row in static_better.iterrows():
    print(f"{row['Disease']:25} Static: {row['static_10yr_AUC']:.4f}  Dynamic: {row['10yr_AUC']:.4f}  Diff: {row['difference']:+.4f}")

print("\n" + "="*80)
print("DISEASES WHERE DYNAMIC IS BETTER:")
print("="*80)
dynamic_better = df[~df['static_better']].sort_values('difference', ascending=True)
for _, row in dynamic_better.iterrows():
    print(f"{row['Disease']:25} Static: {row['static_10yr_AUC']:.4f}  Dynamic: {row['10yr_AUC']:.4f}  Diff: {row['difference']:+.4f}")
