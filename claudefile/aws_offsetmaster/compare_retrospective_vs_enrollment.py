#!/usr/bin/env python3
"""
Compare Fixed_Retrospective_Pooled vs Fixed_Enrollment_Pooled performance
"""

import pandas as pd
import sys
import os

# Get the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv10_path = os.path.join(base_dir, 'pyScripts/new_oct_revision/new_notebooks/comparison_all_approaches_10yr.csv')
csv30_path = os.path.join(base_dir, 'pyScripts/new_oct_revision/new_notebooks/comparison_all_approaches_30yr.csv')

# Read both files
df10 = pd.read_csv(csv10_path)
df30 = pd.read_csv(csv30_path)

# Calculate comparisons
df10['Retro_vs_Enroll'] = df10['Fixed_Retrospective_Pooled'] - df10['Fixed_Enrollment_Pooled']
df30['Retro_vs_Enroll'] = df30['Fixed_Retrospective_Pooled'] - df30['Fixed_Enrollment_Pooled']

print('='*70)
print('10-YEAR PREDICTIONS: Fixed_Retrospective_Pooled vs Fixed_Enrollment_Pooled')
print('='*70)
retro_better_10 = (df10['Retro_vs_Enroll'] > 0).sum()
enroll_better_10 = (df10['Retro_vs_Enroll'] < 0).sum()
total_diseases = len(df10) - 1  # Subtract 1 for header row if needed

print(f"Retrospective better: {retro_better_10} / {total_diseases} diseases")
print(f"Enrollment better: {enroll_better_10} / {total_diseases} diseases")
print(f"Ties (difference < 0.001): {(abs(df10['Retro_vs_Enroll']) < 0.001).sum()} diseases")
print(f"Mean difference: {df10['Retro_vs_Enroll'].mean():.4f}")
print(f"Median difference: {df10['Retro_vs_Enroll'].median():.4f}")
print(f"Max advantage (Retrospective): {df10['Retro_vs_Enroll'].max():.4f}")
print(f"Max advantage (Enrollment): {abs(df10['Retro_vs_Enroll'].min()):.4f}")
print()

print('Top 5 where Retrospective is MUCH better (>0.05 difference):')
df10_retro_adv = df10[df10['Retro_vs_Enroll'] > 0.05].nlargest(5, 'Retro_vs_Enroll')
if len(df10_retro_adv) > 0:
    for idx, row in df10_retro_adv.iterrows():
        print(f"  {row.iloc[0]}: Retro={row['Fixed_Retrospective_Pooled']:.4f}, Enroll={row['Fixed_Enrollment_Pooled']:.4f}, Diff={row['Retro_vs_Enroll']:.4f}")
else:
    print("  None with >0.05 difference")
print()

print('Top 5 where Enrollment is MUCH better (>0.05 difference):')
df10_enroll_adv = df10[df10['Retro_vs_Enroll'] < -0.05].nsmallest(5, 'Retro_vs_Enroll')
if len(df10_enroll_adv) > 0:
    for idx, row in df10_enroll_adv.iterrows():
        print(f"  {row.iloc[0]}: Retro={row['Fixed_Retrospective_Pooled']:.4f}, Enroll={row['Fixed_Enrollment_Pooled']:.4f}, Diff={row['Retro_vs_Enroll']:.4f}")
else:
    print("  None with >0.05 difference")
print()

print('='*70)
print('30-YEAR PREDICTIONS: Fixed_Retrospective_Pooled vs Fixed_Enrollment_Pooled')
print('='*70)
retro_better_30 = (df30['Retro_vs_Enroll'] > 0).sum()
enroll_better_30 = (df30['Retro_vs_Enroll'] < 0).sum()

print(f"Retrospective better: {retro_better_30} / {len(df30)-1} diseases")
print(f"Enrollment better: {enroll_better_30} / {len(df30)-1} diseases")
print(f"Ties (difference < 0.001): {(abs(df30['Retro_vs_Enroll']) < 0.001).sum()} diseases")
print(f"Mean difference: {df30['Retro_vs_Enroll'].mean():.4f}")
print(f"Median difference: {df30['Retro_vs_Enroll'].median():.4f}")
print(f"Max advantage (Retrospective): {df30['Retro_vs_Enroll'].max():.4f}")
print(f"Max advantage (Enrollment): {abs(df30['Retro_vs_Enroll'].min()):.4f}")
print()

print('Top 5 where Retrospective is MUCH better (>0.05 difference):')
df30_retro_adv = df30[df30['Retro_vs_Enroll'] > 0.05].nlargest(5, 'Retro_vs_Enroll')
if len(df30_retro_adv) > 0:
    for idx, row in df30_retro_adv.iterrows():
        print(f"  {row.iloc[0]}: Retro={row['Fixed_Retrospective_Pooled']:.4f}, Enroll={row['Fixed_Enrollment_Pooled']:.4f}, Diff={row['Retro_vs_Enroll']:.4f}")
else:
    print("  None with >0.05 difference")
print()

print('Top 5 where Enrollment is MUCH better (>0.05 difference):')
df30_enroll_adv = df30[df30['Retro_vs_Enroll'] < -0.05].nsmallest(5, 'Retro_vs_Enroll')
if len(df30_enroll_adv) > 0:
    for idx, row in df30_enroll_adv.iterrows():
        print(f"  {row.iloc[0]}: Retro={row['Fixed_Retrospective_Pooled']:.4f}, Enroll={row['Fixed_Enrollment_Pooled']:.4f}, Diff={row['Retro_vs_Enroll']:.4f}")
else:
    print("  None with >0.05 difference")
print()

print('='*70)
print('RECOMMENDATION')
print('='*70)

# Count meaningful differences (>0.01)
meaningful_retro_10 = (df10['Retro_vs_Enroll'] > 0.01).sum()
meaningful_enroll_10 = (df10['Retro_vs_Enroll'] < -0.01).sum()

meaningful_retro_30 = (df30['Retro_vs_Enroll'] > 0.01).sum()
meaningful_enroll_30 = (df30['Retro_vs_Enroll'] < -0.01).sum()

print(f"10-year: Retrospective better by >0.01 in {meaningful_retro_10} diseases")
print(f"10-year: Enrollment better by >0.01 in {meaningful_enroll_10} diseases")
print(f"30-year: Retrospective better by >0.01 in {meaningful_retro_30} diseases")
print(f"30-year: Enrollment better by >0.01 in {meaningful_enroll_30} diseases")
print()

if meaningful_retro_10 > meaningful_enroll_10 and meaningful_retro_30 > meaningful_enroll_30:
    print("RETROSPECTIVE appears consistently better")
    print("BUT: Enrollment analysis still valuable for:")
    print("  - Real-world applicability (using enrollment-age data only)")
    print("  - Demonstrating model works with limited data")
    print("  - Comparison/validation purposes")
elif meaningful_enroll_10 > meaningful_retro_10 and meaningful_enroll_30 > meaningful_retro_30:
    print("ENROLLMENT appears consistently better - definitely continue!")
else:
    print("Mixed results - both analyses provide value")
    print("Consider continuing enrollment analysis for completeness")




