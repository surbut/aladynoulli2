#!/usr/bin/env python3
"""
Compare Aladynoulli 1-year predictions with Delphi 1-year predictions.
Compares 0-year and 1-year washout periods, matching the style of the old comparison.

Usage in notebook:
    %run compare_delphi_1yr_import.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. LOAD ALADYNOULLI RESULTS
# =============================================================================

print("="*100)
print("ALADYNOULLI vs DELPHI: 1-YEAR PREDICTION COMPARISON")
print("="*100)

print("\nLoading Aladynoulli results...")

# Load washout results (1-year predictions with 0-year and 1-year washout)
results_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/washout/pooled_retrospective')

washout_0yr = pd.read_csv(results_dir / 'washout_0yr_results.csv')
washout_1yr = pd.read_csv(results_dir / 'washout_1yr_results.csv')

# Create Aladynoulli results DataFrames
aladynoulli_0gap = washout_0yr[['Disease', 'AUC']].copy()
aladynoulli_0gap.columns = ['Disease', 'Aladynoulli_1yr_0gap']

aladynoulli_1gap = washout_1yr[['Disease', 'AUC']].copy()
aladynoulli_1gap.columns = ['Disease', 'Aladynoulli_1yr_1gap']

# Merge
aladynoulli_all = aladynoulli_0gap.merge(aladynoulli_1gap, on='Disease', how='outer')

print(f"Loaded Aladynoulli results for {len(aladynoulli_all)} diseases")

# =============================================================================
# 2. EXTRACT DELPHI RESULTS FROM SUPPLEMENTARY TABLE
# =============================================================================

print("\nExtracting Delphi results from supplementary table...")

# Load Delphi supplementary table
delphi_supp = pd.read_csv('/Users/sarahurbut/Downloads/41586_2025_9529_MOESM3_ESM.csv')

# Define disease category to ICD-10 code mappings (from notebook)
disease_icd_mapping = {
    'ASCVD': ['I21', 'I25'],  # Myocardial infarction, Coronary atherosclerosis
    'Diabetes': ['E11'],  # Type 2 diabetes
    'Atrial_Fib': ['I48'],  # Atrial fibrillation
    'CKD': ['N18'],  # Chronic renal failure
    'All_Cancers': ['C18', 'C50', 'D07'],  # Colon, Breast, Prostate
    'Stroke': ['I63'],  # Cerebral infarction
    'Heart_Failure': ['I50'],  # Heart failure
    'Pneumonia': ['J18'],  # Pneumonia
    'COPD': ['J44'],  # Chronic obstructive pulmonary disease
    'Osteoporosis': ['M81'],  # Osteoporosis
    'Anemia': ['D50'],  # Iron deficiency anemia
    'Colorectal_Cancer': ['C18'],  # Colon cancer
    'Breast_Cancer': ['C50'],  # Breast cancer
    'Prostate_Cancer': ['C61'],  # Prostate cancer
    'Lung_Cancer': ['C34'],  # Lung cancer
    'Bladder_Cancer': ['C67'],  # Bladder cancer
    'Secondary_Cancer': ['C79'],  # Secondary malignant neoplasm
    'Depression': ['F32', 'F33'],  # Depressive disorders
    'Anxiety': ['F41'],  # Anxiety disorders
    'Bipolar_Disorder': ['F31'],  # Bipolar disorder
    'Rheumatoid_Arthritis': ['M05', 'M06'],  # Rheumatoid arthritis
    'Psoriasis': ['L40'],  # Psoriasis
    'Ulcerative_Colitis': ['K51'],  # Ulcerative colitis
    'Crohns_Disease': ['K50'],  # Crohn's disease
    'Asthma': ['J45'],  # Asthma
    'Parkinsons': ['G20'],  # Parkinson's disease
    'Multiple_Sclerosis': ['G35'],  # Multiple sclerosis
    'Thyroid_Disorders': ['E03']  # Hypothyroidism
}

# Extract Delphi AUCs for each disease category
delphi_results = []

for disease_name, icd_codes in disease_icd_mapping.items():
    matching_rows = []
    
    for icd_code in icd_codes:
        # Find ICD-10 codes that start with the pattern
        matches = delphi_supp[delphi_supp['Name'].str.contains(f'^{icd_code}', regex=True, na=False)]
        if len(matches) > 0:
            matching_rows.append(matches)
    
    if len(matching_rows) > 0:
        # Combine all matching rows
        combined = pd.concat(matching_rows)
        
        # Average AUCs for 0-year gap (no gap) - both male and female
        female_aucs_0gap = combined['AUC Female, (no gap)'].dropna()
        male_aucs_0gap = combined['AUC Male, (no gap)'].dropna()
        all_aucs_0gap = pd.concat([female_aucs_0gap, male_aucs_0gap])
        avg_auc_0gap = all_aucs_0gap.mean() if len(all_aucs_0gap) > 0 else np.nan
        
        # Average AUCs for 1-year gap - both male and female
        female_aucs_1gap = combined['AUC Female, (1 year gap)'].dropna()
        male_aucs_1gap = combined['AUC Male, (1 year gap)'].dropna()
        all_aucs_1gap = pd.concat([female_aucs_1gap, male_aucs_1gap])
        avg_auc_1gap = all_aucs_1gap.mean() if len(all_aucs_1gap) > 0 else np.nan
        
        if not (pd.isna(avg_auc_0gap) and pd.isna(avg_auc_1gap)):
            delphi_results.append({
                'Disease': disease_name,
                'Delphi_1yr_0gap': avg_auc_0gap,
                'Delphi_1yr_1gap': avg_auc_1gap,
                'N_ICD_codes': len(combined)
            })

delphi_df = pd.DataFrame(delphi_results)
print(f"Extracted Delphi results for {len(delphi_df)} diseases")

# =============================================================================
# 3. MERGE AND COMPARE
# =============================================================================

print("\nCreating comparison...")

# Merge all results
comparison_0gap = aladynoulli_all[['Disease', 'Aladynoulli_1yr_0gap']].merge(
    delphi_df[['Disease', 'Delphi_1yr_0gap']], on='Disease', how='outer')
comparison_0gap['Diff_0gap'] = comparison_0gap['Aladynoulli_1yr_0gap'] - comparison_0gap['Delphi_1yr_0gap']

comparison_1gap = aladynoulli_all[['Disease', 'Aladynoulli_1yr_1gap']].merge(
    delphi_df[['Disease', 'Delphi_1yr_1gap']], on='Disease', how='outer')
comparison_1gap['Diff_1gap'] = comparison_1gap['Aladynoulli_1yr_1gap'] - comparison_1gap['Delphi_1yr_1gap']

# Combined comparison
comparison = comparison_0gap.merge(comparison_1gap, on='Disease', how='outer')

# Sort by 0-year gap difference (NaNs go to end)
comparison = comparison.sort_values('Diff_0gap', ascending=False, na_position='last')

# =============================================================================
# 4. IDENTIFY WINS (0-YEAR GAP)
# =============================================================================

wins_0gap = comparison[comparison['Diff_0gap'] > 0].copy()

print("\n" + "=" * 100)
print("ALADYNOULLI vs DELPHI: DISEASES WHERE ALADYNOULLI WINS (1-YEAR, 0-YEAR GAP)")
print("=" * 100)
print(f"\nTotal wins: {len(wins_0gap)} out of {comparison['Diff_0gap'].notna().sum()} diseases")
print(f"Win rate: {len(wins_0gap)/comparison['Diff_0gap'].notna().sum()*100:.1f}%\n")

pd.set_option('display.float_format', '{:.4f}'.format)

print(f"{'Disease':<25} {'Aladynoulli':>12} {'Delphi':>12} {'Advantage':>12} {'Percent':>10}")
print("-" * 100)

for idx, row in wins_0gap.iterrows():
    disease = row['Disease']
    ala = row['Aladynoulli_1yr_0gap']
    delp = row['Delphi_1yr_0gap']
    diff = row['Diff_0gap']
    pct = (diff / delp * 100) if delp > 0 else 0
    
    print(f"{disease:<25} {ala:>12.4f} {delp:>12.4f} {diff:>12.4f} {pct:>9.1f}%")

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 100)
print("SUMMARY STATISTICS (0-YEAR GAP)")
print("=" * 100)

valid_0gap = comparison[comparison[['Aladynoulli_1yr_0gap', 'Delphi_1yr_0gap']].notna().all(axis=1)]

print(f"\n1-YEAR PREDICTIONS (0-YEAR GAP):")
print(f"  Aladynoulli mean (all):  {valid_0gap['Aladynoulli_1yr_0gap'].mean():.4f}")
print(f"  Delphi mean (all):       {valid_0gap['Delphi_1yr_0gap'].mean():.4f}")
print(f"  Overall difference:      {valid_0gap['Diff_0gap'].mean():.4f}")

if len(wins_0gap) > 0:
    print(f"\n  Aladynoulli mean (wins): {wins_0gap['Aladynoulli_1yr_0gap'].mean():.4f}")
    print(f"  Delphi mean (wins):      {wins_0gap['Delphi_1yr_0gap'].mean():.4f}")
    print(f"  Average advantage:       {wins_0gap['Diff_0gap'].mean():.4f}")

# 1-year gap summary
valid_1gap = comparison[comparison[['Aladynoulli_1yr_1gap', 'Delphi_1yr_1gap']].notna().all(axis=1)]
wins_1gap = comparison[comparison['Diff_1gap'] > 0].copy()

print(f"\n1-YEAR PREDICTIONS (1-YEAR GAP):")
print(f"  Aladynoulli mean (all):  {valid_1gap['Aladynoulli_1yr_1gap'].mean():.4f}")
print(f"  Delphi mean (all):       {valid_1gap['Delphi_1yr_1gap'].mean():.4f}")
print(f"  Overall difference:      {valid_1gap['Diff_1gap'].mean():.4f}")

if len(wins_1gap) > 0:
    print(f"\n  Aladynoulli mean (wins): {wins_1gap['Aladynoulli_1yr_1gap'].mean():.4f}")
    print(f"  Delphi mean (wins):      {wins_1gap['Delphi_1yr_1gap'].mean():.4f}")
    print(f"  Average advantage:       {wins_1gap['Diff_1gap'].mean():.4f}")

# =============================================================================
# 6. TOP WINS
# =============================================================================

print("\n" + "=" * 100)
print("TOP 5 BIGGEST WINS (0-YEAR GAP)")
print("=" * 100)

top5_0gap = wins_0gap.head(5)
for i, (idx, row) in enumerate(top5_0gap.iterrows(), 1):
    print(f"\n{i}. {row['Disease']}")
    print(f"   Aladynoulli: {row['Aladynoulli_1yr_0gap']:.4f}")
    print(f"   Delphi:      {row['Delphi_1yr_0gap']:.4f}")
    print(f"   Advantage:   +{row['Diff_0gap']:.4f} ({row['Diff_0gap']/row['Delphi_1yr_0gap']*100:.1f}% better)")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

print("\n" + "=" * 100)
print("SAVING RESULTS")
print("=" * 100)

output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons/pooled_retrospective')
output_dir.mkdir(parents=True, exist_ok=True)

# Save full comparison
comparison.to_csv(output_dir / 'delphi_comparison_1yr_full.csv', index=False)
print(f"Full comparison saved to: {output_dir / 'delphi_comparison_1yr_full.csv'}")

# Save wins only
wins_0gap.to_csv(output_dir / 'delphi_comparison_1yr_wins_0gap.csv', index=False)
wins_1gap.to_csv(output_dir / 'delphi_comparison_1yr_wins_1gap.csv', index=False)
print(f"Wins (0-year gap) saved to: {output_dir / 'delphi_comparison_1yr_wins_0gap.csv'}")
print(f"Wins (1-year gap) saved to: {output_dir / 'delphi_comparison_1yr_wins_1gap.csv'}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

# Make variables available
print("\nAvailable variables:")
print("  - comparison: Full comparison DataFrame")
print("  - wins_0gap: Diseases where Aladynoulli wins (0-year gap)")
print("  - wins_1gap: Diseases where Aladynoulli wins (1-year gap)")
print("  - delphi_df: Extracted Delphi results")
print("  - aladynoulli_all: Aladynoulli results")

