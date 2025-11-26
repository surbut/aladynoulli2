#!/usr/bin/env python3
"""
Compare Aladynoulli predictions across multiple time horizons (5yr, 10yr, 30yr) 
with Delphi 1-year predictions.

This shows how Aladynoulli performance changes across different prediction horizons
compared to Delphi's 1-year predictions.

Usage in notebook:
    %run compare_delphi_multihorizon.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. LOAD ALADYNOULLI RESULTS (MULTIPLE HORIZONS)
# =============================================================================

print("="*100)
print("ALADYNOULLI vs DELPHI: MULTI-HORIZON COMPARISON")
print("="*100)

print("\nLoading Aladynoulli multi-horizon results...")

# Load time horizon results
time_horizons_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/time_horizons/pooled_retrospective')

# Load individual horizon results
aladynoulli_5yr = pd.read_csv(time_horizons_dir / '5yr_results.csv')
aladynoulli_10yr = pd.read_csv(time_horizons_dir / '10yr_results.csv')
aladynoulli_30yr = pd.read_csv(time_horizons_dir / '30yr_results.csv')
aladynoulli_static10yr = pd.read_csv(time_horizons_dir / 'static_10yr_results.csv')

# Also load 1-year washout results for direct comparison
washout_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/washout/pooled_retrospective')
aladynoulli_1yr_0gap = pd.read_csv(washout_dir / 'washout_0yr_results.csv')
aladynoulli_1yr_1gap = pd.read_csv(washout_dir / 'washout_1yr_results.csv')

# Create Aladynoulli results DataFrame
aladynoulli_all = pd.DataFrame({'Disease': aladynoulli_5yr['Disease']})
aladynoulli_all = aladynoulli_all.merge(
    aladynoulli_1yr_0gap[['Disease', 'AUC']].rename(columns={'AUC': 'Aladynoulli_1yr_0gap'}),
    on='Disease', how='outer'
)
aladynoulli_all = aladynoulli_all.merge(
    aladynoulli_1yr_1gap[['Disease', 'AUC']].rename(columns={'AUC': 'Aladynoulli_1yr_1gap'}),
    on='Disease', how='outer'
)
aladynoulli_all = aladynoulli_all.merge(
    aladynoulli_5yr[['Disease', 'AUC']].rename(columns={'AUC': 'Aladynoulli_5yr'}),
    on='Disease', how='outer'
)
aladynoulli_all = aladynoulli_all.merge(
    aladynoulli_10yr[['Disease', 'AUC']].rename(columns={'AUC': 'Aladynoulli_10yr'}),
    on='Disease', how='outer'
)
aladynoulli_all = aladynoulli_all.merge(
    aladynoulli_30yr[['Disease', 'AUC']].rename(columns={'AUC': 'Aladynoulli_30yr'}),
    on='Disease', how='outer'
)
aladynoulli_all = aladynoulli_all.merge(
    aladynoulli_static10yr[['Disease', 'AUC']].rename(columns={'AUC': 'Aladynoulli_static10yr'}),
    on='Disease', how='outer'
)

print(f"Loaded Aladynoulli results for {len(aladynoulli_all)} diseases")
print(f"Horizons available: 1yr (0gap), 1yr (1gap), 5yr, 10yr, 30yr, static10yr")

# =============================================================================
# 2. EXTRACT DELPHI RESULTS FROM SUPPLEMENTARY TABLE
# =============================================================================

print("\nExtracting Delphi results from supplementary table...")

# Load Delphi supplementary table
delphi_supp = pd.read_csv('/Users/sarahurbut/Downloads/41586_2025_9529_MOESM3_ESM.csv')

# Define disease category to ICD-10 code mappings (same as 1yr comparison)
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
# 3. MERGE AND CREATE COMPARISON
# =============================================================================

print("\nCreating multi-horizon comparison...")

# Merge with Delphi results
comparison = aladynoulli_all.merge(
    delphi_df[['Disease', 'Delphi_1yr_0gap', 'Delphi_1yr_1gap']],
    on='Disease', how='outer'
)

# Calculate differences vs Delphi (1yr, 0gap) for each horizon
comparison['Diff_1yr_0gap'] = comparison['Aladynoulli_1yr_0gap'] - comparison['Delphi_1yr_0gap']
comparison['Diff_1yr_1gap'] = comparison['Aladynoulli_1yr_1gap'] - comparison['Delphi_1yr_1gap']
comparison['Diff_5yr'] = comparison['Aladynoulli_5yr'] - comparison['Delphi_1yr_0gap']
comparison['Diff_10yr'] = comparison['Aladynoulli_10yr'] - comparison['Delphi_1yr_0gap']
comparison['Diff_30yr'] = comparison['Aladynoulli_30yr'] - comparison['Delphi_1yr_0gap']
comparison['Diff_static10yr'] = comparison['Aladynoulli_static10yr'] - comparison['Delphi_1yr_0gap']

# Sort by 10-year difference (or 5-year if 10-year not available)
sort_col = 'Diff_10yr' if 'Diff_10yr' in comparison.columns else 'Diff_5yr'
comparison = comparison.sort_values(sort_col, ascending=False, na_position='last')

# =============================================================================
# 4. DISPLAY RESULTS BY HORIZON
# =============================================================================

print("\n" + "=" * 100)
print("ALADYNOULLI PERFORMANCE ACROSS HORIZONS vs DELPHI (1-YEAR, 0-GAP)")
print("=" * 100)

pd.set_option('display.float_format', '{:.4f}'.format)

# Create display columns
display_cols = ['Disease', 'Delphi_1yr_0gap', 'Aladynoulli_1yr_0gap', 'Aladynoulli_5yr', 
                'Aladynoulli_10yr', 'Aladynoulli_30yr', 'Aladynoulli_static10yr']
display_cols = [c for c in display_cols if c in comparison.columns]

# Filter to diseases with Delphi comparison
valid_comparison = comparison[comparison['Delphi_1yr_0gap'].notna()].copy()

if len(valid_comparison) > 0:
    print(f"\n{'Disease':<25} {'Delphi':>10} {'Ala_1yr':>10} {'Ala_5yr':>10} {'Ala_10yr':>10} {'Ala_30yr':>10} {'Ala_st10yr':>10}")
    print("-" * 100)
    
    for idx, row in valid_comparison.iterrows():
        disease = row['Disease']
        delphi = row.get('Delphi_1yr_0gap', np.nan)
        ala_1yr = row.get('Aladynoulli_1yr_0gap', np.nan)
        ala_5yr = row.get('Aladynoulli_5yr', np.nan)
        ala_10yr = row.get('Aladynoulli_10yr', np.nan)
        ala_30yr = row.get('Aladynoulli_30yr', np.nan)
        ala_st10yr = row.get('Aladynoulli_static10yr', np.nan)
        
        print(f"{disease:<25} {delphi:>10.4f} {ala_1yr:>10.4f} {ala_5yr:>10.4f} "
              f"{ala_10yr:>10.4f} {ala_30yr:>10.4f} {ala_st10yr:>10.4f}")

# =============================================================================
# 5. SUMMARY STATISTICS BY HORIZON
# =============================================================================

print("\n" + "=" * 100)
print("SUMMARY STATISTICS: ALADYNOULLI vs DELPHI BY HORIZON")
print("=" * 100)

horizons = [
    ('1yr (0gap)', 'Aladynoulli_1yr_0gap', 'Delphi_1yr_0gap', 'Diff_1yr_0gap'),
    ('1yr (1gap)', 'Aladynoulli_1yr_1gap', 'Delphi_1yr_1gap', 'Diff_1yr_1gap'),
    ('5yr', 'Aladynoulli_5yr', 'Delphi_1yr_0gap', 'Diff_5yr'),
    ('10yr', 'Aladynoulli_10yr', 'Delphi_1yr_0gap', 'Diff_10yr'),
    ('30yr', 'Aladynoulli_30yr', 'Delphi_1yr_0gap', 'Diff_30yr'),
    ('static10yr', 'Aladynoulli_static10yr', 'Delphi_1yr_0gap', 'Diff_static10yr')
]

for horizon_name, ala_col, delphi_col, diff_col in horizons:
    if ala_col in comparison.columns and delphi_col in comparison.columns:
        valid = comparison[[ala_col, delphi_col, diff_col]].dropna()
        if len(valid) > 0:
            wins = valid[valid[diff_col] > 0]
            print(f"\n{horizon_name.upper()}:")
            print(f"  Aladynoulli mean:  {valid[ala_col].mean():.4f}")
            print(f"  Delphi mean:       {valid[delphi_col].mean():.4f}")
            print(f"  Mean difference:   {valid[diff_col].mean():.4f}")
            print(f"  Wins:              {len(wins)}/{len(valid)} ({len(wins)/len(valid)*100:.1f}%)")
            if len(wins) > 0:
                print(f"  Avg advantage:     {wins[diff_col].mean():.4f}")

# =============================================================================
# 6. HORIZON TRENDS (HOW AUC CHANGES WITH TIME)
# =============================================================================

print("\n" + "=" * 100)
print("HORIZON TRENDS: HOW ALADYNOULLI AUC CHANGES WITH PREDICTION HORIZON")
print("=" * 100)

# For diseases with all horizons available
horizon_cols = ['Aladynoulli_1yr_0gap', 'Aladynoulli_5yr', 'Aladynoulli_10yr', 'Aladynoulli_30yr']
horizon_cols = [c for c in horizon_cols if c in comparison.columns]

if len(horizon_cols) >= 3:
    # Find diseases with at least 3 horizons
    complete_diseases = comparison[horizon_cols].dropna(how='any')
    
    if len(complete_diseases) > 0:
        print(f"\nDiseases with complete horizon data: {len(complete_diseases)}")
        print(f"\n{'Disease':<25} {'1yr':>8} {'5yr':>8} {'10yr':>8} {'30yr':>8} {'Change_1to30':>12}")
        print("-" * 100)
        
        for idx, row in complete_diseases.iterrows():
            disease = comparison.loc[idx, 'Disease']
            ala_1yr = row.get('Aladynoulli_1yr_0gap', np.nan)
            ala_5yr = row.get('Aladynoulli_5yr', np.nan)
            ala_10yr = row.get('Aladynoulli_10yr', np.nan)
            ala_30yr = row.get('Aladynoulli_30yr', np.nan)
            
            change = ala_30yr - ala_1yr if not (pd.isna(ala_1yr) or pd.isna(ala_30yr)) else np.nan
            
            print(f"{disease:<25} {ala_1yr:>8.4f} {ala_5yr:>8.4f} {ala_10yr:>8.4f} "
                  f"{ala_30yr:>8.4f} {change:>12.4f}")
        
        # Summary of changes
        changes = []
        for idx, row in complete_diseases.iterrows():
            ala_1yr = row.get('Aladynoulli_1yr_0gap', np.nan)
            ala_30yr = row.get('Aladynoulli_30yr', np.nan)
            if not (pd.isna(ala_1yr) or pd.isna(ala_30yr)):
                changes.append(ala_30yr - ala_1yr)
        
        if len(changes) > 0:
            print(f"\nSummary of AUC change from 1yr to 30yr:")
            print(f"  Mean change: {np.mean(changes):.4f}")
            print(f"  Median change: {np.median(changes):.4f}")
            print(f"  Diseases improving: {sum(1 for c in changes if c > 0)}/{len(changes)}")
            print(f"  Diseases declining: {sum(1 for c in changes if c < 0)}/{len(changes)}")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

print("\n" + "=" * 100)
print("SAVING RESULTS")
print("=" * 100)

output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons/pooled_retrospective')
output_dir.mkdir(parents=True, exist_ok=True)

# Save full comparison
comparison.to_csv(output_dir / 'delphi_comparison_multihorizon_full.csv', index=False)
print(f"Full comparison saved to: {output_dir / 'delphi_comparison_multihorizon_full.csv'}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

# Make variables available
print("\nAvailable variables:")
print("  - comparison: Full multi-horizon comparison DataFrame")
print("  - aladynoulli_all: Aladynoulli results across all horizons")
print("  - delphi_df: Extracted Delphi results")


