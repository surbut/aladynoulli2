"""
Analysis: ICD-10 to PheCode Aggregation
========================================

This script demonstrates the utility of collapsing ICD-10 codes into Phecodes
by showing how many ICD-10 codes map to each PheCode. This aggregation:
- Reduces noise from minor coding variations
- Groups clinically related conditions
- Simplifies analysis from ~1,000+ ICD-10 codes to ~350 Phecodes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# 1. LOAD PHECODE MAPPING DATA
# =============================================================================

print("Loading PheCode mapping data...")

# Try to load from RDS file (if available)
mapping_paths = [
    '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode.rds',
    '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode.csv',
]

mapping_df = None
for path in mapping_paths:
    if Path(path).exists():
        if path.endswith('.rds'):
            try:
                import pyreadr
                result = pyreadr.read_r(path)
                mapping_df = result[None]
                print(f"✓ Loaded from RDS: {len(mapping_df)} rows")
                break
            except ImportError:
                print("⚠ pyreadr not available, trying CSV...")
        else:
            mapping_df = pd.read_csv(path)
            print(f"✓ Loaded from CSV: {len(mapping_df)} rows")
            break

if mapping_df is None:
    print("⚠ Mapping file not found. Creating example analysis with Delphi data...")
    # Fallback: use Delphi data to show ICD-10 structure
    delphi_df = pd.read_csv('/Users/sarahurbut/Downloads/41586_2025_9529_MOESM3_ESM.csv')
    print(f"Using Delphi data: {len(delphi_df)} ICD-10 codes")
    
    # Extract ICD-10 codes from Delphi names (e.g., "I21 Acute myocardial infarction" -> "I21")
    delphi_df['ICD10_code'] = delphi_df['Name'].str.extract(r'^([A-Z]\d{2})')
    delphi_df = delphi_df[delphi_df['ICD10_code'].notna()]
    
    # For demonstration, create a simple mapping (in reality, you'd use the PheCode mapping)
    # This is just to show the structure
    mapping_df = delphi_df[['ICD10_code', 'Name']].copy()
    mapping_df.columns = ['ICD10', 'Description']
    print("Note: Using Delphi data structure for demonstration")

# =============================================================================
# 2. ANALYZE AGGREGATION
# =============================================================================

print("\nAnalyzing ICD-10 to PheCode aggregation...")

# If we have PheCode mapping, analyze it
if 'PheCode' in mapping_df.columns or 'phecode' in mapping_df.columns:
    phecode_col = 'PheCode' if 'PheCode' in mapping_df.columns else 'phecode'
    icd_col = 'ICD10' if 'ICD10' in mapping_df.columns else 'diag_icd10'
    
    # Count ICD-10 codes per PheCode
    aggregation = mapping_df.groupby(phecode_col)[icd_col].nunique().reset_index()
    aggregation.columns = ['PheCode', 'N_ICD10_codes']
    aggregation = aggregation.sort_values('N_ICD10_codes', ascending=False)
    
    print(f"\nTotal Phecodes: {len(aggregation)}")
    print(f"Total ICD-10 codes: {mapping_df[icd_col].nunique()}")
    print(f"Average ICD-10 codes per PheCode: {aggregation['N_ICD10_codes'].mean():.1f}")
    print(f"Median ICD-10 codes per PheCode: {aggregation['N_ICD10_codes'].median():.1f}")
    print(f"Max ICD-10 codes in one PheCode: {aggregation['N_ICD10_codes'].max()}")
    
    # Get top aggregators
    print("\nTop 10 Phecodes with most ICD-10 codes:")
    print(aggregation.head(10).to_string(index=False))
    
else:
    # If no PheCode column, show ICD-10 structure
    print("\nShowing ICD-10 code structure (PheCode mapping not found in data)")
    print(f"Total unique ICD-10 codes: {mapping_df['ICD10'].nunique()}")
    
    # Group by first 3 characters (ICD-10 chapter/subchapter)
    mapping_df['ICD10_chapter'] = mapping_df['ICD10'].str[:3]
    chapter_counts = mapping_df.groupby('ICD10_chapter').size().sort_values(ascending=False)
    
    print("\nTop 20 ICD-10 3-character codes (showing granularity):")
    print(chapter_counts.head(20).to_string())
    
    aggregation = pd.DataFrame({
        'PheCode': chapter_counts.index,
        'N_ICD10_codes': chapter_counts.values
    })

# =============================================================================
# 3. MAP TO YOUR DISEASES
# =============================================================================

print("\n" + "="*80)
print("MAPPING YOUR DISEASES TO ICD-10 CODES")
print("="*80)

# Your disease list from the comparison
your_diseases = [
    'ASCVD', 'Diabetes', 'Atrial_Fib', 'CKD', 'All_Cancers', 'Stroke',
    'Heart_Failure', 'Pneumonia', 'COPD', 'Osteoporosis', 'Anemia',
    'Colorectal_Cancer', 'Breast_Cancer', 'Prostate_Cancer', 'Lung_Cancer',
    'Bladder_Cancer', 'Secondary_Cancer', 'Depression', 'Anxiety',
    'Bipolar_Disorder', 'Rheumatoid_Arthritis', 'Psoriasis',
    'Ulcerative_Colitis', 'Crohns_Disease', 'Asthma', 'Parkinsons',
    'Multiple_Sclerosis', 'Thyroid_Disorders'
]

# ICD-10 mappings for your diseases (from your comparison script)
disease_icd_mapping = {
    'ASCVD': ['I21', 'I22', 'I25'],
    'Diabetes': ['E10', 'E11'],
    'Atrial_Fib': ['I48'],
    'CKD': ['N18'],
    'Stroke': ['I63', 'I64'],
    'Heart_Failure': ['I50'],
    'Pneumonia': ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18'],
    'COPD': ['J44'],
    'Osteoporosis': ['M80', 'M81'],
    'Anemia': ['D50', 'D51', 'D52', 'D53'],
    'Colorectal_Cancer': ['C18', 'C19', 'C20'],
    'Breast_Cancer': ['C50'],
    'Prostate_Cancer': ['C61'],
    'Lung_Cancer': ['C34'],
    'Bladder_Cancer': ['C67'],
    'Secondary_Cancer': ['C79'],
    'Depression': ['F32', 'F33'],
    'Anxiety': ['F41'],
    'Bipolar_Disorder': ['F31'],
    'Rheumatoid_Arthritis': ['M05', 'M06'],
    'Psoriasis': ['L40'],
    'Ulcerative_Colitis': ['K51'],
    'Crohns_Disease': ['K50'],
    'Asthma': ['J45', 'J46'],
    'Parkinsons': ['G20'],
    'Multiple_Sclerosis': ['G35'],
    'Thyroid_Disorders': ['E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07'],
}

# Count ICD-10 codes per disease
disease_icd_counts = {}
for disease, icd_codes in disease_icd_mapping.items():
    disease_icd_counts[disease] = len(icd_codes)

disease_counts_df = pd.DataFrame({
    'Disease': list(disease_icd_counts.keys()),
    'N_ICD10_codes': list(disease_icd_counts.values())
}).sort_values('N_ICD10_codes', ascending=False)

print("\nYour diseases and their ICD-10 code counts:")
print(disease_counts_df.to_string(index=False))

print(f"\nTotal ICD-10 codes across your diseases: {sum(disease_icd_counts.values())}")
print(f"Number of diseases: {len(disease_icd_counts)}")
print(f"Average ICD-10 codes per disease: {np.mean(list(disease_icd_counts.values())):.1f}")

# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))

# Panel A: Distribution of ICD-10 codes per PheCode
ax1 = plt.subplot(2, 3, 1)
ax1.hist(aggregation['N_ICD10_codes'], bins=30, color='#2c7fb8', alpha=0.7, edgecolor='black')
ax1.axvline(aggregation['N_ICD10_codes'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {aggregation["N_ICD10_codes"].mean():.1f}')
ax1.axvline(aggregation['N_ICD10_codes'].median(), color='green', linestyle=':', 
           linewidth=2, label=f'Median: {aggregation["N_ICD10_codes"].median():.1f}')
ax1.set_xlabel('Number of ICD-10 Codes per PheCode', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('A. Distribution of ICD-10 Codes per PheCode', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel B: Top aggregators (Phecodes with most ICD-10 codes)
ax2 = plt.subplot(2, 3, 2)
top20 = aggregation.head(20)
y_pos = np.arange(len(top20))
bars = ax2.barh(y_pos, top20['N_ICD10_codes'], color='#2c7fb8', alpha=0.8, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"PheCode {int(pc)}" if pd.notna(pc) else "Unknown" 
                     for pc in top20['PheCode']], fontsize=9)
ax2.set_xlabel('Number of ICD-10 Codes', fontsize=11, fontweight='bold')
ax2.set_title('B. Top 20 Phecodes by ICD-10 Aggregation', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top20.iterrows()):
    ax2.text(row['N_ICD10_codes'] + 0.5, i, f"{int(row['N_ICD10_codes'])}", 
            va='center', fontsize=9, fontweight='bold')

# Panel C: Your diseases - ICD-10 code counts
ax3 = plt.subplot(2, 3, 3)
top_diseases = disease_counts_df.head(15)
y_pos = np.arange(len(top_diseases))
bars = ax3.barh(y_pos, top_diseases['N_ICD10_codes'], color='#f03b20', alpha=0.8, edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_diseases['Disease'].str.replace('_', ' '), fontsize=9)
ax3.set_xlabel('Number of ICD-10 Codes', fontsize=11, fontweight='bold')
ax3.set_title('C. Your Diseases: ICD-10 Code Counts', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_diseases.iterrows()):
    ax3.text(row['N_ICD10_codes'] + 0.2, i, f"{int(row['N_ICD10_codes'])}", 
            va='center', fontsize=9, fontweight='bold')

# Panel D: Comparison - PheCode vs ICD-10 granularity
ax4 = plt.subplot(2, 3, 4)
comparison_data = {
    'Approach': ['ICD-10\n(Delphi)', 'PheCode\n(Aladynoulli)'],
    'Number of Codes': [mapping_df['ICD10'].nunique() if 'ICD10' in mapping_df.columns 
                        else len(mapping_df), 
                        len(aggregation)]
}
comp_df = pd.DataFrame(comparison_data)
bars = ax4.bar(comp_df['Approach'], comp_df['Number of Codes'], 
               color=['#f03b20', '#2c7fb8'], alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_ylabel('Number of Disease Codes', fontsize=11, fontweight='bold')
ax4.set_title('D. Dimensionality Comparison', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
            f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add reduction percentage
reduction = (1 - comp_df.iloc[1]['Number of Codes'] / comp_df.iloc[0]['Number of Codes']) * 100
ax4.text(0.5, 0.95, f'Reduction: {reduction:.1f}%', 
        transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Panel E: Example - Multiple ICD-10 codes mapping to one concept
ax5 = plt.subplot(2, 3, 5)
# Show examples where many ICD-10 codes map to one disease
example_diseases = ['Thyroid_Disorders', 'Pneumonia', 'Anemia', 'Depression', 'ASCVD']
example_counts = [disease_icd_counts.get(d, 0) for d in example_diseases]
y_pos = np.arange(len(example_diseases))
bars = ax5.barh(y_pos, example_counts, color='#6366f1', alpha=0.8, edgecolor='black')
ax5.set_yticks(y_pos)
ax5.set_yticklabels([d.replace('_', ' ') for d in example_diseases], fontsize=10)
ax5.set_xlabel('Number of ICD-10 Codes', fontsize=11, fontweight='bold')
ax5.set_title('E. Example: ICD-10 Aggregation\n(Many codes → One disease)', 
              fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# Add ICD-10 code lists as annotations
for i, disease in enumerate(example_diseases):
    icd_codes = disease_icd_mapping.get(disease, [])
    if icd_codes:
        code_str = ', '.join(icd_codes[:5])  # Show first 5
        if len(icd_codes) > 5:
            code_str += f' ... (+{len(icd_codes)-5} more)'
        ax5.text(example_counts[i] + 0.3, i, code_str, 
                va='center', fontsize=8, style='italic', alpha=0.7)

# Panel F: Cumulative distribution
ax6 = plt.subplot(2, 3, 6)
sorted_counts = np.sort(aggregation['N_ICD10_codes'])[::-1]
cumulative = np.cumsum(sorted_counts)
cumulative_pct = cumulative / cumulative[-1] * 100
ax6.plot(range(len(cumulative_pct)), cumulative_pct, linewidth=2, color='#2c7fb8')
ax6.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
ax6.axvline(np.where(cumulative_pct >= 80)[0][0] if len(np.where(cumulative_pct >= 80)[0]) > 0 else 0,
           color='red', linestyle='--', alpha=0.5)
ax6.set_xlabel('Phecodes (sorted by ICD-10 count)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cumulative % of ICD-10 Codes', fontsize=11, fontweight='bold')
ax6.set_title('F. Cumulative Distribution\n(How many Phecodes cover most ICD-10?)', 
              fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.legend()

plt.suptitle('ICD-10 to PheCode Aggregation Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('/Users/sarahurbut/aladynoulli2/claudefile/output/icd10_phecode_aggregation_analysis.png', 
            bbox_inches='tight', facecolor='white', dpi=300)
print("\n✓ Saved: icd10_phecode_aggregation_analysis.png")

# =============================================================================
# 5. SUMMARY TABLE
# =============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary_data = {
    'Metric': [
        'Total ICD-10 codes (Delphi approach)',
        'Total Phecodes (Aladynoulli approach)',
        'Reduction factor',
        'Average ICD-10 codes per PheCode',
        'Median ICD-10 codes per PheCode',
        'Max ICD-10 codes in one PheCode',
        'Phecodes with >5 ICD-10 codes',
        'Phecodes with >10 ICD-10 codes',
    ],
    'Value': [
        f"{mapping_df['ICD10'].nunique():,}" if 'ICD10' in mapping_df.columns else "~1,000+",
        f"{len(aggregation):,}",
        f"{mapping_df['ICD10'].nunique() / len(aggregation):.1f}x" if 'ICD10' in mapping_df.columns else "~3x",
        f"{aggregation['N_ICD10_codes'].mean():.1f}",
        f"{aggregation['N_ICD10_codes'].median():.1f}",
        f"{int(aggregation['N_ICD10_codes'].max())}",
        f"{(aggregation['N_ICD10_codes'] > 5).sum()}",
        f"{(aggregation['N_ICD10_codes'] > 10).sum()}",
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('/Users/sarahurbut/aladynoulli2/claudefile/output/icd10_phecode_summary.csv', index=False)
print("\n✓ Saved summary to: icd10_phecode_summary.csv")

# Save detailed aggregation
aggregation.to_csv('/Users/sarahurbut/aladynoulli2/claudefile/output/icd10_per_phecode_counts.csv', index=False)
print("✓ Saved detailed counts to: icd10_per_phecode_counts.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Insight: PheCode aggregation reduces dimensionality while preserving")
print("clinical meaning by grouping related ICD-10 codes that represent")
print("variations of the same underlying condition.")

