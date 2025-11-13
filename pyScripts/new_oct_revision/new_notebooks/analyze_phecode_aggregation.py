"""
Analyze ICD-10 to PheCode Aggregation
======================================

This script replicates the R mapping logic to show how many ICD-10 codes
map to each PheCode, demonstrating the utility of collapsing granular
ICD-10 codes into clinically meaningful Phecodes.

Based on the R function:
- Joins with phecode_icd10cm (exact match)
- Falls back to phecode_icd10 (4-char match)
- Falls back to short_icd10
- Uses parent_phecode as final fallback
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
# 1. LOAD MAPPING TABLES
# =============================================================================

print("Loading PheCode mapping tables...")

# Try to find mapping files
base_paths = [
    '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/',
    '/Users/sarahurbut/Library/CloudStorage/Dropbox/',
]

mapping_files = {
    'phecode_icd10cm': None,
    'phecode_icd10': None,
    'short_icd10': None,
}

# Search for mapping files
for base in base_paths:
    if Path(base).exists():
        for file in Path(base).glob('*phecode*.rds'):
            if 'icd10cm' in file.name.lower():
                mapping_files['phecode_icd10cm'] = file
            elif 'icd10' in file.name.lower() and 'cm' not in file.name.lower():
                mapping_files['phecode_icd10'] = file
        for file in Path(base).glob('*short*icd*.rds'):
            mapping_files['short_icd10'] = file

# Also try CSV versions
for base in base_paths:
    if Path(base).exists():
        for file in Path(base).glob('*phecode*.csv'):
            if mapping_files['phecode_icd10cm'] is None and 'icd10cm' in file.name.lower():
                mapping_files['phecode_icd10cm'] = file
            if mapping_files['phecode_icd10'] is None and 'icd10' in file.name.lower() and 'cm' not in file.name.lower():
                mapping_files['phecode_icd10'] = file

# Load mapping tables
def load_mapping(file_path):
    """Load RDS or CSV mapping file"""
    if file_path is None:
        return None
    try:
        if file_path.suffix == '.rds':
            import pyreadr
            result = pyreadr.read_r(str(file_path))
            return result[None]
        else:
            return pd.read_csv(file_path)
    except Exception as e:
        print(f"  ⚠ Could not load {file_path.name}: {e}")
        return None

phecode_icd10cm = load_mapping(mapping_files['phecode_icd10cm'])
phecode_icd10 = load_mapping(mapping_files['phecode_icd10'])
short_icd10 = load_mapping(mapping_files['short_icd10'])

# If we have the merged file, use that
merged_file = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode.rds')
if merged_file.exists() and phecode_icd10cm is None:
    print(f"Loading merged mapping file: {merged_file.name}")
    try:
        import pyreadr
        result = pyreadr.read_r(str(merged_file))
        merged_df = result[None]
        print(f"  ✓ Loaded {len(merged_df)} rows")
        
        # Extract mapping information
        if 'ICD10' in merged_df.columns and 'PheCode' in merged_df.columns:
            phecode_icd10cm = merged_df[['ICD10', 'PheCode']].drop_duplicates()
            print(f"  ✓ Extracted {len(phecode_icd10cm)} ICD-10 to PheCode mappings")
    except Exception as e:
        print(f"  ⚠ Error loading merged file: {e}")

# =============================================================================
# 2. REPLICATE R MAPPING LOGIC
# =============================================================================

print("\nReplicating R mapping logic...")

# Collect all ICD-10 to PheCode mappings
all_mappings = []

# Method 1: Exact match from phecode_icd10cm
if phecode_icd10cm is not None:
    icd_col = 'ICD10' if 'ICD10' in phecode_icd10cm.columns else 'diag_icd10'
    phec_col = 'PheCode' if 'PheCode' in phecode_icd10cm.columns else 'phecode'
    
    if icd_col in phecode_icd10cm.columns and phec_col in phecode_icd10cm.columns:
        exact_mappings = phecode_icd10cm[[icd_col, phec_col]].dropna()
        exact_mappings['mapping_method'] = 'exact'
        all_mappings.append(exact_mappings)
        print(f"  ✓ Exact mappings: {len(exact_mappings)}")

# Method 2: 4-character match from phecode_icd10
if phecode_icd10 is not None:
    icd_col = 'ICD10' if 'ICD10' in phecode_icd10.columns else 'diag_icd10'
    phec_col = 'PheCode' if 'PheCode' in phecode_icd10.columns else 'phecode'
    
    if icd_col in phecode_icd10.columns and phec_col in phecode_icd10.columns:
        # Create 4-character versions
        fourchar_mappings = phecode_icd10[[icd_col, phec_col]].copy()
        fourchar_mappings['ICD10_4char'] = fourchar_mappings[icd_col].astype(str).str[:4]
        fourchar_mappings = fourchar_mappings[['ICD10_4char', phec_col]].dropna()
        fourchar_mappings.columns = [icd_col, phec_col]
        fourchar_mappings['mapping_method'] = '4char'
        all_mappings.append(fourchar_mappings)
        print(f"  ✓ 4-character mappings: {len(fourchar_mappings)}")

# Method 3: Short ICD-10
if short_icd10 is not None:
    icd_col = 'ICD10' if 'ICD10' in short_icd10.columns else 'diag_icd10'
    phec_col = 'PheCode' if 'PheCode' in short_icd10.columns else 'phecode'
    
    if icd_col in short_icd10.columns and phec_col in short_icd10.columns:
        short_mappings = short_icd10[[icd_col, phec_col]].dropna()
        short_mappings['mapping_method'] = 'short'
        all_mappings.append(short_mappings)
        print(f"  ✓ Short mappings: {len(short_mappings)}")

# Combine all mappings
if all_mappings:
    combined_mappings = pd.concat(all_mappings, ignore_index=True)
    # Remove duplicates (keep first method that worked)
    combined_mappings = combined_mappings.drop_duplicates(subset=[icd_col], keep='first')
    
    print(f"\n  Total unique ICD-10 codes mapped: {combined_mappings[icd_col].nunique()}")
    print(f"  Total unique Phecodes: {combined_mappings[phec_col].nunique()}")
else:
    print("\n  ⚠ No mapping files found. Using Delphi data structure for demonstration.")
    # Fallback: use Delphi data to show structure
    delphi_df = pd.read_csv('/Users/sarahurbut/Downloads/41586_2025_9529_MOESM3_ESM.csv')
    delphi_df['ICD10'] = delphi_df['Name'].str.extract(r'^([A-Z]\d{2})')
    delphi_df = delphi_df[delphi_df['ICD10'].notna()]
    
    # Create a simple grouping by first 3 characters (simulating PheCode aggregation)
    delphi_df['PheCode_sim'] = delphi_df['ICD10'].str[:3]
    combined_mappings = delphi_df[['ICD10', 'PheCode_sim']].drop_duplicates()
    combined_mappings.columns = ['ICD10', 'PheCode']
    print(f"  Using Delphi data structure: {len(combined_mappings)} mappings")

# =============================================================================
# 3. COUNT ICD-10 CODES PER PHECODE
# =============================================================================

print("\n" + "="*80)
print("AGGREGATION ANALYSIS")
print("="*80)

aggregation = combined_mappings.groupby('PheCode')['ICD10'].agg([
    ('N_ICD10_codes', 'nunique'),
    ('ICD10_list', lambda x: ', '.join(sorted(x.unique())[:10]))  # First 10 codes
]).reset_index()

aggregation = aggregation.sort_values('N_ICD10_codes', ascending=False)

print(f"\nTotal Phecodes: {len(aggregation)}")
print(f"Total ICD-10 codes: {combined_mappings['ICD10'].nunique()}")
print(f"Average ICD-10 codes per PheCode: {aggregation['N_ICD10_codes'].mean():.1f}")
print(f"Median ICD-10 codes per PheCode: {aggregation['N_ICD10_codes'].median():.1f}")
print(f"Max ICD-10 codes in one PheCode: {aggregation['N_ICD10_codes'].max()}")

print("\nTop 20 Phecodes with most ICD-10 codes:")
print(aggregation[['PheCode', 'N_ICD10_codes', 'ICD10_list']].head(20).to_string(index=False))

# =============================================================================
# 4. MAP YOUR DISEASES
# =============================================================================

print("\n" + "="*80)
print("YOUR DISEASES: ICD-10 AGGREGATION")
print("="*80)

# Your disease to ICD-10 mapping (from comparison script)
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

disease_summary = []
for disease, icd_codes in disease_icd_mapping.items():
    # Count how many ICD-10 codes map to this disease's PheCode
    # (In reality, these ICD codes would map to one or more Phecodes)
    disease_summary.append({
        'Disease': disease,
        'N_ICD10_codes': len(icd_codes),
        'ICD10_codes': ', '.join(icd_codes)
    })

disease_df = pd.DataFrame(disease_summary).sort_values('N_ICD10_codes', ascending=False)

print("\nYour diseases and their ICD-10 code counts:")
print(disease_df.to_string(index=False))

print(f"\nTotal ICD-10 codes across your {len(disease_df)} diseases: {disease_df['N_ICD10_codes'].sum()}")
print(f"Average ICD-10 codes per disease: {disease_df['N_ICD10_codes'].mean():.1f}")

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))

# Panel A: Distribution
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

# Panel B: Top aggregators
ax2 = plt.subplot(2, 3, 2)
top20 = aggregation.head(20)
y_pos = np.arange(len(top20))
bars = ax2.barh(y_pos, top20['N_ICD10_codes'], color='#2c7fb8', alpha=0.8, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"PheCode {int(pc)}" if pd.notna(pc) else "Unknown" 
                     for pc in top20['PheCode']], fontsize=9)
ax2.set_xlabel('Number of ICD-10 Codes', fontsize=11, fontweight='bold')
ax2.set_title('B. Top 20 Phecodes by Aggregation', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top20.iterrows()):
    ax2.text(row['N_ICD10_codes'] + 0.5, i, f"{int(row['N_ICD10_codes'])}", 
            va='center', fontsize=9, fontweight='bold')

# Panel C: Your diseases
ax3 = plt.subplot(2, 3, 3)
top_diseases = disease_df.head(15)
y_pos = np.arange(len(top_diseases))
bars = ax3.barh(y_pos, top_diseases['N_ICD10_codes'], color='#f03b20', alpha=0.8, edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_diseases['Disease'].str.replace('_', ' '), fontsize=9)
ax3.set_xlabel('Number of ICD-10 Codes', fontsize=11, fontweight='bold')
ax3.set_title('C. Your Diseases: ICD-10 Code Counts', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top_diseases.iterrows()):
    ax3.text(row['N_ICD10_codes'] + 0.2, i, f"{int(row['N_ICD10_codes'])}", 
            va='center', fontsize=9, fontweight='bold')

# Panel D: Comparison
ax4 = plt.subplot(2, 3, 4)
comparison_data = {
    'Approach': ['ICD-10\n(Delphi)', 'PheCode\n(Aladynoulli)'],
    'Number of Codes': [
        combined_mappings['ICD10'].nunique(),
        len(aggregation)
    ]
}
comp_df = pd.DataFrame(comparison_data)
bars = ax4.bar(comp_df['Approach'], comp_df['Number of Codes'], 
               color=['#f03b20', '#2c7fb8'], alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_ylabel('Number of Disease Codes', fontsize=11, fontweight='bold')
ax4.set_title('D. Dimensionality Comparison', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
            f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

reduction = (1 - comp_df.iloc[1]['Number of Codes'] / comp_df.iloc[0]['Number of Codes']) * 100
ax4.text(0.5, 0.95, f'Reduction: {reduction:.1f}%', 
        transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Panel E: Examples with ICD-10 lists
ax5 = plt.subplot(2, 3, 5)
example_diseases = ['Thyroid_Disorders', 'Pneumonia', 'Anemia', 'Depression', 'ASCVD']
example_data = []
for disease in example_diseases:
    icd_codes = disease_icd_mapping.get(disease, [])
    example_data.append({
        'Disease': disease,
        'N_ICD10': len(icd_codes),
        'ICD10_list': ', '.join(icd_codes)
    })
example_df = pd.DataFrame(example_data)

y_pos = np.arange(len(example_df))
bars = ax5.barh(y_pos, example_df['N_ICD10'], color='#6366f1', alpha=0.8, edgecolor='black')
ax5.set_yticks(y_pos)
ax5.set_yticklabels([d.replace('_', ' ') for d in example_df['Disease']], fontsize=10)
ax5.set_xlabel('Number of ICD-10 Codes', fontsize=11, fontweight='bold')
ax5.set_title('E. Example: Multiple ICD-10 → One Disease', fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# Add ICD-10 code annotations
for i, (idx, row) in enumerate(example_df.iterrows()):
    ax5.text(row['N_ICD10'] + 0.3, i, row['ICD10_list'], 
            va='center', fontsize=8, style='italic', alpha=0.7)

# Panel F: Cumulative
ax6 = plt.subplot(2, 3, 6)
sorted_counts = np.sort(aggregation['N_ICD10_codes'])[::-1]
cumulative = np.cumsum(sorted_counts)
cumulative_pct = cumulative / cumulative[-1] * 100
ax6.plot(range(len(cumulative_pct)), cumulative_pct, linewidth=2, color='#2c7fb8')
ax6.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
if len(np.where(cumulative_pct >= 80)[0]) > 0:
    ax6.axvline(np.where(cumulative_pct >= 80)[0][0], color='red', linestyle='--', alpha=0.5)
ax6.set_xlabel('Phecodes (sorted by ICD-10 count)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cumulative % of ICD-10 Codes', fontsize=11, fontweight='bold')
ax6.set_title('F. Cumulative Distribution', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.legend()

plt.suptitle('ICD-10 to PheCode Aggregation: Demonstrating Clinical Grouping', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('/Users/sarahurbut/aladynoulli2/claudefile/output/icd10_phecode_aggregation.png', 
            bbox_inches='tight', facecolor='white', dpi=300)
print("\n✓ Saved: icd10_phecode_aggregation.png")

# Save detailed results
aggregation.to_csv('/Users/sarahurbut/aladynoulli2/claudefile/output/icd10_per_phecode_counts.csv', index=False)
disease_df.to_csv('/Users/sarahurbut/aladynoulli2/claudefile/output/disease_icd10_counts.csv', index=False)

print("✓ Saved detailed counts to CSV files")
print("\n" + "="*80)
print("KEY INSIGHT: PheCode aggregation reduces noise from minor ICD-10")
print("coding variations while preserving clinical meaning.")
print("="*80)


