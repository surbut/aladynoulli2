"""
Create a comparison file showing Ensembl annotations vs original annotations
for each variant.
"""

import pandas as pd
from pathlib import Path

# Load files
ensembl_file = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/unique_loci_annotated.csv")
original_file = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/all_loci_annotated.tsv")

print("Loading files...")
ensembl_df = pd.read_csv(ensembl_file)
original_df = pd.read_csv(original_file, sep='\t')

print(f"Loaded {len(ensembl_df)} loci from Ensembl annotations")
print(f"Loaded {len(original_df)} loci from original annotations\n")

# Remove any existing original_nearestgene column from ensembl_df to avoid duplicates
if 'original_nearestgene' in ensembl_df.columns:
    ensembl_df = ensembl_df.drop(columns=['original_nearestgene'])

# Merge on UID (most reliable identifier)
merged = ensembl_df.merge(
    original_df[['UID', 'nearestgene', 'KNOWN', 'LOCUS_CHR', 'LOCUS_FROM', 'LOCUS_TO', 
                 'LOCUS_SNP', 'cytoband', 'EAF', 'BETA', 'SE']], 
    on='UID', 
    how='left',
    suffixes=('', '_original')
)

# Rename for clarity - handle both possible column names after merge
if 'nearestgene_original' in merged.columns:
    # If there was a conflict, use the _original suffix version
    merged = merged.rename(columns={'nearestgene_original': 'original_nearestgene'})
    # Remove the other one if it exists
    if 'nearestgene' in merged.columns:
        merged = merged.drop(columns=['nearestgene'])
elif 'nearestgene' in merged.columns:
    # If no conflict, just rename it
    merged = merged.rename(columns={'nearestgene': 'original_nearestgene'})

# Remove any duplicate columns (keep first occurrence)
merged = merged.loc[:, ~merged.columns.duplicated()]

# Rename other columns
merged = merged.rename(columns={
    'KNOWN': 'known_status',  # 0 = novel, 1 = known
    'LOCUS_CHR': 'locus_chr',
    'LOCUS_FROM': 'locus_start',
    'LOCUS_TO': 'locus_end',
    'LOCUS_SNP': 'locus_n_snps',
    'EAF': 'effect_allele_freq',
    'BETA': 'beta',
    'SE': 'se'
})

# Calculate distance to nearest gene if we have locus info
def calculate_gene_distance(row):
    """Calculate approximate distance from variant to gene"""
    try:
        if pd.isna(row.get('locus_start')) or pd.isna(row.get('locus_end')):
            return None
        pos = float(row['POS'])
        start = float(row['locus_start'])
        end = float(row['locus_end'])
        # Distance is minimum distance to locus boundaries
        dist = min(abs(pos - start), abs(pos - end))
        return dist / 1000.0  # Return in kb
    except (ValueError, TypeError):
        return None

merged['distance_to_locus_kb'] = merged.apply(calculate_gene_distance, axis=1)
merged['distance_to_locus_kb'] = merged['distance_to_locus_kb'].apply(
    lambda x: f"{x:.1f}" if x is not None and pd.notna(x) else 'N/A'
)

# Ensure we have the required columns
if 'original_nearestgene' not in merged.columns:
    print("WARNING: original_nearestgene column not found after merge!")

# Helper function to safely extract scalar values
def to_scalar(val):
    """Convert value to scalar, handling Series and NaN"""
    if isinstance(val, pd.Series):
        if len(val) == 0:
            return None
        val = val.iloc[0]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return val

# Create comparison columns
def compare_annotations(row):
    """Compare Ensembl vs original annotation"""
    try:
        # Safely extract values - directly access the column (row is a Series, so use .get() or direct access)
        ensembl_val = row.get('ensembl_gene_symbol') if 'ensembl_gene_symbol' in row.index else None
        original_val = row.get('original_nearestgene') if 'original_nearestgene' in row.index else None
        
        # Convert to scalar if needed
        ensembl_val = to_scalar(ensembl_val)
        original_val = to_scalar(original_val)
        
        # Handle Ensembl gene
        if ensembl_val is None:
            ensembl_gene = None
        else:
            ensembl_str = str(ensembl_val).strip()
            if ensembl_str == 'N/A' or ensembl_str == '' or ensembl_str == 'nan':
                ensembl_gene = None
            else:
                ensembl_gene = ensembl_str
        
        # Handle original gene
        if original_val is None:
            original_gene = None
        else:
            original_str = str(original_val).strip()
            if original_str == 'N/A' or original_str == '.' or original_str == '' or original_str == 'nan':
                original_gene = None
            else:
                original_gene = original_str
        
        # Compare
        if ensembl_gene is None and original_gene is None:
            return 'Both_N/A'
        elif ensembl_gene is None and original_gene is not None:
            return 'Ensembl_missing'
        elif ensembl_gene is not None and original_gene is None:
            return 'Original_missing'
        elif ensembl_gene.upper() == original_gene.upper():
            return 'Match'
        else:
            return 'Different'
    except Exception as e:
        # Fallback if something goes wrong
        return f'Error: {str(e)}'

merged['annotation_comparison'] = merged.apply(compare_annotations, axis=1)

# Select and reorder columns for output - simplified as requested
output_columns = [
    'Signature',
    'CHR',
    'POS',
    'UID',
    'rsid',
    'LOG10P',
    'beta',
    'se',
    'effect_allele_freq',
    'ensembl_gene_symbol',
    'ensembl_gene_id',
    'original_nearestgene',
    'annotation_comparison',
    'locus_chr',
    'locus_start',
    'locus_end',
    'locus_n_snps',
    'distance_to_locus_kb'
]

# Create final comparison dataframe - only include columns that exist
available_columns = [col for col in output_columns if col in merged.columns]
missing_columns = [col for col in output_columns if col not in merged.columns]
if missing_columns:
    print(f"WARNING: Missing columns: {missing_columns}")

comparison_df = merged[available_columns].copy()

# Sort by signature and significance
comparison_df = comparison_df.sort_values(['Signature', 'LOG10P'], ascending=[True, False])

# Save comparison file
output_file = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/variant_annotation_comparison.csv")
comparison_df.to_csv(output_file, index=False)

print("="*80)
print("ANNOTATION COMPARISON FILE CREATED")
print("="*80)
print(f"✓ Saved to: {output_file}")
print(f"\nTotal variants: {len(comparison_df)}")

# Summary statistics
print(f"\nAnnotation comparison summary:")
comp_summary = comparison_df['annotation_comparison'].value_counts()
for status, count in comp_summary.items():
    pct = count / len(comparison_df) * 100
    print(f"  {status}: {count} ({pct:.1f}%)")

print(f"\nKnown vs Novel (from original file):")
if 'known_status' in comparison_df.columns:
    known_summary = comparison_df['known_status'].value_counts()
    for status, count in known_summary.items():
        pct = count / len(comparison_df) * 100
        status_label = 'Known' if status == 1 else 'Novel' if status == 0 else 'Unknown'
        print(f"  {status_label}: {count} ({pct:.1f}%)")

print(f"\nVariants with Ensembl annotation: {(comparison_df['ensembl_gene_symbol'] != 'N/A').sum()}")
print(f"Variants with original annotation: {(comparison_df['original_nearestgene'].notna() & (comparison_df['original_nearestgene'] != 'N/A')).sum()}")

print("\nDone! Check the output file for detailed comparison.")

