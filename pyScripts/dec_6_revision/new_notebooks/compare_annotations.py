"""
Compare Ensembl annotations with original annotations from all_loci_annotated.tsv
"""

import pandas as pd
from pathlib import Path

# Load files
ensembl_file = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/unique_loci_annotated.csv")
original_file = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/all_loci_annotated.tsv")

ensembl_df = pd.read_csv(ensembl_file)
original_df = pd.read_csv(original_file, sep='\t')

print(f"Loaded {len(ensembl_df)} loci from Ensembl annotations")
print(f"Loaded {len(original_df)} loci from original annotations\n")

# Merge on UID
merged = ensembl_df.merge(
    original_df[['UID', 'nearestgene']], 
    on='UID', 
    how='left',
    suffixes=('_ensembl', '_original')
)

# Clean up gene names for comparison
def clean_gene_name(name):
    if pd.isna(name) or name == 'N/A':
        return None
    return str(name).strip().upper()

merged['ensembl_clean'] = merged['ensembl_gene_symbol'].apply(clean_gene_name)
merged['original_clean'] = merged['nearestgene_original'].apply(clean_gene_name)

# Compare
merged['match'] = merged.apply(
    lambda row: 'Match' if (
        row['ensembl_clean'] is not None and 
        row['original_clean'] is not None and
        row['ensembl_clean'] == row['original_clean']
    ) else (
        'Ensembl_only' if row['ensembl_clean'] is not None and row['original_clean'] is None
        else 'Original_only' if row['ensembl_clean'] is None and row['original_clean'] is not None
        else 'Both_missing' if row['ensembl_clean'] is None and row['original_clean'] is None
        else 'Different'
    ),
    axis=1
)

# Summary statistics
print("="*80)
print("ANNOTATION COMPARISON SUMMARY")
print("="*80)
print(f"\nTotal loci compared: {len(merged)}")
print(f"\nMatch status:")
match_counts = merged['match'].value_counts()
for status, count in match_counts.items():
    pct = count / len(merged) * 100
    print(f"  {status}: {count} ({pct:.1f}%)")

# Show matches
matches = merged[merged['match'] == 'Match']
print(f"\n✓ Exact matches: {len(matches)}")
if len(matches) > 0:
    print("\nExample matches:")
    print(matches[['rsid', 'ensembl_gene_symbol', 'nearestgene_original']].head(10).to_string())

# Show differences
differences = merged[merged['match'] == 'Different']
print(f"\n⚠️  Different annotations: {len(differences)}")
if len(differences) > 0:
    print("\nExample differences:")
    diff_display = differences[['rsid', 'ensembl_gene_symbol', 'nearestgene_original', 'CHR', 'POS']].head(15)
    print(diff_display.to_string())

# Show Ensembl-only (new annotations)
ensembl_only = merged[merged['match'] == 'Ensembl_only']
print(f"\n✓ Ensembl-only (new annotations): {len(ensembl_only)}")
if len(ensembl_only) > 0:
    print("\nExample new annotations:")
    print(ensembl_only[['rsid', 'ensembl_gene_symbol', 'CHR', 'POS']].head(10).to_string())

# Show original-only (Ensembl missed)
original_only = merged[merged['match'] == 'Original_only']
print(f"\n⚠️  Original-only (Ensembl missed): {len(original_only)}")
if len(original_only) > 0:
    print("\nExample missed annotations:")
    print(original_only[['rsid', 'nearestgene_original', 'CHR', 'POS']].head(10).to_string())

# Save comparison
output_file = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/annotation_comparison.csv")
comparison_cols = [
    'Signature', 'CHR', 'POS', 'UID', 'rsid', 
    'ensembl_gene_symbol', 'nearestgene_original', 'match',
    'annotation_method'
]
merged[comparison_cols].to_csv(output_file, index=False)
print(f"\n✓ Saved comparison to: {output_file}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
total_with_annotation = len(merged[(merged['ensembl_clean'].notna()) | (merged['original_clean'].notna())])
agreement_rate = len(matches) / total_with_annotation * 100 if total_with_annotation > 0 else 0
print(f"Agreement rate (when both have annotations): {agreement_rate:.1f}%")
print(f"Ensembl provides new annotations for {len(ensembl_only)} loci that were missing in original")

