"""
Check if all_loci_annotated.tsv contains all unique loci from lead-for-paper analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the file
loci_file = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/all_loci_annotated.tsv")
df = pd.read_csv(loci_file, sep='\t')

print('='*80)
print('ANALYSIS OF all_loci_annotated.tsv')
print('='*80)
print(f'\nTotal rows (variants): {len(df)}')
print(f'Total unique signatures: {df["SIG"].nunique()}')

# Check for unique loci using locus_id
if 'locus_id' in df.columns:
    print(f'\nUnique locus_id values: {df["locus_id"].nunique()}')
    locus_counts = df['locus_id'].value_counts()
    print(f'Rows per unique locus_id:')
    print(f'  Mean: {locus_counts.mean():.2f}')
    print(f'  Median: {locus_counts.median():.0f}')
    print(f'  Max: {locus_counts.max()}')
    print(f'  Loci with >1 variant: {(locus_counts > 1).sum()}')
    
    # Show examples of loci with multiple variants
    multi_variant_loci = locus_counts[locus_counts > 1].head(5)
    if len(multi_variant_loci) > 0:
        print(f'\nExamples of loci with multiple variants:')
        for locus_id, count in multi_variant_loci.items():
            locus_data = df[df['locus_id'] == locus_id][['SIG', 'nearestgene', 'LOG10P', 'rsid']].head(3)
            print(f'\n  Locus ID {locus_id} ({count} variants):')
            for _, row in locus_data.iterrows():
                print(f'    {row["SIG"]:6} {row["nearestgene"]:15} LOG10P={row["LOG10P"]:6.2f} {row["rsid"]}')

# Check using LOCUS_CHR, LOCUS_FROM, LOCUS_TO
if all(col in df.columns for col in ['LOCUS_CHR', 'LOCUS_FROM', 'LOCUS_TO']):
    df['locus_key'] = df['LOCUS_CHR'].astype(str) + ':' + df['LOCUS_FROM'].astype(str) + '-' + df['LOCUS_TO'].astype(str)
    print(f'\nUnique loci (by LOCUS_CHR:FROM-TO): {df["locus_key"].nunique()}')
    
    # Check if each locus has exactly one lead variant
    locus_variant_counts = df['locus_key'].value_counts()
    print(f'  Loci with exactly 1 variant: {(locus_variant_counts == 1).sum()}')
    print(f'  Loci with >1 variant: {(locus_variant_counts > 1).sum()}')
    
    # Show examples
    if (locus_variant_counts > 1).sum() > 0:
        print(f'\nExamples of loci with multiple variants (by genomic region):')
        multi_loci = locus_variant_counts[locus_variant_counts > 1].head(3)
        for locus_key, count in multi_loci.items():
            locus_data = df[df['locus_key'] == locus_key][['SIG', 'nearestgene', 'LOG10P', 'rsid', 'POS']].head(3)
            print(f'\n  {locus_key} ({count} variants):')
            for _, row in locus_data.iterrows():
                print(f'    {row["SIG"]:6} {row["nearestgene"]:15} POS={row["POS"]:10} LOG10P={row["LOG10P"]:6.2f} {row["rsid"]}')

# Count by signature
print(f'\nVariants per signature:')
sig_counts = df['SIG'].value_counts().sort_index()
for sig, count in sig_counts.items():
    print(f'  {sig:6}: {count:3} variants')

# Check if these are all genome-wide significant
print(f'\nSignificance thresholds:')
genome_wide_threshold = -np.log10(5e-8)
print(f'  Variants with LOG10P >= {genome_wide_threshold:.2f} (p < 5e-8): {(df["LOG10P"] >= genome_wide_threshold).sum()}')
print(f'  Variants with LOG10P < {genome_wide_threshold:.2f}: {(df["LOG10P"] < genome_wide_threshold).sum()}')
print(f'  Minimum LOG10P: {df["LOG10P"].min():.2f}')
print(f'  Maximum LOG10P: {df["LOG10P"].max():.2f}')

# Check for unique combinations of signature and locus
if 'locus_id' in df.columns:
    sig_locus_combos = df.groupby(['SIG', 'locus_id']).size()
    print(f'\nUnique signature-locus combinations: {len(sig_locus_combos)}')
    print(f'  Combinations with >1 variant: {(sig_locus_combos > 1).sum()}')

print('\n' + '='*80)
print('CONCLUSION:')
print('='*80)
if 'locus_id' in df.columns:
    unique_loci = df['locus_id'].nunique()
    total_variants = len(df)
    if unique_loci == total_variants:
        print('✓ Each row represents a UNIQUE locus (one lead variant per locus)')
        print(f'  Total unique loci: {unique_loci}')
    else:
        print(f'⚠ File contains {total_variants} variants from {unique_loci} unique loci')
        print(f'  Some loci have multiple variants (mean: {locus_counts.mean():.2f} variants per locus)')
        print(f'  This could be due to:')
        print(f'    - Multiple independent signals in the same locus')
        print(f'    - Different signatures mapping to the same locus')
        print(f'    - Multiple lead variants from fine-mapping')
        print(f'  → This appears to be LEAD VARIANTS per SIGNATURE per LOCUS')
        print(f'  → Each row = one lead variant for a signature at a specific locus')
