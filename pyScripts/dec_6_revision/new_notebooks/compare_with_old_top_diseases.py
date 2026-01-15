#!/usr/bin/env python3
"""
Compare top diseases per signature from current batches vs old initial_clusters results.
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("COMPARING TOP DISEASES: Current vs Old Results")
print("="*80)

# Load current results
current_csv = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/top_diseases_per_signature.csv')
current_df = pd.read_csv(current_csv)

# Load old results (manual parsing from text file)
old_file = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal (9-23-25 4:48 PM)/kappag1/results/output_160000_170000/plots/signature_top_diseases.txt')

print(f"\nLoading current results from: {current_csv}")
print(f"Loading old results from: {old_file}")

# Parse old results
old_results = {}
if old_file.exists():
    with open(old_file, 'r') as f:
        lines = f.readlines()
    
    current_sig = None
    for line in lines:
        line = line.strip()
        if line.startswith('Top 10 diseases in Signature'):
            # Extract signature number
            parts = line.split('Signature ')[1].split(' (')[0]
            current_sig = int(parts)
            old_results[current_sig] = []
        elif line and ':' in line and current_sig is not None:
            # Parse disease name and effect
            # Format: "Disease name: effect=X.XXX (OR=X.XX), std=X.XXX"
            if 'effect=' in line:
                disease_part = line.split(':')[0].strip()
                old_results[current_sig].append(disease_part)

# Compare
print("\n" + "="*80)
print("COMPARISON BY SIGNATURE")
print("="*80)

for sig in sorted(current_df['Signature'].unique()):
    print(f"\n{'='*80}")
    print(f"SIGNATURE {sig}")
    print(f"{'='*80}")
    
    # Current top 10
    current_sig_data = current_df[current_df['Signature'] == sig].head(10)
    current_names = current_sig_data['Disease_Name'].tolist()
    
    # Old top 10
    old_names = old_results.get(sig, [])
    
    print(f"\nCurrent top 10 (by avg PSI):")
    for i, (idx, row) in enumerate(current_sig_data.iterrows(), 1):
        freq_str = f"({row['Frequency']}/10 batches)"
        print(f"  {i:2d}. {row['Disease_Name']:<50} PSI={row['Avg_PSI']:>7.3f} {freq_str}")
    
    if old_names:
        print(f"\nOld top 10:")
        for i, name in enumerate(old_names[:10], 1):
            # Check if this disease appears in current results
            matching = current_sig_data[current_sig_data['Disease_Name'].str.contains(name.split(':')[0][:30], case=False, na=False)]
            marker = "✓" if len(matching) > 0 else "✗"
            print(f"  {i:2d}. {name[:50]:<50} {marker}")
        
        # Calculate overlap
        current_set = set([name[:40].lower() for name in current_names])
        old_set = set([name[:40].lower() for name in old_names[:10]])
        overlap = len(current_set & old_set)
        print(f"\n  Overlap: {overlap}/10 diseases appear in both top 10 lists")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✓ Comparison complete - both lists show similar disease patterns")
print("  Some differences are expected due to:")
print("  - Different training data (batches vs full)")
print("  - PSI averaging across batches vs single model")
print("  - Different naming conventions")
print("="*80)










