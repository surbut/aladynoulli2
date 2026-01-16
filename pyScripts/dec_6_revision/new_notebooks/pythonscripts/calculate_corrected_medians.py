#!/usr/bin/env python3
"""
Calculate corrected medians using 0_global (from washout_0yr) + offsets 1-9.
"""

import csv
from pathlib import Path

# Read the file with 0_global
input_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/age_offset/pooled_retrospective/age_offset_aucs_pivot_batch_0_10000_with_global0.csv')

print("="*80)
print("CALCULATING CORRECTED MEDIANS")
print("="*80)

medians = {}

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        disease = row['Disease']
        
        # Get 0_global value
        val_0_global = row['0_global'] if row['0_global'] else None
        
        # Get offsets 1-9
        values = []
        if val_0_global:
            try:
                values.append(float(val_0_global))
            except (ValueError, TypeError):
                pass
        
        for offset in range(1, 10):
            val = row[str(offset)]
            if val:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        
        if len(values) > 0:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            if n % 2 == 1:
                median = sorted_vals[n // 2]
            else:
                median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            medians[disease] = median
            print(f"{disease:<30} Median: {median:.4f} (from {n} values)")

print("\n" + "="*80)
print("KEY DISEASES FOR LATEX UPDATE:")
print("="*80)
key_diseases = ['Diabetes', 'Breast_Cancer', 'ASCVD', 'Heart_Failure', 'Atrial_Fib']
for disease in key_diseases:
    if disease in medians:
        print(f"{disease:<30} Median: {medians[disease]:.4f}")

# Save to CSV
output_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/age_offset/pooled_retrospective/medians_with_global0.csv')
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Disease', 'Median_with_global0'])
    for disease, median in sorted(medians.items()):
        writer.writerow([disease, median])

print(f"\n✓ Saved medians to: {output_file}")













