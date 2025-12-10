#!/usr/bin/env python3
"""
Verify age-stratified table values against CSV files.
"""

import csv
import statistics

# Read pivot table for 1-year medians
pivot_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/age_offset/pooled_retrospective/age_stratified/age_stratified_age_offset_aucs_pivot_batch_0_10000.csv'
# Read 10-year static values
static_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/age_stratified/pooled_retrospective/age_stratified_auc_results.csv'

# Diseases to check
diseases = {
    'ASCVD': 'ASCVD',
    'Heart_Failure': 'Heart Failure',
    'Breast_Cancer': 'Breast Cancer',
    'Colorectal_Cancer': 'Colorectal Cancer',
    'Atrial_Fib': 'Atrial Fibrillation',
    'Stroke': 'Stroke',
    'CKD': 'CKD',
    'COPD': 'COPD'
}

age_groups = ['39-50', '50-60', '60-71']  # Pivot table uses 60-71
age_group_map = {'39-50': '39-50', '50-60': '50-60', '60-71': '60-72'}  # Static CSV uses 60-72

print("="*80)
print("VERIFYING AGE-STRATIFIED TABLE VALUES")
print("="*80)

# Read pivot table
pivot_data = {}
with open(pivot_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        age = row['Age_Group']
        disease = row['Disease']
        if disease not in pivot_data:
            pivot_data[disease] = {}
        if age not in pivot_data[disease]:
            pivot_data[disease][age] = []
        
        # Collect all offset values (0-9)
        for offset in range(10):
            val = row[str(offset)]
            if val and val.strip():
                try:
                    pivot_data[disease][age].append(float(val))
                except ValueError:
                    pass

# Read 10-year static values
static_data = {}
with open(static_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Time_Horizon'] == '10yr_static':
            age = row['Age_Group']
            disease = row['Disease']
            if disease not in static_data:
                static_data[disease] = {}
            static_data[disease][age] = float(row['AUC'])

# Calculate medians and verify
print("\n1-YEAR MEDIANS (from pivot table, offsets 0-9):")
print("-"*80)
for csv_disease, display_name in diseases.items():
    print(f"\n{display_name}:")
    for age in age_groups:
        if csv_disease in pivot_data and age in pivot_data[csv_disease]:
            values = pivot_data[csv_disease][age]
            if len(values) > 0:
                median = statistics.median(values)
                print(f"  {age}: median = {median:.3f} (from {len(values)} values)")
            else:
                print(f"  {age}: No values")
        else:
            print(f"  {age}: Not found")

print("\n\n10-YEAR STATIC VALUES:")
print("-"*80)
for csv_disease, display_name in diseases.items():
    print(f"\n{display_name}:")
    for age in age_groups:
        static_age = age_group_map[age]  # Map to static CSV age group
        if csv_disease in static_data and static_age in static_data[csv_disease]:
            auc = static_data[csv_disease][static_age]
            print(f"  {age} (mapped to {static_age}): {auc:.3f}")
        else:
            print(f"  {age} (mapped to {static_age}): Not found")

print("\n" + "="*80)
print("COMPARISON WITH TABLE VALUES:")
print("="*80)
print("\nCurrent table values vs calculated:")
print("-"*80)

# Table values from LaTeX (what's currently there)
table_values = {
    'ASCVD': {
        '1yr': {'39-50': 0.925, '50-60': 0.910, '60-72': 0.907},
        '10yr': {'39-50': 0.651, '50-60': 0.689, '60-72': 0.707}
    },
    'Heart Failure': {
        '1yr': {'39-50': 0.827, '50-60': 0.801, '60-72': 0.900},
        '10yr': {'39-50': 0.574, '50-60': 0.587, '60-72': 0.614}
    },
    'Breast Cancer': {
        '1yr': {'39-50': 0.928, '50-60': 0.951, '60-72': 0.968},
        '10yr': {'39-50': 0.540, '50-60': 0.556, '60-72': 0.559}
    },
    'Colorectal Cancer': {
        '1yr': {'39-50': 0.743, '50-60': 0.928, '60-72': 0.946},
        '10yr': {'39-50': 0.599, '50-60': 0.565, '60-72': 0.579}
    },
    'Atrial Fibrillation': {
        '1yr': {'39-50': 0.873, '50-60': 0.798, '60-72': 0.851},
        '10yr': {'39-50': 0.580, '50-60': 0.595, '60-72': 0.609}
    },
    'Stroke': {
        '1yr': {'39-50': 0.551, '50-60': 0.592, '60-72': 0.737},
        '10yr': {'39-50': 0.544, '50-60': 0.562, '60-72': 0.603}
    },
    'CKD': {
        '1yr': {'39-50': 0.518, '50-60': 0.718, '60-72': 0.810},
        '10yr': {'39-50': 0.539, '50-60': 0.565, '60-72': 0.600}
    },
    'COPD': {
        '1yr': {'39-50': 0.696, '50-60': 0.664, '60-72': 0.896},
        '10yr': {'39-50': 0.550, '50-60': 0.567, '60-72': 0.575}
    }
}

for display_name, csv_disease in diseases.items():
    print(f"\n{display_name}:")
    # Check 1-year medians
    for age in age_groups:
        if csv_disease in pivot_data and age in pivot_data[csv_disease]:
            values = pivot_data[csv_disease][age]
            if len(values) > 0:
                calculated_median = statistics.median(values)
                table_age = '60-72' if age == '60-71' else age  # Table uses 60-72
                table_median = table_values[display_name]['1yr'][table_age]
                diff = abs(calculated_median - table_median)
                match = "✓" if diff < 0.01 else "✗"
                print(f"  1yr {age} (table={table_age}): Table={table_median:.3f}, Calculated={calculated_median:.3f}, Diff={diff:.3f} {match}")
    
    # Check 10-year static
    for age in age_groups:
        static_age = age_group_map[age]  # Map to static CSV age group
        table_age = '60-72' if age == '60-71' else age  # Table uses 60-72
        if csv_disease in static_data and static_age in static_data[csv_disease]:
            calculated_auc = static_data[csv_disease][static_age]
            table_auc = table_values[display_name]['10yr'][table_age]
            diff = abs(calculated_auc - table_auc)
            match = "✓" if diff < 0.01 else "✗"
            print(f"  10yr {age} (static={static_age}, table={table_age}): Table={table_auc:.3f}, Calculated={calculated_auc:.3f}, Diff={diff:.3f} {match}")

