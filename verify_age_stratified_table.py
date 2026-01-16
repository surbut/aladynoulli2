#!/usr/bin/env python3
"""
Verify age-stratified table values in current.tex against CSV data.
Uses the full population CSV (age_stratified_auc_results.csv) instead of the 10K pivot.
"""

import csv
import re
from pathlib import Path

# File paths
csv_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/age_stratified/pooled_retrospective/age_stratified_auc_results.csv'
latex_file = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/current.tex'

# Disease name mapping: CSV name -> LaTeX display name
disease_mapping = {
    'ASCVD': 'ASCVD',
    'Atrial_Fib': 'Atrial Fibrillation',
    'Breast_Cancer': 'Breast Cancer',
    'Colorectal_Cancer': 'Colorectal Cancer',
    'CKD': 'CKD',
    'Diabetes': 'Diabetes',
    'Heart_Failure': 'Heart Failure',
    'Stroke': 'Stroke',
    'Parkinsons': "Parkinson's",
    'Prostate_Cancer': 'Prostate Cancer',
    'Rheumatoid_Arthritis': 'Rheumatoid Arthritis',
    'Depression': 'Depression',
    'All_Cancers': 'All Cancers',
    'COPD': 'COPD',
}

# Age group mapping: CSV uses 60-72, LaTeX table uses 60-72
age_group_mapping = {'39-50': '39-50', '50-60': '50-60', '60-72': '60-72'}

def parse_latex_table(latex_file):
    """Extract table values from LaTeX file."""
    with open(latex_file, 'r') as f:
        content = f.read()
    
    table_values = {}
    
    # Extract all three age group sections
    age_sections = {
        '39-50': r'\\textbf\{Ages 39--50 years\}.*?\\end\{tabular\}',
        '50-60': r'\\textbf\{Ages 50--60 years\}.*?\\end\{tabular\}',
        '60-72': r'\\textbf\{Ages 60--72 years\}.*?\\end\{tabular\}',
    }
    
    for age_group, pattern in age_sections.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            section = match.group(0)
            # Extract rows - pattern: Disease & value (CI) & value (CI) \\
            # Match lines between \midrule and \bottomrule
            row_pattern = r'([A-Za-z\s\']+?)\s+&\s+([0-9.]+)\s+\([^)]+\)\s+&\s+([0-9.-]+|\-\-)\s*\\\\'
            rows = re.findall(row_pattern, section)
            
            table_values[age_group] = {}
            for disease, val10yr, val1yr in rows:
                disease = disease.strip()
                try:
                    val10yr_float = float(val10yr)
                    if val1yr.strip() == '--':
                        val1yr_float = None
                    else:
                        val1yr_float = float(val1yr)
                    table_values[age_group][disease] = {
                        '10yr': val10yr_float,
                        '1yr': val1yr_float
                    }
                except ValueError:
                    continue
    
    return table_values

def load_csv_data(csv_file):
    """Load data from CSV file."""
    data = {}
    with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
            age = row['Age_Group']
            disease = row['Disease']
            horizon = row['Time_Horizon']
            auc = float(row['AUC'])
            
            if age not in data:
                data[age] = {}
            if disease not in data[age]:
                data[age][disease] = {}
            
            data[age][disease][horizon] = auc
    
    return data

def compare_values():
    """Compare CSV data with LaTeX table values."""
    print("="*80)
    print("VERIFYING AGE-STRATIFIED TABLE VALUES")
    print("="*80)
    print(f"\nCSV file: {csv_file}")
    print(f"LaTeX file: {latex_file}")
    
    # Load data
    print("\nLoading CSV data...")
    csv_data = load_csv_data(csv_file)
    
    print("Parsing LaTeX table...")
    latex_table = parse_latex_table(latex_file)

print("\n" + "="*80)
    print("COMPARISON RESULTS")
print("="*80)
    
    all_match = True
    
    # Compare for each age group
    for age_csv, age_latex in age_group_mapping.items():
        if age_latex not in latex_table:
            print(f"\n⚠ Age group {age_latex} not found in LaTeX table")
            continue
        
        print(f"\n{'='*80}")
        print(f"AGE GROUP: {age_latex}")
        print('='*80)
        
        latex_diseases = latex_table[age_latex]
        
        # Check each disease in the LaTeX table
        for disease_latex, latex_values in latex_diseases.items():
            # Find matching CSV disease
            csv_disease = None
            for csv_name, latex_name in disease_mapping.items():
                if latex_name == disease_latex:
                    csv_disease = csv_name
                    break
            
            if csv_disease is None:
                print(f"\n⚠ {disease_latex}: No CSV mapping found")
                continue
            
            if age_csv not in csv_data or csv_disease not in csv_data[age_csv]:
                print(f"\n⚠ {disease_latex}: Not found in CSV data")
                continue
            
            csv_values = csv_data[age_csv][csv_disease]
            
            # Compare 10-year values
            if '10yr_static' in csv_values:
                csv_10yr = csv_values['10yr_static']
                latex_10yr = latex_values['10yr']
                diff_10yr = abs(csv_10yr - latex_10yr)
                match_10yr = "✓" if diff_10yr < 0.002 else "✗"
                print(f"\n{disease_latex} (10yr):")
                print(f"  CSV: {csv_10yr:.6f}, LaTeX: {latex_10yr:.3f}, Diff: {diff_10yr:.6f} {match_10yr}")
                if diff_10yr >= 0.002:
                    all_match = False
            
            # Compare 1-year values
            if latex_values['1yr'] is not None:
                if '1yr' in csv_values:
                    csv_1yr = csv_values['1yr']
                    latex_1yr = latex_values['1yr']
                    diff_1yr = abs(csv_1yr - latex_1yr)
                    match_1yr = "✓" if diff_1yr < 0.002 else "✗"
                    print(f"  (1yr):")
                    print(f"  CSV: {csv_1yr:.6f}, LaTeX: {latex_1yr:.3f}, Diff: {diff_1yr:.6f} {match_1yr}")
                    if diff_1yr >= 0.002:
                        all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("✓ ALL VALUES MATCH (within 0.002 tolerance)")
    else:
        print("✗ SOME VALUES DO NOT MATCH")
    print("="*80)

if __name__ == '__main__':
    compare_values()
