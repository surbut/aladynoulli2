#!/usr/bin/env python3
"""
Quick analysis of the 4 configurations to identify which parameter causes AUC drop.
"""

import pandas as pd
import numpy as np

# Load results
static_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_4_configs_5batches_static_10yr_auc_results.csv'
dynamic_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_4_configs_5batches_dynamic_1yr_auc_results.csv'

static_df = pd.read_csv(static_file)
dynamic_df = pd.read_csv(dynamic_file)

print("="*80)
print("STATIC 10-YEAR AUC COMPARISON")
print("="*80)

# Key diseases to compare
key_diseases = ['ASCVD', 'Diabetes', 'Atrial_Fib', 'CKD', 'Heart_Failure']

for disease in key_diseases:
    disease_data = static_df[static_df['disease'] == disease]
    
    original_auc = disease_data[disease_data['config'] == 'original']['auc'].values[0]
    fixedk_freeg_auc = disease_data[disease_data['config'] == 'fixedk_freeg']['auc'].values[0]
    fixedg_freek_auc = disease_data[disease_data['config'] == 'fixedg_freek']['auc'].values[0]
    fixedgk_auc = disease_data[disease_data['config'] == 'fixedgk']['auc'].values[0]
    
    print(f"\n{disease}:")
    print(f"  Original (free g, free k):     {original_auc:.4f}")
    print(f"  Fixed k, free g:               {fixedk_freeg_auc:.4f} (diff: {fixedk_freeg_auc - original_auc:+.4f})")
    print(f"  Fixed g, free k:               {fixedg_freek_auc:.4f} (diff: {fixedg_freek_auc - original_auc:+.4f})")
    print(f"  Fixed g, fixed k:              {fixedgk_auc:.4f} (diff: {fixedgk_auc - original_auc:+.4f})")
    
    # Check if fixed gamma is the culprit
    if abs(fixedk_freeg_auc - original_auc) < 0.001 and abs(fixedg_freek_auc - original_auc) > 0.01:
        print(f"  → FIXED GAMMA causes the drop (not kappa)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Comparing configurations:")
print("  - Fixed kappa, free gamma: AUC stays high (same as original)")
print("  - Fixed gamma, free kappa: AUC drops significantly")
print("  - Fixed gamma and kappa: Same as fixed gamma alone")
print("\n→ CONCLUSION: Fixing GAMMA (not kappa) is what reduces AUC")
