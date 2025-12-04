#!/usr/bin/env python3
"""Compare two AUC result files to check consistency"""

import pandas as pd
import numpy as np
import sys

# Load both files
df1 = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_40_70_auc_results_10am.csv')
df2 = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_40_70_auc_results.csv')

# Merge on Age, Offset, Disease
merged = df1.merge(df2, on=['Age', 'Offset', 'Disease'], suffixes=('_10am', '_original'))

# Filter to rows where both have AUC values
merged_valid = merged[(merged['AUC_10am'].notna()) & (merged['AUC_original'].notna())]

print('='*80)
print('CONSISTENCY COMPARISON: Two runs with same pi predictions')
print('='*80)
print(f'\nTotal rows in 10am file: {len(df1)}')
print(f'Total rows in original file: {len(df2)}')
print(f'Rows with matching Age/Offset/Disease: {len(merged)}')
print(f'Rows with valid AUC in both: {len(merged_valid)}')

if len(merged_valid) > 0:
    # Calculate differences
    merged_valid['AUC_diff'] = merged_valid['AUC_10am'] - merged_valid['AUC_original']
    merged_valid['AUC_diff_abs'] = merged_valid['AUC_diff'].abs()
    
    print(f'\nAUC Statistics:')
    print(f'  Mean absolute difference: {merged_valid["AUC_diff_abs"].mean():.6f}')
    print(f'  Median absolute difference: {merged_valid["AUC_diff_abs"].median():.6f}')
    print(f'  Max absolute difference: {merged_valid["AUC_diff_abs"].max():.6f}')
    print(f'  Min absolute difference: {merged_valid["AUC_diff_abs"].min():.6f}')
    print(f'  Std of differences: {merged_valid["AUC_diff"].std():.6f}')
    
    # Count exact matches
    exact_matches = (merged_valid['AUC_diff_abs'] < 1e-10).sum()
    print(f'\nExact matches (diff < 1e-10): {exact_matches}/{len(merged_valid)} ({100*exact_matches/len(merged_valid):.1f}%)')
    
    # Show largest differences
    print(f'\nTop 10 largest absolute differences:')
    top_diff = merged_valid.nlargest(10, 'AUC_diff_abs')[['Age', 'Offset', 'Disease', 'AUC_10am', 'AUC_original', 'AUC_diff_abs']]
    print(top_diff.to_string(index=False))
    
    # Check CI differences
    merged_valid['CI_Lower_diff'] = merged_valid['CI_Lower_10am'] - merged_valid['CI_Lower_original']
    merged_valid['CI_Upper_diff'] = merged_valid['CI_Upper_10am'] - merged_valid['CI_Upper_original']
    
    print(f'\nCI Statistics:')
    print(f'  Mean abs CI_Lower diff: {merged_valid["CI_Lower_diff"].abs().mean():.6f}')
    print(f'  Mean abs CI_Upper diff: {merged_valid["CI_Upper_diff"].abs().mean():.6f}')
    print(f'  Max abs CI_Lower diff: {merged_valid["CI_Lower_diff"].abs().max():.6f}')
    print(f'  Max abs CI_Upper diff: {merged_valid["CI_Upper_diff"].abs().max():.6f}')
    
    # Check N_Events consistency
    events_match = (merged_valid['N_Events_10am'] == merged_valid['N_Events_original']).sum()
    print(f'\nN_Events matches: {events_match}/{len(merged_valid)} ({100*events_match/len(merged_valid):.1f}%)')
    
    if events_match < len(merged_valid):
        print('\nRows with different N_Events:')
        diff_events = merged_valid[merged_valid['N_Events_10am'] != merged_valid['N_Events_original']]
        print(diff_events[['Age', 'Offset', 'Disease', 'N_Events_10am', 'N_Events_original']].head(10).to_string(index=False))
    
    # Check if differences are due to bootstrap sampling
    print(f'\n{"="*80}')
    print('INTERPRETATION:')
    print('='*80)
    if merged_valid['AUC_diff_abs'].max() < 0.001:
        print('✓ Excellent consistency: AUC values match within 0.001')
    elif merged_valid['AUC_diff_abs'].max() < 0.01:
        print('✓ Good consistency: AUC values match within 0.01 (likely due to bootstrap sampling)')
    else:
        print('⚠ Warning: Some AUC differences > 0.01 - may indicate different bootstrap samples')
    
    if merged_valid['CI_Lower_diff'].abs().max() > 0.01 or merged_valid['CI_Upper_diff'].abs().max() > 0.01:
        print('⚠ CI bounds differ significantly - this is expected with bootstrap resampling')
        print('  (Different bootstrap samples will produce different confidence intervals)')

