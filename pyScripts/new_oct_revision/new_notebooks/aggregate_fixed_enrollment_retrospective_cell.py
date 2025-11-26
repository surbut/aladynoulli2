# ===== AGGREGATE AND SAVE RESULTS FOR FIXED ENROLLMENT & RETROSPECTIVE =====
# Paste this cell right after line 158 in lifetime.ipynb (after the print statements)

print(f"\n{'='*80}")
print("AGGREGATING RESULTS ACROSS ALL 40 BATCHES")
print("(Fixed Enrollment & Fixed Retrospective: 10yr, 30yr, Static)")
print(f"{'='*80}")

def aggregate_results_to_dataframe(results_list, analysis_name):
    """
    Aggregate results across batches into a DataFrame.
    Each result is a dict with disease names as keys and metrics as values.
    """
    if not results_list:
        print(f"Warning: No results found for {analysis_name}")
        return pd.DataFrame()
    
    # Get all disease names (excluding metadata keys)
    disease_names_list = [k for k in results_list[0].keys() 
                         if k not in ['batch_idx', 'analysis_type']]
    
    # Collect all metrics across batches
    aggregated_data = []
    for disease in disease_names_list:
        aucs = []
        ci_lowers = []
        ci_uppers = []
        n_events_list = []
        event_rates = []
        
        for result in results_list:
            if disease in result and isinstance(result[disease], dict):
                if 'auc' in result[disease] and not np.isnan(result[disease]['auc']):
                    aucs.append(result[disease]['auc'])
                if 'ci_lower' in result[disease] and not np.isnan(result[disease]['ci_lower']):
                    ci_lowers.append(result[disease]['ci_lower'])
                if 'ci_upper' in result[disease] and not np.isnan(result[disease]['ci_upper']):
                    ci_uppers.append(result[disease]['ci_upper'])
                if 'n_events' in result[disease]:
                    n_events_list.append(result[disease]['n_events'])
                if 'event_rate' in result[disease] and result[disease]['event_rate'] is not None:
                    event_rates.append(result[disease]['event_rate'])
        
        if aucs:  # Only add if we have at least one valid AUC
            aggregated_data.append({
                'Disease': disease,
                'AUC_median': np.median(aucs),
                'AUC_mean': np.mean(aucs),
                'AUC_std': np.std(aucs),
                'AUC_min': np.min(aucs),
                'AUC_max': np.max(aucs),
                'CI_lower_median': np.median(ci_lowers) if ci_lowers else np.nan,
                'CI_upper_median': np.median(ci_uppers) if ci_uppers else np.nan,
                'CI_lower_min': np.min(ci_lowers) if ci_lowers else np.nan,
                'CI_upper_max': np.max(ci_uppers) if ci_uppers else np.nan,
                'Total_Events': np.sum(n_events_list) if n_events_list else np.nan,
                'Mean_Event_Rate': np.mean(event_rates) if event_rates else np.nan,
                'N_Batches': len(aucs)
            })
    
    df = pd.DataFrame(aggregated_data)
    if not df.empty:
        df = df.set_index('Disease').sort_values('AUC_median', ascending=False)
    return df

# Aggregate all 6 result lists
print("\nAggregating Fixed Enrollment results...")
fixed_enrollment_10yr_df = aggregate_results_to_dataframe(fixed_enrollment_10yr_results, "Fixed Enrollment 10yr")
fixed_enrollment_30yr_df = aggregate_results_to_dataframe(fixed_enrollment_30yr_results, "Fixed Enrollment 30yr")
fixed_enrollment_static_10yr_df = aggregate_results_to_dataframe(fixed_enrollment_static_10yr_results, "Fixed Enrollment Static 10yr")

print("Aggregating Fixed Retrospective results...")
fixed_retrospective_10yr_df = aggregate_results_to_dataframe(fixed_retrospective_10yr_results, "Fixed Retrospective 10yr")
fixed_retrospective_30yr_df = aggregate_results_to_dataframe(fixed_retrospective_30yr_results, "Fixed Retrospective 30yr")
fixed_retrospective_static_10yr_df = aggregate_results_to_dataframe(fixed_retrospective_static_10yr_results, "Fixed Retrospective Static 10yr")

# Save individual DataFrames
output_dir = '/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/'
print(f"\nSaving aggregated results to {output_dir}...")

fixed_enrollment_10yr_df.to_csv(f'{output_dir}pooled_fixed_enrollment_10yr.csv')
fixed_enrollment_30yr_df.to_csv(f'{output_dir}pooled_fixed_enrollment_30yr.csv')
fixed_enrollment_static_10yr_df.to_csv(f'{output_dir}pooled_fixed_enrollment_static_10yr.csv')

fixed_retrospective_10yr_df.to_csv(f'{output_dir}pooled_fixed_retrospective_10yr.csv')
fixed_retrospective_30yr_df.to_csv(f'{output_dir}pooled_fixed_retrospective_30yr.csv')
fixed_retrospective_static_10yr_df.to_csv(f'{output_dir}pooled_fixed_retrospective_static_10yr.csv')

print("✓ Saved individual result files")

# Create a combined comparison DataFrame
print("\nCreating combined comparison DataFrame...")
all_diseases = set()
for df in [fixed_enrollment_10yr_df, fixed_enrollment_30yr_df, fixed_enrollment_static_10yr_df, 
           fixed_retrospective_10yr_df, fixed_retrospective_30yr_df, fixed_retrospective_static_10yr_df]:
    if not df.empty:
        all_diseases.update(df.index)

comparison_df = pd.DataFrame(index=sorted(all_diseases))
comparison_df['Fixed_Enrollment_10yr'] = fixed_enrollment_10yr_df['AUC_median']
comparison_df['Fixed_Enrollment_30yr'] = fixed_enrollment_30yr_df['AUC_median']
comparison_df['Fixed_Enrollment_Static_10yr'] = fixed_enrollment_static_10yr_df['AUC_median']
comparison_df['Fixed_Retrospective_10yr'] = fixed_retrospective_10yr_df['AUC_median']
comparison_df['Fixed_Retrospective_30yr'] = fixed_retrospective_30yr_df['AUC_median']
comparison_df['Fixed_Retrospective_Static_10yr'] = fixed_retrospective_static_10yr_df['AUC_median']

comparison_df.to_csv(f'{output_dir}pooled_comparison_fixed_enrollment_vs_retrospective.csv')
print("✓ Saved combined comparison file: pooled_comparison_fixed_enrollment_vs_retrospective.csv")

# Print summary
print(f"\n{'='*80}")
print("SUMMARY OF AGGREGATED RESULTS (40 batches)")
print(f"{'='*80}")
print(f"\nFixed Enrollment:")
print(f"  10yr: {len(fixed_enrollment_10yr_df)} diseases, {len(fixed_enrollment_10yr_results)} batches")
print(f"  30yr: {len(fixed_enrollment_30yr_df)} diseases, {len(fixed_enrollment_30yr_results)} batches")
print(f"  Static 10yr: {len(fixed_enrollment_static_10yr_df)} diseases, {len(fixed_enrollment_static_10yr_results)} batches")
print(f"\nFixed Retrospective:")
print(f"  10yr: {len(fixed_retrospective_10yr_df)} diseases, {len(fixed_retrospective_10yr_results)} batches")
print(f"  30yr: {len(fixed_retrospective_30yr_df)} diseases, {len(fixed_retrospective_30yr_results)} batches")
print(f"  Static 10yr: {len(fixed_retrospective_static_10yr_df)} diseases, {len(fixed_retrospective_static_10yr_results)} batches")

print(f"\n{'='*80}")
print("TOP 10 DISEASES BY AUC (Fixed Enrollment Static 10yr)")
print(f"{'='*80}")
if not fixed_enrollment_static_10yr_df.empty:
    print(fixed_enrollment_static_10yr_df[['AUC_median', 'CI_lower_median', 'CI_upper_median', 'N_Batches']].head(10).round(4))

print(f"\n{'='*80}")
print("TOP 10 DISEASES BY AUC (Fixed Retrospective Static 10yr)")
print(f"{'='*80}")
if not fixed_retrospective_static_10yr_df.empty:
    print(fixed_retrospective_static_10yr_df[['AUC_median', 'CI_lower_median', 'CI_upper_median', 'N_Batches']].head(10).round(4))

print(f"\n{'='*80}")
print("COMPARISON: Fixed Enrollment vs Fixed Retrospective (Static 10yr)")
print(f"{'='*80}")
if not fixed_enrollment_static_10yr_df.empty and not fixed_retrospective_static_10yr_df.empty:
    static_comparison = pd.DataFrame({
        'Fixed_Enrollment_Static_10yr': fixed_enrollment_static_10yr_df['AUC_median'],
        'Fixed_Retrospective_Static_10yr': fixed_retrospective_static_10yr_df['AUC_median'],
        'Difference': fixed_retrospective_static_10yr_df['AUC_median'] - fixed_enrollment_static_10yr_df['AUC_median']
    }).sort_values('Difference', ascending=False)
    print(static_comparison.round(4))
    print(f"\nMean difference (Retrospective - Enrollment): {static_comparison['Difference'].mean():.4f}")
    print(f"Diseases where Retrospective > Enrollment: {(static_comparison['Difference'] > 0).sum()} / {len(static_comparison)}")

print(f"\n{'='*80}")
print("All results saved successfully!")
print(f"{'='*80}")

