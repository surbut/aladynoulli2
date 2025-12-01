# ===== AGGREGATE AND SAVE RESULTS FOR JOINT AND OLD FIXED APPROACHES =====
# Paste this cell after your loop completes (after line 147 in lifetime.ipynb)

print(f"\n{'='*80}")
print("AGGREGATING RESULTS ACROSS ALL BATCHES (JOINT & OLD FIXED)")
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

# Aggregate all result lists
print("\nAggregating Joint Phi results...")
joint_10yr_df = aggregate_results_to_dataframe(joint_10yr_results, "Joint 10yr")
joint_30yr_df = aggregate_results_to_dataframe(joint_30yr_results, "Joint 30yr")
joint_static_10yr_df = aggregate_results_to_dataframe(joint_static_10yr_results, "Joint Static 10yr")

print("Aggregating Old Fixed Phi results...")
fixed_10yr_df = aggregate_results_to_dataframe(fixed_10yr_results, "Fixed 10yr")
fixed_30yr_df = aggregate_results_to_dataframe(fixed_30yr_results, "Fixed 30yr")
fixed_static_10yr_df = aggregate_results_to_dataframe(fixed_static_10yr_results, "Fixed Static 10yr")

# Save individual DataFrames
output_dir = '/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/'
print(f"\nSaving aggregated results to {output_dir}...")

joint_10yr_df.to_csv(f'{output_dir}pooled_joint_10yr.csv')
joint_30yr_df.to_csv(f'{output_dir}pooled_joint_30yr.csv')
joint_static_10yr_df.to_csv(f'{output_dir}pooled_joint_static_10yr.csv')

fixed_10yr_df.to_csv(f'{output_dir}pooled_old_fixed_10yr.csv')
fixed_30yr_df.to_csv(f'{output_dir}pooled_old_fixed_30yr.csv')
fixed_static_10yr_df.to_csv(f'{output_dir}pooled_old_fixed_static_10yr.csv')

print("✓ Saved individual result files")

# Create a combined comparison DataFrame
print("\nCreating combined comparison DataFrame...")
all_diseases = set()
for df in [joint_10yr_df, joint_30yr_df, joint_static_10yr_df, 
           fixed_10yr_df, fixed_30yr_df, fixed_static_10yr_df]:
    if not df.empty:
        all_diseases.update(df.index)

comparison_df = pd.DataFrame(index=sorted(all_diseases))
comparison_df['Joint_10yr'] = joint_10yr_df['AUC_median']
comparison_df['Joint_30yr'] = joint_30yr_df['AUC_median']
comparison_df['Joint_Static_10yr'] = joint_static_10yr_df['AUC_median']
comparison_df['Old_Fixed_10yr'] = fixed_10yr_df['AUC_median']
comparison_df['Old_Fixed_30yr'] = fixed_30yr_df['AUC_median']
comparison_df['Old_Fixed_Static_10yr'] = fixed_static_10yr_df['AUC_median']

comparison_df.to_csv(f'{output_dir}pooled_joint_and_old_fixed_comparison.csv')
print("✓ Saved combined comparison file: pooled_joint_and_old_fixed_comparison.csv")

# Print summary
print(f"\n{'='*80}")
print("SUMMARY OF AGGREGATED RESULTS")
print(f"{'='*80}")
print(f"\nJoint - 10yr: {len(joint_10yr_df)} diseases")
print(f"Joint - 30yr: {len(joint_30yr_df)} diseases")
print(f"Joint - Static 10yr: {len(joint_static_10yr_df)} diseases")
print(f"Old Fixed - 10yr: {len(fixed_10yr_df)} diseases")
print(f"Old Fixed - 30yr: {len(fixed_30yr_df)} diseases")
print(f"Old Fixed - Static 10yr: {len(fixed_static_10yr_df)} diseases")

print(f"\n{'='*80}")
print("TOP 10 DISEASES BY AUC (Joint 10yr)")
print(f"{'='*80}")
if not joint_10yr_df.empty:
    print(joint_10yr_df[['AUC_median', 'CI_lower_median', 'CI_upper_median', 'N_Batches']].head(10).round(4))

print(f"\n{'='*80}")
print("TOP 10 DISEASES BY AUC (Old Fixed 10yr)")
print(f"{'='*80}")
if not fixed_10yr_df.empty:
    print(fixed_10yr_df[['AUC_median', 'CI_lower_median', 'CI_upper_median', 'N_Batches']].head(10).round(4))

print(f"\n{'='*80}")
print("COMPARISON: Joint vs Old Fixed (Static 10yr)")
print(f"{'='*80}")
if not joint_static_10yr_df.empty and not fixed_static_10yr_df.empty:
    static_comparison = pd.DataFrame({
        'Joint_Static_10yr': joint_static_10yr_df['AUC_median'],
        'Old_Fixed_Static_10yr': fixed_static_10yr_df['AUC_median'],
        'Difference': fixed_static_10yr_df['AUC_median'] - joint_static_10yr_df['AUC_median']
    }).sort_values('Difference', ascending=False)
    print(static_comparison.round(4))
    print(f"\nMean difference (Old Fixed - Joint): {static_comparison['Difference'].mean():.4f}")
    print(f"Diseases where Old Fixed > Joint: {(static_comparison['Difference'] > 0).sum()} / {len(static_comparison)}")

print(f"\n{'='*80}")
print("All results saved successfully!")
print(f"{'='*80}")

