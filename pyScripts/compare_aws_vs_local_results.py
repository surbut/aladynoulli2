#!/usr/bin/env python3
"""
Compare AWS vs Local results for overlapping age range (40-60 years)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_compare_results():
    """Load both result files and compare overlapping age ranges."""
    
    # Load AWS results (ages 40-59, offsets 40-59)
    aws_df = pd.read_csv('all_offsets_results_fixedphi_from40_59_aws.csv')
    
    # Load local results (ages 40-70, offsets 0-30)
    local_df = pd.read_csv('all_offsets_results_fixedphi_from40_70.csv')
    
    print("AWS results shape:", aws_df.shape)
    print("Local results shape:", local_df.shape)
    
    # AWS file has offsets 40-59 (ages 40-59)
    # Local file has offsets 0-30 (ages 40-70)
    # For comparison, we need to map:
    # AWS offset 40 -> Local offset 0 (both age 40)
    # AWS offset 59 -> Local offset 19 (both age 59)
    
    # Create mapping for comparison
    aws_comparison = aws_df.copy()
    aws_comparison['age'] = aws_comparison['offset']
    aws_comparison['source'] = 'AWS'
    
    local_comparison = local_df[local_df['offset'] <= 19].copy()  # Only offsets 0-19 (ages 40-59)
    local_comparison['age'] = local_comparison['offset'] + 40  # Convert offset to actual age
    local_comparison['source'] = 'Local'
    
    print(f"\nAWS age range: {aws_comparison['age'].min()}-{aws_comparison['age'].max()}")
    print(f"Local comparison age range: {local_comparison['age'].min()}-{local_comparison['age'].max()}")
    
    # Get unique diseases
    diseases = sorted(aws_comparison['Disease'].unique())
    print(f"\nNumber of diseases: {len(diseases)}")
    
    # Compare results for each disease
    comparison_results = []
    
    for disease in diseases:
        aws_data = aws_comparison[aws_comparison['Disease'] == disease]
        local_data = local_comparison[local_comparison['Disease'] == disease]
        
        if len(aws_data) > 0 and len(local_data) > 0:
            # Calculate average AUC for each age group
            aws_avg_auc = aws_data['auc'].mean() if not aws_data['auc'].isna().all() else np.nan
            local_avg_auc = local_data['auc'].mean() if not local_data['auc'].isna().all() else np.nan
            
            # Calculate correlation if we have multiple age points
            if len(aws_data) > 1 and len(local_data) > 1:
                # Align by age
                merged = pd.merge(aws_data, local_data, on='age', suffixes=('_aws', '_local'))
                if len(merged) > 1:
                    correlation = merged[['auc_aws', 'auc_local']].corr().iloc[0, 1]
                else:
                    correlation = np.nan
            else:
                correlation = np.nan
            
            comparison_results.append({
                'Disease': disease,
                'AWS_avg_auc': aws_avg_auc,
                'Local_avg_auc': local_avg_auc,
                'AUC_difference': local_avg_auc - aws_avg_auc if not (pd.isna(aws_avg_auc) or pd.isna(local_avg_auc)) else np.nan,
                'Correlation': correlation,
                'AWS_age_points': len(aws_data),
                'Local_age_points': len(local_data)
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    return aws_comparison, local_comparison, comparison_df

def analyze_comparison(aws_comparison, local_comparison, comparison_df):
    """Analyze the comparison results."""
    
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS: AWS vs Local Results (Ages 40-59)")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  AWS results: {len(aws_comparison)} data points")
    print(f"  Local results: {len(local_comparison)} data points")
    
    # Diseases with significant differences
    significant_diff = comparison_df[comparison_df['AUC_difference'].abs() > 0.05]
    print(f"\nDiseases with AUC difference > 0.05:")
    if len(significant_diff) > 0:
        for _, row in significant_diff.iterrows():
            print(f"  {row['Disease']}: AWS={row['AWS_avg_auc']:.3f}, Local={row['Local_avg_auc']:.3f}, Diff={row['AUC_difference']:.3f}")
    else:
        print("  None found")
    
    # Correlation analysis
    valid_correlations = comparison_df[comparison_df['Correlation'].notna()]
    if len(valid_correlations) > 0:
        print(f"\nCorrelation Analysis:")
        print(f"  Mean correlation: {valid_correlations['Correlation'].mean():.3f}")
        print(f"  Median correlation: {valid_correlations['Correlation'].median():.3f}")
        print(f"  Diseases with correlation > 0.8: {len(valid_correlations[valid_correlations['Correlation'] > 0.8])}")
        print(f"  Diseases with correlation < 0.5: {len(valid_correlations[valid_correlations['Correlation'] < 0.5])}")
    
    # Summary by disease category
    print(f"\nSummary by Disease Category:")
    
    # Cardiovascular diseases
    cardio_diseases = ['ASCVD', 'Stroke', 'Heart_Failure', 'Atrial_Fib']
    cardio_aws = comparison_df[comparison_df['Disease'].isin(cardio_diseases)]['AWS_avg_auc'].mean()
    cardio_local = comparison_df[comparison_df['Disease'].isin(cardio_diseases)]['Local_avg_auc'].mean()
    print(f"  Cardiovascular: AWS={cardio_aws:.3f}, Local={cardio_local:.3f}, Diff={cardio_local-cardio_aws:.3f}")
    
    # Cancer diseases
    cancer_diseases = ['All_Cancers', 'Breast_Cancer', 'Prostate_Cancer', 'Colorectal_Cancer', 'Lung_Cancer']
    cancer_aws = comparison_df[comparison_df['Disease'].isin(cancer_diseases)]['AWS_avg_auc'].mean()
    cancer_local = comparison_df[comparison_df['Disease'].isin(cancer_diseases)]['Local_avg_auc'].mean()
    print(f"  Cancer: AWS={cancer_aws:.3f}, Local={cancer_local:.3f}, Diff={cancer_local-cancer_aws:.3f}")
    
    # Mental health diseases
    mental_diseases = ['Depression', 'Anxiety', 'Bipolar_Disorder']
    mental_aws = comparison_df[comparison_df['Disease'].isin(mental_diseases)]['AWS_avg_auc'].mean()
    mental_local = comparison_df[comparison_df['Disease'].isin(mental_diseases)]['Local_avg_auc'].mean()
    print(f"  Mental Health: AWS={mental_aws:.3f}, Local={mental_local:.3f}, Diff={mental_local-mental_aws:.3f}")
    
    return comparison_df

def create_visualizations(aws_comparison, local_comparison, comparison_df):
    """Create visualizations for the comparison."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AWS vs Local Results Comparison (Ages 40-59)', fontsize=16)
    
    # 1. Scatter plot of AWS vs Local AUCs
    ax1 = axes[0, 0]
    valid_comparison = comparison_df[comparison_df['AWS_avg_auc'].notna() & comparison_df['Local_avg_auc'].notna()]
    ax1.scatter(valid_comparison['AWS_avg_auc'], valid_comparison['Local_avg_auc'], alpha=0.7)
    
    # Add diagonal line
    min_val = min(valid_comparison['AWS_avg_auc'].min(), valid_comparison['Local_avg_auc'].min())
    max_val = max(valid_comparison['AWS_avg_auc'].max(), valid_comparison['Local_avg_auc'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    ax1.set_xlabel('AWS Average AUC')
    ax1.set_ylabel('Local Average AUC')
    ax1.set_title('AWS vs Local AUC Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 2. AUC difference distribution
    ax2 = axes[0, 1]
    valid_diff = comparison_df['AUC_difference'].dropna()
    ax2.hist(valid_diff, bins=15, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('AUC Difference (Local - AWS)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of AUC Differences')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation distribution
    ax3 = axes[1, 0]
    valid_corr = comparison_df['Correlation'].dropna()
    ax3.hist(valid_corr, bins=15, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Correlations')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top diseases by absolute difference
    ax4 = axes[1, 1]
    top_diff = comparison_df[comparison_df['AUC_difference'].notna()].nlargest(10, 'AUC_difference')
    top_diff = top_diff.sort_values('AUC_difference', key=abs, ascending=False)
    
    y_pos = np.arange(len(top_diff))
    ax4.barh(y_pos, top_diff['AUC_difference'])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_diff['Disease'])
    ax4.set_xlabel('AUC Difference (Local - AWS)')
    ax4.set_title('Top 10 Diseases by Absolute AUC Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aws_vs_local_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main analysis function."""
    
    print("Loading and comparing AWS vs Local results...")
    
    # Load and compare results
    aws_comparison, local_comparison, comparison_df = load_and_compare_results()
    
    # Analyze the comparison
    comparison_df = analyze_comparison(aws_comparison, local_comparison, comparison_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = create_visualizations(aws_comparison, local_comparison, comparison_df)
    
    # Save detailed results
    comparison_df.to_csv('aws_vs_local_comparison_results.csv', index=False)
    print(f"\nDetailed results saved to 'aws_vs_local_comparison_results.csv'")
    
    # Summary conclusion
    print("\n" + "="*80)
    print("SUMMARY CONCLUSION")
    print("="*80)
    
    valid_comparison = comparison_df[comparison_df['AUC_difference'].notna()]
    mean_diff = valid_comparison['AUC_difference'].mean()
    std_diff = valid_comparison['AUC_difference'].std()
    
    print(f"Overall, the local results show a mean AUC difference of {mean_diff:.3f} ± {std_diff:.3f}")
    print(f"compared to AWS results for the overlapping age range (40-59 years).")
    
    if abs(mean_diff) < 0.05:
        print("This suggests good agreement between the two implementations.")
    elif abs(mean_diff) < 0.1:
        print("This suggests moderate agreement between the two implementations.")
    else:
        print("This suggests notable differences between the two implementations.")
    
    valid_corr = comparison_df['Correlation'].dropna()
    if len(valid_corr) > 0:
        mean_corr = valid_corr.mean()
        print(f"The mean correlation between AWS and local results is {mean_corr:.3f}.")
        
        if mean_corr > 0.8:
            print("This indicates strong consistency in the temporal patterns.")
        elif mean_corr > 0.6:
            print("This indicates moderate consistency in the temporal patterns.")
        else:
            print("This indicates weak consistency in the temporal patterns.")

if __name__ == "__main__":
    main()
