# ============================================================================
# SAVE LOO VALIDATION RESULTS TO CSV AND CREATE VISUALIZATIONS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Extract all differences from the comparison results
def extract_differences(loo_aucs, full_aucs, prediction_type):
    """Extract differences between LOO and Full Pooled AUCs"""
    all_differences = []
    
    for batch_idx in sorted(set(list(loo_aucs.keys()) & list(full_aucs.keys()))):
        loo_batch = loo_aucs.get(batch_idx, {})
        full_batch = full_aucs.get(batch_idx, {})
        
        common_diseases = set(loo_batch.keys()) & set(full_batch.keys())
        
        for disease in common_diseases:
            loo_auc = loo_batch[disease]
            full_auc = full_batch[disease]
            diff = abs(loo_auc - full_auc)
            
            all_differences.append({
                'batch_idx': batch_idx,
                'disease': disease,
                'loo_auc': loo_auc,
                'full_pooled_auc': full_auc,
                'difference': diff,
                'prediction_type': prediction_type
            })
    
    return all_differences

# Extract differences for all prediction types
all_diffs = []
all_diffs.extend(extract_differences(loo_10yr_aucs, full_10yr_aucs, '10-Year'))
all_diffs.extend(extract_differences(loo_30yr_aucs, full_30yr_aucs, '30-Year'))
all_diffs.extend(extract_differences(loo_static_10yr_aucs, full_static_10yr_aucs, 'Static 10-Year'))

# Create DataFrame
df_loo = pd.DataFrame(all_diffs)

# Save to CSV
csv_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/loo_validation_results.csv'
df_loo.to_csv(csv_path, index=False)
print(f"✓ Saved LOO validation results to: {csv_path}")
print(f"  Total comparisons: {len(df_loo)}")
print(f"  Batches: {sorted(df_loo['batch_idx'].unique())}")
print(f"  Prediction types: {df_loo['prediction_type'].unique()}")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(16, 10))

# 1. Distribution of differences by prediction type
ax1 = plt.subplot(2, 3, 1)
for pred_type in df_loo['prediction_type'].unique():
    diffs = df_loo[df_loo['prediction_type'] == pred_type]['difference'] * 1000
    ax1.hist(diffs, bins=30, alpha=0.6, label=pred_type, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('AUC Difference (×1000)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Differences\nby Prediction Type', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, linestyle='--')

# 2. Box plot of differences by prediction type
ax2 = plt.subplot(2, 3, 2)
df_loo['difference_x1000'] = df_loo['difference'] * 1000
sns.boxplot(data=df_loo, x='prediction_type', y='difference_x1000', ax=ax2, palette='Set2')
ax2.set_xlabel('Prediction Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('AUC Difference (×1000)', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Differences\n(Box Plot)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 3. Max difference per batch
ax3 = plt.subplot(2, 3, 3)
batch_max_diffs = df_loo.groupby(['batch_idx', 'prediction_type'])['difference'].max().reset_index()
for pred_type in df_loo['prediction_type'].unique():
    batch_data = batch_max_diffs[batch_max_diffs['prediction_type'] == pred_type]
    ax3.plot(batch_data['batch_idx'], batch_data['difference'] * 1000, 
             marker='o', linewidth=2, markersize=8, label=pred_type)

ax3.set_xlabel('Excluded Batch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Max AUC Difference (×1000)', fontsize=11, fontweight='bold')
ax3.set_title('Max Difference per Excluded Batch', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_xticks(sorted(df_loo['batch_idx'].unique()))

# 4. Summary statistics bar chart
ax4 = plt.subplot(2, 3, 4)
summary_stats = df_loo.groupby('prediction_type')['difference'].agg(['mean', 'max', 'std']).reset_index()
x = np.arange(len(summary_stats))
width = 0.25

bars1 = ax4.bar(x - width, summary_stats['mean'] * 1000, width, label='Mean', color='#2E86AB', alpha=0.8)
bars2 = ax4.bar(x, summary_stats['max'] * 1000, width, label='Max', color='#A23B72', alpha=0.8)
bars3 = ax4.bar(x + width, summary_stats['std'] * 1000, width, label='Std', color='#F18F01', alpha=0.8)

ax4.set_ylabel('AUC Difference (×1000)', fontsize=11, fontweight='bold')
ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(summary_stats['prediction_type'], rotation=15, ha='right')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

# 5. Percentage within thresholds
ax5 = plt.subplot(2, 3, 5)
thresholds = [0.001, 0.005, 0.01, 0.05]
threshold_data = []
for pred_type in df_loo['prediction_type'].unique():
    pred_diffs = df_loo[df_loo['prediction_type'] == pred_type]['difference']
    total = len(pred_diffs)
    for thresh in thresholds:
        within = (pred_diffs < thresh).sum()
        threshold_data.append({
            'prediction_type': pred_type,
            'threshold': f'{thresh*1000:.1f}',
            'percentage': within / total * 100
        })

df_thresh = pd.DataFrame(threshold_data)
pivot_thresh = df_thresh.pivot(index='prediction_type', columns='threshold', values='percentage')

sns.heatmap(pivot_thresh, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax5, 
            cbar_kws={'label': '% Within Threshold'}, vmin=95, vmax=100)
ax5.set_xlabel('Threshold (×1000)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Prediction Type', fontsize=11, fontweight='bold')
ax5.set_title('Percentage Within Thresholds', fontsize=12, fontweight='bold')

# 6. Scatter plot: LOO vs Full Pooled AUCs
ax6 = plt.subplot(2, 3, 6)
for pred_type in df_loo['prediction_type'].unique():
    pred_data = df_loo[df_loo['prediction_type'] == pred_type]
    ax6.scatter(pred_data['full_pooled_auc'], pred_data['loo_auc'], 
               alpha=0.5, s=30, label=pred_type)

# Add diagonal line
min_val = min(df_loo['loo_auc'].min(), df_loo['full_pooled_auc'].min())
max_val = max(df_loo['loo_auc'].max(), df_loo['full_pooled_auc'].max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='y=x')

ax6.set_xlabel('Full Pooled AUC', fontsize=11, fontweight='bold')
ax6.set_ylabel('Leave-One-Out AUC', fontsize=11, fontweight='bold')
ax6.set_title('LOO vs Full Pooled AUCs\n(Perfect match = diagonal)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9, loc='lower right')
ax6.grid(alpha=0.3, linestyle='--')

plt.suptitle('Leave-One-Out Validation: Robustness of Pooled Phi Approach', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save figure
fig_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/loo_validation_plot.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved figure to: {fig_path}")

plt.show()

# Print summary
print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("="*80)
for pred_type in df_loo['prediction_type'].unique():
    pred_data = df_loo[df_loo['prediction_type'] == pred_type]
    print(f"\n{pred_type}:")
    print(f"  Mean difference: {pred_data['difference'].mean()*1000:.3f} (×1000)")
    print(f"  Max difference: {pred_data['difference'].max()*1000:.3f} (×1000)")
    print(f"  Std difference: {pred_data['difference'].std()*1000:.3f} (×1000)")
    print(f"  Median difference: {pred_data['difference'].median()*1000:.3f} (×1000)")
    print(f"  Comparisons < 0.001: {(pred_data['difference'] < 0.001).sum()}/{len(pred_data)} ({(pred_data['difference'] < 0.001).sum()/len(pred_data)*100:.1f}%)")
    print(f"  Comparisons < 0.01: {(pred_data['difference'] < 0.01).sum()}/{len(pred_data)} ({(pred_data['difference'] < 0.01).sum()/len(pred_data)*100:.1f}%)")
print("\n" + "="*80)




