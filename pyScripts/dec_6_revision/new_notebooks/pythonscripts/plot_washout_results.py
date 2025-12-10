#!/usr/bin/env python3
"""
Plot washout analysis results showing performance across different washout windows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Load data
data_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/washout_comparison_1yr_10yr.csv'
df = pd.read_csv(data_path, index_col=0)

# Key diseases to highlight
key_diseases = ['ASCVD', 'Diabetes', 'Heart_Failure', 'Breast_Cancer', 'Atrial_Fib', 
                'Stroke', 'COPD', 'Colorectal_Cancer', 'All_Cancers']

# Filter to key diseases
df_plot = df.loc[df.index.isin(key_diseases)].copy()

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Washout Analysis: Performance Impact of Removing Events Within 1-6 Months of Enrollment', 
             fontsize=14, fontweight='bold', y=0.995)

# 1-year predictions
ax1 = axes[0]
washout_periods = ['No Washout', '1 Month', '3 Months', '6 Months']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

x_pos = np.arange(len(key_diseases))
width = 0.2

for i, period in enumerate(['no_washout', '1month', '3month', '6month']):
    aucs = df_plot[f'{period}_1yr_AUC'].values
    aucs_clean = [x if not pd.isna(x) else 0 for x in aucs]
    ax1.bar(x_pos + i*width, aucs_clean, width, label=washout_periods[i], 
            color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Disease', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUC (1-Year Predictions)', fontsize=12, fontweight='bold')
ax1.set_title('1-Year Predictions: Small Performance Drops with Washout', 
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(key_diseases, rotation=45, ha='right')
ax1.set_ylim([0.4, 1.0])
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 10-year predictions
ax2 = axes[1]
for i, period in enumerate(['no_washout', '1month', '3month', '6month']):
    aucs = df_plot[f'{period}_10yr_AUC'].values
    aucs_clean = [x if not pd.isna(x) else 0 for x in aucs]
    ax2.bar(x_pos + i*width, aucs_clean, width, label=washout_periods[i], 
            color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Disease', fontsize=12, fontweight='bold')
ax2.set_ylabel('AUC (10-Year Predictions)', fontsize=12, fontweight='bold')
ax2.set_title('10-Year Predictions: Minimal Performance Impact with Washout', 
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(x_pos + width * 1.5)
ax2.set_xticklabels(key_diseases, rotation=45, ha='right')
ax2.set_ylim([0.4, 0.8])
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()

# Save figure
output_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/washout_performance_plot.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: {output_path}")

# Also create a summary table plot
fig2, ax = plt.subplots(figsize=(12, 8))

# Calculate drops from no_washout to 6month
summary_data = []
for disease in key_diseases:
    if disease in df_plot.index:
        auc_1yr_no = df_plot.loc[disease, 'no_washout_1yr_AUC']
        auc_1yr_6m = df_plot.loc[disease, '6month_1yr_AUC']
        auc_10yr_no = df_plot.loc[disease, 'no_washout_10yr_AUC']
        auc_10yr_6m = df_plot.loc[disease, '6month_10yr_AUC']
        
        if not (pd.isna(auc_1yr_no) or pd.isna(auc_1yr_6m)):
            drop_1yr = auc_1yr_no - auc_1yr_6m
        else:
            drop_1yr = np.nan
            
        if not (pd.isna(auc_10yr_no) or pd.isna(auc_10yr_6m)):
            drop_10yr = auc_10yr_no - auc_10yr_6m
        else:
            drop_10yr = np.nan
            
        summary_data.append({
            'Disease': disease,
            '1-Year Drop': drop_1yr,
            '10-Year Drop': drop_10yr,
            '1-Year AUC (No Washout)': auc_1yr_no,
            '1-Year AUC (6-Month Washout)': auc_1yr_6m,
            '10-Year AUC (No Washout)': auc_10yr_no,
            '10-Year AUC (6-Month Washout)': auc_10yr_6m
        })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('1-Year Drop', ascending=False)

# Create bar plot of drops
x_pos = np.arange(len(summary_df))
width = 0.35

bars1 = ax.bar(x_pos - width/2, summary_df['1-Year Drop'], width, 
               label='1-Year Predictions', color='#C73E1D', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, summary_df['10-Year Drop'], width, 
               label='10-Year Predictions', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Disease', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC Drop (No Washout → 6-Month Washout)', fontsize=12, fontweight='bold')
ax.set_title('Performance Impact of 6-Month Washout Window\n(Small Drops Indicate Robustness to Reverse Causation)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(summary_df['Disease'], rotation=45, ha='right')
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()

output_path2 = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/washout_drop_summary.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ Saved summary plot to: {output_path2}")

# Print summary statistics
print(f"\n{'='*80}")
print("WASHOUT IMPACT SUMMARY")
print(f"{'='*80}")
print(f"\n1-Year Predictions (No Washout → 6-Month Washout):")
print(f"  Mean AUC drop: {summary_df['1-Year Drop'].mean():.4f}")
print(f"  Median AUC drop: {summary_df['1-Year Drop'].median():.4f}")
print(f"  Max AUC drop: {summary_df['1-Year Drop'].max():.4f} ({summary_df.loc[summary_df['1-Year Drop'].idxmax(), 'Disease']})")
print(f"  Min AUC drop: {summary_df['1-Year Drop'].min():.4f} ({summary_df.loc[summary_df['1-Year Drop'].idxmin(), 'Disease']})")

print(f"\n10-Year Predictions (No Washout → 6-Month Washout):")
print(f"  Mean AUC drop: {summary_df['10-Year Drop'].mean():.4f}")
print(f"  Median AUC drop: {summary_df['10-Year Drop'].median():.4f}")
print(f"  Max AUC drop: {summary_df['10-Year Drop'].max():.4f} ({summary_df.loc[summary_df['10-Year Drop'].idxmax(), 'Disease']})")
print(f"  Min AUC drop: {summary_df['10-Year Drop'].min():.4f} ({summary_df.loc[summary_df['10-Year Drop'].idxmin(), 'Disease']})")

print(f"\n✓ All plots saved successfully!")
plt.show()

