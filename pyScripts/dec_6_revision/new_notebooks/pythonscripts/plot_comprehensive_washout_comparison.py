#!/usr/bin/env python
"""
Create comprehensive washout comparison plot showing 1-year and 10-year AUCs
for all washout periods: 0, 1mo, 3mo, 6mo, 1yr, 2yr, 3yr
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Load data
monthly_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/washout_comparison_1yr_10yr.csv'
yearly_file = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/washout_comparison_1yr_2yr_3yr_vs_baseline.csv'

df_monthly = pd.read_csv(monthly_file, index_col=0)
df_yearly = pd.read_csv(yearly_file, index_col=0)

# Extract AUC columns
diseases = df_monthly.index.tolist()

# Prepare data for plotting
plot_data = []

for disease in diseases:
    if disease not in df_yearly.index:
        continue
    
    # 1-year AUCs
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '0 (baseline)',
        'AUC': df_monthly.loc[disease, 'no_washout_1yr_AUC'],
        'Order': 0
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '1 month',
        'AUC': df_monthly.loc[disease, '1month_1yr_AUC'],
        'Order': 1
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '3 months',
        'AUC': df_monthly.loc[disease, '3month_1yr_AUC'],
        'Order': 2
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '6 months',
        'AUC': df_monthly.loc[disease, '6month_1yr_AUC'],
        'Order': 3
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '1 year',
        'AUC': df_yearly.loc[disease, '1yr_1yr_AUC'],
        'Order': 4
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '2 years',
        'AUC': df_yearly.loc[disease, '2yr_1yr_AUC'],
        'Order': 5
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '1-year',
        'Washout': '3 years',
        'AUC': df_yearly.loc[disease, '3yr_1yr_AUC'],
        'Order': 6
    })
    
    # 10-year AUCs
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '0 (baseline)',
        'AUC': df_monthly.loc[disease, 'no_washout_10yr_AUC'],
        'Order': 0
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '1 month',
        'AUC': df_monthly.loc[disease, '1month_10yr_AUC'],
        'Order': 1
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '3 months',
        'AUC': df_monthly.loc[disease, '3month_10yr_AUC'],
        'Order': 2
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '6 months',
        'AUC': df_monthly.loc[disease, '6month_10yr_AUC'],
        'Order': 3
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '1 year',
        'AUC': df_yearly.loc[disease, '1yr_10yr_AUC'],
        'Order': 4
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '2 years',
        'AUC': df_yearly.loc[disease, '2yr_10yr_AUC'],
        'Order': 5
    })
    plot_data.append({
        'Disease': disease,
        'Horizon': '10-year',
        'Washout': '3 years',
        'AUC': df_yearly.loc[disease, '3yr_10yr_AUC'],
        'Order': 6
    })

df_plot = pd.DataFrame(plot_data)

# Focus on key diseases
key_diseases = ['ASCVD', 'All_Cancers', 'Atrial_Fib', 'Diabetes', 'CKD', 'Heart_Failure', 
                'Breast_Cancer', 'Colorectal_Cancer', 'Lung_Cancer', 'Parkinsons']  # 10 key diseases

# Filter to key diseases and check for missing data
valid_diseases = []
for disease in key_diseases:
    if disease in df_yearly.index and disease in df_monthly.index:
        # Check if all values are present
        if (not pd.isna(df_yearly.loc[disease, '1yr_1yr_AUC']) and
            not pd.isna(df_yearly.loc[disease, '2yr_1yr_AUC']) and
            not pd.isna(df_yearly.loc[disease, '3yr_1yr_AUC'])):
            valid_diseases.append(disease)

df_plot = df_plot[df_plot['Disease'].isin(valid_diseases)]

# Sort diseases by baseline 1-year AUC (descending)
baseline_1yr = df_plot[(df_plot['Horizon'] == '1-year') & (df_plot['Washout'] == '0 (baseline)')].set_index('Disease')['AUC']
disease_order = baseline_1yr.sort_values(ascending=False).index.tolist()
df_plot['Disease'] = pd.Categorical(df_plot['Disease'], categories=disease_order, ordered=True)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Color palette for washout periods
washout_colors = {
    '0 (baseline)': '#2E86AB',
    '1 month': '#A23B72',
    '3 months': '#F18F01',
    '6 months': '#C73E1D',
    '1 year': '#6A994E',
    '2 years': '#BC4749',
    '3 years': '#D62828'
}

washout_order = ['0 (baseline)', '1 month', '3 months', '6 months', '1 year', '2 years', '3 years']

# Plot 1-year AUCs
ax1 = axes[0]
df_1yr = df_plot[df_plot['Horizon'] == '1-year'].copy()
df_1yr['Washout'] = pd.Categorical(df_1yr['Washout'], categories=washout_order, ordered=True)

# Grouped bar plot
x = np.arange(len(disease_order))
width = 0.12

for i, washout in enumerate(washout_order):
    washout_data = df_1yr[df_1yr['Washout'] == washout]
    aucs = [washout_data[washout_data['Disease'] == d]['AUC'].values[0] if len(washout_data[washout_data['Disease'] == d]) > 0 else np.nan 
            for d in disease_order]
    offset = (i - len(washout_order)/2) * width + width/2
    ax1.bar(x + offset, aucs, width, label=washout, color=washout_colors[washout], alpha=0.8)

ax1.set_xlabel('Disease', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUC', fontsize=12, fontweight='bold')
ax1.set_title('1-Year Prediction AUC by Washout Period', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(disease_order, rotation=45, ha='right', fontsize=10)
ax1.legend(title='Washout Period', fontsize=9, title_fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.3, 1.0])
ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot 10-year AUCs
ax2 = axes[1]
df_10yr = df_plot[df_plot['Horizon'] == '10-year'].copy()
df_10yr['Washout'] = pd.Categorical(df_10yr['Washout'], categories=washout_order, ordered=True)

for i, washout in enumerate(washout_order):
    washout_data = df_10yr[df_10yr['Washout'] == washout]
    aucs = [washout_data[washout_data['Disease'] == d]['AUC'].values[0] if len(washout_data[washout_data['Disease'] == d]) > 0 else np.nan 
            for d in disease_order]
    offset = (i - len(washout_order)/2) * width + width/2
    ax2.bar(x + offset, aucs, width, label=washout, color=washout_colors[washout], alpha=0.8)

ax2.set_xlabel('Disease', fontsize=12, fontweight='bold')
ax2.set_ylabel('AUC', fontsize=12, fontweight='bold')
ax2.set_title('10-Year Static Prediction AUC by Washout Period', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(disease_order, rotation=45, ha='right', fontsize=10)
ax2.legend(title='Washout Period', fontsize=9, title_fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0.3, 1.0])
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()

# Save
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/plots')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'comprehensive_washout_comparison_1yr_10yr.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"✓ Saved plot to {output_path}")

# Also create a line plot version
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))

for disease in disease_order:  # All key diseases
    df_disease_1yr = df_1yr[df_1yr['Disease'] == disease].sort_values('Order')
    df_disease_10yr = df_10yr[df_10yr['Disease'] == disease].sort_values('Order')
    
    axes2[0].plot(df_disease_1yr['Order'], df_disease_1yr['AUC'], 
                  marker='o', label=disease, alpha=0.7, linewidth=1.5)
    axes2[1].plot(df_disease_10yr['Order'], df_disease_10yr['AUC'], 
                  marker='o', label=disease, alpha=0.7, linewidth=1.5)

axes2[0].set_xlabel('Washout Period', fontsize=12, fontweight='bold')
axes2[0].set_ylabel('AUC', fontsize=12, fontweight='bold')
axes2[0].set_title('1-Year Prediction AUC by Washout Period', fontsize=14, fontweight='bold')
axes2[0].set_xticks(range(len(washout_order)))
axes2[0].set_xticklabels(washout_order, rotation=45, ha='right', fontsize=9)
axes2[0].legend(fontsize=9, ncol=2, loc='best')
axes2[0].grid(True, alpha=0.3)
axes2[0].set_ylim([0.3, 1.0])
axes2[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

axes2[1].set_xlabel('Washout Period', fontsize=12, fontweight='bold')
axes2[1].set_ylabel('AUC', fontsize=12, fontweight='bold')
axes2[1].set_title('10-Year Static Prediction AUC by Washout Period', fontsize=14, fontweight='bold')
axes2[1].set_xticks(range(len(washout_order)))
axes2[1].set_xticklabels(washout_order, rotation=45, ha='right', fontsize=9)
axes2[1].legend(fontsize=9, ncol=2, loc='best')
axes2[1].grid(True, alpha=0.3)
axes2[1].set_ylim([0.3, 1.0])
axes2[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()

output_path2 = output_dir / 'comprehensive_washout_comparison_line_plot.png'
plt.savefig(output_path2, bbox_inches='tight')
print(f"✓ Saved line plot to {output_path2}")

plt.show()

print(f"\n✓ Comparison plots created successfully!")
print(f"  Bar plot: {output_path}")
print(f"  Line plot: {output_path2}")

