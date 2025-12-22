#!/usr/bin/env python3
"""
Create multipanel figure for washout analyses:
- Panel A: Reverse causation (1, 3, 6 month exclusion) - 10-year predictions
- Panel B: Time horizon analysis (10-year predictions with 1-year exclusion)
- Panel C: Temporal leakage heatmaps for key diseases (0-, 1-, 2-year washout across timepoints)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# LOAD DATA
# ============================================================================

# Reverse causation data (1, 3, 6 months)
reverse_causation_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/washout_comparison_1yr_10yr.csv')
df_reverse = pd.read_csv(reverse_causation_path, index_col=0)

# Time horizon analysis data
time_horizons_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/time_horizons/pooled_retrospective')
washout_time_horizons_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_time_horizons/pooled_retrospective')

baseline_10yr_path = time_horizons_dir / 'static_10yr_results.csv'
washout_10yr_path = washout_time_horizons_dir / 'washout_1yr_10yr_static_results.csv'

# Temporal leakage data (0-, 1-, 2-year washout)
temporal_leakage_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_fixed_timepoint/pooled_retrospective/washout_results_by_disease_pivot.csv')
df_temporal = pd.read_csv(temporal_leakage_path)

# Get all available diseases from temporal leakage data
all_temporal_diseases = sorted(df_temporal['Disease'].unique())

# Key diseases to highlight (expanded list)
key_diseases = ['ASCVD', 'Diabetes', 'Heart_Failure', 'Breast_Cancer', 'Atrial_Fib', 
                'All_Cancers', 'CKD', 'Colorectal_Cancer', 'Stroke', 'COPD', 
                'Prostate_Cancer', 'Parkinsons', 'Lung_Cancer', 'Bladder_Cancer']

# ============================================================================
# CREATE MULTIPANEL FIGURE
# ============================================================================

fig = plt.figure(figsize=(18, 10))
# Layout: Top row has 2 panels (A and B), bottom row has 4 heatmaps (C)
gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35, 
                      left=0.06, right=0.97, top=0.94, bottom=0.08)

# ============================================================================
# PANEL A: Reverse Causation - 10-Year Predictions
# ============================================================================

ax1 = fig.add_subplot(gs[0, :2])  # Span first 2 columns
df_plot = df_reverse.loc[df_reverse.index.isin(key_diseases)].copy()

washout_periods = ['No Washout', '1 Month', '3 Months', '6 Months']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
x_pos = np.arange(len(key_diseases))
width = 0.2

for i, period in enumerate(['no_washout', '1month', '3month', '6month']):
    aucs = df_plot[f'{period}_10yr_AUC'].values
    aucs_clean = [x if not pd.isna(x) else 0 for x in aucs]
    ax1.bar(x_pos + i*width, aucs_clean, width, label=washout_periods[i], 
            color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Disease', fontsize=11, fontweight='bold')
ax1.set_ylabel('AUC', fontsize=11, fontweight='bold')
ax1.set_title('A. Reverse Causation: 10-Year Predictions\n(Excluding events 1-6 months before enrollment)', 
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels([d.replace('_', ' ') for d in key_diseases], rotation=45, ha='right', fontsize=9)
ax1.set_ylim([0.4, 0.8])
ax1.legend(loc='upper right', frameon=True, fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# ============================================================================
# PANEL B: Time Horizon Analysis - 10-Year Predictions with 1-Year Exclusion
# ============================================================================

ax2 = fig.add_subplot(gs[0, 2:])  # Span last 2 columns

if baseline_10yr_path.exists() and washout_10yr_path.exists():
    baseline_10yr = pd.read_csv(baseline_10yr_path)
    washout_10yr = pd.read_csv(washout_10yr_path)
    
    # Merge for comparison
    comparison = baseline_10yr[['Disease', 'AUC']].merge(
        washout_10yr[['Disease', 'AUC']],
        on='Disease',
        suffixes=('_baseline', '_washout')
    )
    comparison['AUC_drop'] = comparison['AUC_baseline'] - comparison['AUC_washout']
    
    # Filter to key diseases for visibility
    comparison_plot = comparison[comparison['Disease'].isin(key_diseases)].copy()
    
    # Scatter plot
    ax2.scatter(comparison_plot['AUC_baseline'], comparison_plot['AUC_washout'], 
               alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
    
    # Add diagonal line
    min_auc = min(comparison_plot['AUC_baseline'].min(), comparison_plot['AUC_washout'].min())
    max_auc = max(comparison_plot['AUC_baseline'].max(), comparison_plot['AUC_washout'].max())
    ax2.plot([min_auc, max_auc], [min_auc, max_auc], 'r--', linewidth=2, label='No change', alpha=0.7)
    
    # Add disease labels
    for _, row in comparison_plot.iterrows():
        ax2.annotate(row['Disease'].replace('_', ' '), 
                    (row['AUC_baseline'], row['AUC_washout']),
                    fontsize=8, alpha=0.8, ha='center', va='bottom')
    
    ax2.set_xlabel('AUC (10-year, no washout)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('AUC (10-year, 1-year exclusion)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Time Horizon Analysis: 10-Year Predictions\n(1-year exclusion vs. baseline)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
else:
    ax2.text(0.5, 0.5, 'Time horizon data not found', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('B. Time Horizon Analysis', fontsize=12, fontweight='bold')

# ============================================================================
# PANEL C: Temporal Leakage - Heatmaps for Key Diseases
# ============================================================================

# Select key diseases for heatmaps
heatmap_diseases = ['ASCVD', 'Diabetes', 'Heart_Failure', 'Atrial_Fib']

# Reshape data for heatmap: convert wide format to long format
df_long = []
for _, row in df_temporal.iterrows():
    disease = row['Disease']
    timepoint = row['Timepoint']
    for washout_col in ['Washout_0yr', 'Washout_1yr', 'Washout_2yr']:
        if pd.notna(row[washout_col]):
            washout_years = int(washout_col.split('_')[1].replace('yr', ''))
            df_long.append({
                'Disease': disease,
                'Timepoint': timepoint,
                'Washout_years': washout_years,
                'AUC': row[washout_col]
            })

df_comprehensive = pd.DataFrame(df_long)

# Create heatmaps for each disease (4 heatmaps in bottom row)
for idx, disease in enumerate(heatmap_diseases[:4]):
    ax = fig.add_subplot(gs[1, idx])
    
    disease_df = df_comprehensive[df_comprehensive['Disease'] == disease]
    if len(disease_df) > 0:
        pivot = disease_df.pivot(index='Timepoint', columns='Washout_years', values='AUC')
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.4, vmax=1.0, ax=ax, cbar_kws={'label': 'AUC'})
        
        ax.set_title(f'{disease.replace("_", " ")}\nAUC by Timepoint and Washout', 
                    fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel('Washout (years)', fontsize=9)
        ax.set_ylabel('Prediction Timepoint\n(enrollment + N)', fontsize=9)
    else:
        ax.text(0.5, 0.5, f'{disease}\nNo data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f'{disease.replace("_", " ")}', fontsize=10, fontweight='bold')

# Add overall title
fig.suptitle('Washout Analyses: Reverse Causation and Temporal Leakage Assessment', 
             fontsize=14, fontweight='bold', y=0.97)

# Save figure
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'washout_multipanel.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved multipanel washout figure to: {output_path}")

# Also save as PNG
output_path_png = output_dir / 'washout_multipanel.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"✓ Saved multipanel washout figure to: {output_path_png}")

plt.close()

