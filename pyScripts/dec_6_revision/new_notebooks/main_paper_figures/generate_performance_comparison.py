"""
Generate publication-ready performance comparison figure and summary table:
- Highlights 10-year static as the ideal model
- Compares Aladynoulli models (1yr baseline, 1yr median, 10yr static)
- Includes baseline comparisons (Cox, external scores)
- Creates both a comprehensive plot and summary table
- Note: Delphi excluded from plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']

# ============================================================================
# LOAD DATA
# ============================================================================

results_dir = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/")
output_dir = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig5")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading performance comparison data...")

# 1. Aladynoulli 1yr at baseline (washout 0yr)
washout_0yr = pd.read_csv(results_dir / "washout/pooled_retrospective/washout_0yr_results.csv")
washout_0yr = washout_0yr.rename(columns={'AUC': 'Aladynoulli_1yr_baseline', 
                                          'CI_lower': 'Aladynoulli_1yr_baseline_CI_lower',
                                          'CI_upper': 'Aladynoulli_1yr_baseline_CI_upper'})

# 2. Aladynoulli 1yr median (over ten different 1-year risk periods)
median_1yr = pd.read_csv(results_dir / "age_offset/pooled_retrospective/medians_with_global0.csv")
median_1yr = median_1yr.rename(columns={'Median_with_global0': 'Aladynoulli_1yr_median'})

# 3. Aladynoulli 10yr - choose best (static vs dynamic) per disease
time_horizons = pd.read_csv(results_dir / "time_horizons/pooled_retrospective/comparison_all_horizons.csv", index_col=0)
time_horizons = time_horizons.reset_index()
time_horizons = time_horizons.rename(columns={'index': 'Disease'})

# Compare static vs dynamic and choose best for each disease
time_horizons['Aladynoulli_10yr_best'] = np.nan
time_horizons['Aladynoulli_10yr_best_CI_lower'] = np.nan
time_horizons['Aladynoulli_10yr_best_CI_upper'] = np.nan
time_horizons['Aladynoulli_10yr_type'] = ''  # 'static' or 'dynamic'

for idx, row in time_horizons.iterrows():
    static_auc = row.get('static_10yr_AUC', np.nan)
    dynamic_auc = row.get('10yr_AUC', np.nan)
    
    if pd.notna(static_auc) and pd.notna(dynamic_auc):
        # Choose the better one
        if static_auc >= dynamic_auc:
            time_horizons.loc[idx, 'Aladynoulli_10yr_best'] = static_auc
            time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_lower'] = row.get('static_10yr_CI_lower', np.nan)
            time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_upper'] = row.get('static_10yr_CI_upper', np.nan)
            time_horizons.loc[idx, 'Aladynoulli_10yr_type'] = 'static'
        else:
            time_horizons.loc[idx, 'Aladynoulli_10yr_best'] = dynamic_auc
            time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_lower'] = row.get('10yr_CI_lower', np.nan)
            time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_upper'] = row.get('10yr_CI_upper', np.nan)
            time_horizons.loc[idx, 'Aladynoulli_10yr_type'] = 'dynamic'
    elif pd.notna(static_auc):
        time_horizons.loc[idx, 'Aladynoulli_10yr_best'] = static_auc
        time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_lower'] = row.get('static_10yr_CI_lower', np.nan)
        time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_upper'] = row.get('static_10yr_CI_upper', np.nan)
        time_horizons.loc[idx, 'Aladynoulli_10yr_type'] = 'static'
    elif pd.notna(dynamic_auc):
        time_horizons.loc[idx, 'Aladynoulli_10yr_best'] = dynamic_auc
        time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_lower'] = row.get('10yr_CI_lower', np.nan)
        time_horizons.loc[idx, 'Aladynoulli_10yr_best_CI_upper'] = row.get('10yr_CI_upper', np.nan)
        time_horizons.loc[idx, 'Aladynoulli_10yr_type'] = 'dynamic'

# 4. Cox models
cox_comparison = pd.read_csv(results_dir / "comparisons/pooled_retrospective/cox_baseline_comparison_static10yr_full.csv")
cox_comparison = cox_comparison.rename(columns={'Cox_AUC': 'Cox_without_Aladynoulli',
                                                'Aladynoulli_AUC': 'Cox_with_Aladynoulli',
                                                'CI_lower': 'Cox_with_Aladynoulli_CI_lower',
                                                'CI_upper': 'Cox_with_Aladynoulli_CI_upper'})

# 5. External scores (Delphi excluded from plot)
external_scores = pd.read_csv(results_dir / "comparisons/pooled_retrospective/external_scores_comparison.csv", index_col=0)

print(f"Loaded data for {len(washout_0yr)} diseases")

# ============================================================================
# MERGE DATA
# ============================================================================

# Start with washout_0yr as base (has most diseases)
merged = washout_0yr[['Disease', 'Aladynoulli_1yr_baseline', 'Aladynoulli_1yr_baseline_CI_lower', 
                      'Aladynoulli_1yr_baseline_CI_upper']].copy()

# Merge median 1yr
merged = merged.merge(median_1yr[['Disease', 'Aladynoulli_1yr_median']], on='Disease', how='left')

# Merge 10yr best (chooses static or dynamic per disease)
merged = merged.merge(time_horizons[['Disease', 'Aladynoulli_10yr_best', 
                                     'Aladynoulli_10yr_best_CI_lower', 
                                     'Aladynoulli_10yr_best_CI_upper',
                                     'Aladynoulli_10yr_type']], on='Disease', how='left')

# Merge Cox models
merged = merged.merge(cox_comparison[['Disease', 'Cox_without_Aladynoulli', 'Cox_with_Aladynoulli',
                                     'Cox_with_Aladynoulli_CI_lower', 'Cox_with_Aladynoulli_CI_upper']], 
                     on='Disease', how='left')

# Add external scores for specific diseases
# ASCVD: PCE, PREVENT, QRISK3
if 'ASCVD_10yr' in external_scores.index:
    ascvd_row = external_scores.loc['ASCVD_10yr']
    merged.loc[merged['Disease'] == 'ASCVD', 'PCE_10yr'] = ascvd_row.get('PCE_AUC', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'PCE_10yr_CI_lower'] = ascvd_row.get('PCE_CI_lower', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'PCE_10yr_CI_upper'] = ascvd_row.get('PCE_CI_upper', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'PREVENT_10yr'] = ascvd_row.get('PREVENT_10yr_AUC', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'PREVENT_10yr_CI_lower'] = ascvd_row.get('PREVENT_10yr_CI_lower', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'PREVENT_10yr_CI_upper'] = ascvd_row.get('PREVENT_10yr_CI_upper', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'QRISK3_10yr'] = ascvd_row.get('QRISK3_AUC', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'QRISK3_10yr_CI_lower'] = ascvd_row.get('QRISK3_CI_lower', np.nan)
    merged.loc[merged['Disease'] == 'ASCVD', 'QRISK3_10yr_CI_upper'] = ascvd_row.get('QRISK3_CI_upper', np.nan)

# Breast Cancer: GAIL (1yr and 10yr)
if 'Breast_Cancer_1yr' in external_scores.index:
    breast_1yr_row = external_scores.loc['Breast_Cancer_1yr']
    merged.loc[merged['Disease'] == 'Breast_Cancer', 'GAIL_1yr'] = breast_1yr_row.get('Gail_AUC', np.nan)
    merged.loc[merged['Disease'] == 'Breast_Cancer', 'GAIL_1yr_CI_lower'] = breast_1yr_row.get('Gail_CI_lower', np.nan)
    merged.loc[merged['Disease'] == 'Breast_Cancer', 'GAIL_1yr_CI_upper'] = breast_1yr_row.get('Gail_CI_upper', np.nan)

if 'Breast_Cancer_10yr' in external_scores.index:
    breast_10yr_row = external_scores.loc['Breast_Cancer_10yr']
    merged.loc[merged['Disease'] == 'Breast_Cancer', 'GAIL_10yr'] = breast_10yr_row.get('Gail_AUC', np.nan)
    merged.loc[merged['Disease'] == 'Breast_Cancer', 'GAIL_10yr_CI_lower'] = breast_10yr_row.get('Gail_CI_lower', np.nan)
    merged.loc[merged['Disease'] == 'Breast_Cancer', 'GAIL_10yr_CI_upper'] = breast_10yr_row.get('Gail_CI_upper', np.nan)

print(f"Merged data for {len(merged)} diseases")

# ============================================================================
# CREATE SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("CREATING SUMMARY TABLE")
print("="*80)

# Select key diseases for plot prioritization
key_diseases = ['ASCVD', 'Breast_Cancer', 'Prostate_Cancer', 'Colorectal_Cancer', 
                'Heart_Failure', 'Diabetes', 'Atrial_Fib', 'CKD', 'Stroke',
                'Parkinsons', 'Rheumatoid_Arthritis', 'Depression', 'All_Cancers']

# Filter to available diseases
key_diseases = [d for d in key_diseases if d in merged['Disease'].values]

# Use ALL diseases for summary table (not just key diseases)
all_diseases_for_table = sorted(merged['Disease'].values.tolist())

# Create summary table
summary_data = []
for disease in all_diseases_for_table:
    row = merged[merged['Disease'] == disease].iloc[0]
    
    # Format AUC with CI
    def format_auc(auc_val, ci_lower=None, ci_upper=None):
        if pd.isna(auc_val):
            return "—"
        if pd.notna(ci_lower) and pd.notna(ci_upper):
            return f"{auc_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        else:
            return f"{auc_val:.3f}"
    
    # Get 10yr type for labeling
    tenyr_type = row.get('Aladynoulli_10yr_type', '')
    tenyr_type_label = f" ({tenyr_type.capitalize()})" if tenyr_type else ""
    
    summary_row = {
        'Disease': disease.replace('_', ' '),
        'Aladynoulli 10yr (Best)': format_auc(
            row.get('Aladynoulli_10yr_best'),
            row.get('Aladynoulli_10yr_best_CI_lower'),
            row.get('Aladynoulli_10yr_best_CI_upper')
        ) + tenyr_type_label,
        'Aladynoulli 1yr (Baseline)': format_auc(
            row.get('Aladynoulli_1yr_baseline'),
            row.get('Aladynoulli_1yr_baseline_CI_lower'),
            row.get('Aladynoulli_1yr_baseline_CI_upper')
        ),
        'Aladynoulli 1yr (Median)': format_auc(row.get('Aladynoulli_1yr_median')),
        'Cox 10yr (Baseline)': format_auc(row.get('Cox_without_Aladynoulli'))
    }
    
    # Add external scores where applicable
    if disease == 'ASCVD':
        summary_row['PCE 10yr'] = format_auc(
            row.get('PCE_10yr'),
            row.get('PCE_10yr_CI_lower'),
            row.get('PCE_10yr_CI_upper')
        )
        summary_row['PREVENT 10yr'] = format_auc(
            row.get('PREVENT_10yr'),
            row.get('PREVENT_10yr_CI_lower'),
            row.get('PREVENT_10yr_CI_upper')
        )
        summary_row['QRISK3 10yr'] = format_auc(
            row.get('QRISK3_10yr'),
            row.get('QRISK3_10yr_CI_lower'),
            row.get('QRISK3_10yr_CI_upper')
        )
    elif disease == 'Breast_Cancer':
        summary_row['GAIL 1yr'] = format_auc(
            row.get('GAIL_1yr'),
            row.get('GAIL_1yr_CI_lower'),
            row.get('GAIL_1yr_CI_upper')
        )
        summary_row['GAIL 10yr'] = format_auc(
            row.get('GAIL_10yr'),
            row.get('GAIL_10yr_CI_lower'),
            row.get('GAIL_10yr_CI_upper')
        )
    
    summary_data.append(summary_row)

summary_df = pd.DataFrame(summary_data)

# Save summary table

summary_table_file = output_dir / "performance_summary_table.csv"
summary_df.to_csv(summary_table_file, index=False)
print(f"✓ Saved summary table to: {summary_table_file}")

# Print summary table
print("\n" + "="*100)
print("PERFORMANCE SUMMARY TABLE")
print("="*100)
print(summary_df.to_string(index=False))
print("="*100)

# ============================================================================
# CREATE PUBLICATION-READY MULTI-FACETED PLOT
# ============================================================================

print("\n" + "="*80)
print("CREATING PUBLICATION-READY MULTI-FACETED PLOT")
print("="*80)

# Select diseases for main plot (prioritize key diseases)
plot_diseases = key_diseases + [d for d in merged['Disease'].values if d not in key_diseases]
plot_diseases = [d for d in plot_diseases if d in merged['Disease'].values]

# Calculate grid dimensions for multi-faceted plot
n_diseases = len(plot_diseases)
n_cols = 4  # 4 panels per row
n_rows = int(np.ceil(n_diseases / n_cols))

# Create figure with multiple subplots (one per disease)
fig = plt.figure(figsize=(20, 5 * n_rows))

# Color scheme
colors = {
    'Aladynoulli_10yr_best': '#8E44AD',       # Purple
    'Aladynoulli_1yr_baseline': '#E74C3C',    # Red
    'Aladynoulli_1yr_median': '#3498DB',      # Blue
    'Cox_without_Aladynoulli': '#F39C12',     # Orange
    'PCE_10yr': '#95A5A6',                    # Grey
    'PREVENT_10yr': '#34495E',                # Dark grey
    'QRISK3_10yr': '#7F8C8D',                 # Medium grey
    'GAIL_1yr': '#16A085',                    # Green
    'GAIL_10yr': '#27AE60'                    # Dark green
}

markers = {
    'Aladynoulli_10yr_best': '^',             # Triangle
    'Aladynoulli_1yr_baseline': 'o',          # Circle
    'Aladynoulli_1yr_median': 's',            # Square
    'Cox_without_Aladynoulli': 'D',           # Diamond
    'PCE_10yr': 'p',                          # Pentagon
    'PREVENT_10yr': 'p',                      # Pentagon
    'QRISK3_10yr': 'p',                       # Pentagon
    'GAIL_1yr': 'h',                          # Hexagon
    'GAIL_10yr': 'h'                          # Hexagon
}

# Define model order:
# 1. 1 year (at enrollment)
# 2. 1 year median
# 3. 1 year gail (if applies)
# 4. 10 year (best)
# 5. External 10-year scores (Gail/PCE/PREVENT/QRISK when applied)
# 6. 10 year cox (without Aladynoulli)

# Plot each disease in its own panel
for disease_idx, disease in enumerate(plot_diseases):
    row = disease_idx // n_cols
    col = disease_idx % n_cols
    ax = fig.add_subplot(n_rows, n_cols, disease_idx + 1)
    
    disease_data = merged[merged['Disease'] == disease].iloc[0]
    
    # Collect all models to plot for this disease in the specified order
    models_to_plot = []
    x_positions = []
    auc_values = []
    ci_lowers = []
    ci_uppers = []
    
    def add_model(model_name, auc_val, ci_lower=None, ci_upper=None):
        """Helper function to add a model if AUC is available"""
        if pd.notna(auc_val):
            models_to_plot.append(model_name)
            x_positions.append(len(models_to_plot) - 1)
            auc_values.append(auc_val)
            if ci_lower is None:
                ci_lower = auc_val
            if ci_upper is None:
                ci_upper = auc_val
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
    
    # 1. 1 year (at enrollment)
    auc_val = disease_data.get('Aladynoulli_1yr_baseline')
    if pd.notna(auc_val):
        ci_lower = disease_data.get('Aladynoulli_1yr_baseline_CI_lower', auc_val)
        ci_upper = disease_data.get('Aladynoulli_1yr_baseline_CI_upper', auc_val)
        add_model('Aladynoulli_1yr_baseline', auc_val, ci_lower, ci_upper)
    
    # 2. 1 year median
    auc_val = disease_data.get('Aladynoulli_1yr_median')
    if pd.notna(auc_val):
        add_model('Aladynoulli_1yr_median', auc_val)
    
    # 3. 1 year gail (if applies - only for Breast_Cancer)
    if disease == 'Breast_Cancer':
        auc_val = disease_data.get('GAIL_1yr')
        if pd.notna(auc_val):
            ci_lower = disease_data.get('GAIL_1yr_CI_lower', auc_val)
            ci_upper = disease_data.get('GAIL_1yr_CI_upper', auc_val)
            add_model('GAIL_1yr', auc_val, ci_lower, ci_upper)
    
    # 4. 10 year (best)
    auc_val = disease_data.get('Aladynoulli_10yr_best')
    if pd.notna(auc_val):
        ci_lower = disease_data.get('Aladynoulli_10yr_best_CI_lower', auc_val)
        ci_upper = disease_data.get('Aladynoulli_10yr_best_CI_upper', auc_val)
        add_model('Aladynoulli_10yr_best', auc_val, ci_lower, ci_upper)
    
    # 5. External 10-year scores (when applied)
    if disease == 'ASCVD':
        # PCE, PREVENT, QRISK3 (in that order)
        for model in ['PCE_10yr', 'PREVENT_10yr', 'QRISK3_10yr']:
            auc_val = disease_data.get(model)
            if pd.notna(auc_val):
                ci_lower = disease_data.get(f'{model}_CI_lower', auc_val)
                ci_upper = disease_data.get(f'{model}_CI_upper', auc_val)
                add_model(model, auc_val, ci_lower, ci_upper)
    elif disease == 'Breast_Cancer':
        # GAIL 10yr
        auc_val = disease_data.get('GAIL_10yr')
        if pd.notna(auc_val):
            ci_lower = disease_data.get('GAIL_10yr_CI_lower', auc_val)
            ci_upper = disease_data.get('GAIL_10yr_CI_upper', auc_val)
            add_model('GAIL_10yr', auc_val, ci_lower, ci_upper)
    
    # 6. 10 year cox (without Aladynoulli)
    auc_val = disease_data.get('Cox_without_Aladynoulli')
    if pd.notna(auc_val):
        add_model('Cox_without_Aladynoulli', auc_val)
    
    # Plot all models for this disease
    for i, model in enumerate(models_to_plot):
        x_pos = x_positions[i]
        auc_val = auc_values[i]
        ci_lower = ci_lowers[i]
        ci_upper = ci_uppers[i]
        
        color = colors[model]
        marker = markers[model]
        
        # Plot all models with same style
        ax.scatter(x_pos, auc_val, color=color, marker=marker, s=100, 
                  alpha=0.8, zorder=3, edgecolors='white', linewidths=1)
        
        # Error bars
        if ci_lower != ci_upper:
            ax.errorbar(x_pos, auc_val, yerr=[[auc_val - ci_lower], [ci_upper - auc_val]], 
                       color=color, alpha=0.6, capsize=3, capthick=1.5, zorder=2, linestyle='-')
    
    # Formatting per panel
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_ylim(0.4, 1.0)
    if len(models_to_plot) > 0:
        ax.set_xlim(-0.5, len(models_to_plot) - 0.5)
        ax.set_xticks(range(len(models_to_plot)))
        # Short labels for x-axis
        labels = []
        for model in models_to_plot:
            if model == 'Aladynoulli_1yr_baseline':
                labels.append('1yr enroll')
            elif model == 'Aladynoulli_1yr_median':
                labels.append('1yr med')
            elif model == 'GAIL_1yr':
                labels.append('GAIL 1yr')
            elif model == 'Aladynoulli_10yr_best':
                labels.append('10yr best')
            elif model == 'PCE_10yr':
                labels.append('PCE')
            elif model == 'PREVENT_10yr':
                labels.append('PREVENT')
            elif model == 'QRISK3_10yr':
                labels.append('QRISK3')
            elif model == 'GAIL_10yr':
                labels.append('GAIL 10yr')
            elif model == 'Cox_without_Aladynoulli':
                labels.append('Cox 10yr')
            else:
                labels.append(model.replace('_', ' ')[:8])
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    ax.set_ylabel('AUC', fontsize=9, fontweight='bold')
    ax.set_title(disease.replace('_', ' '), fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Add overall title
fig.suptitle('Predictive Performance Comparison: Aladynoulli vs Baseline Models and Clinical Risk Scores', 
             fontsize=14, fontweight='bold', y=0.995)

# Create legend in the specified order
from matplotlib.lines import Line2D
legend_elements = []
legend_order = [
    'Aladynoulli_1yr_baseline',
    'Aladynoulli_1yr_median',
    'GAIL_1yr',
    'Aladynoulli_10yr_best',  # Highlight with star
    'PCE_10yr',
    'PREVENT_10yr',
    'QRISK3_10yr',
    'GAIL_10yr',
    'Cox_without_Aladynoulli'
]

for model_name in legend_order:
    if model_name in colors:
        marker = markers[model_name]
        color = colors[model_name]
        
        # Create readable labels
        if model_name == 'Aladynoulli_1yr_baseline':
            label = 'Aladynoulli 1yr (Enrollment)'
        elif model_name == 'Aladynoulli_1yr_median':
            label = 'Aladynoulli 1yr (Median)'
        elif model_name == 'GAIL_1yr':
            label = 'GAIL 1yr'
        elif model_name == 'Aladynoulli_10yr_best':
            label = 'Aladynoulli 10yr (Best)'
        elif model_name == 'Cox_without_Aladynoulli':
            label = 'Cox 10yr'
        else:
            label = model_name.replace('_', ' ')
        
        # Special formatting for ideal model
        if model_name == 'Aladynoulli_10yr_best':
            edgecolor = 'black'
            linewidth = 2
        else:
            edgecolor = 'white'
            linewidth = 1
        
        legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                     markerfacecolor=color, markersize=10,
                                     label=label, markeredgecolor=edgecolor, 
                                     markeredgewidth=linewidth))

# Add legend to the figure (positioned at bottom center)
fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
          fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space for legend

# Save figure as PDF only
output_file = output_dir / "performance_comparison_publication.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n✓ Saved publication figure to: {output_file}")

# Also save summary table with type information
summary_with_type = summary_df.copy()
# Add a column showing which type was chosen for each disease
type_info = []
for disease in all_diseases_for_table:
    row = merged[merged['Disease'] == disease].iloc[0]
    tenyr_type = row.get('Aladynoulli_10yr_type', '')
    type_info.append(tenyr_type.capitalize() if tenyr_type else 'N/A')
summary_with_type.insert(2, '10yr Type', type_info)

summary_table_detailed = output_dir / "performance_summary_table_detailed.csv"
summary_with_type.to_csv(summary_table_detailed, index=False)
print(f"✓ Saved detailed summary table to: {summary_table_detailed}")

plt.show()

# ============================================================================
# PRINT FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total diseases in comparison: {len(merged)}")
print(f"Diseases in summary table: {len(all_diseases_for_table)}")
print(f"\nModels compared:")
print(f"  - Aladynoulli 10yr Best: {merged['Aladynoulli_10yr_best'].notna().sum()} diseases")
# Print breakdown of static vs dynamic
static_count = (merged['Aladynoulli_10yr_type'] == 'static').sum()
dynamic_count = (merged['Aladynoulli_10yr_type'] == 'dynamic').sum()
print(f"    - Static chosen: {static_count} diseases")
print(f"    - Dynamic chosen: {dynamic_count} diseases")
if static_count > 0:
    static_diseases = merged[merged['Aladynoulli_10yr_type'] == 'static']['Disease'].tolist()
    print(f"      Static diseases: {', '.join(static_diseases[:10])}{'...' if len(static_diseases) > 10 else ''}")
if dynamic_count > 0:
    dynamic_diseases = merged[merged['Aladynoulli_10yr_type'] == 'dynamic']['Disease'].tolist()
    print(f"      Dynamic diseases: {', '.join(dynamic_diseases[:10])}{'...' if len(dynamic_diseases) > 10 else ''}")
print(f"  - Aladynoulli 1yr Baseline: {merged['Aladynoulli_1yr_baseline'].notna().sum()} diseases")
print(f"  - Aladynoulli 1yr Median: {merged['Aladynoulli_1yr_median'].notna().sum()} diseases")
print(f"  - Cox 10yr (Baseline): {merged['Cox_without_Aladynoulli'].notna().sum()} diseases")
print(f"\nExternal scores:")
print(f"  - PCE 10yr (ASCVD): {merged['PCE_10yr'].notna().sum()} disease")
print(f"  - PREVENT 10yr (ASCVD): {merged['PREVENT_10yr'].notna().sum()} disease")
print(f"  - QRISK3 10yr (ASCVD): {merged['QRISK3_10yr'].notna().sum()} disease")
print(f"  - GAIL 1yr (Breast Cancer): {merged['GAIL_1yr'].notna().sum()} disease")
print(f"  - GAIL 10yr (Breast Cancer): {merged['GAIL_10yr'].notna().sum()} disease")
print("="*80)
