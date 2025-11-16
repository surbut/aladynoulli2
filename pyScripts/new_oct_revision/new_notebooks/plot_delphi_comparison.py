"""
Visualization of Aladynoulli vs Delphi-2M Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Load comparison data
comparison = pd.read_csv('/Users/sarahurbut/aladynoulli2/claudefile/output/comparison_aladynoulli_vs_delphi_full.csv')

# Filter to diseases with both results
comparison_valid = comparison.dropna(subset=['Aladynoulli_1yr', 'Delphi_1yr']).copy()
comparison_valid['Wins'] = comparison_valid['Diff_1yr'] > 0

# Sort by difference
comparison_valid = comparison_valid.sort_values('Diff_1yr', ascending=True)

# =============================================================================
# 1. SIDE-BY-SIDE BAR CHART
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

y_pos = np.arange(len(comparison_valid))
bar_width = 0.35

# Colors
ala_color = '#2c7fb8'  # Blue
delphi_color = '#f03b20'  # Red

bars1 = ax.barh(y_pos - bar_width/2, comparison_valid['Aladynoulli_1yr'], 
                bar_width, label='Aladynoulli', color=ala_color, alpha=0.8)
bars2 = ax.barh(y_pos + bar_width/2, comparison_valid['Delphi_1yr'], 
                bar_width, label='Delphi-2M', color=delphi_color, alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_valid['Disease'], fontsize=10)
ax.set_xlabel('AUC (1-Year Prediction)', fontsize=12, fontweight='bold')
ax.set_title('Aladynoulli vs Delphi-2M: 1-Year Prediction Performance', 
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(0.7, color='green', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlim(0.4, 1.0)
ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax.grid(axis='x', alpha=0.3)

# Add difference annotations
for i, (idx, row) in enumerate(comparison_valid.iterrows()):
    diff = row['Diff_1yr']
    if abs(diff) > 0.01:  # Only annotate meaningful differences
        x_pos = max(row['Aladynoulli_1yr'], row['Delphi_1yr']) + 0.02
        color = 'green' if diff > 0 else 'red'
        marker = '▲' if diff > 0 else '▼'
        ax.text(x_pos, i, marker, color=color, fontsize=12, va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_comparison_bars.png', 
            bbox_inches='tight', facecolor='white')
plt.show()

# =============================================================================
# 2. SCATTER PLOT WITH DIAGONAL
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 10))

# Color by win/loss
colors = ['#2c7fb8' if win else '#f03b20' for win in comparison_valid['Wins']]
sizes = [150 if win else 100 for win in comparison_valid['Wins']]

scatter = ax.scatter(comparison_valid['Delphi_1yr'], comparison_valid['Aladynoulli_1yr'],
                     c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add diagonal line (y=x)
min_val = min(comparison_valid[['Aladynoulli_1yr', 'Delphi_1yr']].min())
max_val = max(comparison_valid[['Aladynoulli_1yr', 'Delphi_1yr']].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal performance')

# Add disease labels for significant differences
for idx, row in comparison_valid.iterrows():
    if abs(row['Diff_1yr']) > 0.05:  # Only label large differences
        ax.annotate(row['Disease'], 
                   (row['Delphi_1yr'], row['Aladynoulli_1yr']),
                   fontsize=8, alpha=0.7, 
                   xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Delphi-2M AUC', fontsize=12, fontweight='bold')
ax.set_ylabel('Aladynoulli AUC', fontsize=12, fontweight='bold')
ax.set_title('Aladynoulli vs Delphi-2M: 1-Year Prediction Comparison', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0.4, 1.0)
ax.grid(alpha=0.3)
ax.legend(['Equal performance', 'Aladynoulli wins', 'Delphi wins'], 
          loc='lower right', fontsize=10, frameon=True)

# Add quadrant labels
ax.text(0.95, 0.45, 'Delphi\nBetter', ha='right', va='bottom', 
        fontsize=10, alpha=0.5, style='italic')
ax.text(0.45, 0.95, 'Aladynoulli\nBetter', ha='left', va='top', 
        fontsize=10, alpha=0.5, style='italic')

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_comparison_scatter.png', 
            bbox_inches='tight', facecolor='white')
plt.show()

# =============================================================================
# 3. DIFFERENCE BAR CHART (RANKED)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 10))

# Sort by difference
diff_sorted = comparison_valid.sort_values('Diff_1yr', ascending=True)

colors = ['#2c7fb8' if diff > 0 else '#f03b20' for diff in diff_sorted['Diff_1yr']]

bars = ax.barh(range(len(diff_sorted)), diff_sorted['Diff_1yr'], color=colors, alpha=0.8)

ax.set_yticks(range(len(diff_sorted)))
ax.set_yticklabels(diff_sorted['Disease'], fontsize=10)
ax.set_xlabel('AUC Difference (Aladynoulli - Delphi-2M)', fontsize=12, fontweight='bold')
ax.set_title('Aladynoulli Advantage: 1-Year Predictions', 
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(0, color='black', linestyle='-', linewidth=2)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(diff_sorted.iterrows()):
    diff = row['Diff_1yr']
    x_pos = diff + (0.01 if diff > 0 else -0.01)
    ha = 'left' if diff > 0 else 'right'
    ax.text(x_pos, i, f'{diff:+.3f}', 
           va='center', ha=ha, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_comparison_differences.png', 
            bbox_inches='tight', facecolor='white')
plt.show()

# =============================================================================
# 4. SUMMARY STATISTICS VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Mean AUCs
ax = axes[0]
means = [comparison_valid['Aladynoulli_1yr'].mean(), 
         comparison_valid['Delphi_1yr'].mean()]
stds = [comparison_valid['Aladynoulli_1yr'].std(), 
        comparison_valid['Delphi_1yr'].std()]
bars = ax.bar(['Aladynoulli', 'Delphi-2M'], means, yerr=stds, 
              color=[ala_color, delphi_color], alpha=0.8, capsize=10)
ax.set_ylabel('Mean AUC', fontsize=11, fontweight='bold')
ax.set_title('A. Average Performance', fontsize=12, fontweight='bold')
ax.set_ylim(0.5, 0.8)
ax.grid(axis='y', alpha=0.3)
for i, (bar, mean) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2, mean + stds[i] + 0.01,
           f'{mean:.3f}', ha='center', fontweight='bold', fontsize=11)

# Panel B: Win/Loss Count
ax = axes[1]
wins_count = comparison_valid['Wins'].sum()
losses_count = len(comparison_valid) - wins_count
ax.bar(['Wins', 'Losses'], [wins_count, losses_count], 
       color=['#2c7fb8', '#f03b20'], alpha=0.8)
ax.set_ylabel('Number of Diseases', fontsize=11, fontweight='bold')
ax.set_title('B. Win/Loss Count', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, count in enumerate([wins_count, losses_count]):
    ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold', fontsize=12)

# Panel C: Distribution of Differences
ax = axes[2]
ax.hist(comparison_valid['Diff_1yr'], bins=15, color='#6366f1', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.axvline(comparison_valid['Diff_1yr'].mean(), color='green', 
          linestyle=':', linewidth=2, label=f'Mean: {comparison_valid["Diff_1yr"].mean():.3f}')
ax.set_xlabel('AUC Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('C. Distribution of Differences', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Aladynoulli vs Delphi-2M: Summary Statistics', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_comparison_summary.png', 
            bbox_inches='tight', facecolor='white')
plt.show()

print("\n✓ All plots saved to claudefile/output/")
print("  - delphi_comparison_bars.png")
print("  - delphi_comparison_scatter.png")
print("  - delphi_comparison_differences.png")
print("  - delphi_comparison_summary.png")





