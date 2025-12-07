#!/usr/bin/env python3
"""
Compare corrected E vs enrollment E results visually
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pathlib import Path
import sys

# Get absolute paths
base_dir = Path('/Users/sarahurbut/aladynoulli2')
new_washout_path = base_dir / 'pyScripts/dec_6_revision/new_notebooks/results/washout_time_horizons/pooled_retrospective/washout_1yr_comparison_all_horizons.csv'
new_time_path = base_dir / 'pyScripts/dec_6_revision/new_notebooks/results/time_horizons/pooled_retrospective/comparison_all_horizons.csv'
old_time_path = base_dir / 'pyScripts/new_oct_revision/new_notebooks/results/time_horizons/pooled_retrospective/comparison_all_horizons.csv'
old_washout_path = base_dir / 'pyScripts/new_oct_revision/new_notebooks/results/washout/pooled_retrospective_withlocal/washout_comparison_all_offsets.csv'

# Load data
new_washout = pd.read_csv(new_washout_path, index_col=0)
new_time = pd.read_csv(new_time_path, index_col=0)
old_time = pd.read_csv(old_time_path, index_col=0)
old_washout = pd.read_csv(old_washout_path, index_col=0)

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Corrected E vs Enrollment E: AUC Comparison', fontsize=16, fontweight='bold')

# 1. Time Horizons (10yr, 30yr) - No Washout
ax1 = axes[0, 0]
diseases = new_time.index
x = np.arange(len(diseases))
width = 0.35

new_10yr = new_time['10yr_AUC'].values
old_10yr = old_time['10yr_AUC'].values
new_30yr = new_time['30yr_AUC'].values
old_30yr = old_time['30yr_AUC'].values

ax1.bar(x - width/2, new_10yr, width, label='Corrected E (10yr)', alpha=0.8, color='#2E86AB')
ax1.bar(x + width/2, old_10yr, width, label='Enrollment E (10yr)', alpha=0.8, color='#A23B72')
ax1.set_xlabel('Disease')
ax1.set_ylabel('AUC')
ax1.set_title('10-Year Horizon (No Washout)')
ax1.set_xticks(x)
ax1.set_xticklabels(diseases, rotation=45, ha='right', fontsize=8)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# 2. 30-Year Horizon Comparison
ax2 = axes[0, 1]
ax2.bar(x - width/2, new_30yr, width, label='Corrected E (30yr)', alpha=0.8, color='#2E86AB')
ax2.bar(x + width/2, old_30yr, width, label='Enrollment E (30yr)', alpha=0.8, color='#A23B72')
ax2.set_xlabel('Disease')
ax2.set_ylabel('AUC')
ax2.set_title('30-Year Horizon (No Washout)')
ax2.set_xticks(x)
ax2.set_xticklabels(diseases, rotation=45, ha='right', fontsize=8)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# 3. Washout 10yr Dynamic
ax3 = axes[1, 0]
new_washout_10yr = new_washout['10yr_dynamic_AUC'].values
old_washout_0yr = old_washout['0yr_AUC'].values  # 0yr washout is closest comparison

# Match diseases
common_diseases = sorted(set(new_washout.index) & set(old_washout.index))
new_washout_10yr_matched = [new_washout.loc[d, '10yr_dynamic_AUC'] for d in common_diseases]
old_washout_0yr_matched = [old_washout.loc[d, '0yr_AUC'] for d in common_diseases]

x3 = np.arange(len(common_diseases))
ax3.bar(x3 - width/2, new_washout_10yr_matched, width, label='Corrected E (10yr, 1yr washout)', alpha=0.8, color='#2E86AB')
ax3.bar(x3 + width/2, old_washout_0yr_matched, width, label='Enrollment E (0yr washout)', alpha=0.8, color='#A23B72')
ax3.set_xlabel('Disease')
ax3.set_ylabel('AUC')
ax3.set_title('Washout Comparison (10yr dynamic vs 0yr)')
ax3.set_xticks(x3)
ax3.set_xticklabels(common_diseases, rotation=45, ha='right', fontsize=8)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# 4. Difference plot (10yr horizon)
ax4 = axes[1, 1]
diff_10yr = new_10yr - old_10yr
colors = ['green' if d > 0 else 'red' for d in diff_10yr]
ax4.bar(x, diff_10yr, color=colors, alpha=0.7)
ax4.set_xlabel('Disease')
ax4.set_ylabel('AUC Difference (Corrected E - Enrollment E)')
ax4.set_title('10-Year Horizon: Difference')
ax4.set_xticks(x)
ax4.set_xticklabels(diseases, rotation=45, ha='right', fontsize=8)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = base_dir / 'pyScripts/dec_6_revision/new_notebooks/results/comparison_correctedE_vs_enrollmentE.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison plot to: {output_path}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS: 10-Year Horizon (No Washout)")
print("="*80)
print(f"Mean AUC - Corrected E: {new_10yr.mean():.4f}")
print(f"Mean AUC - Enrollment E: {old_10yr.mean():.4f}")
print(f"Mean Difference: {diff_10yr.mean():.4f}")
print(f"Median Difference: {np.median(diff_10yr):.4f}")
print(f"Min Difference: {diff_10yr.min():.4f}")
print(f"Max Difference: {diff_10yr.max():.4f}")
print(f"Std Difference: {diff_10yr.std():.4f}")

print("\n" + "="*80)
print("TOP 5 DISEASES WITH LARGEST DECREASES (Corrected E < Enrollment E)")
print("="*80)
decreases = diff_10yr[diff_10yr < 0]
decrease_indices = np.where(diff_10yr < 0)[0]
decrease_diseases = [diseases[i] for i in decrease_indices]
decrease_values = diff_10yr[decrease_indices]
sorted_decreases = sorted(zip(decrease_values, decrease_diseases))
for val, disease in sorted_decreases[:5]:
    print(f"  {disease:25s} {val:+.4f}")

print("\n" + "="*80)
print("TOP 5 DISEASES WITH LARGEST INCREASES (Corrected E > Enrollment E)")
print("="*80)
increases = diff_10yr[diff_10yr > 0]
increase_indices = np.where(diff_10yr > 0)[0]
increase_diseases = [diseases[i] for i in increase_indices]
increase_values = diff_10yr[increase_indices]
sorted_increases = sorted(zip(increase_values, increase_diseases), reverse=True)
for val, disease in sorted_increases[:5]:
    print(f"  {disease:25s} {val:+.4f}")

plt.show()

