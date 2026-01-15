"""
Generate figure comparing Digital Twin Matching vs Propensity Score Matching
for statin effect demonstration.

This figure shows:
1. Hazard ratios for different matching approaches
2. Comparison to RCT benchmark
3. Confounding reduction metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Data from reviewer response analysis
methods = [
    'Naive\n(No adjustment)',
    'Clinical\nMatching Only',
    'Static Signature\nMatching',
    'Time-Varying\nSignature Matching',
    'Full Temporal +\nPS + PCE + Scripts',
    'RCT Benchmark'
]

unmatched_hr = [1.70, 1.55, 1.38, 1.38, 1.05, None]
matched_hr = [None, 1.45, 1.32, 1.02, 0.95, 0.75]
confounding_reduction = [None, 15, 35, 90, 95, None]

# Colors for different approaches
colors_unmatched = ['#d62728', '#ff7f0e', '#2ca02c', '#2ca02c', '#1f77b4', '#9467bd']
colors_matched = [None, '#ff7f0e', '#2ca02c', '#2ca02c', '#1f77b4', '#9467bd']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ============================================
# Panel A: Hazard Ratios Comparison
# ============================================
x_pos = np.arange(len(methods))
width = 0.35

# Plot unmatched HRs (where available)
bars1 = ax1.bar(x_pos - width/2, 
                [h if h is not None else 0 for h in unmatched_hr],
                width, label='Unmatched HR', 
                color=colors_unmatched, alpha=0.7, edgecolor='black', linewidth=1.5)

# Plot matched HRs (where available)
bars2 = ax1.bar(x_pos + width/2, 
                [h if h is not None else 0 for h in matched_hr],
                width, label='Matched HR', 
                color=colors_matched, alpha=0.9, edgecolor='black', linewidth=1.5)

# Add horizontal line at HR=1.0 (no effect)
ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

# Add RCT benchmark line
ax1.axhline(y=0.75, color='#9467bd', linestyle=':', linewidth=2, 
            label='RCT Benchmark (HR=0.75)', alpha=0.8, zorder=0)

# Customize x-axis
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, rotation=15, ha='right')
ax1.set_ylabel('Hazard Ratio', fontweight='bold')
ax1.set_title('A. Statin Effect: Digital Twin vs Propensity Score Matching', 
              fontweight='bold', pad=15)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0.5, 1.8])

# Add value labels on bars
for i, (u_hr, m_hr) in enumerate(zip(unmatched_hr, matched_hr)):
    if u_hr is not None:
        ax1.text(i - width/2, u_hr + 0.05, f'{u_hr:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    if m_hr is not None:
        ax1.text(i + width/2, m_hr + 0.05, f'{m_hr:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add annotation for best result
ax1.annotate('Approaches RCT\nbenchmark', 
            xy=(4, 0.95), xytext=(3.5, 0.6),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# ============================================
# Panel B: Confounding Reduction
# ============================================
# Filter out None values for plotting
methods_filtered = [m for m, cr in zip(methods, confounding_reduction) if cr is not None]
confounding_filtered = [cr for cr in confounding_reduction if cr is not None]
colors_filtered = [c for c, cr in zip(colors_matched, confounding_reduction) if cr is not None]

x_pos2 = np.arange(len(methods_filtered))
bars3 = ax2.bar(x_pos2, confounding_filtered, 
                color=colors_filtered, alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xticks(x_pos2)
ax2.set_xticklabels(methods_filtered, rotation=15, ha='right')
ax2.set_ylabel('Confounding Reduction (%)', fontweight='bold')
ax2.set_title('B. Confounding Reduction by Matching Approach', 
              fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 100])

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, confounding_filtered)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight time-varying signature matching
ax2.annotate('90% reduction\nvs 35% for\nstatic matching', 
            xy=(2, 90), xytext=(1, 60),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('statin_effect_digital_twin_vs_psm.png', dpi=300, bbox_inches='tight')
plt.savefig('statin_effect_digital_twin_vs_psm.pdf', bbox_inches='tight')
print("Figure saved as 'statin_effect_digital_twin_vs_psm.png' and '.pdf'")
plt.show()








