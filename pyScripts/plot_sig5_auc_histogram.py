#!/usr/bin/env python3
"""
Create histogram of Signature 5 AUC phenotypes to demonstrate
that signatures are continuous phenotypes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data_path = '/Users/sarahurbut/Downloads/signature_auc_phenotypes.txt'
df = pd.read_csv(data_path, sep='\t')

print(f"Loaded {len(df)} individuals")
print(f"\nSIG5_AUC statistics:")
print(df['SIG5_AUC'].describe())

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram
sig5_values = df['SIG5_AUC'].values
n, bins, patches = ax.hist(sig5_values, bins=100, color='#2E86AB', edgecolor='white', 
                           linewidth=0.5, alpha=0.8)

# Add mean and median lines
mean_val = np.mean(sig5_values)
median_val = np.median(sig5_values)
ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

# Labels and title
ax.set_xlabel('Signature 5 (Cardiovascular) Lifetime Exposure (AUC)', fontsize=12)
ax.set_ylabel('Number of Individuals', fontsize=12)
ax.set_title('Distribution of Cardiovascular Signature (SIG5) Lifetime Exposure\n'
             'Continuous phenotype used for heritability estimation', fontsize=13)

# Add text box with statistics
stats_text = (f'N = {len(sig5_values):,}\n'
              f'Mean = {mean_val:.3f}\n'
              f'Median = {median_val:.3f}\n'
              f'SD = {np.std(sig5_values):.3f}\n'
              f'Min = {np.min(sig5_values):.3f}\n'
              f'Max = {np.max(sig5_values):.3f}')
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(loc='upper left')

# Add note about continuous phenotype
ax.text(0.5, -0.12, 
        'Note: This continuous distribution demonstrates why observed-scale heritability\n'
        'is the only applicable scale for signatures (no liability transformation for continuous traits).',
        transform=ax.transAxes, fontsize=9, ha='center', style='italic', color='gray')

plt.tight_layout()

# Save
output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/sig5_auc_histogram.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved histogram to: {output_path}")

# Also save PNG
output_png = '/Users/sarahurbut/aladynoulli2/pyScripts/sig5_auc_histogram.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Saved PNG to: {output_png}")

plt.show()
