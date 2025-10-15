import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Washout results from your analysis
washout_data = {
    'Disease': [
        'ASCVD', 'Diabetes', 'Atrial Fib', 'CKD', 'All Cancers', 
        'Stroke', 'Heart Failure', 'Pneumonia', 'COPD', 'Osteoporosis',
        'Colorectal Cancer', 'Breast Cancer', 'Prostate Cancer', 
        'Lung Cancer'
    ],
    '0yr': [0.898, 0.715, 0.814, 0.848, 0.783, 0.628, 0.795, 0.626, 0.760, 0.794, 0.949, 0.805, 0.855, 0.691],
    '1yr': [0.701, 0.603, 0.673, 0.711, 0.684, 0.725, 0.607, 0.582, 0.668, 0.704, 0.669, 0.632, 0.719, 0.780],
    '2yr': [0.680, 0.603, 0.659, 0.702, 0.675, 0.708, 0.706, 0.661, 0.662, 0.665, 0.685, 0.531, 0.695, 0.666]
}

df = pd.DataFrame(washout_data)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ============================================================================
# PANEL A: Line plot showing AUC decay over washout periods
# ============================================================================
colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
washout_periods = [0, 1, 2]

for idx, row in df.iterrows():
    aucs = [row['0yr'], row['1yr'], row['2yr']]
    ax1.plot(washout_periods, aucs, 'o-', linewidth=2.5, markersize=8, 
             label=row['Disease'], color=colors[idx], alpha=0.8)

# Add reference line at 0.5 (random chance)
ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, 
            label='Random Chance', alpha=0.5)

# Add clinical utility threshold
ax1.axhline(y=0.7, color='red', linestyle=':', linewidth=1.5, 
            label='Clinical Utility (0.70)', alpha=0.5)

ax1.set_xlabel('Washout Period (Years)', fontsize=14, fontweight='bold')
ax1.set_ylabel('AUC', fontsize=14, fontweight='bold')
ax1.set_title('A. Model Performance with Temporal Washout', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(washout_periods)
ax1.set_xticklabels(['Immediate\n(0 yr)', '1-Year\nWashout', '2-Year\nWashout'])
ax1.set_ylim(0.45, 1.0)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
           framealpha=0.9)
ax1.tick_params(labelsize=12)

# ============================================================================
# PANEL B: Bar plot comparing performance drop
# ============================================================================
# Calculate performance drop from 0yr to 2yr
df['drop'] = df['0yr'] - df['2yr']
df['retention'] = (df['2yr'] / df['0yr']) * 100
df_sorted = df.sort_values('2yr', ascending=True)

y_pos = np.arange(len(df_sorted))
bars = ax2.barh(y_pos, df_sorted['2yr'], alpha=0.7, color='steelblue', 
                edgecolor='black', linewidth=1.5, label='2-Year Washout AUC')

# Add 0-year performance as comparison
ax2.barh(y_pos, df_sorted['0yr'], alpha=0.3, color='lightcoral', 
         edgecolor='black', linewidth=1.5, label='Immediate (0-Year) AUC')

# Add reference lines
ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.axvline(x=0.7, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(df_sorted['Disease'], fontsize=11)
ax2.set_xlabel('AUC', fontsize=14, fontweight='bold')
ax2.set_title('B. Predictive Performance After 2-Year Washout', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlim(0.45, 1.0)
ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
ax2.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax2.tick_params(labelsize=12)

# Add text annotations for retention percentage
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    retention = (row['2yr'] / row['0yr']) * 100
    ax2.text(row['2yr'] + 0.02, i, f"{retention:.0f}%", 
             va='center', fontsize=9, color='darkblue', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/washout_analysis.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/washout_analysis.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved washout analysis plots!")

# ============================================================================
# BONUS: Summary table
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create summary table
summary_data = []
for idx, row in df.iterrows():
    summary_data.append([
        row['Disease'],
        f"{row['0yr']:.3f}",
        f"{row['1yr']:.3f}",
        f"{row['2yr']:.3f}",
        f"{row['drop']:.3f}",
        f"{(row['2yr']/row['0yr'])*100:.0f}%"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Disease', 'Immediate\n(0-Year)', '1-Year\nWashout', 
                          '2-Year\nWashout', 'AUC Drop', 'Retention'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code retention percentages
for i, (idx, row) in enumerate(df.iterrows()):
    retention = (row['2yr'] / row['0yr']) * 100
    if retention >= 75:
        color = '#C6EFCE'  # Light green
    elif retention >= 70:
        color = '#FFEB9C'  # Light yellow
    else:
        color = '#FFC7CE'  # Light red
    table[(i+1, 5)].set_facecolor(color)

plt.title('Washout Analysis Summary: Model Performance Retention', 
          fontsize=16, fontweight='bold', pad=20)
plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/washout_table.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/washout_table.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved washout summary table!")

plt.show()

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*80)
print("WASHOUT ANALYSIS SUMMARY STATISTICS")
print("="*80)
print(f"Mean AUC at 0-year: {df['0yr'].mean():.3f} (SD: {df['0yr'].std():.3f})")
print(f"Mean AUC at 1-year: {df['1yr'].mean():.3f} (SD: {df['1yr'].std():.3f})")
print(f"Mean AUC at 2-year: {df['2yr'].mean():.3f} (SD: {df['2yr'].std():.3f})")
print(f"\nMean AUC drop: {df['drop'].mean():.3f} (SD: {df['drop'].std():.3f})")
print(f"Mean retention: {df['retention'].mean():.1f}% (SD: {df['retention'].std():.1f}%)")
print(f"\nDiseases with AUC > 0.7 at 2-year washout: {len(df[df['2yr'] > 0.7])}/{len(df)}")
print(f"Diseases with AUC > 0.65 at 2-year washout: {len(df[df['2yr'] > 0.65])}/{len(df)}")
print(f"Diseases with AUC > 0.6 at 2-year washout: {len(df[df['2yr'] > 0.6])}/{len(df)}")
print("="*80)

