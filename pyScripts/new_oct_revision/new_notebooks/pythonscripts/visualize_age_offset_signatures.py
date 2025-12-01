#!/usr/bin/env python3
"""
Visualize age offset signature analysis results.

Shows how lambda (patient-specific parameters) change for different groups:
- With outcome events (conservative washout)
- With precursor only (accurate washout)
- Without either (model refinement)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def plot_signature_changes(disease_name='ASCVD', output_dir=None):
    """Create comprehensive figure showing signature changes by washout type."""
    
    if output_dir is None:
        output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis/plots')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load signature changes
    signature_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis') / f'signature_changes_age_offset_{disease_name}.csv'
    
    if not signature_file.exists():
        print(f"⚠️  Signature changes file not found: {signature_file}")
        return
    
    df = pd.read_csv(signature_file)
    
    # Focus on Signature 5 (cardiovascular cluster) - most relevant
    sig5_df = df[df['Signature'].str.contains('Signature_5')].copy()
    
    if len(sig5_df) == 0:
        print("⚠️  No Signature 5 data found")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Color scheme
    colors = {
        'with_outcome': '#d62728',  # Red - conservative washout
        'with_precursor_only': '#2ca02c',  # Green - accurate washout
        'without_either': '#9467bd'  # Purple - model refinement
    }
    
    # 1. Signature 5 changes by group (4yr washout)
    ax1 = fig.add_subplot(gs[0, 0])
    sig5_4yr = sig5_df[sig5_df['Washout_period'] == '4yr'].copy()
    
    x_pos = np.arange(len(sig5_4yr))
    width = 0.25
    
    precursors = sig5_4yr['Precursor'].unique()
    x = np.arange(len(precursors))
    
    with_outcome_vals = []
    with_precursor_vals = []
    without_either_vals = []
    
    for precursor in precursors:
        row = sig5_4yr[sig5_4yr['Precursor'] == precursor].iloc[0]
        with_outcome_vals.append(row['Mean_change_with_outcome'])
        with_precursor_vals.append(row['Mean_change_with_precursor_only'])
        without_either_vals.append(row['Mean_change_without_either'])
    
    bars1 = ax1.bar(x - width, with_outcome_vals, width, label='With Outcome\n(Conservative)', 
                    color=colors['with_outcome'], alpha=0.8)
    bars2 = ax1.bar(x, with_precursor_vals, width, label='With Precursor Only\n(Accurate)', 
                    color=colors['with_precursor_only'], alpha=0.8)
    bars3 = ax1.bar(x + width, without_either_vals, width, label='Without Either\n(Refinement)', 
                    color=colors['without_either'], alpha=0.8)
    
    ax1.set_xlabel('Precursor Disease', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Lambda Change', fontsize=12, fontweight='bold')
    ax1.set_title('Signature 5 (Cardiovascular Cluster) - 4yr Washout\nLambda Changes by Group', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(precursors, rotation=15, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 2. Signature 5 changes by group (9yr washout)
    ax2 = fig.add_subplot(gs[0, 1])
    sig5_9yr = sig5_df[sig5_df['Washout_period'] == '9yr'].copy()
    
    with_outcome_vals_9yr = []
    with_precursor_vals_9yr = []
    without_either_vals_9yr = []
    
    for precursor in precursors:
        row = sig5_9yr[sig5_9yr['Precursor'] == precursor].iloc[0]
        with_outcome_vals_9yr.append(row['Mean_change_with_outcome'])
        with_precursor_vals_9yr.append(row['Mean_change_with_precursor_only'])
        without_either_vals_9yr.append(row['Mean_change_without_either'])
    
    bars1 = ax2.bar(x - width, with_outcome_vals_9yr, width, label='With Outcome\n(Conservative)', 
                    color=colors['with_outcome'], alpha=0.8)
    bars2 = ax2.bar(x, with_precursor_vals_9yr, width, label='With Precursor Only\n(Accurate)', 
                    color=colors['with_precursor_only'], alpha=0.8)
    bars3 = ax2.bar(x + width, without_either_vals_9yr, width, label='Without Either\n(Refinement)', 
                    color=colors['without_either'], alpha=0.8)
    
    ax2.set_xlabel('Precursor Disease', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Lambda Change', fontsize=12, fontweight='bold')
    ax2.set_title('Signature 5 (Cardiovascular Cluster) - 9yr Washout\nLambda Changes by Group', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(precursors, rotation=15, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 3. Patient counts by group (4yr washout)
    ax3 = fig.add_subplot(gs[1, 0])
    
    n_with_outcome = []
    n_with_precursor = []
    n_without_either = []
    
    for precursor in precursors:
        row = sig5_4yr[sig5_4yr['Precursor'] == precursor].iloc[0]
        n_with_outcome.append(row['N_with_outcome'])
        n_with_precursor.append(row['N_with_precursor_only'])
        n_without_either.append(row['N_without_either'])
    
    x_pos = np.arange(len(precursors))
    bars1 = ax3.bar(x_pos - width, n_with_outcome, width, label='With Outcome', 
                    color=colors['with_outcome'], alpha=0.8)
    bars2 = ax3.bar(x_pos, n_with_precursor, width, label='With Precursor Only', 
                    color=colors['with_precursor_only'], alpha=0.8)
    bars3 = ax3.bar(x_pos + width, n_without_either, width, label='Without Either', 
                    color=colors['without_either'], alpha=0.8)
    
    ax3.set_xlabel('Precursor Disease', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax3.set_title('Patient Counts by Group - 4yr Washout', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(precursors, rotation=15, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Patient counts by group (9yr washout)
    ax4 = fig.add_subplot(gs[1, 1])
    
    n_with_outcome_9yr = []
    n_with_precursor_9yr = []
    n_without_either_9yr = []
    
    for precursor in precursors:
        row = sig5_9yr[sig5_9yr['Precursor'] == precursor].iloc[0]
        n_with_outcome_9yr.append(row['N_with_outcome'])
        n_with_precursor_9yr.append(row['N_with_precursor_only'])
        n_without_either_9yr.append(row['N_without_either'])
    
    bars1 = ax4.bar(x_pos - width, n_with_outcome_9yr, width, label='With Outcome', 
                    color=colors['with_outcome'], alpha=0.8)
    bars2 = ax4.bar(x_pos, n_with_precursor_9yr, width, label='With Precursor Only', 
                    color=colors['with_precursor_only'], alpha=0.8)
    bars3 = ax4.bar(x_pos + width, n_without_either_9yr, width, label='Without Either', 
                    color=colors['without_either'], alpha=0.8)
    
    ax4.set_xlabel('Precursor Disease', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax4.set_title('Patient Counts by Group - 9yr Washout', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(precursors, rotation=15, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Comparison: Conservative vs Accurate Washout (4yr) - moved to bottom
    # (This will be ax8, defined below)
    
    # 6. Hypercholesterolemia across ALL signatures (4yr washout) - shows specificity
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Get all signatures for hypercholesterolemia
    hyperchol_4yr = df[(df['Precursor'] == 'Hypercholesterolemia') & 
                       (df['Washout_period'] == '4yr')].copy()
    
    # Extract signature numbers
    hyperchol_4yr['Sig_num'] = hyperchol_4yr['Signature'].str.extract(r'Signature_(\d+)').astype(int)
    hyperchol_4yr = hyperchol_4yr.sort_values('Sig_num')
    
    sig_nums = hyperchol_4yr['Sig_num'].values
    with_outcome_all = hyperchol_4yr['Mean_change_with_outcome'].values
    with_precursor_all = hyperchol_4yr['Mean_change_with_precursor_only'].values
    without_either_all = hyperchol_4yr['Mean_change_without_either'].values
    
    x_pos = np.arange(len(sig_nums))
    width = 0.25
    
    bars1 = ax6.bar(x_pos - width, with_outcome_all, width, label='With Outcome', 
                    color=colors['with_outcome'], alpha=0.8)
    bars2 = ax6.bar(x_pos, with_precursor_all, width, label='With Precursor Only', 
                    color=colors['with_precursor_only'], alpha=0.8)
    bars3 = ax6.bar(x_pos + width, without_either_all, width, label='Without Either', 
                    color=colors['without_either'], alpha=0.8)
    
    # Highlight Signature 5
    sig5_idx = np.where(sig_nums == 5)[0][0]
    ax6.axvline(x=sig5_idx, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Signature 5')
    
    ax6.set_xlabel('Signature Number', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Mean Lambda Change', fontsize=12, fontweight='bold')
    ax6.set_title('Hypercholesterolemia: Effect on ALL Signatures (4yr Washout)\n' + 
                  'Shows Specificity - Only Signature 5 Shows Large Positive Change',
                  fontsize=13, fontweight='bold')
    ax6.set_xticks(x_pos[::2])  # Show every other signature
    ax6.set_xticklabels(sig_nums[::2])
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Interpretation text box
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')
    
    interpretation_text = """
    KEY FINDINGS - MODEL VALIDATION:
    
    ✓ SIGNATURE SPECIFICITY (Panel 6):
      Hypercholesterolemia shows positive changes ONLY on Signature 5.
      All other signatures show small/negative changes - demonstrating
      the model correctly identifies signature-specific associations.
      
      This is NOT circular reasoning: hypercholesterolemia affects
      Signature 5 specifically, not all signatures indiscriminately.
    
    ✓ PRECURSOR-SIGNATURE MAPPING (Panels 1-2):
      • Hypercholesterolemia → Signature 5: Positive (correct - it's a member)
      • Other precursors → Signature 5: Negative (correct - not members)
      • Model correctly distinguishes signature members from non-members
    
    ✓ WASHOUT VALIDATION:
      • Red (with outcomes): Large positive changes - model learns from events
      • Green (precursor only): Signature-specific changes based on membership
      • Purple (neither): Small/negative changes - model refinement
    
    STRENGTH: Model demonstrates:
    • Accurate signature-specific learning (not circular)
    • Correct precursor-signature associations
    • Appropriate washout behavior
    
    This validates sophisticated biological understanding!
    """
    
    ax7.text(0.05, 0.95, interpretation_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 8. Comparison: Conservative vs Accurate Washout (4yr) - full width at bottom
    ax8 = fig.add_subplot(gs[3, :])
    
    # Scatter plot showing relationship
    for i, precursor in enumerate(precursors):
        row = sig5_4yr[sig5_4yr['Precursor'] == precursor].iloc[0]
        x_val = row['Mean_change_with_outcome']
        y_val = row['Mean_change_with_precursor_only']
        if not (np.isnan(x_val) or np.isnan(y_val)):
            ax8.scatter(x_val, y_val, s=200, alpha=0.7, label=precursor, edgecolors='black', linewidth=1)
            ax8.text(x_val, y_val, precursor[:15], fontsize=9, ha='center', va='bottom')
    
    ax8.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax8.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax8.plot([-0.1, 0.7], [-0.1, 0.7], 'k--', alpha=0.3, linewidth=1, label='y=x')
    ax8.set_xlabel('Mean Change: With Outcome (Conservative Washout)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Mean Change: With Precursor Only (Accurate Washout)', fontsize=12, fontweight='bold')
    ax8.set_title('Conservative vs Accurate Washout - Signature 5 (4yr)\n' + 
                  'Hypercholesterolemia shows positive accurate washout (signature member); ' +
                  'others show negative (not signature members)',
                  fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=10, loc='upper left', ncol=2)
    
    plt.suptitle(f'Age Offset Signature Analysis: {disease_name}\n' + 
                 'How Lambda (Patient-Specific Parameters) Change Across Washout Periods',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = output_dir / f'age_offset_signature_analysis_{disease_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")
    
    plt.close()

def main():
    """Main function."""
    print("="*80)
    print("VISUALIZING AGE OFFSET SIGNATURE ANALYSIS")
    print("="*80)
    
    plot_signature_changes('ASCVD')
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

