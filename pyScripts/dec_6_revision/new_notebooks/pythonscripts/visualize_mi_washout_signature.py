#!/usr/bin/env python3
"""
Visualize MI washout analysis with signature-based learning.

Shows:
- Predictions at t9 from models m0, m5, m9
- Signature 5 loadings over time
- Categorization by washout type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 11

def plot_mi_washout_analysis(output_dir=None):
    """Create comprehensive figure for MI washout analysis."""
    
    if output_dir is None:
        output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/analysis/plots')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis') / 'mi_washout_analysis_batch_0_10000.csv'
    
    if not data_file.exists():
        print(f"⚠️  Data file not found: {data_file}")
        print("Please run analyze_mi_washout_signature.py first")
        return
    
    df = pd.read_csv(data_file)
    
    # Remove rows with missing predictions
    df = df.dropna(subset=['m0t9', 'm5t9', 'm9t9'])
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Predictions at t9: m0 vs m5 vs m9
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Color by washout category
    for category, color in [('conservative', 'red'), ('accurate', 'green'), ('unrelated', 'orange'), ('neither', 'gray')]:
        mask = df['washout_category'] == category
        if mask.sum() > 0:
            sample = df.loc[mask].sample(min(500, len(df[mask])), random_state=42) if mask.sum() > 500 else df.loc[mask]
            ax1.scatter(sample['m0t9'], sample['m5t9'], 
                       alpha=0.5, s=30, label=f'{category.capitalize()}', 
                       color=color, edgecolors='black', linewidth=0.3)
    
    ax1.plot([0, 0.3], [0, 0.3], 'k--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('m0t9\n(Model trained to t0, predicts at t9)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('m5t9\n(Model trained to t5, predicts at t9)', fontsize=11, fontweight='bold')
    ax1.set_title('MI Predictions at t9: m0 vs m5\n(Same timepoint, different training data)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.3)
    ax1.set_ylim(0, 0.3)
    
    # 2. Predictions at t9: m5 vs m9
    ax2 = fig.add_subplot(gs[0, 1])
    
    for category, color in [('conservative', 'red'), ('accurate', 'green'), ('unrelated', 'orange'), ('neither', 'gray')]:
        mask = df['washout_category'] == category
        if mask.sum() > 0:
            sample = df.loc[mask].sample(min(500, len(df[mask])), random_state=42) if mask.sum() > 500 else df.loc[mask]
            ax2.scatter(sample['m5t9'], sample['m9t9'], 
                       alpha=0.5, s=30, label=f'{category.capitalize()}', 
                       color=color, edgecolors='black', linewidth=0.3)
    
    ax2.plot([0, 0.3], [0, 0.3], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('m5t9\n(Model trained to t5, predicts at t9)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('m9t9\n(Model trained to t9, predicts at t9)', fontsize=11, fontweight='bold')
    ax2.set_title('MI Predictions at t9: m5 vs m9\n(Same timepoint, different training data)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.3)
    ax2.set_ylim(0, 0.3)
    
    # 3. Mean predictions by washout category
    ax3 = fig.add_subplot(gs[0, 2])
    
    summary_data = []
    for category in ['conservative', 'accurate', 'unrelated', 'neither']:
        mask = df['washout_category'] == category
        if mask.sum() > 0:
            summary_data.append({
                'Category': category.replace('conservative', 'Conservative\n(Had MI)')
                                  .replace('accurate', 'Accurate\n(Sig5 Precursor)')
                                  .replace('unrelated', 'Unrelated')
                                  .replace('neither', 'Neither'),
                'm0t9': df.loc[mask, 'm0t9'].mean(),
                'm5t9': df.loc[mask, 'm5t9'].mean(),
                'm9t9': df.loc[mask, 'm9t9'].mean(),
                'N': mask.sum()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) > 0:
        x_pos = np.arange(len(summary_df))
        width = 0.25
        
        bars1 = ax3.bar(x_pos - width, summary_df['m0t9'], width, label='m0t9', color='blue', alpha=0.7)
        bars2 = ax3.bar(x_pos, summary_df['m5t9'], width, label='m5t9', color='orange', alpha=0.7)
        bars3 = ax3.bar(x_pos + width, summary_df['m9t9'], width, label='m9t9', color='green', alpha=0.7)
        
        ax3.set_xlabel('Washout Category', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Mean MI Probability at t9', fontsize=11, fontweight='bold')
        ax3.set_title('Mean MI Predictions at t9\nBy Washout Category', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(summary_df['Category'], rotation=0, ha='center')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Signature 5 loadings: m0sig5t5 vs m0sig5t9
    ax4 = fig.add_subplot(gs[1, 0])
    
    df_sig = df.dropna(subset=['m0sig5t5', 'm0sig5t9'])
    if len(df_sig) > 0:
        for category, color in [('conservative', 'red'), ('accurate', 'green'), ('unrelated', 'orange'), ('neither', 'gray')]:
            mask = df_sig['washout_category'] == category
            if mask.sum() > 0:
                sample = df_sig.loc[mask].sample(min(500, len(df_sig[mask])), random_state=42) if mask.sum() > 500 else df_sig.loc[mask]
                ax4.scatter(sample['m0sig5t5'], sample['m0sig5t9'], 
                           alpha=0.5, s=30, label=f'{category.capitalize()}', 
                           color=color, edgecolors='black', linewidth=0.3)
        
        ax4.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, linewidth=1)
        ax4.set_xlabel('m0sig5t5\n(Signature 5 loading at t5)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('m0sig5t9\n(Signature 5 loading at t9)', fontsize=11, fontweight='bold')
        ax4.set_title('Signature 5 Loadings: t5 vs t9\n(Model m0)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # 5. Signature 5 loadings: m0sig5t9 vs m9sig5t9
    ax5 = fig.add_subplot(gs[1, 1])
    
    df_sig2 = df.dropna(subset=['m0sig5t9', 'm9sig5t9'])
    if len(df_sig2) > 0:
        for category, color in [('conservative', 'red'), ('accurate', 'green'), ('unrelated', 'orange'), ('neither', 'gray')]:
            mask = df_sig2['washout_category'] == category
            if mask.sum() > 0:
                sample = df_sig2.loc[mask].sample(min(500, len(df_sig2[mask])), random_state=42) if mask.sum() > 500 else df_sig2.loc[mask]
                ax5.scatter(sample['m0sig5t9'], sample['m9sig5t9'], 
                           alpha=0.5, s=30, label=f'{category.capitalize()}', 
                           color=color, edgecolors='black', linewidth=0.3)
        
        ax5.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, linewidth=1)
        ax5.set_xlabel('m0sig5t9\n(Model m0, Signature 5 at t9)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('m9sig5t9\n(Model m9, Signature 5 at t9)', fontsize=11, fontweight='bold')
        ax5.set_title('Signature 5 Loadings at t9\n(m0 vs m9)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # 6. Prediction changes: m5t9 - m0t9 vs m9t9 - m5t9
    ax6 = fig.add_subplot(gs[1, 2])
    
    df['change_m0_to_m5'] = df['m5t9'] - df['m0t9']
    df['change_m5_to_m9'] = df['m9t9'] - df['m5t9']
    
    df_changes = df.dropna(subset=['change_m0_to_m5', 'change_m5_to_m9'])
    if len(df_changes) > 0:
        for category, color in [('conservative', 'red'), ('accurate', 'green'), ('unrelated', 'orange'), ('neither', 'gray')]:
            mask = df_changes['washout_category'] == category
            if mask.sum() > 0:
                sample = df_changes.loc[mask].sample(min(500, len(df_changes[mask])), random_state=42) if mask.sum() > 500 else df_changes.loc[mask]
                ax6.scatter(sample['change_m0_to_m5'], sample['change_m5_to_m9'], 
                           alpha=0.5, s=30, label=f'{category.capitalize()}', 
                           color=color, edgecolors='black', linewidth=0.3)
        
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax6.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax6.set_xlabel('Change: m5t9 - m0t9\n(More training data: t0→t5)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Change: m9t9 - m5t9\n(More training data: t5→t9)', fontsize=11, fontweight='bold')
        ax6.set_title('Prediction Changes\n(How predictions evolve with more data)', 
                      fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
    
    # 7. 3x3 Structure: Models × Time Periods (MI + Precursors)
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Create 3x3 matrix: rows = models (m0t9, m5t9, m9t9), cols = time periods (baseline, t0-t5, t5-t9)
    # For each model, show mean prediction for patients who developed MI OR precursors in each time period
    matrix_data = []
    matrix_counts = []  # Also track patient counts
    
    models = ['m0t9', 'm5t9', 'm9t9']
    model_labels = ['m0t9\n(trained to t0)', 'm5t9\n(trained to t5)', 'm9t9\n(trained to t9)']
    time_period_labels = ['Baseline\n(before t0)', 'Interval\nt0-t5', 'Interval\nt5-t9']
    
    # Check for MI or precursors in each period
    period_masks = [
        (df['MI_at_baseline'] == True) | (df['n_precursors_at_baseline'] > 0),  # Baseline
        (df['MI_between_t0_t5'] == True) | (df['n_precursors_between_t0_t5'] > 0),  # t0-t5
        (df['MI_between_t5_t9'] == True) | (df['n_precursors_between_t5_t9'] > 0)  # t5-t9
    ]
    
    for model_col in models:
        row_data = []
        row_counts = []
        for mask in period_masks:
            if mask.sum() > 0:
                mean_pred = df.loc[mask, model_col].mean()
                n_patients = mask.sum()
            else:
                mean_pred = 0
                n_patients = 0
            row_data.append(mean_pred)
            row_counts.append(n_patients)
        matrix_data.append(row_data)
        matrix_counts.append(row_counts)
    
    matrix = np.array(matrix_data)
    counts_matrix = np.array(matrix_counts)
    
    # Create heatmap
    vmax = matrix.max() if matrix.max() > 0 else 0.01
    im = ax7.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    
    # Set ticks and labels
    ax7.set_xticks(np.arange(len(time_period_labels)))
    ax7.set_yticks(np.arange(len(model_labels)))
    ax7.set_xticklabels(time_period_labels, fontsize=10)
    ax7.set_yticklabels(model_labels, fontsize=10)
    
    # Add text annotations (prediction + count)
    for i in range(len(models)):
        for j in range(len(time_period_labels)):
            pred_text = f'{matrix[i, j]:.4f}'
            count_text = f'\n(n={counts_matrix[i, j]})'
            text = ax7.text(j, i, pred_text + count_text,
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax7.set_title('3×3: Mean MI Predictions by Model × Time Period\n(Patients with MI or Precursors in Each Period)', 
                  fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Mean MI Probability', fontsize=10)
    
    # 8. Distribution of prediction changes by category
    ax8 = fig.add_subplot(gs[2, 1])
    
    for category, color in [('conservative', 'red'), ('accurate', 'green'), ('unrelated', 'orange'), ('neither', 'gray')]:
        mask = df['washout_category'] == category
        changes = df.loc[mask, 'change_m0_to_m5'].dropna()
        if len(changes) > 0:
            ax8.hist(changes, bins=50, alpha=0.5, label=f'{category.capitalize()}', 
                    color=color, density=True)
    
    ax8.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax8.set_xlabel('Prediction Change: m5t9 - m0t9', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax8.set_title('Distribution of Prediction Changes\n(m0→m5, by category)', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Explanation text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    explanation_text = """
    MI WASHOUT ANALYSIS - SIGNATURE-BASED LEARNING
    
    KEY METRICS:
    • m0t9, m5t9, m9t9: MI probability at t9 from models
      trained to t0, t5, t9 (all predict at SAME timepoint)
    
    • m0sig5t5, m0sig5t9, m9sig5t9: Signature 5 loadings
      at different timepoints from different models
    
    • MI_at_t0, t5, t9: Cumulative MI status
    
    • Precursors_at_t0, t5, t9: Signature 5 precursor diseases
      (excluding MI itself)
    
    WASHOUT CATEGORIES:
    
    1. CONSERVATIVE (Red):
       • Patient got MI between t0-t5 or t5-t9
       • Prediction changes because model learned from real outcome
    
    2. ACCURATE (Green):
       • Patient got Signature 5 precursor (but NOT MI)
       • Prediction changes because model detected pre-clinical signal
       • Shows signature-based learning!
    
    3. UNRELATED (Orange):
       • Patient got non-Signature 5 disease
    
    4. NEITHER (Gray):
       • Patient got nothing new
    
    KEY INSIGHT:
    Signature 5 precursors (cardiovascular diseases) should
    show "accurate washout" - demonstrating the model learns
    from signature-related pre-clinical signals!
    """
    
    ax9.text(0.05, 0.95, explanation_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('MI Washout Analysis: Signature-Based Learning\n' + 
                 'Predictions at t9 from Models Trained with Different Amounts of Data',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = output_dir / 'mi_washout_signature_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")
    
    plt.close()

def main():
    print("="*80)
    print("VISUALIZING MI WASHOUT ANALYSIS")
    print("="*80)
    
    plot_mi_washout_analysis()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

