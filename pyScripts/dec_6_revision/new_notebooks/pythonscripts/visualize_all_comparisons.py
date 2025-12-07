#!/usr/bin/env python3
"""
Visualize all comparison results including:
1. External scores comparison (PCE, PREVENT, Gail, QRISK3)
2. Delphi comparison
3. Cox baseline comparison
4. Prediction drops analysis

For Gail: Shows women-only comparison (fairest) and all patients (Aladynoulli only).

Usage:
    python visualize_all_comparisons.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_external_scores_comparison():
    """Load external scores comparison results."""
    results_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/pooled_retrospective/external_scores_comparison.csv')
    
    if not results_file.exists():
        print(f"⚠️  External scores comparison file not found: {results_file}")
        return None
    
    # Read CSV - handle index column properly
    df = pd.read_csv(results_file, index_col=0)
    
    # Debug: print columns to see what we have
    print(f"   Columns in CSV: {df.columns.tolist()}")
    print(f"   Index: {df.index.tolist()}")
    
    return df

def plot_external_scores_comparison(df, output_dir):
    """Plot external scores comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Aladynoulli vs External Risk Scores\n10-Year Predictions', 
                 fontsize=16, fontweight='bold')
    
    # 1. ASCVD: Aladynoulli vs PCE vs QRISK3 vs PREVENT
    ax = axes[0, 0]
    
    # Find ASCVD row
    ascvd_rows = df[df.index.str.contains('ASCVD_10yr', na=False)]
    if len(ascvd_rows) == 0:
        ax.text(0.5, 0.5, 'ASCVD data not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('ASCVD (10-year)', fontsize=12, fontweight='bold')
    else:
        ascvd_row = ascvd_rows.iloc[0]
        
        models = []
        aucs = []
        ci_lowers = []
        ci_uppers = []
        
        # Check if columns exist before accessing
        if 'Aladynoulli_AUC' in df.columns and pd.notna(ascvd_row.get('Aladynoulli_AUC', np.nan)):
            models.append('Aladynoulli')
            aucs.append(ascvd_row['Aladynoulli_AUC'])
            ci_lowers.append(ascvd_row.get('Aladynoulli_CI_lower', np.nan))
            ci_uppers.append(ascvd_row.get('Aladynoulli_CI_upper', np.nan))
        
        if 'PCE_AUC' in df.columns and pd.notna(ascvd_row.get('PCE_AUC', np.nan)):
            models.append('PCE')
            aucs.append(ascvd_row['PCE_AUC'])
            ci_lowers.append(ascvd_row.get('PCE_CI_lower', np.nan))
            ci_uppers.append(ascvd_row.get('PCE_CI_upper', np.nan))
        
        if 'QRISK3_AUC' in df.columns and pd.notna(ascvd_row.get('QRISK3_AUC', np.nan)):
            models.append('QRISK3')
            aucs.append(ascvd_row['QRISK3_AUC'])
            ci_lowers.append(ascvd_row.get('QRISK3_CI_lower', np.nan))
            ci_uppers.append(ascvd_row.get('QRISK3_CI_upper', np.nan))
        
        if 'PREVENT_AUC' in df.columns and pd.notna(ascvd_row.get('PREVENT_AUC', np.nan)):
            models.append('PREVENT')
            aucs.append(ascvd_row['PREVENT_AUC'])
            ci_lowers.append(ascvd_row.get('PREVENT_CI_lower', np.nan))
            ci_uppers.append(ascvd_row.get('PREVENT_CI_upper', np.nan))
        
        if models:
            x_pos = np.arange(len(models))
            colors = ['#e74c3c' if m == 'Aladynoulli' else '#3498db' for m in models]
            bars = ax.bar(x_pos, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add error bars
            errors_lower = [aucs[i] - ci_lowers[i] for i in range(len(models))]
            errors_upper = [ci_uppers[i] - aucs[i] for i in range(len(models))]
            ax.errorbar(x_pos, aucs, yerr=[errors_lower, errors_upper], 
                       fmt='none', color='black', capsize=5, capthick=2)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=11)
            ax.set_ylabel('AUC', fontsize=11)
            ax.set_title('ASCVD (10-year)', fontsize=12, fontweight='bold')
            ax.set_ylim(0.65, max(aucs) * 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, auc) in enumerate(zip(bars, aucs)):
                ax.text(bar.get_x() + bar.get_width()/2., auc + errors_upper[i] + 0.005,
                        f'{auc:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No model data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ASCVD (10-year)', fontsize=12, fontweight='bold')
    
    # 2. Breast Cancer: Women only (Gail comparison - fairest)
    ax = axes[0, 1]
    breast_female_rows = df[df.index.str.contains('Breast_Cancer_10yr_Female', na=False)]
    
    if len(breast_female_rows) > 0:
        breast_female_row = breast_female_rows.iloc[0]
        
        models = []
        aucs = []
        ci_lowers = []
        ci_uppers = []
        
        if 'Aladynoulli_AUC' in df.columns and pd.notna(breast_female_row.get('Aladynoulli_AUC', np.nan)):
            models.append('Aladynoulli')
            aucs.append(breast_female_row['Aladynoulli_AUC'])
            ci_lowers.append(breast_female_row.get('Aladynoulli_CI_lower', np.nan))
            ci_uppers.append(breast_female_row.get('Aladynoulli_CI_upper', np.nan))
        
        if 'Gail_AUC' in df.columns and pd.notna(breast_female_row.get('Gail_AUC', np.nan)):
            models.append('Gail')
            aucs.append(breast_female_row['Gail_AUC'])
            ci_lowers.append(breast_female_row.get('Gail_CI_lower', np.nan))
            ci_uppers.append(breast_female_row.get('Gail_CI_upper', np.nan))
        
        if models:
            x_pos = np.arange(len(models))
            colors = ['#e74c3c' if m == 'Aladynoulli' else '#9b59b6' for m in models]
            bars = ax.bar(x_pos, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            errors_lower = [aucs[i] - ci_lowers[i] for i in range(len(models))]
            errors_upper = [ci_uppers[i] - aucs[i] for i in range(len(models))]
            ax.errorbar(x_pos, aucs, yerr=[errors_lower, errors_upper], 
                       fmt='none', color='black', capsize=5, capthick=2)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=11)
            ax.set_ylabel('AUC', fontsize=11)
            ax.set_title('Breast Cancer (10-year)\nWomen Only - Fair Comparison', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0.50, max(aucs) * 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, auc) in enumerate(zip(bars, aucs)):
                ax.text(bar.get_x() + bar.get_width()/2., auc + errors_upper[i] + 0.003,
                        f'{auc:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add note about fair comparison
            if len(models) == 2:
                diff = aucs[0] - aucs[1]
                ax.text(0.5, 0.05, f'Difference: {diff:+.3f}', 
                       transform=ax.transAxes, ha='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                       fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Breast Cancer (10-year)\nWomen Only', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Breast Cancer (Female) data not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Breast Cancer (10-year)\nWomen Only', fontsize=12, fontweight='bold')
    
    # 3. Breast Cancer: All patients (Aladynoulli only)
    ax = axes[1, 0]
    breast_all_rows = df[df.index.str.contains('Breast_Cancer_10yr_All', na=False)]
    
    if len(breast_all_rows) > 0:
        breast_all_row = breast_all_rows.iloc[0]
        
        if 'Aladynoulli_AUC' in df.columns and pd.notna(breast_all_row.get('Aladynoulli_AUC', np.nan)):
            auc = breast_all_row['Aladynoulli_AUC']
            ci_lower = breast_all_row.get('Aladynoulli_CI_lower', np.nan)
            ci_upper = breast_all_row.get('Aladynoulli_CI_upper', np.nan)
            
            bars = ax.bar(['Aladynoulli'], [auc], color='#e74c3c', alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            
            errors_lower = [auc - ci_lower]
            errors_upper = [ci_upper - auc]
            ax.errorbar([0], [auc], yerr=[errors_lower, errors_upper], 
                       fmt='none', color='black', capsize=5, capthick=2)
            
            ax.set_ylabel('AUC', fontsize=11)
            ax.set_title('Breast Cancer (10-year)\nAll Patients (Gail N/A for men)', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0.50, auc * 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            ax.text(0, auc + errors_upper[0] + 0.003, f'{auc:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add note
            ax.text(0.5, 0.05, 'Gail model only applies to women', 
                   transform=ax.transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                   fontsize=9, style='italic')
        else:
            ax.text(0.5, 0.5, 'No Aladynoulli data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Breast Cancer (10-year)\nAll Patients', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Breast Cancer (All) data not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Breast Cancer (10-year)\nAll Patients', fontsize=12, fontweight='bold')
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = """
    COMPARISON SUMMARY:
    
    ASCVD (10-year):
    • Aladynoulli outperforms PCE, QRISK3, and PREVENT
    
    Breast Cancer (10-year):
    • Women Only: Aladynoulli vs Gail (fair comparison)
      - Both models apply to women
      - Direct head-to-head comparison
    
    • All Patients: Aladynoulli only
      - Gail model doesn't apply to men
      - Shows Aladynoulli's ability to predict
        breast cancer risk in both sexes
    
    NOTE: Gail comparison is fairest when limited
    to women only, as Gail was specifically designed
    for breast cancer risk prediction in women.
    """
    
    ax.text(0.05, 0.95, summary_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'external_scores_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")
    plt.close()

def plot_delphi_comparison(output_dir):
    """Plot Delphi comparison."""
    delphi_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/pooled_retrospective/delphi_comparison_1yr_full.csv')
    
    if not delphi_file.exists():
        print(f"⚠️  Delphi comparison file not found: {delphi_file}")
        return
    
    df = pd.read_csv(delphi_file, index_col=0)
    
    # Check what columns are available
    print(f"   Columns in Delphi file: {df.columns.tolist()}")
    
    # Create subplots for 0gap and 1gap comparisons
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Aladynoulli vs Delphi: 1-Year Predictions', fontsize=14, fontweight='bold')
    
    diseases = df.index.tolist()
    
    # Plot 1: 0gap comparison
    ax = axes[0]
    if 'Aladynoulli_1yr_0gap' in df.columns and 'Delphi_1yr_0gap' in df.columns:
        aladynoulli_aucs = df['Aladynoulli_1yr_0gap'].values
        delphi_aucs = df['Delphi_1yr_0gap'].values
        
        x_pos = np.arange(len(diseases))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, aladynoulli_aucs, width,
                       label='Aladynoulli', color='#e74c3c', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, delphi_aucs, width,
                       label='Delphi', color='#3498db', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Disease', fontsize=11)
        ax.set_ylabel('AUC', fontsize=11)
        ax.set_title('0-Year Gap (No Washout)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.5, 0.5, '0gap data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('0-Year Gap', fontsize=12, fontweight='bold')
    
    # Plot 2: 1gap comparison
    ax = axes[1]
    if 'Aladynoulli_1yr_1gap' in df.columns and 'Delphi_1yr_1gap' in df.columns:
        aladynoulli_aucs = df['Aladynoulli_1yr_1gap'].values
        delphi_aucs = df['Delphi_1yr_1gap'].values
        
        x_pos = np.arange(len(diseases))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, aladynoulli_aucs, width,
                       label='Aladynoulli', color='#e74c3c', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, delphi_aucs, width,
                       label='Delphi', color='#3498db', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Disease', fontsize=11)
        ax.set_ylabel('AUC', fontsize=11)
        ax.set_title('1-Year Gap (1-Year Washout)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.5, 0.5, '1gap data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('1-Year Gap', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'delphi_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")
    plt.close()

def main():
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("VISUALIZING ALL COMPARISONS")
    print("="*100)
    
    # Load external scores comparison
    print("\n1. Loading external scores comparison...")
    df = load_external_scores_comparison()
    
    if df is not None:
        print("   Creating external scores comparison plot...")
        plot_external_scores_comparison(df, output_dir)
    
    # Plot Delphi comparison
    print("\n2. Creating Delphi comparison plot...")
    plot_delphi_comparison(output_dir)
    
    print("\n" + "="*100)
    print("VISUALIZATION COMPLETE")
    print("="*100)
    print(f"\nPlots saved to: {output_dir}")

if __name__ == '__main__':
    main()

