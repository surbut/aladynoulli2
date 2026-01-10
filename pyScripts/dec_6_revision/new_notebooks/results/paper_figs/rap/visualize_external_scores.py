# Load and display external scores comparison results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

results_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/pooled_retrospective')
external_scores_file = results_dir / 'external_scores_comparison.csv'

# Load static 10-year results to get women-only breast cancer AUC
static_results_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/time_horizons/pooled_retrospective/static_10yr_results.csv')
breast_cancer_women_only_auc = None
breast_cancer_women_only_ci_lower = None
breast_cancer_women_only_ci_upper = None

if static_results_file.exists():
    static_df = pd.read_csv(static_results_file)
    breast_row = static_df[static_df['Disease'] == 'Breast_Cancer']
    if len(breast_row) > 0:
        breast_cancer_women_only_auc = breast_row.iloc[0]['AUC']
        breast_cancer_women_only_ci_lower = breast_row.iloc[0]['CI_lower']
        breast_cancer_women_only_ci_upper = breast_row.iloc[0]['CI_upper']
        print(f"Loaded women-only 10-year breast cancer AUC: {breast_cancer_women_only_auc:.4f}")

if external_scores_file.exists():
    df = pd.read_csv(external_scores_file, index_col=0)
    
    print("="*80)
    print("COMPARISON WITH ESTABLISHED CLINICAL RISK SCORES")
    print("="*80)
    
    # Display summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    
    # ASCVD 10-year
    if 'ASCVD_10yr' in df.index:
        row = df.loc['ASCVD_10yr']
        summary_data.append({
            'Outcome': 'ASCVD (10-year)',
            'Aladynoulli AUC': f"{row['Aladynoulli_AUC']:.4f}",
            'PCE AUC': f"{row['PCE_AUC']:.4f}" if pd.notna(row.get('PCE_AUC')) else 'N/A',
            'QRISK3 AUC': f"{row['QRISK3_AUC']:.4f}" if pd.notna(row.get('QRISK3_AUC')) else 'N/A',
            'PREVENT (10yr) AUC': f"{row['PREVENT_10yr_AUC']:.4f}" if pd.notna(row.get('PREVENT_10yr_AUC')) else 'N/A',
            'N Patients': int(row['N_patients'])
        })
    
    # Breast Cancer 10-year (use women-only AUC for fair comparison)
    if 'Breast_Cancer_10yr' in df.index:
        row = df.loc['Breast_Cancer_10yr']
        # Use women-only AUC if available, otherwise fall back to full population
        aladynoulli_auc = breast_cancer_women_only_auc if breast_cancer_women_only_auc is not None else row['Aladynoulli_AUC']
        summary_data.append({
            'Outcome': 'Breast Cancer (10-year, women only)',
            'Aladynoulli AUC': f"{aladynoulli_auc:.4f}",
            'PCE AUC': 'N/A',
            'QRISK3 AUC': 'N/A',
            'PREVENT (10yr) AUC': 'N/A',
            'GAIL AUC': f"{row['Gail_AUC']:.4f}" if pd.notna(row.get('Gail_AUC')) else 'N/A',
            'N Patients': int(row.get('N_patients_gail', row['N_patients']))
        })
    
    # Breast Cancer 1-year
    if 'Breast_Cancer_1yr' in df.index:
        row = df.loc['Breast_Cancer_1yr']
        summary_data.append({
            'Outcome': 'Breast Cancer (1-year)',
            'Aladynoulli AUC': f"{row['Aladynoulli_AUC']:.4f}",
            'PCE AUC': 'N/A',
            'QRISK3 AUC': 'N/A',
            'PREVENT (10yr) AUC': 'N/A',
            'GAIL AUC': f"{row['Gail_AUC']:.4f}" if pd.notna(row.get('Gail_AUC')) else 'N/A',
            'N Patients': int(row['N_patients'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    # 10-year ASCVD
    if 'ASCVD_10yr' in df.index:
        row = df.loc['ASCVD_10yr']
        print(f"\n10-YEAR ASCVD PREDICTION:")
        print(f"  Aladynoulli:  {row['Aladynoulli_AUC']:.4f} ({row['Aladynoulli_CI_lower']:.4f}-{row['Aladynoulli_CI_upper']:.4f})")
        
        if pd.notna(row.get('PCE_AUC')):
            print(f"  PCE:          {row['PCE_AUC']:.4f} ({row['PCE_CI_lower']:.4f}-{row['PCE_CI_upper']:.4f})")
            diff = row['Difference']
            pct = (diff / row['PCE_AUC']) * 100
            print(f"  Difference:   {diff:+.4f} ({pct:+.2f}%)")
        
        if pd.notna(row.get('QRISK3_AUC')):
            print(f"  QRISK3:       {row['QRISK3_AUC']:.4f} ({row['QRISK3_CI_lower']:.4f}-{row['QRISK3_CI_upper']:.4f})")
            qrisk3_diff = row['QRISK3_Difference']
            qrisk3_pct = (qrisk3_diff / row['QRISK3_AUC']) * 100
            print(f"  Difference:   {qrisk3_diff:+.4f} ({qrisk3_pct:+.2f}%)")
        
        if pd.notna(row.get('PREVENT_10yr_AUC')):
            print(f"  PREVENT (10yr): {row['PREVENT_10yr_AUC']:.4f} ({row['PREVENT_10yr_CI_lower']:.4f}-{row['PREVENT_10yr_CI_upper']:.4f})")
            prevent_diff = row['PREVENT_10yr_Difference']
            prevent_pct = (prevent_diff / row['PREVENT_10yr_AUC']) * 100
            print(f"  Difference:     {prevent_diff:+.4f} ({prevent_pct:+.2f}%)")
        
        print(f"  N patients:   {int(row['N_patients'])}")
        print(f"  N events:     {int(row['N_events'])}")
    
    # Breast Cancer 10-year (use women-only AUC for fair comparison)
    print("\n" + "="*80)
    print("BREAST CANCER PREDICTIONS (10-YEAR, WOMEN ONLY)")
    print("="*80)
    
    if 'Breast_Cancer_10yr' in df.index:
        row = df.loc['Breast_Cancer_10yr']
        # Use women-only AUC if available, otherwise fall back to full population
        if breast_cancer_women_only_auc is not None:
            aladynoulli_auc = breast_cancer_women_only_auc
            aladynoulli_ci_lower = breast_cancer_women_only_ci_lower
            aladynoulli_ci_upper = breast_cancer_women_only_ci_upper
            print(f"\nCOMPARISON (Women Only - Fair Comparison):")
            print(f"  Aladynoulli (Women Only):     {aladynoulli_auc:.4f} ({aladynoulli_ci_lower:.4f}-{aladynoulli_ci_upper:.4f})")
        else:
            aladynoulli_auc = row['Aladynoulli_AUC']
            aladynoulli_ci_lower = row['Aladynoulli_CI_lower']
            aladynoulli_ci_upper = row['Aladynoulli_CI_upper']
            print(f"\nCOMPARISON:")
            print(f"  Aladynoulli (Full Population):  {aladynoulli_auc:.4f} ({aladynoulli_ci_lower:.4f}-{aladynoulli_ci_upper:.4f})")
        
        if pd.notna(row.get('Gail_AUC')):
            print(f"  GAIL (Women Only):            {row['Gail_AUC']:.4f} ({row['Gail_CI_lower']:.4f}-{row['Gail_CI_upper']:.4f})")
            diff = aladynoulli_auc - row['Gail_AUC']
            pct = (diff / row['Gail_AUC']) * 100
            print(f"  Difference:                   {diff:+.4f} ({pct:+.2f}%)")
            if breast_cancer_women_only_auc is not None:
                print(f"\n  Note: Both Aladynoulli and GAIL use women only for fair comparison")
            else:
                print(f"\n  Note: Aladynoulli uses full population (men + women), GAIL uses women only")
        
        if 'N_patients_gail' in row and pd.notna(row['N_patients_gail']):
            print(f"  N patients:                   {int(row['N_patients_gail'])}")
        else:
            print(f"  N patients (Aladynoulli):     {int(row['N_patients'])}")
        if 'N_events_gail' in row and pd.notna(row['N_events_gail']):
            print(f"  N events:                     {int(row['N_events_gail'])}")
        else:
            print(f"  N events (Aladynoulli):       {int(row['N_events'])}")
    
    # Breast Cancer 1-year
    print("\n" + "="*80)
    print("BREAST CANCER PREDICTIONS (1-YEAR)")
    print("="*80)
    
    if 'Breast_Cancer_1yr' in df.index:
        row = df.loc['Breast_Cancer_1yr']
        print(f"\nCOMPARISON (Women Only):")
        print(f"  Aladynoulli (washout 0yr):  {row['Aladynoulli_AUC']:.4f} ({row['Aladynoulli_CI_lower']:.4f}-{row['Aladynoulli_CI_upper']:.4f})")
        
        if pd.notna(row.get('Gail_AUC')):
            print(f"  GAIL (1-year):               {row['Gail_AUC']:.4f} ({row['Gail_CI_lower']:.4f}-{row['Gail_CI_upper']:.4f})")
            if 'Difference' in row and pd.notna(row['Difference']):
                diff = row['Difference']
            else:
                diff = row['Aladynoulli_AUC'] - row['Gail_AUC']
            pct = (diff / row['Gail_AUC']) * 100
            print(f"  Difference:                  {diff:+.4f} ({pct:+.2f}%)")
            print(f"\n  Note: Both Aladynoulli (washout 0yr) and GAIL use women only")
        
        print(f"  N patients:   {int(row['N_patients'])}")
        print(f"  N events:     {int(row['N_events'])}")
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Aladynoulli vs Established Clinical Risk Scores', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. ASCVD 10-year: Aladynoulli vs PCE vs QRISK3 vs PREVENT
    ax1 = fig.add_subplot(gs[0, 0])
    if 'ASCVD_10yr' in df.index:
        row = df.loc['ASCVD_10yr']
        models = []
        aucs = []
        ci_lowers = []
        ci_uppers = []
        
        if pd.notna(row.get('Aladynoulli_AUC')):
            models.append('Aladynoulli')
            aucs.append(row['Aladynoulli_AUC'])
            ci_lowers.append(row['Aladynoulli_CI_lower'])
            ci_uppers.append(row['Aladynoulli_CI_upper'])
        
        if pd.notna(row.get('PCE_AUC')):
            models.append('PCE')
            aucs.append(row['PCE_AUC'])
            ci_lowers.append(row['PCE_CI_lower'])
            ci_uppers.append(row['PCE_CI_upper'])
        
        if pd.notna(row.get('QRISK3_AUC')):
            models.append('QRISK3')
            aucs.append(row['QRISK3_AUC'])
            ci_lowers.append(row['QRISK3_CI_lower'])
            ci_uppers.append(row['QRISK3_CI_upper'])
        
        if pd.notna(row.get('PREVENT_10yr_AUC')):
            models.append('PREVENT\n(10yr)')
            aucs.append(row['PREVENT_10yr_AUC'])
            ci_lowers.append(row['PREVENT_10yr_CI_lower'])
            ci_uppers.append(row['PREVENT_10yr_CI_upper'])
        
        if models:
            x_pos = np.arange(len(models))
            colors = ['#2c7fb8' if m == 'Aladynoulli' else '#e74c3c' if m == 'PCE' else '#f39c12' if m == 'QRISK3' else '#27ae60' for m in models]
            bars = ax1.bar(x_pos, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            errors_lower = [aucs[i] - ci_lowers[i] for i in range(len(models))]
            errors_upper = [ci_uppers[i] - aucs[i] for i in range(len(models))]
            ax1.errorbar(x_pos, aucs, yerr=[errors_lower, errors_upper], 
                       fmt='none', color='black', capsize=5, capthick=2)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(models, fontsize=11)
            ax1.set_ylabel('AUC', fontsize=12, fontweight='bold')
            ax1.set_title('ASCVD 10-Year Prediction', fontsize=13, fontweight='bold')
            ax1.set_ylim(0.60, max(aucs) * 1.05)
            ax1.grid(axis='y', alpha=0.3)
            
            for i, (bar, auc) in enumerate(zip(bars, aucs)):
                ax1.text(bar.get_x() + bar.get_width()/2., auc + errors_upper[i] + 0.003,
                        f'{auc:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Combined Breast Cancer: 10-year and 1-year in one panel
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for both 10-year and 1-year
    breast_10yr_data = None
    breast_1yr_data = None
    
    # Get 10-year data
    if 'Breast_Cancer_10yr' in df.index:
        row_10yr = df.loc['Breast_Cancer_10yr']
        if breast_cancer_women_only_auc is not None:
            breast_10yr_data = {
                'Aladynoulli': (breast_cancer_women_only_auc, 
                              breast_cancer_women_only_ci_lower, 
                              breast_cancer_women_only_ci_upper),
                'GAIL': (row_10yr['Gail_AUC'], 
                        row_10yr['Gail_CI_lower'], 
                        row_10yr['Gail_CI_upper']) if pd.notna(row_10yr.get('Gail_AUC')) else None
            }
        elif pd.notna(row_10yr.get('Aladynoulli_AUC')):
            breast_10yr_data = {
                'Aladynoulli': (row_10yr['Aladynoulli_AUC'], 
                              row_10yr['Aladynoulli_CI_lower'], 
                              row_10yr['Aladynoulli_CI_upper']),
                'GAIL': (row_10yr['Gail_AUC'], 
                        row_10yr['Gail_CI_lower'], 
                        row_10yr['Gail_CI_upper']) if pd.notna(row_10yr.get('Gail_AUC')) else None
            }
    
    # Get 1-year data
    if 'Breast_Cancer_1yr' in df.index:
        row_1yr = df.loc['Breast_Cancer_1yr']
        if pd.notna(row_1yr.get('Aladynoulli_AUC')) and pd.notna(row_1yr.get('Gail_AUC')):
            breast_1yr_data = {
                'Aladynoulli': (row_1yr['Aladynoulli_AUC'], 
                              row_1yr['Aladynoulli_CI_lower'], 
                              row_1yr['Aladynoulli_CI_upper']),
                'GAIL': (row_1yr['Gail_AUC'], 
                        row_1yr['Gail_CI_lower'], 
                        row_1yr['Gail_CI_upper'])
            }
    
    # Create grouped bar chart
    if breast_10yr_data or breast_1yr_data:
        x_groups = []
        aladynoulli_aucs = []
        aladynoulli_ci_lowers = []
        aladynoulli_ci_uppers = []
        gail_aucs = []
        gail_ci_lowers = []
        gail_ci_uppers = []
        
        if breast_10yr_data:
            x_groups.append('10-Year')
            aladynoulli_aucs.append(breast_10yr_data['Aladynoulli'][0])
            aladynoulli_ci_lowers.append(breast_10yr_data['Aladynoulli'][1])
            aladynoulli_ci_uppers.append(breast_10yr_data['Aladynoulli'][2])
            if breast_10yr_data['GAIL']:
                gail_aucs.append(breast_10yr_data['GAIL'][0])
                gail_ci_lowers.append(breast_10yr_data['GAIL'][1])
                gail_ci_uppers.append(breast_10yr_data['GAIL'][2])
            else:
                gail_aucs.append(np.nan)
                gail_ci_lowers.append(np.nan)
                gail_ci_uppers.append(np.nan)
        
        if breast_1yr_data:
            x_groups.append('1-Year')
            aladynoulli_aucs.append(breast_1yr_data['Aladynoulli'][0])
            aladynoulli_ci_lowers.append(breast_1yr_data['Aladynoulli'][1])
            aladynoulli_ci_uppers.append(breast_1yr_data['Aladynoulli'][2])
            gail_aucs.append(breast_1yr_data['GAIL'][0])
            gail_ci_lowers.append(breast_1yr_data['GAIL'][1])
            gail_ci_uppers.append(breast_1yr_data['GAIL'][2])
        
        # Set up grouped bar positions
        n_groups = len(x_groups)
        x = np.arange(n_groups)
        width = 0.35  # Width of bars
        
        # Plot bars
        bars1 = ax2.bar(x - width/2, aladynoulli_aucs, width, 
                        label='Aladynoulli', color='#2c7fb8', alpha=0.7, 
                        edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, gail_aucs, width, 
                        label='GAIL', color='#9b59b6', alpha=0.7, 
                        edgecolor='black', linewidth=1.5)
        
        # Add error bars
        errors_lower_ala = [aladynoulli_aucs[i] - aladynoulli_ci_lowers[i] for i in range(len(aladynoulli_aucs))]
        errors_upper_ala = [aladynoulli_ci_uppers[i] - aladynoulli_aucs[i] for i in range(len(aladynoulli_aucs))]
        errors_lower_gail = [gail_aucs[i] - gail_ci_lowers[i] if not np.isnan(gail_aucs[i]) else 0 for i in range(len(gail_aucs))]
        errors_upper_gail = [gail_ci_uppers[i] - gail_aucs[i] if not np.isnan(gail_aucs[i]) else 0 for i in range(len(gail_aucs))]
        
        ax2.errorbar(x - width/2, aladynoulli_aucs, 
                    yerr=[errors_lower_ala, errors_upper_ala],
                    fmt='none', color='black', capsize=5, capthick=2)
        ax2.errorbar(x + width/2, gail_aucs, 
                    yerr=[errors_lower_gail, errors_upper_gail],
                    fmt='none', color='black', capsize=5, capthick=2)
        
        # Add value labels on bars
        for i, (bar, auc) in enumerate(zip(bars1, aladynoulli_aucs)):
            ax2.text(bar.get_x() + bar.get_width()/2., auc + errors_upper_ala[i] + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for i, (bar, auc) in enumerate(zip(bars2, gail_aucs)):
            if not np.isnan(auc):
                ax2.text(bar.get_x() + bar.get_width()/2., auc + errors_upper_gail[i] + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax2.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
        ax2.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax2.set_title('Breast Cancer Prediction (Women Only)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_groups, fontsize=11)
        ax2.legend(loc='upper left', fontsize=10)
        max_auc = max(max(aladynoulli_aucs), max([g for g in gail_aucs if not np.isnan(g)]))
        ax2.set_ylim(0.50, max_auc * 1.1)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'external_scores_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")
    plt.close()
    
    # Optionally show (comment out if running in non-interactive mode)
    # plt.show()

    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    if 'ASCVD_10yr' in df.index:
        row = df.loc['ASCVD_10yr']
        if pd.notna(row.get('PCE_AUC')):
            print("✓ Aladynoulli outperforms PCE for 10-year ASCVD prediction")
        if pd.notna(row.get('QRISK3_AUC')):
            print("✓ Aladynoulli outperforms QRISK3 for 10-year ASCVD prediction")
        if pd.notna(row.get('PREVENT_10yr_AUC')):
            print("✓ Aladynoulli outperforms PREVENT for 10-year ASCVD prediction")
    if 'Breast_Cancer_10yr' in df.index:
        row = df.loc['Breast_Cancer_10yr']
        if pd.notna(row.get('Gail_AUC')):
            if breast_cancer_women_only_auc is not None:
                diff = breast_cancer_women_only_auc - row['Gail_AUC']
                if diff > 0:
                    print("✓ Aladynoulli (women only) outperforms GAIL (women only) for 10-year breast cancer prediction")
                else:
                    print("✓ Aladynoulli (women only) vs GAIL (women only) for 10-year breast cancer prediction")
            else:
                print("✓ Aladynoulli (full population) outperforms GAIL (women only) for 10-year breast cancer prediction")
    if 'Breast_Cancer_1yr' in df.index:
        row = df.loc['Breast_Cancer_1yr']
        if pd.notna(row.get('Gail_AUC')):
            print("✓ Aladynoulli (washout 0yr, women only) substantially outperforms GAIL (1-year, women only) for 1-year breast cancer prediction")
    
else:
    print("⚠️  Results file not found. Please run the comparison script first.")
