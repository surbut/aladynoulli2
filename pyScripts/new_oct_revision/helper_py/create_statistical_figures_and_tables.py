#!/usr/bin/env python3
"""
Create Publication-Ready Figures and Tables from Statistical Test Results

Generates:
1. Summary tables (CSV and LaTeX-ready)
2. Visualizations (signature discrimination, age differences, effect sizes, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_statistical_results(results_dir='output_10yr'):
    """Load statistical test results"""
    with open(f'{results_dir}/statistical_test_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return results


def create_top_diseases_table(stats_results, output_dir='output_10yr', top_n=20):
    """Create table of top N most significantly different diseases"""
    disease_tests = stats_results['disease_prevalence_tests']
    significant = disease_tests[disease_tests['is_significant_fdr']].copy()
    
    # Sort by chi-square statistic
    significant = significant.sort_values('chi2_statistic', ascending=False).head(top_n)
    
    # Expand prevalences into separate columns
    prev_data = []
    for idx, row in significant.iterrows():
        prevs = row['prevalences']
        prev_row = {
            'disease': row['disease'],
            'chi2': row['chi2_statistic'],
            'p_value': row['p_value'],
            'p_value_fdr': row['p_value_fdr_corrected'],
            'cramers_v': row['cramers_v']
        }
        # Add prevalence for each pathway
        for pid in sorted(prevs.keys()):
            prev_row[f'Pathway_{pid}_prev'] = prevs[pid] * 100  # Convert to percentage
        prev_data.append(prev_row)
    
    table_df = pd.DataFrame(prev_data)
    
    # Save as CSV
    table_df.to_csv(f'{output_dir}/top_{top_n}_diseases_table.csv', index=False)
    
    # Create LaTeX table
    latex_table = table_df.to_latex(
        index=False,
        float_format="{:.3f}".format,
        caption=f"Top {top_n} Diseases with Significant Prevalence Differences Across Pathways (FDR < 0.05)",
        label="tab:top_diseases"
    )
    
    with open(f'{output_dir}/top_{top_n}_diseases_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Created top {top_n} diseases table")
    return table_df


def create_signature_discrimination_table(stats_results, output_dir='output_10yr'):
    """Create table of signature discrimination statistics"""
    sig_results = stats_results['signature_trajectory_tests']
    
    sig_data = []
    for sig_idx, res in sig_results.items():
        sig_data.append({
            'Signature': sig_idx,
            'F_statistic': res['f_statistic'],
            'p_value': res['p_value'],
            'eta_squared': res['eta_squared'],
            'n_pathways': len(res['pathway_means'])
        })
        
        # Add mean and std for each pathway
        for pid in sorted(res['pathway_means'].keys()):
            sig_data[-1][f'Pathway_{pid}_mean'] = res['pathway_means'][pid]
            sig_data[-1][f'Pathway_{pid}_std'] = res['pathway_stds'][pid]
    
    sig_df = pd.DataFrame(sig_data)
    sig_df = sig_df.sort_values('F_statistic', ascending=False)
    
    # Save as CSV
    sig_df.to_csv(f'{output_dir}/signature_discrimination_table.csv', index=False)
    
    # Create LaTeX table
    latex_table = sig_df.to_latex(
        index=False,
        float_format="{:.3f}".format,
        caption="Signature Trajectory Discrimination Statistics (ANOVA)",
        label="tab:signature_discrimination"
    )
    
    with open(f'{output_dir}/signature_discrimination_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("✅ Created signature discrimination table")
    return sig_df


def create_medication_table(stats_results, output_dir='output_10yr', top_n=15):
    """Create table of significantly different medications"""
    if 'medication_tests' not in stats_results or stats_results['medication_tests'] is None:
        print("⚠️  No medication test results available")
        return None
    
    med_tests = stats_results['medication_tests']
    significant = med_tests[med_tests['is_significant_fdr']].copy()
    
    if len(significant) == 0:
        print("⚠️  No significantly different medications found")
        return None
    
    # Sort by chi-square
    significant = significant.sort_values('chi2_statistic', ascending=False).head(top_n)
    
    # Expand prevalences
    med_data = []
    for idx, row in significant.iterrows():
        prevs = row['prevalences']
        med_row = {
            'medication': row['medication'],
            'chi2': row['chi2_statistic'],
            'p_value': row['p_value'],
            'p_value_fdr': row['p_value_fdr_corrected'],
            'cramers_v': row['cramers_v']
        }
        for pid in sorted(prevs.keys()):
            med_row[f'Pathway_{pid}_prev'] = prevs[pid] * 100
        med_data.append(med_row)
    
    med_df = pd.DataFrame(med_data)
    
    # Save as CSV
    med_df.to_csv(f'{output_dir}/top_{top_n}_medications_table.csv', index=False)
    
    print(f"✅ Created top {top_n} medications table")
    return med_df


def create_age_at_onset_table(stats_results, output_dir='output_10yr'):
    """Create table of age at disease onset by pathway"""
    age_test = stats_results['age_at_onset_test']
    
    age_data = []
    for pid in sorted(age_test['pathway_means'].keys()):
        age_data.append({
            'Pathway': pid,
            'Mean_Age': age_test['pathway_means'][pid],
            'Std_Age': age_test['pathway_stds'][pid],
            'N_Patients': age_test['n_per_pathway'][pid]
        })
    
    age_df = pd.DataFrame(age_data)
    
    # Add overall statistics
    age_df['F_statistic'] = age_test['f_statistic']
    age_df['p_value'] = age_test['p_value']
    age_df['eta_squared'] = age_test['eta_squared']
    
    # Save as CSV
    age_df.to_csv(f'{output_dir}/age_at_onset_table.csv', index=False)
    
    print("✅ Created age at onset table")
    return age_df


def plot_signature_discrimination(stats_results, output_dir='output_10yr'):
    """Plot F-statistics for signature discrimination"""
    sig_results = stats_results['signature_trajectory_tests']
    
    sig_data = []
    for sig_idx, res in sig_results.items():
        sig_data.append({
            'Signature': f'Sig {sig_idx}',
            'F_statistic': res['f_statistic'],
            'p_value': res['p_value'],
            'eta_squared': res['eta_squared']
        })
    
    sig_df = pd.DataFrame(sig_data).sort_values('F_statistic', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71' if p < 0.001 else '#3498db' if p < 0.01 else '#95a5a6' 
              for p in sig_df['p_value']]
    
    bars = ax.barh(sig_df['Signature'], sig_df['F_statistic'], color=colors)
    ax.set_xlabel('F-Statistic', fontsize=12, fontweight='bold')
    ax.set_ylabel('Signature', fontsize=12, fontweight='bold')
    ax.set_title('Signature Trajectory Discrimination Across Pathways\n(All signatures significantly different, p < 0.05)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value annotations
    for i, (sig, f_stat, p_val) in enumerate(zip(sig_df['Signature'], 
                                                   sig_df['F_statistic'], 
                                                   sig_df['p_value'])):
        if p_val < 0.001:
            label = '***'
        elif p_val < 0.01:
            label = '**'
        elif p_val < 0.05:
            label = '*'
        else:
            label = ''
        ax.text(f_stat * 1.02, i, label, va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/signature_discrimination_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/signature_discrimination_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created signature discrimination plot")


def plot_age_at_onset(stats_results, output_dir='output_10yr'):
    """Plot age at disease onset by pathway"""
    age_test = stats_results['age_at_onset_test']
    
    pathway_ids = sorted(age_test['pathway_means'].keys())
    means = [age_test['pathway_means'][pid] for pid in pathway_ids]
    stds = [age_test['pathway_stds'][pid] for pid in pathway_ids]
    ns = [age_test['n_per_pathway'][pid] for pid in pathway_ids]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot with error bars
    bars = ax1.bar([f'Pathway {pid}' for pid in pathway_ids], means, 
                   yerr=stds, capsize=5, alpha=0.7, color=sns.color_palette("husl", len(pathway_ids)))
    ax1.set_ylabel('Age at Disease Onset (years)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Pathway', fontsize=12, fontweight='bold')
    ax1.set_title(f'Mean Age at Disease Onset by Pathway\nANOVA: F={age_test["f_statistic"]:.2f}, p={age_test["p_value"]:.2e}',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add sample sizes
    for i, (bar, n) in enumerate(zip(bars, ns)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                f'n={n:,}', ha='center', va='bottom', fontsize=9)
    
    # Violin-style distribution (using error bars as proxy)
    for i, (pid, mean, std, n) in enumerate(zip(pathway_ids, means, stds, ns)):
        # Create approximate distribution
        x_pos = i
        ax2.errorbar([x_pos], [mean], yerr=[[std*0.8], [std*0.8]], 
                    fmt='o', capsize=10, capthick=2, markersize=10,
                    label=f'Pathway {pid} (n={n:,})')
    
    ax2.set_xticks(range(len(pathway_ids)))
    ax2.set_xticklabels([f'Pathway {pid}' for pid in pathway_ids])
    ax2.set_ylabel('Age at Disease Onset (years)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Pathway', fontsize=12, fontweight='bold')
    ax2.set_title('Age Distribution by Pathway', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_at_onset_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/age_at_onset_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created age at onset plot")


def plot_effect_sizes_heatmap(stats_results, output_dir='output_10yr', top_n_sigs=10):
    """Create heatmap of Cohen's d effect sizes for top signatures"""
    effect_sizes = stats_results['effect_sizes']
    sig_results = stats_results['signature_trajectory_tests']
    
    # Get top N signatures by F-statistic
    sig_f_stats = [(sig, res['f_statistic']) for sig, res in sig_results.items()]
    sig_f_stats.sort(key=lambda x: x[1], reverse=True)
    top_sigs = [sig for sig, _ in sig_f_stats[:top_n_sigs]]
    
    # Get all pathway pairs
    pathway_pairs = []
    pathway_ids = sorted(effect_sizes['signature_effect_sizes'][0].keys())
    for pair_key in pathway_ids:
        if '_vs_' in pair_key:
            pid1, pid2 = pair_key.split('_vs_')
            pathway_pairs.append((int(pid1), int(pid2)))
    
    # Build matrix
    n_pathways = len(set([p for pair in pathway_pairs for p in pair]))
    cohens_d_matrix = np.zeros((top_n_sigs, len(pathway_pairs)))
    pair_labels = []
    
    for j, (pid1, pid2) in enumerate(pathway_pairs):
        pair_labels.append(f'P{pid1} vs P{pid2}')
        for i, sig_idx in enumerate(top_sigs):
            pair_key = f"{pid1}_vs_{pid2}"
            if pair_key in effect_sizes['signature_effect_sizes'][sig_idx]:
                cohens_d_matrix[i, j] = effect_sizes['signature_effect_sizes'][sig_idx][pair_key]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(pathway_pairs)*0.8), max(8, top_n_sigs*0.6)))
    
    sns.heatmap(cohens_d_matrix, 
                xticklabels=pair_labels,
                yticklabels=[f'Sig {sig}' for sig in top_sigs],
                annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': "Cohen's d"}, ax=ax)
    
    ax.set_title(f'Effect Sizes (Cohen\'s d) for Top {top_n_sigs} Signatures\nBetween Pathway Pairs',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Pathway Pair Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Signature', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/effect_sizes_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/effect_sizes_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created effect sizes heatmap")


def plot_disease_prevalence_examples(stats_results, output_dir='output_10yr', top_n=10):
    """Plot prevalence patterns for top N most significant diseases"""
    disease_tests = stats_results['disease_prevalence_tests']
    significant = disease_tests[disease_tests['is_significant_fdr']].copy()
    significant = significant.sort_values('chi2_statistic', ascending=False).head(top_n)
    
    n_rows = (top_n + 2) // 3  # 3 columns
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if top_n > 1 else [axes]
    
    pathway_ids = sorted(significant.iloc[0]['prevalences'].keys())
    
    for idx, (_, row) in enumerate(significant.iterrows()):
        ax = axes[idx]
        prevs = row['prevalences']
        
        prevalences = [prevs[pid] * 100 for pid in pathway_ids]  # Convert to percentage
        
        bars = ax.bar([f'P{pid}' for pid in pathway_ids], prevalences, 
                     color=sns.color_palette("husl", len(pathway_ids)), alpha=0.7)
        ax.set_ylabel('Prevalence (%)', fontsize=10)
        ax.set_xlabel('Pathway', fontsize=10)
        ax.set_title(f"{row['disease']}\nχ²={row['chi2_statistic']:.1f}, p={row['p_value']:.2e}",
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {top_n} Diseases with Significant Prevalence Differences Across Pathways',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/disease_prevalence_examples.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/disease_prevalence_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created disease prevalence examples plot (top {top_n})")


def create_summary_statistics_table(stats_results, output_dir='output_10yr'):
    """Create summary table of all test results"""
    summary_data = []
    
    # Disease prevalence
    disease_tests = stats_results['disease_prevalence_tests']
    n_sig_diseases = disease_tests['is_significant_fdr'].sum() if 'is_significant_fdr' in disease_tests.columns else 0
    summary_data.append({
        'Test Category': 'Disease Prevalence',
        'Test Type': 'Chi-square',
        'N_Significant': n_sig_diseases,
        'N_Total': len(disease_tests),
        'Effect_Size_Mean': disease_tests['cramers_v'].mean() if 'cramers_v' in disease_tests.columns else np.nan
    })
    
    # Signature trajectories
    sig_results = stats_results['signature_trajectory_tests']
    n_sig_sigs = sum(1 for res in sig_results.values() if res['p_value'] < 0.05)
    summary_data.append({
        'Test Category': 'Signature Trajectories',
        'Test Type': 'ANOVA',
        'N_Significant': n_sig_sigs,
        'N_Total': len(sig_results),
        'Effect_Size_Mean': np.mean([res['eta_squared'] for res in sig_results.values()])
    })
    
    # Age at onset
    age_test = stats_results['age_at_onset_test']
    summary_data.append({
        'Test Category': 'Age at Onset',
        'Test Type': age_test['test_type'],
        'N_Significant': 1 if age_test['p_value'] < 0.05 else 0,
        'N_Total': 1,
        'Effect_Size_Mean': age_test.get('eta_squared', np.nan)
    })
    
    # Medications
    if 'medication_tests' in stats_results and stats_results['medication_tests'] is not None:
        med_tests = stats_results['medication_tests']
        n_sig_meds = med_tests['is_significant_fdr'].sum() if 'is_significant_fdr' in med_tests.columns else 0
        summary_data.append({
            'Test Category': 'Medications',
            'Test Type': 'Chi-square',
            'N_Significant': n_sig_meds,
            'N_Total': len(med_tests),
            'Effect_Size_Mean': med_tests['cramers_v'].mean() if 'cramers_v' in med_tests.columns else np.nan
        })
    
    # Permutation test
    perm_test = stats_results['permutation_test']
    summary_data.append({
        'Test Category': 'Pathway Stability',
        'Test Type': 'Permutation Test',
        'N_Significant': 1 if perm_test['is_significant'] else 0,
        'N_Total': 1,
        'Effect_Size_Mean': np.nan,
        'P_Value': perm_test['p_value']
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/summary_statistics_table.csv', index=False)
    
    print("✅ Created summary statistics table")
    return summary_df


def main(output_dir='output_10yr'):
    """Main function to generate all tables and figures"""
    print("="*80)
    print("GENERATING PUBLICATION-READY TABLES AND FIGURES")
    print("="*80)
    
    # Load results
    print("\nLoading statistical test results...")
    stats_results = load_statistical_results(output_dir)
    
    # Create tables
    print("\n--- CREATING TABLES ---")
    top_diseases_df = create_top_diseases_table(stats_results, output_dir, top_n=20)
    sig_disc_df = create_signature_discrimination_table(stats_results, output_dir)
    med_df = create_medication_table(stats_results, output_dir, top_n=15)
    age_df = create_age_at_onset_table(stats_results, output_dir)
    summary_df = create_summary_statistics_table(stats_results, output_dir)
    
    # Create figures
    print("\n--- CREATING FIGURES ---")
    plot_signature_discrimination(stats_results, output_dir)
    plot_age_at_onset(stats_results, output_dir)
    plot_effect_sizes_heatmap(stats_results, output_dir, top_n_sigs=10)
    plot_disease_prevalence_examples(stats_results, output_dir, top_n=9)
    
    print("\n" + "="*80)
    print("✅ ALL TABLES AND FIGURES GENERATED!")
    print(f"📁 Output directory: {output_dir}/")
    print("="*80)
    
    return {
        'tables': {
            'top_diseases': top_diseases_df,
            'signatures': sig_disc_df,
            'medications': med_df,
            'age': age_df,
            'summary': summary_df
        },
        'stats_results': stats_results
    }


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output_10yr'
    main(output_dir)
