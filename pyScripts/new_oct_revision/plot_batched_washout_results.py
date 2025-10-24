#!/usr/bin/env python3
"""
Plot Batched Washout Results

This script processes and visualizes the results from the full batched washout analysis
that processes all 400K patients in batches.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

def load_batched_washout_results(results_file_path=None):
    """
    Load the batched washout results from the full analysis
    
    Parameters:
    -----------
    results_file_path : str, optional
        Path to the saved washout results. If None, tries to find it.
    """
    if results_file_path is None:
        results_file_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/full_washout_results_0_30k.pt"
    
    if not os.path.exists(results_file_path):
        print(f"❌ Results file not found: {results_file_path}")
        print("Please run the full washout analysis first or specify the correct path.")
        return None
    
    try:
        washout_results = torch.load(results_file_path, map_location='cpu')
        print(f"✅ Loaded batched washout results from {results_file_path}")
        return washout_results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def process_batched_results(washout_results):
    """
    Process the batched washout results into a clean DataFrame
    
    Parameters:
    -----------
    washout_results : dict
        Results from batched washout analysis with structure:
        {
            '0yr': {disease: {'aucs': [...], 'cis': [...], 'events': [...], 'rates': [...]}},
            '1yr': {disease: {'aucs': [...], 'cis': [...], 'events': [...], 'rates': [...]}},
            '2yr': {disease: {'aucs': [...], 'cis': [...], 'events': [...], 'rates': [...]}}
        }
    """
    print("\n=== PROCESSING BATCHED WASHOUT RESULTS ===")
    
    # Extract diseases that have results across all washout periods
    all_diseases = set()
    for washout_period in ['0yr', '1yr', '2yr']:
        all_diseases.update(washout_results[washout_period].keys())
    
    print(f"Found {len(all_diseases)} diseases with washout data")
    
    # Process each disease
    processed_data = []
    
    for disease in all_diseases:
        disease_data = {'Disease': disease}
        
        for washout_period in ['0yr', '1yr', '2yr']:
            if disease in washout_results[washout_period]:
                disease_results = washout_results[washout_period][disease]
                
                # Calculate mean AUC from all batches
                aucs = [a for a in disease_results['aucs'] if not pd.isna(a)]
                if aucs:
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)
                    n_batches = len(aucs)
                    
                    disease_data[washout_period] = mean_auc
                    disease_data[f'{washout_period}_std'] = std_auc
                    disease_data[f'{washout_period}_batches'] = n_batches
                    
                    # Calculate overall CI from individual CIs
                    all_cis = [ci for ci in disease_results['cis'] if ci is not None]
                    if all_cis:
                        ci_lowers = [ci[0] for ci in all_cis]
                        ci_uppers = [ci[1] for ci in all_cis]
                        disease_data[f'{washout_period}_ci_lower'] = np.mean(ci_lowers)
                        disease_data[f'{washout_period}_ci_upper'] = np.mean(ci_uppers)
                else:
                    disease_data[washout_period] = np.nan
            else:
                disease_data[washout_period] = np.nan
        
        # Only include diseases with data for all washout periods
        if not any(pd.isna([disease_data['0yr'], disease_data['1yr'], disease_data['2yr']])):
            processed_data.append(disease_data)
    
    df = pd.DataFrame(processed_data)
    print(f"Processed {len(df)} diseases with complete washout data")
    
    return df

def plot_batched_washout_results(df, save_plots=True):
    """
    Create comprehensive washout analysis plots from batched results
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed washout results DataFrame
    save_plots : bool
        Whether to save plots to files
    """
    print("\n=== CREATING BATCHED WASHOUT VISUALIZATIONS ===")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # ============================================================================
    # PANEL A: Line plot showing AUC decay over washout periods
    # ============================================================================
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    washout_periods = [0, 1, 2]
    
    for idx, row in df.iterrows():
        aucs = [row['0yr'], row['1yr'], row['2yr']]
        auc_stds = [row['0yr_std'], row['1yr_std'], row['2yr_std']]
        
        # Plot line with error bars
        ax1.errorbar(washout_periods, aucs, yerr=auc_stds, 
                    fmt='o-', linewidth=2.5, markersize=8, 
                    label=row['Disease'], color=colors[idx], alpha=0.8,
                    capsize=5, capthick=2)
    
    # Add reference lines
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, 
                label='Random Chance', alpha=0.5)
    ax1.axhline(y=0.7, color='red', linestyle=':', linewidth=1.5, 
                label='Clinical Utility (0.70)', alpha=0.5)
    
    ax1.set_xlabel('Washout Period (Years)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AUC', fontsize=14, fontweight='bold')
    ax1.set_title('A. Model Performance with Temporal Washout\n(Batched Analysis: 0-400K Patients)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(washout_periods)
    ax1.set_xticklabels(['Immediate\n(0 yr)', '1-Year\nWashout', '2-Year\nWashout'])
    ax1.set_ylim(0.45, 1.0)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
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
    
    # Plot bars with error bars
    bars = ax2.barh(y_pos, df_sorted['2yr'], alpha=0.7, color='steelblue', 
                    edgecolor='black', linewidth=1.5, label='2-Year Washout AUC')
    
    # Add error bars for 2-year washout
    ax2.errorbar(df_sorted['2yr'], y_pos, xerr=df_sorted['2yr_std'], 
                 fmt='none', color='black', capsize=3, capthick=1)
    
    # Add 0-year performance as comparison
    ax2.barh(y_pos, df_sorted['0yr'], alpha=0.3, color='lightcoral', 
             edgecolor='black', linewidth=1.5, label='Immediate (0-Year) AUC')
    
    # Add reference lines
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(x=0.7, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted['Disease'], fontsize=11)
    ax2.set_xlabel('AUC', fontsize=14, fontweight='bold')
    ax2.set_title('B. Predictive Performance After 2-Year Washout\n(Batched Analysis: 0-400K Patients)', 
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
    
    if save_plots:
        plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/batched_washout_analysis.pdf', 
                    dpi=300, bbox_inches='tight')
        plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/batched_washout_analysis.png', 
                    dpi=300, bbox_inches='tight')
        print("✓ Saved batched washout analysis plots!")
    
    plt.show()
    
    return fig

def create_batched_summary_table(df, save_plots=True):
    """
    Create a detailed summary table for batched washout results
    """
    print("\n=== CREATING BATCHED SUMMARY TABLE ===")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create detailed summary table
    summary_data = []
    for idx, row in df.iterrows():
        summary_data.append([
            row['Disease'],
            f"{row['0yr']:.3f}±{row['0yr_std']:.3f}",
            f"{row['1yr']:.3f}±{row['1yr_std']:.3f}",
            f"{row['2yr']:.3f}±{row['2yr_std']:.3f}",
            f"{row['drop']:.3f}",
            f"{(row['2yr']/row['0yr'])*100:.0f}%",
            f"{row['0yr_batches']}/{row['1yr_batches']}/{row['2yr_batches']}"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Disease', 'Immediate\n(0-Year)', '1-Year\nWashout', 
                              '2-Year\nWashout', 'AUC Drop', 'Retention', 'Batches\n(0yr/1yr/2yr)'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(7):
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
    
    plt.title('Batched Washout Analysis Summary: Model Performance Retention\n(0-400K Patients Processed in Batches)', 
              fontsize=16, fontweight='bold', pad=20)
    
    if save_plots:
        plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/batched_washout_table.pdf', 
                    dpi=300, bbox_inches='tight')
        plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/batched_washout_table.png', 
                    dpi=300, bbox_inches='tight')
        print("✓ Saved batched washout summary table!")
    
    plt.show()
    
    return fig

def print_batched_summary_statistics(df):
    """
    Print comprehensive summary statistics for batched washout results
    """
    print("\n" + "="*80)
    print("BATCHED WASHOUT ANALYSIS SUMMARY STATISTICS")
    print("="*80)
    print(f"Analysis Coverage: 0-400K patients processed in batches")
    print(f"Total Diseases Analyzed: {len(df)}")
    
    print(f"\nMean AUC at 0-year: {df['0yr'].mean():.3f} (SD: {df['0yr'].std():.3f})")
    print(f"Mean AUC at 1-year: {df['1yr'].mean():.3f} (SD: {df['1yr'].std():.3f})")
    print(f"Mean AUC at 2-year: {df['2yr'].mean():.3f} (SD: {df['2yr'].std():.3f})")
    
    print(f"\nMean AUC drop: {df['drop'].mean():.3f} (SD: {df['drop'].std():.3f})")
    print(f"Mean retention: {df['retention'].mean():.1f}% (SD: {df['retention'].std():.1f}%)")
    
    print(f"\nPerformance Thresholds at 2-Year Washout:")
    print(f"  AUC > 0.7: {len(df[df['2yr'] > 0.7])}/{len(df)} diseases")
    print(f"  AUC > 0.65: {len(df[df['2yr'] > 0.65])}/{len(df)} diseases")
    print(f"  AUC > 0.6: {len(df[df['2yr'] > 0.6])}/{len(df)} diseases")
    
    print(f"\nBatch Coverage:")
    print(f"  Mean batches per disease: {df[['0yr_batches', '1yr_batches', '2yr_batches']].mean().mean():.1f}")
    print(f"  Min batches: {df[['0yr_batches', '1yr_batches', '2yr_batches']].min().min()}")
    print(f"  Max batches: {df[['0yr_batches', '1yr_batches', '2yr_batches']].max().max()}")
    
    print("="*80)

def main():
    """
    Main function to run the complete batched washout analysis visualization
    """
    print("="*80)
    print("BATCHED WASHOUT RESULTS VISUALIZATION")
    print("="*80)
    
    # Load batched results
    washout_results = load_batched_washout_results()
    if washout_results is None:
        return
    
    # Process results into DataFrame
    df = process_batched_results(washout_results)
    if df.empty:
        print("❌ No valid washout data found")
        return
    
    # Create visualizations
    plot_batched_washout_results(df)
    create_batched_summary_table(df)
    
    # Print summary statistics
    print_batched_summary_statistics(df)
    
    print("\n✅ Batched washout analysis visualization complete!")

if __name__ == "__main__":
    main()
