#!/usr/bin/env python3
"""
Generate PRS-Signature association plots (bar plot and heatmaps) from gamma data.

This replicates the R analysis for visualizing PRS-signature associations.

Usage:
    python generate_prs_signature_associations.py [--gamma_csv <path>] [--prs_names_csv <path>]
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']


# Category colors (matching R code)
CATEGORY_COLORS = {
    "Cardiovascular": "#E74C3C",
    "Metabolic": "#2ECC71",
    "Autoimmune": "#3498DB",
    "Neurological": "#9B59B6",
    "Cancer": "#F39C12",
    "Other": "#95A5A6"
}


def load_prs_names(prs_names_path):
    """Load PRS names from CSV file."""
    if prs_names_path and os.path.exists(prs_names_path):
        df = pd.read_csv(prs_names_path, header=None)
        return df.iloc[:, 0].tolist()
    else:
        # Default PRS names if file not found
        return [f"PRS_{i}" for i in range(36)]


def assign_disease_categories(prs_names):
    """Assign disease categories to PRSs (matching R logic)."""
    categories = []
    
    cardiovascular_prs = ["CAD", "AF", "HT", "LDL_SF"]
    metabolic_prs = ["T1D", "T2D", "BMI", "HBA1C_DF"]
    autoimmune_prs = ["RA", "PSO", "SLE", "CD", "UC"]
    neurological_prs = ["AD", "PD", "MS", "BD", "SCZ"]
    cancer_prs = ["BC", "PC", "CRC", "MEL"]
    
    for prs in prs_names:
        if prs in cardiovascular_prs:
            categories.append("Cardiovascular")
        elif prs in metabolic_prs:
            categories.append("Metabolic")
        elif prs in autoimmune_prs:
            categories.append("Autoimmune")
        elif prs in neurological_prs:
            categories.append("Neurological")
        elif prs in cancer_prs:
            categories.append("Cancer")
        else:
            categories.append("Other")
    
    return categories


def load_gamma_data(gamma_csv_path=None, ukb_params_path=None):
    """
    Load gamma data from CSV or RDS file.
    
    If gamma_csv_path is provided, loads from CSV with columns:
    - prs, signature, effect_mean, effect_se, p_value, significant
    
    If ukb_params_path is provided, loads from RDS (requires rpy2).
    """
    if gamma_csv_path and os.path.exists(gamma_csv_path):
        print(f"Loading gamma data from CSV: {gamma_csv_path}")
        df = pd.read_csv(gamma_csv_path)
        
        # Rename columns if needed
        if 'signature' in df.columns:
            df['signature'] = df['signature'].str.replace('Signature ', 'Sig ')
        
        return df
    
    elif ukb_params_path and os.path.exists(ukb_params_path):
        print(f"Loading gamma data from RDS: {ukb_params_path}")
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            
            ro.r['load'](ukb_params_path)
            ukb_params = ro.r['ukb_params']
            gamma = ro.r['gamma']
            
            # Convert to pandas DataFrame
            gamma_df = pd.DataFrame(gamma)
            return gamma_df
        except ImportError:
            print("Error: rpy2 not available. Please install with: pip install rpy2")
            return None
    else:
        print("Error: No valid gamma data source provided")
        return None


def create_gamma_melted_df(gamma_df, prs_names, disease_categories):
    """Create melted dataframe for plotting."""
    # If gamma_df is already in long format (from CSV)
    if 'prs' in gamma_df.columns and 'signature' in gamma_df.columns:
        gamma_melted = gamma_df.copy()
        
        # Add category if not present
        if 'category' not in gamma_melted.columns:
            prs_to_category = dict(zip(prs_names, disease_categories))
            gamma_melted['category'] = gamma_melted['prs'].map(prs_to_category)
        
        return gamma_melted
    
    # Otherwise, assume wide format (PRS x Signature matrix)
    gamma_melted = gamma_df.melt(
        id_vars=None,
        var_name='signature',
        value_name='effect'
    )
    gamma_melted['prs'] = gamma_melted.index.map(lambda i: prs_names[i] if i < len(prs_names) else f"PRS_{i}")
    gamma_melted['signature'] = gamma_melted['signature'].astype(str).str.replace('Signature ', 'Sig ')
    
    # Add category
    prs_to_category = dict(zip(prs_names, disease_categories))
    gamma_melted['category'] = gamma_melted['prs'].map(prs_to_category)
    
    return gamma_melted


def plot_top_associations_bar(gamma_melted, output_dir, significance_threshold=0.1):
    """Create bar plot of top PRS-signature associations."""
    # Filter for significant associations
    if 'significant' in gamma_melted.columns:
        top_associations = gamma_melted[gamma_melted['significant'] == True].copy()
    elif 'p_value' in gamma_melted.columns:
        # Bonferroni correction
        n_tests = len(gamma_melted)
        bonferroni_threshold = 0.05 / n_tests
        top_associations = gamma_melted[gamma_melted['p_value'] < bonferroni_threshold].copy()
    else:
        # Use effect size threshold
        top_associations = gamma_melted[gamma_melted['effect'].abs() > significance_threshold].copy()
    
    if len(top_associations) == 0:
        print("Warning: No significant associations found. Using effect size threshold.")
        top_associations = gamma_melted[gamma_melted['effect'].abs() > significance_threshold].copy()
    
    # Create label for y-axis
    top_associations['label'] = top_associations['prs'] + " - " + top_associations['signature']
    
    # Sort by absolute effect size
    top_associations = top_associations.sort_values('effect', key=abs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(top_associations) * 0.3)))
    
    # Map categories to colors
    colors = [CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other']) for cat in top_associations['category']]
    
    bars = ax.barh(range(len(top_associations)), top_associations['effect'], color=colors)
    
    ax.set_yticks(range(len(top_associations)))
    ax.set_yticklabels(top_associations['label'], fontsize=9)
    ax.set_xlabel('Effect Size', fontsize=12, fontweight='bold')
    ax.set_title('Top PRS-Signature Associations', fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) 
                      for cat, color in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'top_prs_associations.pdf'
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    
    plt.close(fig)
    return top_associations


def plot_significant_heatmap(gamma_melted, output_dir, significance_threshold=0.1):
    """Create heatmap of significant PRS-signature associations."""
    # Filter for significant effects
    if 'significant' in gamma_melted.columns:
        significant_effects = gamma_melted[gamma_melted['significant'] == True].copy()
    elif 'p_value' in gamma_melted.columns:
        n_tests = len(gamma_melted)
        bonferroni_threshold = 0.05 / n_tests
        significant_effects = gamma_melted[gamma_melted['p_value'] < bonferroni_threshold].copy()
    else:
        significant_effects = gamma_melted[gamma_melted['effect'].abs() > significance_threshold].copy()
    
    if len(significant_effects) == 0:
        print("Warning: No significant associations found. Using effect size threshold.")
        significant_effects = gamma_melted[gamma_melted['effect'].abs() > significance_threshold].copy()
    
    # Create pivot table for heatmap
    heatmap_data = significant_effects.pivot_table(
        values='effect',
        index='prs',
        columns='signature',
        aggfunc='mean'
    )
    
    # Sort PRSs by category
    prs_categories = dict(zip(significant_effects['prs'], significant_effects['category']))
    category_order = ["Cardiovascular", "Metabolic", "Autoimmune", "Neurological", "Cancer", "Other"]
    prs_order = []
    for cat in category_order:
        prs_in_cat = [prs for prs in heatmap_data.index if prs_categories.get(prs) == cat]
        prs_order.extend(sorted(prs_in_cat))
    
    # Reorder
    heatmap_data = heatmap_data.reindex(prs_order)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.8), 
                                     max(10, len(heatmap_data.index) * 0.4)))
    
    sns.heatmap(heatmap_data, 
                cmap='RdBu_r', 
                center=0,
                vmin=-0.3, vmax=0.3,
                cbar_kws={'label': 'Effect Size'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray')
    
    ax.set_title('Significant PRS-Signature Associations', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Signature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Polygenic Risk Score', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'significant_prs_heatmap.pdf'
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    
    plt.close(fig)


def plot_full_heatmap(gamma_melted, output_dir):
    """Create full heatmap with all PRS-signature associations."""
    # Create pivot table
    heatmap_data = gamma_melted.pivot_table(
        values='effect',
        index='prs',
        columns='signature',
        aggfunc='mean'
    )
    
    # Sort PRSs by category
    prs_categories = dict(zip(gamma_melted['prs'], gamma_melted['category']))
    category_order = ["Cardiovascular", "Metabolic", "Autoimmune", "Neurological", "Cancer", "Other"]
    prs_order = []
    for cat in category_order:
        prs_in_cat = [prs for prs in heatmap_data.index if prs_categories.get(prs) == cat]
        prs_order.extend(sorted(prs_in_cat))
    
    heatmap_data = heatmap_data.reindex(prs_order)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.8), 
                                     max(10, len(heatmap_data.index) * 0.4)))
    
    sns.heatmap(heatmap_data, 
                cmap='RdBu_r', 
                center=0,
                vmin=-0.3, vmax=0.3,
                cbar_kws={'label': 'Effect Size'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray')
    
    ax.set_title('Complete PRS-Signature Association Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Signature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Polygenic Risk Score', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'complete_prs_heatmap.pdf'
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate PRS-Signature association plots')
    parser.add_argument('--gamma_csv', type=str, 
                        help='Path to gamma associations CSV file')
    parser.add_argument('--prs_names_csv', type=str,
                        default='prs_names.csv',
                        help='Path to PRS names CSV file')
    parser.add_argument('--ukb_params_rds', type=str,
                        help='Path to ukb_params.rds file (alternative to CSV)')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures',
                        help='Output directory for plots')
    parser.add_argument('--significance_threshold', type=float, default=0.1,
                        help='Effect size threshold for significance (if p-values not available)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load PRS names
    prs_names = load_prs_names(args.prs_names_csv)
    print(f"Loaded {len(prs_names)} PRS names")
    
    # Assign disease categories
    disease_categories = assign_disease_categories(prs_names)
    
    # Load gamma data
    gamma_df = load_gamma_data(args.gamma_csv, args.ukb_params_rds)
    if gamma_df is None:
        print("Error: Could not load gamma data")
        return
    
    print(f"Loaded gamma data: {gamma_df.shape}")
    
    # Create melted dataframe
    gamma_melted = create_gamma_melted_df(gamma_df, prs_names, disease_categories)
    print(f"Created melted dataframe: {gamma_melted.shape}")
    
    # Calculate significance statistics
    if 'significant' in gamma_melted.columns:
        n_significant = gamma_melted['significant'].sum()
    elif 'p_value' in gamma_melted.columns:
        n_tests = len(gamma_melted)
        bonferroni_threshold = 0.05 / n_tests
        n_significant = (gamma_melted['p_value'] < bonferroni_threshold).sum()
    else:
        n_significant = (gamma_melted['effect'].abs() > args.significance_threshold).sum()
    
    total_tests = len(gamma_melted)
    significance_rate = n_significant / total_tests if total_tests > 0 else 0
    
    print(f"\nSignificance Statistics:")
    print(f"  Significant associations: {n_significant} / {total_tests} ({100*significance_rate:.2f}%)")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Top associations bar plot
    print("\n1. Creating top associations bar plot...")
    top_associations = plot_top_associations_bar(gamma_melted, output_dir, args.significance_threshold)
    
    # 2. Significant associations heatmap
    print("\n2. Creating significant associations heatmap...")
    plot_significant_heatmap(gamma_melted, output_dir, args.significance_threshold)
    
    # 3. Full heatmap
    print("\n3. Creating full heatmap...")
    plot_full_heatmap(gamma_melted, output_dir)
    
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

