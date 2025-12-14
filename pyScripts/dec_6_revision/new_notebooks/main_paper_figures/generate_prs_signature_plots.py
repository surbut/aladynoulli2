#!/usr/bin/env python3
"""
Generate PRS-Signature association plots from batch checkpoints or averaged gamma.

Creates:
1. Bar plot of top PRS-signature associations
2. Heatmap of significant associations
3. Full heatmap of all associations

Usage:
    python generate_prs_signature_plots.py --batch_dir <path> [--prs_names_csv <path>]
    python generate_prs_signature_plots.py --gamma_file <path> [--prs_names_csv <path>]
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import argparse
from scipy import stats
from statsmodels.stats.multitest import multipletests

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


def load_prs_names(prs_names_path=None):
    """Load PRS names from CSV file or use defaults."""
    # Default path
    if prs_names_path is None:
        default_path = '/Users/sarahurbut/aladynoulli2/claudefile/aladyn_project/prs_names.csv'
        if os.path.exists(default_path):
            prs_names_path = default_path
    
    if prs_names_path and os.path.exists(prs_names_path):
        df = pd.read_csv(prs_names_path, header=None)
        prs_names = df.iloc[:, 0].tolist()
        print(f"Loaded {len(prs_names)} PRS names from: {prs_names_path}")
        return prs_names
    else:
        # Default PRS names (fallback)
        default_prs = [
            "CAD", "AF", "HT", "LDL_SF", "T1D", "T2D", "BMI", "HBA1C_DF",
            "RA", "PSO", "SLE", "CD", "UC", "AD", "PD", "MS", "BD", "SCZ",
            "BC", "PC", "CRC", "MEL", "HEIGHT", "POAG"
        ]
        # Pad to 36 if needed
        while len(default_prs) < 36:
            default_prs.append(f"PRS_{len(default_prs)}")
        print(f"Warning: Using default PRS names (could not find CSV)")
        return default_prs[:36]


def assign_disease_categories(prs_names):
    """Assign disease categories to PRSs (matching R logic)."""
    categories = []
    
    cardiovascular_prs = ["CAD", "AF", "HT", "LDL_SF", "CVD"]
    metabolic_prs = ["T1D", "T2D", "BMI", "HBA1C_DF", "HBA1C"]
    autoimmune_prs = ["RA", "PSO", "SLE", "CD", "UC"]
    neurological_prs = ["AD", "PD", "MS", "BD", "SCZ"]
    cancer_prs = ["BC", "PC", "CRC", "MEL"]
    
    for prs in prs_names:
        prs_upper = prs.upper()
        if any(cv in prs_upper for cv in [p.upper() for p in cardiovascular_prs]):
            categories.append("Cardiovascular")
        elif any(m in prs_upper for m in [p.upper() for p in metabolic_prs]):
            categories.append("Metabolic")
        elif any(a in prs_upper for a in [p.upper() for p in autoimmune_prs]):
            categories.append("Autoimmune")
        elif any(n in prs_upper for n in [p.upper() for p in neurological_prs]):
            categories.append("Neurological")
        elif any(c in prs_upper for c in [p.upper() for p in cancer_prs]):
            categories.append("Cancer")
        else:
            categories.append("Other")
    
    return categories


def load_gamma_from_batches(batch_dir, pattern="enrollment_model_W0.0001_batch_*_*.pt"):
    """Load and average gamma from multiple batch checkpoints."""
    checkpoint_files = sorted(glob.glob(os.path.join(batch_dir, pattern)))
    
    if not checkpoint_files:
        print(f"No batch checkpoints found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(checkpoint_files)} batch checkpoints. Loading gamma...")
    all_gammas = []
    
    for checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            gamma = None
            if 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
            
            if gamma is not None:
                if torch.is_tensor(gamma):
                    gamma = gamma.detach().cpu().numpy()
                
                # Skip if all zeros
                if not np.allclose(gamma, 0):
                    all_gammas.append(gamma)
                    print(f"  Loaded gamma from {os.path.basename(checkpoint_path)} (shape: {gamma.shape})")
                else:
                    print(f"  Skipping {os.path.basename(checkpoint_path)} - gamma is all zeros")
        except Exception as e:
            print(f"  Warning: Could not load gamma from {os.path.basename(checkpoint_path)}: {e}")
            continue
    
    if not all_gammas:
        print("No non-zero gamma values found!")
        return None
    
    # Stack and average
    gamma_stack = np.stack(all_gammas)
    gamma_mean = np.mean(gamma_stack, axis=0)
    gamma_std = np.std(gamma_stack, axis=0)
    # Standard error of the mean (
    # ) = std / sqrt(n)
    gamma_sem = gamma_std / np.sqrt(len(all_gammas))
    
    print(f"\n✓ Averaged gamma from {len(all_gammas)} batches")
    print(f"  Final shape: {gamma_mean.shape}")
    print(f"  Min: {np.min(gamma_mean):.6f}, Max: {np.max(gamma_mean):.6f}")
    print(f"  Mean: {np.mean(gamma_mean):.6f}, Std: {np.std(gamma_mean):.6f}")
    print(f"  Mean SEM: {np.mean(gamma_sem):.6f}, Median SEM: {np.median(gamma_sem):.6f}")
    
    return gamma_mean, gamma_sem, len(all_gammas)  # Return SEM instead of std


def load_gamma_from_old_structure(results_dir, pattern="output_*_*/model.pt"):
    """Load and average gamma from old folder structure: results/output_*_*/model.pt"""
    model_files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    
    if not model_files:
        print(f"No model files found matching pattern: {pattern}")
        print(f"  Searched in: {results_dir}")
        return None
    
    print(f"Found {len(model_files)} model files. Loading gamma...")
    all_gammas = []
    
    for model_path in model_files:
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            gamma = None
            if 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
            
            if gamma is not None:
                if torch.is_tensor(gamma):
                    gamma = gamma.detach().cpu().numpy()
                
                # Skip if all zeros
                if not np.allclose(gamma, 0):
                    all_gammas.append(gamma)
                    folder_name = os.path.basename(os.path.dirname(model_path))
                    print(f"  Loaded gamma from {folder_name}/model.pt (shape: {gamma.shape})")
                else:
                    folder_name = os.path.basename(os.path.dirname(model_path))
                    print(f"  Skipping {folder_name}/model.pt - gamma is all zeros")
        except Exception as e:
            folder_name = os.path.basename(os.path.dirname(model_path))
            print(f"  Warning: Could not load gamma from {folder_name}/model.pt: {e}")
            continue
    
    if not all_gammas:
        print("No non-zero gamma values found!")
        return None
    
    # Stack and average
    gamma_stack = np.stack(all_gammas)
    gamma_mean = np.mean(gamma_stack, axis=0)
    gamma_std = np.std(gamma_stack, axis=0)
    # Standard error of the mean (SEM) = std / sqrt(n)
    gamma_sem = gamma_std / np.sqrt(len(all_gammas))
    
    print(f"\n✓ Averaged gamma from {len(all_gammas)} batches")
    print(f"  Final shape: {gamma_mean.shape}")
    print(f"  Min: {np.min(gamma_mean):.6f}, Max: {np.max(gamma_mean):.6f}")
    print(f"  Mean: {np.mean(gamma_mean):.6f}, Std: {np.std(gamma_mean):.6f}")
    print(f"  Mean SEM: {np.mean(gamma_sem):.6f}, Median SEM: {np.median(gamma_sem):.6f}")
    
    return gamma_mean, gamma_sem, len(all_gammas)


def load_gamma_from_file(gamma_file_path):
    """Load gamma from a single checkpoint file."""
    if not os.path.exists(gamma_file_path):
        print(f"Error: File not found: {gamma_file_path}")
        return None
    
    try:
        if gamma_file_path.endswith('.npy'):
            gamma = np.load(gamma_file_path)
            print(f"Loaded gamma from numpy file: shape {gamma.shape}")
            return gamma, None, 1
        elif gamma_file_path.endswith('.pt'):
            print(f"Loading checkpoint: {gamma_file_path}")
            checkpoint = torch.load(gamma_file_path, map_location='cpu', weights_only=False)
            
            # Try to find gamma
            gamma = None
            if 'model_state_dict' in checkpoint:
                if 'gamma' in checkpoint['model_state_dict']:
                    gamma = checkpoint['model_state_dict']['gamma']
                    print(f"  Found gamma in model_state_dict")
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
                print(f"  Found gamma in root level")
            
            if gamma is None:
                print(f"  Error: Could not find gamma in checkpoint")
                print(f"  Available keys: {list(checkpoint.keys())}")
                if 'model_state_dict' in checkpoint:
                    print(f"  model_state_dict keys: {list(checkpoint['model_state_dict'].keys())}")
                return None
            
            if torch.is_tensor(gamma):
                gamma = gamma.detach().cpu().numpy()
            
            print(f"  Gamma shape: {gamma.shape}")
            print(f"  Gamma stats: min={np.min(gamma):.6f}, max={np.max(gamma):.6f}, mean={np.mean(gamma):.6f}")
            
            return gamma, None, 1
    except Exception as e:
        print(f"Error loading gamma: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return None


def create_gamma_dataframe(gamma, prs_names, n_signatures=21, gamma_sem=None):
    """Create melted dataframe from gamma matrix.
    
    Note: gamma structure when PCs are included:
      - Rows 0-35: PRS (36 features)
      - Row 36: Sex (1 feature)
      - Rows 37-46: PCs (10 features: f.22009.0.1 through f.22009.0.10)
      Total: 47 features
    
    We only extract the first 36 rows (PRS) for plotting.
    
    Args:
        gamma: (P_full, K) array of effect sizes
        prs_names: List of PRS names
        n_signatures: Number of signatures to include
        gamma_sem: Optional (P_full, K) array of standard errors of the mean
    """
    P_full, K = gamma.shape
    n_prs = len(prs_names)
    
    # Extract only PRS rows (first 36)
    if P_full > n_prs:
        print(f"  Note: Gamma has {P_full} features, extracting first {n_prs} PRS features")
        if P_full == 47:
            print(f"  Structure: rows 0-35=PRS, row 36=sex, rows 37-46=PCs")
        elif P_full == 37:
            print(f"  Structure: rows 0-35=PRS, row 36=sex")
        else:
            print(f"  (Remaining {P_full - n_prs} features are likely sex + PCs)")
        gamma_prs = gamma[:n_prs, :]  # Extract only PRS rows
        if gamma_sem is not None:
            gamma_sem_prs = gamma_sem[:n_prs, :]
        else:
            gamma_sem_prs = None
    else:
        gamma_prs = gamma
        gamma_sem_prs = gamma_sem
    
    P = gamma_prs.shape[0]
    
    # Ensure we have enough PRS names
    if len(prs_names) < P:
        prs_names = prs_names + [f"PRS_{i}" for i in range(len(prs_names), P)]
    prs_names = prs_names[:P]
    
    # Ensure we have enough signatures
    if K > n_signatures:
        K = n_signatures
    
    # Create dataframe - ensure PRS names are strings
    data = []
    for p in range(P):
        for k in range(K):
            effect = float(gamma_prs[p, k])
            sem = float(gamma_sem_prs[p, k]) if gamma_sem_prs is not None else np.nan
            # Calculate z-score (effect / standard error)
            z_score = effect / sem if (gamma_sem_prs is not None and sem > 0) else np.nan
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
            
            data.append({
                'prs': str(prs_names[p]),  # Ensure string type
                'signature': f'Sig {k}',
                'effect': effect,
                'sem': sem,
                'z_score': z_score,  # Effect size / standard error
                'p_value': p_value  # Two-tailed p-value
            })
    
    gamma_df = pd.DataFrame(data)
    
    # Add categories - ensure both keys and values are strings
    disease_categories = assign_disease_categories(prs_names)
    prs_to_category = dict(zip([str(p) for p in prs_names], disease_categories))
    gamma_df['category'] = gamma_df['prs'].map(prs_to_category)
    
    # Apply FDR correction for multiple testing (Benjamini-Hochberg)
    # With 36 PRS × 21 signatures = 756 tests, we need to correct
    valid_p_mask = ~gamma_df['p_value'].isna()
    if valid_p_mask.sum() > 0:
        p_values = gamma_df.loc[valid_p_mask, 'p_value'].values
        # Apply FDR correction
        rejected, p_adjusted, _, _ = multipletests(
            p_values, 
            alpha=0.05, 
            method='fdr_bh'  # Benjamini-Hochberg FDR correction
        )
        
        # Add FDR-corrected p-values
        gamma_df['p_value_fdr'] = np.nan
        gamma_df.loc[valid_p_mask, 'p_value_fdr'] = p_adjusted
        gamma_df['significant_fdr'] = False
        gamma_df.loc[valid_p_mask, 'significant_fdr'] = rejected
        
        n_significant_fdr = rejected.sum()
        print(f"\n  Multiple Testing Correction (FDR):")
        print(f"    Total tests: {len(p_values)}")
        print(f"    Significant at FDR < 0.05: {n_significant_fdr} ({100*n_significant_fdr/len(p_values):.1f}%)")
        print(f"    Significant at uncorrected p < 0.05: {(p_values < 0.05).sum()} ({100*(p_values < 0.05).sum()/len(p_values):.1f}%)")
    else:
        gamma_df['p_value_fdr'] = np.nan
        gamma_df['significant_fdr'] = False
        print(f"\n  Multiple Testing Correction: Not applied (no p-values available)")
    
    return gamma_df


def plot_top_associations_bar(gamma_df, output_dir, n_top=30, fdr_threshold=0.05):
    """Create bar plot of top PRS-signature associations by z-score."""
    # Filter by absolute z-score (only include valid z-scores)
    valid_z = gamma_df['z_score'].dropna()
    if len(valid_z) == 0:
        print("  Warning: No valid z-scores available. Falling back to effect size.")
        gamma_df['abs_effect'] = gamma_df['effect'].abs()
        top_associations = gamma_df.nlargest(n_top, 'abs_effect').copy()
        sort_col = 'effect'
        xlabel = 'Effect Size'
    else:
        gamma_df['abs_z_score'] = gamma_df['z_score'].abs()
        top_associations = gamma_df.nlargest(n_top, 'abs_z_score').copy()
        sort_col = 'z_score'
        xlabel = 'Z-Score (Effect / SEM)'
    
    # Sort by z-score or effect
    top_associations = top_associations.sort_values(sort_col)
    
    # Create label with FDR-corrected p-value if available
    def make_label(row):
        label = row['prs'] + " - " + row['signature']
        # Prefer FDR-corrected p-value, fall back to uncorrected
        pval_fdr = row.get('p_value_fdr', np.nan)
        pval_uncorr = row.get('p_value', np.nan)
        
        if not np.isnan(pval_fdr):
            # Show FDR-corrected p-value
            if pval_fdr < 0.001:
                label += " (FDR<0.001)"
            elif pval_fdr < 0.01:
                label += f" (FDR={pval_fdr:.3f})"
            else:
                label += f" (FDR={pval_fdr:.2f})"
            # Add asterisk if significant after FDR correction
            if row.get('significant_fdr', False):
                label += "*"
        elif not np.isnan(pval_uncorr):
            # Fall back to uncorrected p-value
            if pval_uncorr < 0.001:
                label += " (p<0.001)"
            elif pval_uncorr < 0.01:
                label += f" (p={pval_uncorr:.3f})"
            else:
                label += f" (p={pval_uncorr:.2f})"
        return label
    
    top_associations['label'] = top_associations.apply(make_label, axis=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(top_associations) * 0.35)))
    
    # Map categories to colors
    colors = [CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other']) for cat in top_associations['category']]
    
    bars = ax.barh(range(len(top_associations)), top_associations[sort_col], color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(top_associations)))
    ax.set_yticklabels(top_associations['label'], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_top} PRS-Signature Associations (by Z-Score)', fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat, alpha=0.8) 
                      for cat, color in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'top_prs_associations.pdf'
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    
    plt.close(fig)
    return top_associations


def plot_significant_heatmap(gamma_df, output_dir, fdr_threshold=0.05):
    """Create heatmap of significant PRS-signature associations by FDR-corrected p-value."""
    # Prefer FDR-corrected p-values, fall back to uncorrected if not available
    valid_fdr = gamma_df['p_value_fdr'].dropna()
    if len(valid_fdr) > 0:
        # Use FDR-corrected p-value threshold
        significant_effects = gamma_df[gamma_df['p_value_fdr'] < fdr_threshold].copy()
        value_col = 'z_score'
        cbar_label = 'Z-Score'
        vmin, vmax = -5, 5  # Reasonable z-score range
        print(f"  Filtering by FDR-corrected p-value < {fdr_threshold}")
        print(f"    Found {len(significant_effects)} significant associations")
    else:
        # Fallback to uncorrected p-value
        valid_p = gamma_df['p_value'].dropna()
        if len(valid_p) > 0:
            significant_effects = gamma_df[gamma_df['p_value'] < fdr_threshold].copy()
            value_col = 'z_score'
            cbar_label = 'Z-Score'
            vmin, vmax = -5, 5
            print(f"  FDR correction not available. Filtering by uncorrected p-value < {fdr_threshold}")
        else:
            # Fallback to z-score threshold (|z| > 2 roughly corresponds to p < 0.05)
            if 'abs_z_score' not in gamma_df.columns:
                gamma_df = gamma_df.copy()
                gamma_df['abs_z_score'] = gamma_df['z_score'].abs()
            z_threshold = 2.0
            significant_effects = gamma_df[gamma_df['abs_z_score'] > z_threshold].copy()
            value_col = 'z_score'
            cbar_label = 'Z-Score'
            vmin, vmax = -5, 5
            print(f"  No p-values available. Filtering by |z-score| > {z_threshold}")
    
    if len(significant_effects) == 0:
        print(f"Warning: No significant associations. Using top 50 by |z-score| or |effect|.")
        if 'abs_z_score' in gamma_df.columns:
            significant_effects = gamma_df.nlargest(50, 'abs_z_score').copy()
            value_col = 'z_score'
        else:
            gamma_df['abs_effect'] = gamma_df['effect'].abs()
            significant_effects = gamma_df.nlargest(50, 'abs_effect').copy()
            value_col = 'effect'
    
    # Create pivot table
    try:
        heatmap_data = significant_effects.pivot_table(
            values=value_col,
            index='prs',
            columns='signature',
            aggfunc='mean'
        )
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        print(f"  significant_effects shape: {significant_effects.shape}")
        print(f"  Columns: {significant_effects.columns.tolist()}")
        print(f"  Sample data:\n{significant_effects.head()}")
        raise
    
    # Ensure index is string type (not float)
    if not all(isinstance(idx, str) for idx in heatmap_data.index):
        print(f"Warning: Converting index to string. Index types: {[type(idx) for idx in heatmap_data.index[:5]]}")
        heatmap_data.index = heatmap_data.index.astype(str)
    
    # Sort PRSs by category
    prs_categories = dict(zip(gamma_df['prs'].astype(str), gamma_df['category']))
    category_order = ["Cardiovascular", "Metabolic", "Autoimmune", "Neurological", "Cancer", "Other"]
    prs_order = []
    for cat in category_order:
        prs_in_cat = [str(prs) for prs in heatmap_data.index if prs_categories.get(str(prs)) == cat]
        prs_order.extend(sorted(prs_in_cat))
    
    # Reorder - ensure all indices are strings
    prs_order_clean = [str(p) for p in prs_order if str(p) in heatmap_data.index]
    if len(prs_order_clean) > 0:
        heatmap_data = heatmap_data.reindex(prs_order_clean)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.8), 
                                     max(10, len(heatmap_data.index) * 0.4)))
    
    sns.heatmap(heatmap_data, 
                cmap='RdBu_r', 
                center=0,
                vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label},
                ax=ax,
                linewidths=0.5,
                linecolor='gray',
                fmt='.2f')
    
    title = 'Significant PRS-Signature Associations'
    if 'p_value_fdr' in gamma_df.columns and gamma_df['p_value_fdr'].notna().sum() > 0:
        title += ' (FDR < 0.05)'
    elif 'p_value' in gamma_df.columns and gamma_df['p_value'].notna().sum() > 0:
        title += ' (uncorrected p < 0.05)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
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


def plot_full_heatmap(gamma_df, output_dir):
    """Create full heatmap with all PRS-signature associations (using z-scores)."""
    # Ensure prs column is string type
    gamma_df = gamma_df.copy()
    gamma_df['prs'] = gamma_df['prs'].astype(str)
    
    # Use z-score if available, fall back to effect
    value_col = 'z_score' if 'z_score' in gamma_df.columns and gamma_df['z_score'].notna().sum() > 0 else 'effect'
    cbar_label = 'Z-Score' if value_col == 'z_score' else 'Effect Size'
    vmin, vmax = (-5, 5) if value_col == 'z_score' else (-0.3, 0.3)
    
    # Create pivot table
    try:
        heatmap_data = gamma_df.pivot_table(
            values=value_col,
            index='prs',
            columns='signature',
            aggfunc='mean'
        )
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        print(f"  gamma_df shape: {gamma_df.shape}")
        print(f"  Columns: {gamma_df.columns.tolist()}")
        raise
    
    # Ensure index is string type
    if not all(isinstance(idx, str) for idx in heatmap_data.index):
        heatmap_data.index = heatmap_data.index.astype(str)
    
    # Sort PRSs by category
    prs_categories = dict(zip(gamma_df['prs'].astype(str), gamma_df['category']))
    category_order = ["Cardiovascular", "Metabolic", "Autoimmune", "Neurological", "Cancer", "Other"]
    prs_order = []
    for cat in category_order:
        prs_in_cat = [str(prs) for prs in heatmap_data.index if prs_categories.get(str(prs)) == cat]
        prs_order.extend(sorted(prs_in_cat))
    
    # Reorder - ensure all indices are strings
    prs_order_clean = [str(p) for p in prs_order if str(p) in heatmap_data.index]
    if len(prs_order_clean) > 0:
        heatmap_data = heatmap_data.reindex(prs_order_clean)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.8), 
                                     max(10, len(heatmap_data.index) * 0.4)))
    
    sns.heatmap(heatmap_data, 
                cmap='RdBu_r', 
                center=0,
                vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label},
                ax=ax,
                linewidths=0.5,
                linecolor='gray',
                fmt='.2f')
    
    title = 'Complete PRS-Signature Association Matrix'
    if value_col == 'z_score':
        title += ' (Z-Scores)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
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
    parser.add_argument('--batch_dir', type=str,
                        help='Directory containing batch checkpoints')
    parser.add_argument('--old_results_dir', type=str,
                        help='Directory with old structure: results/output_*_*/model.pt')
    parser.add_argument('--gamma_file', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_noPCS/enrollment_model_VECTORIZED_W0.0001_batch_0_10000.pt',
                        help='Path to single gamma file (.pt or .npy)')
    parser.add_argument('--pattern', type=str,
                        default='enrollment_model_W0.0001_batch_*_*.pt',
                        help='Glob pattern for batch files')
    parser.add_argument('--prs_names_csv', type=str,
                        default='/Users/sarahurbut/aladynoulli2/prs_names.csv',
                        help='Path to PRS names CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures',
                        help='Output directory for plots')
    parser.add_argument('--fdr_threshold', type=float, default=0.05,
                        help='FDR threshold for significance (default: 0.05)')
    parser.add_argument('--n_top', type=int, default=30,
                        help='Number of top associations to show in bar plot')
    parser.add_argument('--n_signatures', type=int, default=21,
                        help='Number of signatures (K)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load PRS names
    prs_names = load_prs_names(args.prs_names_csv)
    print(f"\nLoaded {len(prs_names)} PRS names")
    print(f"  First few: {prs_names[:5]}")
    
    # Load gamma
    gamma = None
    gamma_sem = None
    n_batches = 0
    loading_method = None
    
    # Priority: old_results_dir > batch_dir > gamma_file (single file is a fallback)
    # If old_results_dir is provided, use old folder structure
    if args.old_results_dir:
        print(f"\n{'='*60}")
        print(f"METHOD: Loading gamma from OLD FOLDER STRUCTURE (results/output_*_*/model.pt)")
        print(f"{'='*60}")
        print(f"Directory: {args.old_results_dir}")
        result = load_gamma_from_old_structure(args.old_results_dir)
        if result:
            gamma, gamma_sem, n_batches = result
            loading_method = "old_structure"
            print(f"\n✓ Successfully loaded gamma from {n_batches} batches")
            if gamma_sem is not None:
                print(f"  SEM available: Yes (mean={np.mean(gamma_sem):.6f}, median={np.median(gamma_sem):.6f})")
                print(f"  Z-scores (effect/SEM) will be calculated")
            else:
                print(f"  SEM available: No")
        else:
            print(f"  ⚠ Old structure loading failed, will try other methods...")
    
    # If batch_dir is provided, try batch loading first
    if gamma is None and args.batch_dir:
        print(f"\n{'='*60}")
        print(f"METHOD: Loading gamma from BATCH DIRECTORY (will average across batches)")
        print(f"{'='*60}")
        print(f"Directory: {args.batch_dir}")
        print(f"Pattern: {args.pattern}")
        result = load_gamma_from_batches(args.batch_dir, args.pattern)
        if result:
            gamma, gamma_sem, n_batches = result
            loading_method = "batch"
            print(f"\n✓ Successfully loaded gamma from {n_batches} batches")
            if gamma_sem is not None:
                print(f"  SEM available: Yes (mean={np.mean(gamma_sem):.6f}, median={np.median(gamma_sem):.6f})")
                print(f"  Z-scores (effect/SEM) will be calculated")
            else:
                print(f"  SEM available: No")
        else:
            print(f"  ⚠ Batch loading failed, will try single file fallback...")
    
    # Fallback: if batch loading failed or wasn't attempted, use single file
    if gamma is None and args.gamma_file:
        print(f"\n{'='*60}")
        print(f"METHOD: Loading gamma from SINGLE FILE (fallback - no SEM available)")
        print(f"{'='*60}")
        print(f"File: {args.gamma_file}")
        result = load_gamma_from_file(args.gamma_file)
        if result:
            gamma, gamma_sem, n_batches = result
            loading_method = "single_file"
            print(f"\n✓ Successfully loaded gamma from single file")
            print(f"  SEM available: No (single file - cannot calculate SEM without multiple batches)")
            print(f"  Z-scores cannot be calculated (need SEM)")
        else:
            print(f"  ⚠ Single file loading also failed")
    
    if gamma is None:
        print("Error: Could not load gamma data")
        return
    
    print(f"\n{'='*60}")
    print(f"Loading method: {loading_method.upper()}")
    print(f"{'='*60}")
    
    # Create dataframe
    print(f"\nCreating gamma dataframe...")
    gamma_df = create_gamma_dataframe(gamma, prs_names, args.n_signatures, gamma_sem)
    print(f"  Created dataframe: {gamma_df.shape}")
    print(f"  Effect range: [{gamma_df['effect'].min():.6f}, {gamma_df['effect'].max():.6f}]")
    
    # Print SEM and z-score statistics
    if gamma_sem is not None:
        print(f"\nStandard Error Statistics:")
        print(f"  SEM range: [{gamma_df['sem'].min():.6f}, {gamma_df['sem'].max():.6f}]")
        print(f"  Mean SEM: {gamma_df['sem'].mean():.6f}, Median SEM: {gamma_df['sem'].median():.6f}")
        print(f"\nZ-score Statistics (effect / SEM):")
        valid_z = gamma_df['z_score'].dropna()
        if len(valid_z) > 0:
            print(f"  Z-score range: [{valid_z.min():.2f}, {valid_z.max():.2f}]")
            print(f"  Mean |z-score|: {valid_z.abs().mean():.2f}")
            print(f"  |z-score| > 2: {(valid_z.abs() > 2).sum()} / {len(valid_z)} ({(valid_z.abs() > 2).sum()/len(valid_z)*100:.1f}%)")
            print(f"  |z-score| > 3: {(valid_z.abs() > 3).sum()} / {len(valid_z)} ({(valid_z.abs() > 3).sum()/len(valid_z)*100:.1f}%)")
        else:
            print(f"  No valid z-scores (all SEM values are zero or NaN)")
    else:
        print(f"\nNote: SEM not available (single file), z-scores cannot be calculated")
    
    # Count significant associations (using FDR if available)
    total_tests = len(gamma_df)
    if 'significant_fdr' in gamma_df.columns:
        n_significant_fdr = gamma_df['significant_fdr'].sum()
        print(f"\nSignificance Statistics (FDR-corrected):")
        print(f"  Significant at FDR < {args.fdr_threshold}: {n_significant_fdr} / {total_tests} ({100*n_significant_fdr/total_tests:.2f}%)")
    else:
        print(f"\nSignificance Statistics:")
        print(f"  Total tests: {total_tests}")
        print(f"  (FDR correction not available - need batch data)")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Top associations bar plot
    print("\n1. Creating top associations bar plot (by z-score)...")
    top_associations = plot_top_associations_bar(gamma_df, output_dir, args.n_top, args.fdr_threshold)
    
    # 2. Significant associations heatmap
    print("\n2. Creating significant associations heatmap (FDR < 0.05)...")
    plot_significant_heatmap(gamma_df, output_dir, args.fdr_threshold)
    
    # 3. Full heatmap
    print("\n3. Creating full heatmap...")
    plot_full_heatmap(gamma_df, output_dir)
    
    # Save CSV for reference
    csv_path = output_dir / 'gamma_associations.csv'
    gamma_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved gamma dataframe to: {csv_path}")
    
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

