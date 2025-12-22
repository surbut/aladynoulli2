#!/usr/bin/env python3
"""
Plot signature-gene-disease associations.

For each signature-gene combination from rare variant analysis:
- X-axis: Gene-disease correlation (rare variant burden correlation)
- Y-axis: Signature-disease loading (phi value)

Hypothesis: Diseases with high signature loading should also have high correlation
with rare variant status of the associated gene.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from scipy import stats
from scipy.special import expit as sigmoid  # Inverse logit
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PHI_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt'
DISEASE_NAMES_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv'
GENE_DISEASE_CSV = '/Users/sarahurbut/Downloads/rare_variant_burden_associations.csv'  # Default filename
RARE_VARIANT_RESULTS_DIR = '/Users/sarahurbut/Desktop/SIG/gene_based_analysis/canonical'  # For mask3 results

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'signature_gene_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# Filtering and transformation options
MIN_ABS_CORRELATION = 0.001  # Filter out diseases with very low absolute correlation
MIN_ABS_PHI = None  # Filter out diseases with very low absolute phi (None = no filter)
PHI_TRANSFORM = 'sigmoid'  # Options: 'none', 'standardize', 'center', 'negate', 'sigmoid'
# 'sigmoid': inverse logit transform (phi -> probability) - RECOMMENDED since phi is logit
# 'standardize': z-score transform
# 'center': subtract mean
# 'negate': multiply by -1 (if phi values are negative)
# 'none': no transformation

# Signature-gene combinations from rare variant analysis
SIGNATURE_GENE_COMBOS = [
    (0, 'TTN'),
    (0, 'EEF1A1'),
    (3, 'ADGRG7'),
    (4, 'C10orf67'),
    (5, 'LDLR'),
    (5, 'APOB'),
    (5, 'LPA'),
    (5, 'CDH26'),
    (6, 'BRCA2'),
    (7, 'GNB2'),
    (8, 'RNF216'),
    (10, 'MIP'),
    (11, 'CLPTM1L'),
    (16, 'PKD1'),
    (16, 'TET2'),
    (16, 'BRCA2'),
    (19, 'OCA2'),
    (19, 'RAD52'),
    (20, 'DEFB1'),
]

# =============================================================================
# LOAD DATA
# =============================================================================

def load_phi_and_disease_names():
    """Load phi matrix and disease names."""
    print("Loading phi matrix...")
    ckpt = torch.load(PHI_PATH, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in ckpt and 'phi' in ckpt['model_state_dict']:
        phi = ckpt['model_state_dict']['phi']
    elif 'phi' in ckpt:
        phi = ckpt['phi']
    else:
        raise ValueError(f"No phi found in {PHI_PATH}")
    
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    
    K, D, T = phi.shape
    print(f"  Phi shape: {phi.shape} (K={K}, D={D}, T={T})")
    
    # Take max phi over time (peak association)
    phi_mean = np.max(phi, axis=2)  # (K, D)
    
    print("Loading disease names...")
    disease_names_path = Path(DISEASE_NAMES_PATH)
    
    if disease_names_path.suffix == '.csv':
        # Load from CSV file
        disease_df = pd.read_csv(disease_names_path)
        if 'x' in disease_df.columns:
            disease_names = disease_df['x'].tolist()
        elif len(disease_df.columns) >= 2:
            disease_names = disease_df.iloc[:, 1].tolist()  # Second column
        else:
            disease_names = disease_df.iloc[:, 0].tolist()  # First column
        
        # Remove header if it's a string like 'x'
        if len(disease_names) > 0 and str(disease_names[0]).lower() in ['x', 'disease', 'name']:
            disease_names = disease_names[1:]
        
        # Convert to strings
        disease_names = [str(name) if pd.notna(name) else f"Disease_{i}" for i, name in enumerate(disease_names)]
    else:
        # Load from torch checkpoint
        ckpt_diseases = torch.load(DISEASE_NAMES_PATH, map_location='cpu', weights_only=False)
        disease_names = ckpt_diseases['disease_names']
        
        if isinstance(disease_names, (list, tuple)):
            disease_names = list(disease_names)
        elif hasattr(disease_names, 'values'):
            disease_names = disease_names.values.tolist()
        elif torch.is_tensor(disease_names):
            disease_names = disease_names.tolist()
    
    print(f"  Loaded {len(disease_names)} disease names")
    
    return phi_mean, disease_names


def load_rare_variant_signature_gene_associations(results_dir):
    """Load rare variant association results (mask3) for signature-gene pairs.
    
    Returns a dictionary: (signature, gene) -> {LOG10P, BETA, SE, MAF, ...}
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"  ⚠️  Rare variant results directory not found: {results_dir}")
        return {}
    
    print(f"\nLoading rare variant signature-gene associations from {results_dir}...")
    mask3_files = sorted(results_dir.glob("Mask3_*_significant_canonical.tsv"))
    
    if len(mask3_files) == 0:
        print(f"  ⚠️  No mask3 files found in {results_dir}")
        return {}
    
    mask3_results = []
    for file in mask3_files:
        parts = file.stem.replace("_significant_canonical", "").split("_")
        maf = parts[1] if len(parts) > 1 else "unknown"
        
        df = pd.read_csv(file, sep='\t')
        df['MAF'] = maf
        mask3_results.append(df)
    
    mask3_df = pd.concat(mask3_results, ignore_index=True)
    print(f"  Loaded {len(mask3_df)} associations from mask3")
    
    # Get best result per gene-signature (across MAF thresholds)
    best_results = mask3_df.loc[mask3_df.groupby(['SIG', 'SYMBOL'])['LOG10P'].idxmax()].copy()
    
    # Create lookup dictionary: (signature, gene) -> association info
    sig_gene_associations = {}
    for _, row in best_results.iterrows():
        sig = int(row['SIG'])
        gene = str(row['SYMBOL']).strip().upper()
        key = (sig, gene)
        sig_gene_associations[key] = {
            'LOG10P': float(row['LOG10P']),
            'P_value': 10**(-float(row['LOG10P'])),
            'BETA': float(row.get('BETA', np.nan)),
            'SE': float(row.get('SE', np.nan)),
            'MAF': str(row.get('MAF', 'unknown'))
        }
    
    print(f"  Found {len(sig_gene_associations)} signature-gene associations")
    return sig_gene_associations


def load_gene_disease_correlations(csv_path):
    """Load gene-disease correlation CSV.
    
    Expected format:
    - Columns: gene, disease, correlation (or similar)
    """
    print(f"\nLoading gene-disease correlations from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Genes: {df.iloc[:, 0].nunique()} unique")
    print(f"  Diseases: {df.iloc[:, 1].nunique()} unique")
    
    # Auto-detect column names
    gene_col = None
    disease_col = None
    corr_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'gene' in col_lower or col_lower == 'symbol':
            gene_col = col
        elif 'disease' in col_lower or 'pheno' in col_lower:
            disease_col = col
        elif 'correlation' in col_lower or 'corr' in col_lower or 'r' == col_lower:
            corr_col = col
    
    if gene_col is None:
        gene_col = df.columns[0]
    if disease_col is None:
        disease_col = df.columns[1]
    if corr_col is None:
        # Look for numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr_col = numeric_cols[0]
        else:
            corr_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]
    
    print(f"  Using columns: gene='{gene_col}', disease='{disease_col}', correlation='{corr_col}'")
    
    # Create lookup dictionary: (gene, disease) -> correlation
    correlations = {}
    for _, row in df.iterrows():
        gene = str(row[gene_col]).strip()
        disease = str(row[disease_col]).strip()
        corr = float(row[corr_col])
        correlations[(gene, disease)] = corr
    
    return correlations, gene_col, disease_col, corr_col


def match_disease_names(disease_name_list, csv_disease_name):
    """Try to match disease names between phi and CSV."""
    csv_name_lower = str(csv_disease_name).strip().lower()
    
    # Exact match
    for i, name in enumerate(disease_name_list):
        if str(name).strip().lower() == csv_name_lower:
            return i, name
    
    # Partial match (contains)
    for i, name in enumerate(disease_name_list):
        name_lower = str(name).strip().lower()
        if csv_name_lower in name_lower or name_lower in csv_name_lower:
            return i, name
    
    return None, None


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_signature_gene_association(signature, gene, phi_mean, disease_names, 
                                   gene_disease_correlations, output_dir,
                                   sig_gene_associations=None,
                                   min_abs_correlation=MIN_ABS_CORRELATION,
                                   min_abs_phi=MIN_ABS_PHI,
                                   phi_transform=PHI_TRANSFORM):
    """Plot diseases for a signature-gene combo."""
    
    K, D = phi_mean.shape
    if signature >= K:
        print(f"  ⚠️  Signature {signature} out of range (max={K-1})")
        return None
    
    # Get signature-gene association info (from rare variant analysis)
    sig_gene_info = None
    if sig_gene_associations:
        key = (signature, gene.upper())
        sig_gene_info = sig_gene_associations.get(key)
    
    # Get signature-disease loadings (phi)
    sig_phi = phi_mean[signature, :]  # (D,)
    
    # Get gene-disease correlations for this gene
    gene_corrs = []
    matched_diseases = []
    matched_indices = []
    
    # First, collect all correlations for this gene from CSV
    gene_correlations_dict = {}  # disease_name_from_csv -> correlation
    for (gene_csv, disease_csv), corr in gene_disease_correlations.items():
        if gene_csv.upper() == gene.upper():
            gene_correlations_dict[disease_csv] = corr
    
    if len(gene_correlations_dict) == 0:
        print(f"  ⚠️  Gene {gene} not found in correlation data")
        return None
    
    # Now try to match each CSV disease to our disease_names list
    for disease_csv, corr in gene_correlations_dict.items():
        # Try to find matching disease in our disease_names
        d_idx, matched_name = match_disease_names(disease_names, disease_csv)
        if d_idx is not None:
            gene_corrs.append(corr)
            matched_diseases.append(matched_name)
            matched_indices.append(d_idx)
    
    if len(gene_corrs) == 0:
        print(f"  ⚠️  No matched diseases found for gene {gene}")
        return None
    
    # Get phi values for matched diseases (before filtering)
    sig_phi_values_raw = [sig_phi[idx] for idx in matched_indices]
    
    # Filter uninteresting diseases
    original_n = len(gene_corrs)
    filtered_data = list(zip(gene_corrs, sig_phi_values_raw, matched_diseases, matched_indices))
    
    # Filter by absolute correlation
    if min_abs_correlation is not None and min_abs_correlation > 0:
        filtered_data = [(c, p, d, idx) for c, p, d, idx in filtered_data if abs(c) >= min_abs_correlation]
    
    # Filter by absolute phi
    if min_abs_phi is not None and min_abs_phi > 0:
        filtered_data = [(c, p, d, idx) for c, p, d, idx in filtered_data if abs(p) >= min_abs_phi]
    
    if len(filtered_data) == 0:
        print(f"  ⚠️  No diseases remaining after filtering (original: {original_n})")
        return None
    
    # Unpack filtered data
    gene_corrs, sig_phi_values, matched_diseases, matched_indices = zip(*filtered_data)
    gene_corrs = list(gene_corrs)
    sig_phi_values = np.array(sig_phi_values)
    matched_diseases = list(matched_diseases)
    matched_indices = list(matched_indices)
    
    # Apply phi transformation
    if phi_transform == 'sigmoid':
        # Inverse logit transform (phi is logit, convert to probability)
        sig_phi_values = sigmoid(sig_phi_values)
        phi_label = f'Signature {signature} - Disease Loading (probability)'
    elif phi_transform == 'standardize':
        # Z-score transform
        sig_phi_values = (sig_phi_values - np.mean(sig_phi_values)) / (np.std(sig_phi_values) + 1e-10)
        phi_label = f'Signature {signature} - Disease Loading (φ, standardized)'
    elif phi_transform == 'center':
        # Center around mean
        sig_phi_values = sig_phi_values - np.mean(sig_phi_values)
        phi_label = f'Signature {signature} - Disease Loading (φ, centered)'
    elif phi_transform == 'negate':
        # Negate (if all values are negative)
        sig_phi_values = -sig_phi_values
        phi_label = f'Signature {signature} - Disease Loading (φ, negated)'
    else:
        phi_label = f'Signature {signature} - Disease Loading (φ)'
    
    if original_n > len(gene_corrs):
        print(f"  Filtered from {original_n} to {len(gene_corrs)} diseases")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(gene_corrs, sig_phi_values, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add correlation line and statistics
    r = np.nan
    p_val = np.nan
    if len(gene_corrs) > 2:
        # Calculate correlation and p-value
        r, p_val = stats.pearsonr(gene_corrs, sig_phi_values)
        
        # Add trend line
        z = np.polyfit(gene_corrs, sig_phi_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(gene_corrs), max(gene_corrs), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={r:.3f})')
        ax.legend(fontsize=10)
        
        # Add text annotation
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.2e}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels
    ax.set_xlabel(f'{gene} Rare Variant Burden - Disease Correlation', fontsize=12, fontweight='bold')
    ax.set_ylabel(phi_label, fontsize=12, fontweight='bold')
    
    # Build title with signature-gene association info
    title = f'Signature {signature} - {gene}'
    if sig_gene_info:
        log10p = sig_gene_info['LOG10P']
        pval_str = f"{sig_gene_info['P_value']:.2e}"
        beta = sig_gene_info['BETA']
        title += f'\nGene-Sig: -log₁₀(p)={log10p:.2f} (p={pval_str})'
        if not np.isnan(beta):
            title += f', β={beta:.4f}'
    
    title_note = f'\n(n={len(gene_corrs)} diseases'
    if original_n > len(gene_corrs):
        title_note += f', filtered from {original_n}'
    title_note += ')'
    ax.set_title(title + title_note, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Label top points
    if len(gene_corrs) > 0:
        # Find top points by absolute correlation or phi value
        top_n = min(5, len(gene_corrs))
        # Score = abs(correlation) * abs(phi)
        scores = [abs(corr) * abs(phi_val) for corr, phi_val in zip(gene_corrs, sig_phi_values)]
        top_indices = np.argsort(scores)[-top_n:]
        
        for idx in top_indices:
            ax.annotate(matched_diseases[idx], 
                       (gene_corrs[idx], sig_phi_values[idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'sig{signature}_{gene}_association.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()
    
    return {
        'signature': signature,
        'gene': gene,
        'n_diseases': len(gene_corrs),
        'correlation': r if len(gene_corrs) > 2 else np.nan,
        'p_value': p_val if len(gene_corrs) > 2 else np.nan
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    import sys
    
    # Get CSV path from command line or try default locations
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try default filename in current directory and parent directories
        possible_paths = [
            Path(__file__).parent / GENE_DISEASE_CSV,
            Path(__file__).parent.parent / GENE_DISEASE_CSV,
            Path.cwd() / GENE_DISEASE_CSV,
            Path(__file__).parent / '..' / '..' / '..' / GENE_DISEASE_CSV,
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = str(path.resolve())
                print(f"Found CSV at: {csv_path}")
                break
        
        if csv_path is None:
            csv_path = input(f"Enter path to {GENE_DISEASE_CSV}: ").strip()
            if csv_path.startswith('"') or csv_path.startswith("'"):
                csv_path = csv_path[1:-1]
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load data
    phi_mean, disease_names = load_phi_and_disease_names()
    gene_disease_correlations, gene_col, disease_col, corr_col = load_gene_disease_correlations(csv_path)
    sig_gene_associations = load_rare_variant_signature_gene_associations(RARE_VARIANT_RESULTS_DIR)
    
    print(f"\n{'='*80}")
    print("PLOTTING SIGNATURE-GENE-DISEASE ASSOCIATIONS")
    print(f"{'='*80}")
    
    results = []
    for sig, gene in SIGNATURE_GENE_COMBOS:
        print(f"\nPlotting Signature {sig} - {gene}...")
        result = plot_signature_gene_association(
            sig, gene, phi_mean, disease_names, 
            gene_disease_correlations, OUTPUT_DIR,
            sig_gene_associations=sig_gene_associations,
            min_abs_correlation=MIN_ABS_CORRELATION,
            min_abs_phi=MIN_ABS_PHI,
            phi_transform=PHI_TRANSFORM
        )
        if result:
            results.append(result)
    
    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = OUTPUT_DIR / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Saved summary: {summary_path}")
        print(f"\nSummary statistics:")
        print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"✓ Completed! Plots saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

