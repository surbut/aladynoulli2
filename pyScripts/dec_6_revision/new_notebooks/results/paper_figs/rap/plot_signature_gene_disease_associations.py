#!/usr/bin/env python3
"""
Plot signature-gene-disease associations.

For each signature-gene combination from rare variant analysis:
- X-axis: Gene-disease correlation (rare variant burden correlation)
- Y-axis: Signature-disease loading (max phi over time)

Hypothesis: Diseases with high signature loading should also have high correlation
with rare variant status of the associated gene.

Note: Uses MAX PHI (peak association over time, pooled across batches). This captures
the strongest signature-disease association across all timepoints and is less biased
toward rare diseases compared to psi.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from scipy import stats
from scipy.special import expit as sigmoid  # Inverse logit
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths and options
GENERATE_BOTH_PLOTS = True  # If True, generates both: (1) max phi vs correlation, (2) psi vs log OR
# If False, uses settings below:
USE_PSI_INSTEAD_OF_PHI = False  # If True, pool psi from censor_e_batchrun_vectorized batches; if False, use max phi from master
USE_LOG_OR_FOR_X_AXIS = False  # If True, use log OR on x-axis (requires log_or column in CSV); if False, use correlation

# PHI_PATH is used when USE_PSI_INSTEAD_OF_PHI = False:
#   Can be either:
#   1. A directory containing batch files (phi will be pooled across batches, then max over time)
#      Example: '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full'
#   2. A single checkpoint file (phi will be loaded from that file, then max over time)
#      Example: '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt'
PHI_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt'  # Master checkpoint (already pooled from corrected E batches)

# PSI_BATCH_DIR is used when USE_PSI_INSTEAD_OF_PHI = True:
#   Directory containing batch files with corrected E (psi will be pooled across batches)
PSI_BATCH_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized'
PSI_BATCH_PATTERN = 'enrollment_model_W0.0001_batch_*_*.pt'  # Pattern to match batch files
DISEASE_NAMES_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv'
GENE_DISEASE_CSV = '/Users/sarahurbut/Downloads/rare_variant_burden_associations-2.csv'  # Default filename (with p-values)
RARE_VARIANT_RESULTS_DIR = '/Users/sarahurbut/Desktop/SIG/gene_based_analysis/canonical'  # For mask3 results

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'signature_gene_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# Filtering and transformation options
MIN_ABS_CORRELATION = None  # Filter out diseases with very low absolute correlation
MIN_ABS_PHI = None  # Filter out diseases with very low signature loading (max phi)
# Set to None to disable filtering. If using sigmoid transform, typical thresholds might be 0.002-0.01
# (probability scale after sigmoid transformation)
MAX_P_VALUE = None  # Filter to only significant gene-disease correlations (None = no filter)
PHI_TRANSFORM = 'sigmoid'  # Options: 'none', 'exp', 'standardize', 'center', 'sigmoid'
# 'sigmoid': convert logit (phi) to probability - RECOMMENDED (interprets phi as log-odds)
# 'none': use phi directly (logit/log-odds scale)
# 'exp': convert logit to OR (exp(phi)) - less appropriate
# 'standardize': z-score transform
# 'center': subtract mean

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

def pool_psi_from_batches(batch_dir, pattern, max_batches=None):
    """
    Load and pool psi from all batch files.
    
    Args:
        batch_dir: Directory containing batch files
        pattern: Glob pattern for batch files (e.g., "batch_*.pt")
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled psi (mean across batches) as numpy array, shape (K, D)
    """
    batch_dir = Path(batch_dir)
    all_psis = []
    
    # Find all matching files
    files = sorted(glob.glob(str(batch_dir / pattern)))
    print(f"  Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Extract psi
            if 'model_state_dict' in checkpoint and 'psi' in checkpoint['model_state_dict']:
                psi = checkpoint['model_state_dict']['psi']
            elif 'psi' in checkpoint:
                psi = checkpoint['psi']
            else:
                print(f"    Warning: No psi found in {Path(file_path).name}")
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(psi):
                psi = psi.detach().cpu().numpy()
            
            all_psis.append(psi)
            print(f"    Loaded psi from {Path(file_path).name}, shape: {psi.shape}")
            
        except Exception as e:
            print(f"    Error loading {Path(file_path).name}: {e}")
            continue
    
    if len(all_psis) == 0:
        raise ValueError(f"No psi arrays loaded from {batch_dir / pattern}")
    
    # Stack and compute mean
    psi_stack = np.stack(all_psis, axis=0)  # (n_batches, K, D)
    psi_mean = np.mean(psi_stack, axis=0)  # (K, D)
    psi_std = np.std(psi_stack, axis=0)  # (K, D)
    
    print(f"  ✓ Pooled {len(all_psis)} batches")
    print(f"  ✓ Psi shape: {psi_mean.shape}")
    print(f"  ✓ Psi range: [{psi_mean.min():.4f}, {psi_mean.max():.4f}]")
    print(f"  ✓ Psi mean: {psi_mean.mean():.4f}, std (across batches): {psi_std.mean():.4f}")
    
    return psi_mean

def pool_phi_from_batches(batch_dir, pattern, max_batches=None):
    """
    Load and pool phi from all batch files, then take max over time.
    
    Args:
        batch_dir: Directory containing batch files
        pattern: Glob pattern for batch files (e.g., "enrollment_model_W0.0001_batch_*.pt")
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled max phi (mean across batches, max over time) as numpy array, shape (K, D)
    """
    batch_dir = Path(batch_dir)
    all_phi_maxes = []
    
    # Find all matching files
    files = sorted(glob.glob(str(batch_dir / pattern)))
    print(f"  Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Extract phi
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                print(f"    Warning: No phi found in {Path(file_path).name}")
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            # Take max over time dimension
            if len(phi.shape) == 3:  # (K, D, T)
                phi_max = np.max(phi, axis=2)  # (K, D)
            elif len(phi.shape) == 2:  # (K, D) - already no time dimension
                phi_max = phi
            else:
                raise ValueError(f"Unexpected phi shape: {phi.shape}")
            
            all_phi_maxes.append(phi_max)
            print(f"    Loaded phi from {Path(file_path).name}, shape: {phi.shape} -> max: {phi_max.shape}")
            
        except Exception as e:
            print(f"    Error loading {Path(file_path).name}: {e}")
            continue
    
    if len(all_phi_maxes) == 0:
        raise ValueError(f"No phi arrays loaded from {batch_dir / pattern}")
    
    # Stack and compute mean
    phi_stack = np.stack(all_phi_maxes, axis=0)  # (n_batches, K, D)
    phi_mean = np.mean(phi_stack, axis=0)  # (K, D)
    phi_std = np.std(phi_stack, axis=0)  # (K, D)
    
    print(f"  ✓ Pooled {len(all_phi_maxes)} batches")
    print(f"  ✓ Max phi shape: {phi_mean.shape}")
    print(f"  ✓ Max phi range: [{phi_mean.min():.4f}, {phi_mean.max():.4f}]")
    print(f"  ✓ Max phi mean: {phi_mean.mean():.4f}, std (across batches): {phi_std.mean():.4f}")
    
    return phi_mean

def load_phi_max_and_disease_names():
    """Load phi matrix (max over time) or psi matrix, depending on USE_PSI_INSTEAD_OF_PHI flag.
    
    If USE_PSI_INSTEAD_OF_PHI = True:
        Pools psi from censor_e_batchrun_vectorized batches
    
    If USE_PSI_INSTEAD_OF_PHI = False:
        Supports two modes for phi:
        1. If PHI_PATH is a directory: Pool phi from batch files, then take max over time
        2. If PHI_PATH is a file: Load from single checkpoint, then take max over time
    """
    if USE_PSI_INSTEAD_OF_PHI:
        print("Loading psi matrix (log OR) from censor_e_batchrun_vectorized batches...")
        matrix_mean = pool_psi_from_batches(PSI_BATCH_DIR, PSI_BATCH_PATTERN, max_batches=None)
        K, D = matrix_mean.shape
        print(f"  ✓ Final psi shape: {matrix_mean.shape} (K={K}, D={D})")
        print(f"  ✓ Psi represents log OR (log odds ratio) for signature-disease associations")
    else:
        print("Loading phi matrix (will take max over time)...")
        phi_path = Path(PHI_PATH)
        
        # Check if PHI_PATH is a directory or file
        if phi_path.is_dir():
            # Pool phi from batches
            print(f"  Pooling phi from batches in: {phi_path}")
            # Default pattern for UKB retrospective batches
            pattern = "enrollment_model_W0.0001_batch_*_*.pt"
            try:
                phi_max_mean = pool_phi_from_batches(phi_path, pattern, max_batches=None)
            except ValueError as e:
                # Try alternative pattern
                print(f"  Trying alternative pattern: model_*.pt")
                pattern = "model_*.pt"
                phi_max_mean = pool_phi_from_batches(phi_path, pattern, max_batches=None)
        elif phi_path.is_file():
            # Load from single checkpoint
            print(f"  Loading phi from checkpoint: {phi_path}")
            ckpt = torch.load(phi_path, map_location='cpu', weights_only=False)
    
            # Load phi
    if 'model_state_dict' in ckpt and 'phi' in ckpt['model_state_dict']:
        phi = ckpt['model_state_dict']['phi']
    elif 'phi' in ckpt:
        phi = ckpt['phi']
    else:
                raise ValueError(f"No phi found in {phi_path}")
    
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    
    K, D, T = phi.shape
    print(f"  Phi shape: {phi.shape} (K={K}, D={D}, T={T})")
    # Take max phi over time (peak association)
            phi_max_mean = np.max(phi, axis=2)  # (K, D)
            print(f"  Taking max over time dimension: {phi_max_mean.shape}")
        else:
            raise ValueError(f"PHI_PATH does not exist: {phi_path}")
        
        matrix_mean = phi_max_mean
        K, D = matrix_mean.shape
        print(f"  ✓ Final max phi shape: {matrix_mean.shape} (K={K}, D={D})")
        print(f"  ✓ Max phi represents peak signature-disease association over time")
    
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
    
    return matrix_mean, disease_names


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
    """Load gene-disease correlation CSV (or log OR if available).
    
    Expected format:
    - Columns: gene, disease, correlation, p_value (or similar)
    - Optional: log_or, log_or_p_value (if computed)
    """
    print(f"\nLoading gene-disease associations from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Genes: {df.iloc[:, 0].nunique()} unique")
    print(f"  Diseases: {df.iloc[:, 1].nunique()} unique")
    
    # Auto-detect column names
    gene_col = None
    disease_col = None
    corr_col = None
    pval_col = None
    log_or_col = None
    log_or_pval_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'gene' in col_lower or col_lower == 'symbol':
            gene_col = col
        elif 'disease' in col_lower or 'pheno' in col_lower:
            disease_col = col
        elif 'correlation' in col_lower or 'corr' in col_lower or 'r' == col_lower:
            corr_col = col
        elif 'p' in col_lower and ('value' in col_lower or 'val' in col_lower or col_lower == 'p_value' or col_lower == 'pvalue'):
            pval_col = col
        elif 'log_or' in col_lower or 'logor' in col_lower:
            log_or_col = col
        elif 'log_or_p' in col_lower or 'logor_p' in col_lower:
            log_or_pval_col = col
    
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
    
    if pval_col is None:
        # Try to find p-value column (usually after correlation)
        if len(df.columns) > 3:
            pval_col = df.columns[3]
        elif 'p_value' in df.columns:
            pval_col = 'p_value'
        elif 'pvalue' in df.columns:
            pval_col = 'pvalue'
    
    print(f"  Using columns: gene='{gene_col}', disease='{disease_col}', correlation='{corr_col}'")
    if pval_col:
        print(f"  Found p-value column: '{pval_col}'")
    else:
        print(f"  ⚠️  No p-value column found - significance filtering will be skipped")
    if log_or_col:
        print(f"  Found log OR column: '{log_or_col}'")
    if log_or_pval_col:
        print(f"  Found log OR p-value column: '{log_or_pval_col}'")
    
    # Create lookup dictionaries: (gene, disease) -> correlation/p_value/log_or
    correlations = {}
    p_values = {}
    log_ors = {}
    log_or_p_values = {}
    for _, row in df.iterrows():
        gene = str(row[gene_col]).strip()
        disease = str(row[disease_col]).strip()
        corr = float(row[corr_col])
        correlations[(gene, disease)] = corr
    
        if pval_col and pval_col in df.columns:
            pval = row[pval_col]
            if pd.notna(pval):
                p_values[(gene, disease)] = float(pval)
        
        if log_or_col and log_or_col in df.columns:
            log_or = row[log_or_col]
            if pd.notna(log_or):
                log_ors[(gene, disease)] = float(log_or)
        
        if log_or_pval_col and log_or_pval_col in df.columns:
            log_or_pval = row[log_or_pval_col]
            if pd.notna(log_or_pval):
                log_or_p_values[(gene, disease)] = float(log_or_pval)
    
    return correlations, p_values, gene_col, disease_col, corr_col, pval_col, log_ors, log_or_p_values


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

def plot_signature_gene_association(signature, gene, phi_max_mean, disease_names, 
                                   gene_disease_correlations, output_dir,
                                   sig_gene_associations=None,
                                   gene_disease_p_values=None,
                                   log_ors=None,
                                   log_or_p_values=None,
                                   min_abs_correlation=MIN_ABS_CORRELATION,
                                   min_abs_phi=MIN_ABS_PHI,
                                   max_p_value=MAX_P_VALUE,
                                   phi_transform=PHI_TRANSFORM,
                                   use_log_or_for_x_axis=USE_LOG_OR_FOR_X_AXIS,
                                   use_psi=False):
    """Plot diseases for a signature-gene combo."""
    
    K, D = phi_max_mean.shape
    if signature >= K:
        print(f"  ⚠️  Signature {signature} out of range (max={K-1})")
        return None
    
    # Get signature-gene association info (from rare variant analysis)
    sig_gene_info = None
    if sig_gene_associations:
        key = (signature, gene.upper())
        sig_gene_info = sig_gene_associations.get(key)
    
    # Get signature-disease associations (max phi over time)
    sig_phi_max = phi_max_mean[signature, :]  # (D,)
    
    # Get gene-disease associations for this gene (correlation or log OR)
    gene_corrs = []
    gene_p_values = []
    matched_diseases = []
    matched_indices = []
    
    # Choose which metric to use based on flag and availability
    use_log_or = use_log_or_for_x_axis and log_ors and len(log_ors) > 0
    if use_log_or:
        print(f"  Using log OR on x-axis (matching psi/log OR scale)")
        source_dict = log_ors
        source_p_values = log_or_p_values if log_or_p_values else {}
    else:
        print(f"  Using correlation on x-axis")
        source_dict = gene_disease_correlations
        source_p_values = gene_disease_p_values if gene_disease_p_values else {}
    
    # First, collect all associations for this gene from CSV
    gene_associations_dict = {}  # disease_name_from_csv -> association value
    gene_p_values_dict = {}      # disease_name_from_csv -> p_value
    for (gene_csv, disease_csv), value in source_dict.items():
        if gene_csv.upper() == gene.upper():
            gene_associations_dict[disease_csv] = value
            if source_p_values and (gene_csv, disease_csv) in source_p_values:
                gene_p_values_dict[disease_csv] = source_p_values[(gene_csv, disease_csv)]
    
    if len(gene_associations_dict) == 0:
        metric_name = "log OR" if use_log_or else "correlation"
        print(f"  ⚠️  Gene {gene} not found in {metric_name} data")
        return None
    
    # Now try to match each CSV disease to our disease_names list
    for disease_csv, assoc_value in gene_associations_dict.items():
        # Try to find matching disease in our disease_names
        d_idx, matched_name = match_disease_names(disease_names, disease_csv)
        if d_idx is not None:
            gene_corrs.append(assoc_value)  # Actually correlation or log OR depending on flag
            matched_diseases.append(matched_name)
            matched_indices.append(d_idx)
            # Add p_value if available (None if not available)
            gene_p_values.append(gene_p_values_dict.get(disease_csv, None))
    
    if len(gene_corrs) == 0:
        print(f"  ⚠️  No matched diseases found for gene {gene}")
        return None
    
    # Get phi_max values (max phi over time) for matched diseases (before filtering)
    sig_phi_max_values_raw = [sig_phi_max[idx] for idx in matched_indices]
    
    # Filter uninteresting diseases
    original_n = len(gene_corrs)
    filtered_data = list(zip(gene_corrs, sig_phi_max_values_raw, gene_p_values, matched_diseases, matched_indices))
    
    # Filter by absolute correlation
    if min_abs_correlation is not None and min_abs_correlation > 0:
        filtered_data = [(c, p, pv, d, idx) for c, p, pv, d, idx in filtered_data if abs(c) >= min_abs_correlation]
    
    # Filter by p-value (significance) - only include diseases with significant gene-disease correlation
    if max_p_value is not None and max_p_value > 0:
        n_before_pfilter = len(filtered_data)
        filtered_data = [(c, p, pv, d, idx) for c, p, pv, d, idx in filtered_data 
                        if pv is not None and pv <= max_p_value]
        if len(filtered_data) < n_before_pfilter:
            print(f"  Filtered to {len(filtered_data)} diseases with p-value ≤ {max_p_value}")
    
    if len(filtered_data) == 0:
        print(f"  ⚠️  No diseases remaining after filtering (original: {original_n})")
        return None
    
    # Unpack filtered data
    gene_corrs, sig_phi_max_values, gene_p_values_filtered, matched_diseases, matched_indices = zip(*filtered_data)
    gene_corrs = list(gene_corrs)
    sig_phi_max_values = np.array(sig_phi_max_values)
    matched_diseases = list(matched_diseases)
    matched_indices = list(matched_indices)
    
    # Apply transformation
    # If using psi, it's already log OR - use no transform
    if use_psi:
        # Psi is already log OR, keep on original scale
        phi_label = f'Signature {signature} - Disease Association (log OR)'
    elif phi_transform == 'sigmoid':
        # Inverse logit transform (phi is logit, convert to probability)
        sig_phi_max_values = sigmoid(sig_phi_max_values)
        phi_label = f'Signature {signature} - Disease Loading (probability)'
    elif phi_transform == 'standardize':
        # Z-score transform
        sig_phi_max_values = (sig_phi_max_values - np.mean(sig_phi_max_values)) / (np.std(sig_phi_max_values) + 1e-10)
        phi_label = f'Signature {signature} - Disease Loading (φ, standardized)'
    elif phi_transform == 'center':
        # Center around mean
        sig_phi_max_values = sig_phi_max_values - np.mean(sig_phi_max_values)
        phi_label = f'Signature {signature} - Disease Loading (φ, centered)'
    elif phi_transform == 'exp':
        # Convert logit to OR (if phi represents log OR, though typically phi is logit)
        sig_phi_max_values = np.exp(sig_phi_max_values)
        phi_label = f'Signature {signature} - Disease Loading (OR)'
    else:
        # Default: use phi directly (logit)
        phi_label = f'Signature {signature} - Disease Loading (φ, max over time)'
    
    # Filter by phi AFTER transformation (if specified)
    if min_abs_phi is not None and min_abs_phi > 0:
        n_before_phifilter = len(sig_phi_max_values)
        mask = np.abs(sig_phi_max_values) >= min_abs_phi
        if np.any(mask):
            sig_phi_max_values = sig_phi_max_values[mask]
            gene_corrs = [gene_corrs[i] for i in range(len(gene_corrs)) if mask[i]]
            gene_p_values_filtered = list(gene_p_values_filtered)  # Convert to list if needed
            gene_p_values_filtered = [gene_p_values_filtered[i] for i in range(len(gene_p_values_filtered)) if mask[i]]
            matched_diseases = [matched_diseases[i] for i in range(len(matched_diseases)) if mask[i]]
            matched_indices = [matched_indices[i] for i in range(len(matched_indices)) if mask[i]]
            if len(sig_phi_max_values) < n_before_phifilter:
                print(f"  Filtered to {len(sig_phi_max_values)} diseases with |signature loading| ≥ {min_abs_phi}")
        else:
            print(f"  ⚠️  No diseases remaining after phi filtering (threshold: {min_abs_phi})")
            return None
    else:
        # Convert to list for consistency
        gene_p_values_filtered = list(gene_p_values_filtered)
    
    if original_n > len(gene_corrs):
        print(f"  Filtered from {original_n} to {len(gene_corrs)} diseases")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot - all points are significant (filtered by p ≤ 0.05)
    scatter = ax.scatter(gene_corrs, sig_phi_max_values, alpha=0.6, s=50, 
                        edgecolors='black', linewidth=0.5, color='#3498DB')
    
    # Calculate correlation and p-value (no trend line)
    r = np.nan
    p_val = np.nan
    if len(gene_corrs) > 2:
        # Calculate correlation and p-value
        r, p_val = stats.pearsonr(gene_corrs, sig_phi_max_values)
        
        # Add text annotation (no trend line)
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.2e}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels
    if use_log_or:
        x_label = f'{gene} Rare Variant Burden - Disease Association (log OR)'
    else:
        x_label = f'{gene} Rare Variant Burden - Disease Correlation'
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
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
    
    # Identify and report outliers: high correlation but low signature loading
    if len(gene_corrs) > 0:
        # Find the point with highest correlation (potential far-right outlier)
        corr_array = np.array(gene_corrs)
        phi_array = np.array(sig_phi_max_values)
        
        # Find highest correlation
        max_corr_idx = np.argmax(np.abs(corr_array))
        
        # Check if this point also has very low association
        # Use different thresholds for psi (log OR) vs phi (logit)
        threshold = 0.1 if use_psi else 0.005  # Log OR near 0 = no association; phi logit near 0 = no association
        label_name = "association (log OR)" if use_psi else "loading (max phi)"
        
        if abs(phi_array[max_corr_idx]) < threshold:
            outlier_disease = matched_diseases[max_corr_idx]
            outlier_corr = gene_corrs[max_corr_idx]
            outlier_phi = sig_phi_max_values[max_corr_idx]
            outlier_pval = gene_p_values_filtered[max_corr_idx]
            print(f"  📍 Outlier detected: {outlier_disease}")
            pval_str = f"p={outlier_pval:.4e}" if outlier_pval is not None else "p=N/A"
            print(f"     - {gene} correlation: {outlier_corr:.4f} ({pval_str})")
            print(f"     - Signature {signature} {label_name}: {outlier_phi:.4f}")
            print(f"     - This disease has high {gene} rare variant burden but is NOT associated with Signature {signature}")
        
        # Also find all outliers: high correlation but low phi
        correlation_percentile_90 = np.percentile([abs(c) for c in gene_corrs], 90)
        phi_percentile_10 = np.percentile([abs(p) for p in sig_phi_max_values], 10)
        
        outliers = []
        for i, (corr, phi_val, disease, pval) in enumerate(zip(gene_corrs, sig_phi_max_values, matched_diseases, gene_p_values_filtered)):
            if abs(corr) >= correlation_percentile_90 and abs(phi_val) <= phi_percentile_10:
                outliers.append({
                    'disease': disease,
                    'correlation': corr,
                    'phi_loading': phi_val,
                    'p_value': pval
                })
        
        if len(outliers) > 1:  # Already printed the main one above
            print(f"  ⚠️  Found {len(outliers)} total outlier(s): High {gene} correlation but low Signature {signature} loading")
    
    # Label top points
    if len(gene_corrs) > 0:
        # Find top points by absolute correlation or phi value
        top_n = min(5, len(gene_corrs))
        # Score = abs(correlation) * abs(phi)
        scores = [abs(corr) * abs(phi_val) for corr, phi_val in zip(gene_corrs, sig_phi_max_values)]
        top_indices = np.argsort(scores)[-top_n:]
        
        for idx in top_indices:
            ax.annotate(matched_diseases[idx], 
                       (gene_corrs[idx], sig_phi_max_values[idx]),
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

def run_plots(use_psi, use_log_or_x, output_dir_suffix=""):
    """Run plotting with specific settings."""
    import sys
    global USE_PSI_INSTEAD_OF_PHI, USE_LOG_OR_FOR_X_AXIS
    
    # Set flags for this run
    USE_PSI_INSTEAD_OF_PHI = use_psi
    USE_LOG_OR_FOR_X_AXIS = use_log_or_x
    
    # Use different output directory if suffix provided
    if output_dir_suffix:
        output_dir = OUTPUT_DIR.parent / f'{OUTPUT_DIR.name}{output_dir_suffix}'
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = OUTPUT_DIR
    
    # Get CSV path from command line or try default locations
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try default filename in current directory and parent directories
        possible_paths = [
            Path(GENE_DISEASE_CSV),  # Try absolute path first
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
    matrix_mean, disease_names = load_phi_max_and_disease_names()
    gene_disease_correlations, gene_disease_p_values, gene_col, disease_col, corr_col, pval_col, log_ors, log_or_p_values = load_gene_disease_correlations(csv_path)
    sig_gene_associations = load_rare_variant_signature_gene_associations(RARE_VARIANT_RESULTS_DIR)
    
    print(f"\n{'='*80}")
    print("PLOTTING SIGNATURE-GENE-DISEASE ASSOCIATIONS")
    if use_psi:
        print("Using PSI (log OR, pooled from censor_e_batchrun_vectorized)")
    else:
        print("Using MAX PHI (peak association over time, pooled across batches)")
    if use_log_or_x:
        print("Using log OR on x-axis (matching psi/log OR scale)")
    else:
        print("Using correlation on x-axis")
    if MAX_P_VALUE:
        print(f"Filtering to p-value ≤ {MAX_P_VALUE}")
    if output_dir_suffix:
        print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    results = []
    for sig, gene in SIGNATURE_GENE_COMBOS:
        print(f"\nPlotting Signature {sig} - {gene}...")
        result = plot_signature_gene_association(
            sig, gene, matrix_mean, disease_names, 
            gene_disease_correlations, output_dir,
            sig_gene_associations=sig_gene_associations,
            gene_disease_p_values=gene_disease_p_values,
            log_ors=log_ors,
            log_or_p_values=log_or_p_values,
            min_abs_correlation=MIN_ABS_CORRELATION,
            min_abs_phi=MIN_ABS_PHI,
            max_p_value=MAX_P_VALUE,
            phi_transform=PHI_TRANSFORM,
            use_log_or_for_x_axis=use_log_or_x,
            use_psi=use_psi
        )
        if result:
            results.append(result)
    
    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ Completed! Plots saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return output_dir

def main():
    """Main function."""
    if GENERATE_BOTH_PLOTS:
        print("="*80)
        print("GENERATING BOTH PLOTS")
        print("="*80)
        
        # Plot 1: Max phi (sigmoid) vs correlation
        print("\n" + "="*80)
        print("PLOT 1: Max Phi (sigmoid) vs Correlation")
        print("="*80)
        run_plots(use_psi=False, use_log_or_x=False, output_dir_suffix="_maxphi_vs_corr")
        
        # Plot 2: Psi vs log OR
        print("\n" + "="*80)
        print("PLOT 2: Psi vs Log OR")
        print("="*80)
        run_plots(use_psi=True, use_log_or_x=True, output_dir_suffix="_psi_vs_logor")
        
        print("\n" + "="*80)
        print("✓ BOTH PLOTS COMPLETE!")
        print("="*80)
        print(f"  Plot 1 (Max Phi vs Correlation): {OUTPUT_DIR.parent / (OUTPUT_DIR.name + '_maxphi_vs_corr')}")
        print(f"  Plot 2 (Psi vs Log OR): {OUTPUT_DIR.parent / (OUTPUT_DIR.name + '_psi_vs_logor')}")
    else:
        # Single plot with configured settings
        run_plots(use_psi=USE_PSI_INSTEAD_OF_PHI, use_log_or_x=USE_LOG_OR_FOR_X_AXIS)


if __name__ == '__main__':
    main()

