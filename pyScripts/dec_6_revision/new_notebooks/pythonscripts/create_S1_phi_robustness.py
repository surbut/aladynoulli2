#!/usr/bin/env python3
"""
Create S1: Robustness of φ estimation across subsets

This script:
- Loads phi from all 40 batches in censor_e_batchrun_vectorized
- For selected diseases, plots:
  - Phi for the top signature (as assigned by cluster) across all 40 batches
  - Mean phi from master checkpoint for other signatures
- Shows standard errors across batches

Diseases to plot:
- Disease 112: Unstable angina (Signature 5)
- Disease 47: Type 2 diabetes (Signature 15)
- Disease 110 or 11: (user thinks 19, but we'll check)

Usage:
    python create_S1_phi_robustness.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import glob
from scipy.special import expit as sigmoid

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'

# Configuration
BATCH_DIR = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized")
MASTER_CHECKPOINT = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt")
CLUSTERS_FILE = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt")
DISEASE_NAMES_FILE = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv")
OUTPUT_DIR = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Diseases to plot: (disease_idx, expected_signature, disease_name)
# Note: Using 0-based indexing (Python)
# User wants: MI, Breast cancer, Type 2 diabetes, Atrial fibrillation, and Depression
# Signatures will be looked up from clusters file
DISEASES_TO_PLOT = [
    (112, None, "Myocardial infarction"),  # MI - confirmed index 112 (0-based, 113 in 1-based)
    (16, None, "Breast cancer [female]"),  # Will check signature from clusters
    (47, None, "Type 2 diabetes"),  # Will check signature from clusters
    (127, None, "Atrial fibrillation"),  # Will check signature from clusters
    (66, None, "Depression"),  # Will check signature from clusters (Major depressive disorder)
]

AGE_OFFSET = 30
N_BATCHES = 40
BATCH_SIZE = 10000


def extract_batch_idx(filename):
    """Extract batch start index from filename."""
    basename = Path(filename).name
    # Pattern: model_enroll_fixedphi_sex_0_10000.pt or similar
    parts = basename.replace('.pt', '').split('_')
    numeric_parts = [p for p in parts if p.isdigit()]
    if len(numeric_parts) >= 2:
        return int(numeric_parts[0])  # First number is start index
    return None


def load_disease_names(disease_names_file):
    """Load disease names from CSV file."""
    if disease_names_file.exists():
        disease_names_df = pd.read_csv(disease_names_file)
        # Disease names are in column 1 (the "x" column), not column 0
        disease_names = disease_names_df.iloc[:, 1].tolist()
        # Remove header value "x" if it's the first element
        if len(disease_names) > 0 and str(disease_names[0]).lower() == 'x':
            disease_names = disease_names[1:]
        # Convert all disease names to strings
        disease_names = [str(name) if pd.notna(name) else f"Disease_{i}" for i, name in enumerate(disease_names)]
        print(f"✓ Loaded {len(disease_names)} disease names from CSV")
        return disease_names
    else:
        print(f"⚠️  Disease names file not found: {disease_names_file}")
        return None


def load_phi_from_batches(batch_dir, n_batches=40, clusters_file=None):
    """
    Load phi from all batch checkpoints.
    
    Returns:
        all_phis: list of phi arrays, shape (K, D, T) each
        clusters: disease-to-signature assignments, shape (D,)
    """
    print(f"\n{'='*80}")
    print("LOADING PHI FROM BATCHES")
    print(f"{'='*80}")
    print(f"Batch directory: {batch_dir}")
    
    # Find all batch files - only model checkpoint files, not pi prediction files
    pattern = str(batch_dir / "enrollment_model_W0.0001_batch_*_*.pt")
    batch_files = glob.glob(pattern)
    
    if len(batch_files) == 0:
        # Try alternative pattern for model files
        pattern = str(batch_dir / "model_enroll_fixedphi_sex_*_*.pt")
        batch_files = glob.glob(pattern)
    
    if len(batch_files) == 0:
        raise ValueError(f"No model checkpoint files found in {batch_dir}")
    
    # Filter out pi_fullmode files (these are pi predictions, not model checkpoints)
    batch_files = [f for f in batch_files if 'pi_fullmode' not in Path(f).name]
    
    # Sort by batch index
    batch_files = sorted(batch_files, key=extract_batch_idx)
    batch_files = batch_files[:n_batches]  # Take first n_batches
    
    print(f"Found {len(batch_files)} batch files")
    
    all_phis = []
    clusters = None
    
    for i, batch_file in enumerate(batch_files):
        batch_idx = extract_batch_idx(batch_file)
        print(f"  Loading batch {i+1}/{len(batch_files)}: {Path(batch_file).name}")
        
        try:
            checkpoint = torch.load(batch_file, map_location='cpu', weights_only=False)
            
            # Extract phi
            phi = None
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                print(f"    ⚠️  Warning: No phi found in batch {i+1}")
                continue
            
            # Convert to numpy
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            all_phis.append(phi)
            
            print(f"    ✓ Phi shape: {phi.shape}")
            
        except Exception as e:
            print(f"    ✗ Error loading batch {i+1}: {e}")
            continue
    
    if len(all_phis) == 0:
        raise ValueError("No phi values loaded from any batch!")
    
    # Verify all phis have same shape
    shapes = [phi.shape for phi in all_phis]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"Phi shapes differ! Shapes: {set(shapes)}")
    
    print(f"\n✓ Loaded {len(all_phis)} batches")
    print(f"  Phi shape: {all_phis[0].shape}")
    
    # Load clusters from separate file
    if clusters_file is not None:
        print(f"\nLoading clusters from: {clusters_file}")
        if clusters_file.exists():
            clusters_data = torch.load(clusters_file, map_location='cpu', weights_only=False)
            
            # Handle different formats: dict with 'clusters' key, or direct array/tensor
            if isinstance(clusters_data, dict):
                if 'clusters' in clusters_data:
                    clusters = clusters_data['clusters']
                    if torch.is_tensor(clusters):
                        clusters = clusters.detach().cpu().numpy()
                    print(f"  ✓ Loaded clusters from dict, shape: {clusters.shape}")
                else:
                    print(f"  ⚠️  Warning: No 'clusters' key in {clusters_file}")
                    print(f"  Available keys: {list(clusters_data.keys())}")
                    clusters = None
            elif isinstance(clusters_data, (torch.Tensor, np.ndarray)):
                # File contains array directly
                clusters = clusters_data
                if torch.is_tensor(clusters):
                    clusters = clusters.detach().cpu().numpy()
                print(f"  ✓ Loaded clusters (direct array), shape: {clusters.shape}")
            else:
                print(f"  ⚠️  Warning: Unexpected clusters data type: {type(clusters_data)}")
                clusters = None
        else:
            print(f"  ⚠️  Warning: Clusters file not found: {clusters_file}")
            clusters = None
    else:
        print(f"\n⚠️  Warning: No clusters file specified, will try to extract from checkpoints")
        clusters = None
    
    return all_phis, clusters


def load_master_phi(master_checkpoint):
    """Load pooled phi from master checkpoint."""
    print(f"\n{'='*80}")
    print("LOADING MASTER CHECKPOINT")
    print(f"{'='*80}")
    print(f"Master checkpoint: {master_checkpoint}")
    
    if not master_checkpoint.exists():
        raise FileNotFoundError(f"Master checkpoint not found: {master_checkpoint}")
    
    checkpoint = torch.load(master_checkpoint, map_location='cpu', weights_only=False)
    
    # Extract phi
    phi = None
    if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
        phi = checkpoint['model_state_dict']['phi']
    elif 'phi' in checkpoint:
        phi = checkpoint['phi']
    else:
        raise ValueError(f"No phi found in master checkpoint")
    
    # Convert to numpy
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    
    print(f"✓ Loaded master phi, shape: {phi.shape}")
    
    return phi


def create_s1_figure(all_phis, master_phi, clusters, disease_names, diseases_to_plot):
    """
    Create S1 figure matching the structure:
    - Panel A (left): Phi curves from all batches for selected diseases
    - Panel B (right): Distribution of standard errors across ALL phi parameters
    """
    n_batches = len(all_phis)
    K, D, T = all_phis[0].shape
    
    # Stack all phis: (n_batches, K, D, T)
    phi_stack = np.stack(all_phis, axis=0)
    
    # Create figure with two main panels
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # ===== PANEL A: Phi curves from all batches for selected diseases =====
    ax_left = fig.add_subplot(gs[:, 0])
    
    # Select a few diseases to plot (use the diseases_to_plot list)
    n_diseases_to_show = min(5, len(diseases_to_plot))
    selected_diseases = diseases_to_plot[:n_diseases_to_show]
    
    # Create subplots for each disease
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_left = GridSpecFromSubplotSpec(n_diseases_to_show, 1, subplot_spec=gs[:, 0], hspace=0.4)
    axes_left = [fig.add_subplot(gs_left[i, 0]) for i in range(n_diseases_to_show)]
    
    for idx, (disease_idx, assigned_sig, disease_name) in enumerate(selected_diseases):
        ax = axes_left[idx]
        
        # Extract phi curves for this disease-signature pair across all batches
        # Shape: (n_batches, T)
        phi_curves = phi_stack[:, assigned_sig, disease_idx, :]
        
        # Compute mean and SE across batches
        mean_phi = phi_curves.mean(axis=0)
        se_phi = phi_curves.std(axis=0) / np.sqrt(n_batches)
        
        # Get master phi for this disease-signature pair
        master_phi_curve = master_phi[assigned_sig, disease_idx, :]  # Shape: (T,)
        
        # Plot all batch curves in gray
        for b in range(n_batches):
            ax.plot(phi_curves[b], color='gray', alpha=0.2, linewidth=0.7, zorder=1)
        
        # Plot mean curve
        ax.plot(mean_phi, color='blue', linewidth=2, label='Mean', zorder=3)
        
        # Plot master checkpoint curve
        time_points = np.arange(T)
        ax.plot(time_points, master_phi_curve, color='red', linewidth=2, 
               linestyle='--', label='Master', zorder=4)
        
        # Plot SE band
        ax.fill_between(time_points, mean_phi - se_phi, mean_phi + se_phi, 
                       alpha=0.3, color='lightblue', label='SE', zorder=2)
        
        # Formatting
        ax.set_ylabel('Phi', fontsize=10)
        ax.set_title(f'Signature {assigned_sig}, Disease {disease_idx}: {disease_name}', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        if idx == n_diseases_to_show - 1:
            ax.set_xlabel('Time (yr)', fontsize=10)
        else:
            ax.set_xticklabels([])
    
    # ===== PANEL B: Distribution of standard errors across ALL phi parameters =====
    
    # Compute standard errors for ALL phi parameters
    # For each (k, d, t), compute SE across batches
    # Shape: (K, D, T)
    se_all = np.std(phi_stack, axis=0) / np.sqrt(n_batches)
    
    # Flatten to get all SE values
    se_values = se_all.flatten()
    
    # Remove any NaN or inf values
    se_values = se_values[np.isfinite(se_values)]
    
    mean_se = np.mean(se_values)
    median_se = np.median(se_values)
    p95_se = np.percentile(se_values, 95)
    
    # Top subplot: Histogram (linear scale)
    ax_se1 = fig.add_subplot(gs[0, 1])
    counts, bins, patches = ax_se1.hist(se_values, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax_se1.set_xlabel('Standard error', fontsize=10)
    ax_se1.set_ylabel('Frequency', fontsize=10)
    ax_se1.set_title('Distribution of Standard Errors (Linear Scale)', fontsize=11, fontweight='bold')
    ax_se1.grid(True, alpha=0.3)
    ax_se1.axvline(mean_se, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_se:.4f}')
    ax_se1.axvline(median_se, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_se:.4f}')
    ax_se1.legend(loc='best', fontsize=9)
    
    # Middle subplot: Histogram (log scale)
    ax_se2 = fig.add_subplot(gs[1, 1])
    counts, bins, patches = ax_se2.hist(se_values, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax_se2.set_yscale('log')
    ax_se2.set_xlabel('Standard error', fontsize=10)
    ax_se2.set_ylabel('Frequency (log scale)', fontsize=10)
    ax_se2.set_title('Distribution of Standard Errors (Log Scale)', fontsize=11, fontweight='bold')
    ax_se2.grid(True, alpha=0.3, which='both')
    
    # Bottom subplot: Cumulative distribution
    # We need to add this - let's adjust the grid
    fig.delaxes(ax_se2)  # Remove middle subplot, we'll use the bottom space
    
    # Create new grid for right panel with 3 subplots
    gs_right = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 1], hspace=0.4)
    ax_se1 = fig.add_subplot(gs_right[0, 0])
    ax_se2 = fig.add_subplot(gs_right[1, 0])
    ax_se3 = fig.add_subplot(gs_right[2, 0])
    
    # Replot top (linear)
    counts, bins, patches = ax_se1.hist(se_values, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax_se1.set_xlabel('Standard error', fontsize=10)
    ax_se1.set_ylabel('Frequency', fontsize=10)
    ax_se1.set_title('Distribution of Standard Errors (Linear Scale)', fontsize=11, fontweight='bold')
    ax_se1.grid(True, alpha=0.3)
    ax_se1.axvline(mean_se, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_se:.4f}')
    ax_se1.axvline(median_se, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_se:.4f}')
    ax_se1.legend(loc='best', fontsize=9)
    
    # Middle (log scale)
    counts, bins, patches = ax_se2.hist(se_values, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax_se2.set_yscale('log')
    ax_se2.set_xlabel('Standard error', fontsize=10)
    ax_se2.set_ylabel('Frequency (log scale)', fontsize=10)
    ax_se2.set_title('Distribution of Standard Errors (Log Scale)', fontsize=11, fontweight='bold')
    ax_se2.grid(True, alpha=0.3, which='both')
    
    # Bottom (cumulative)
    sorted_se = np.sort(se_values)
    cumulative = np.arange(1, len(sorted_se) + 1) / len(sorted_se)
    ax_se3.plot(sorted_se, cumulative, linewidth=2, color='green', label='Cumulative distribution')
    ax_se3.axvline(p95_se, color='red', linestyle='--', linewidth=2, label=f'95th percentile: {p95_se:.4f}')
    ax_se3.set_xlabel('Standard error', fontsize=10)
    ax_se3.set_ylabel('Cumulative probability', fontsize=10)
    ax_se3.set_title('Cumulative Distribution of Standard Errors', fontsize=11, fontweight='bold')
    ax_se3.set_ylim(0, 1)
    ax_se3.grid(True, alpha=0.3)
    ax_se3.legend(loc='best', fontsize=9)
    
    plt.suptitle('S1: Robustness of φ Estimation Across Subsets', fontsize=14, fontweight='bold', y=0.98)
    
    return fig, se_values


def plot_phi_robustness_distribution(all_phis, master_phi, clusters, disease_names,
                                     disease_idx, assigned_sig, disease_name):
    """
    Plot phi robustness as distributions (like S1.pdf).
    
    Shows:
    - Panel A: Distribution of phi values across batches (log scale)
    - Panel B: Cumulative distribution of phi values
    """
    K, D, T = all_phis[0].shape
    
    # Stack all phis: (n_batches, K, D, T)
    phi_stack = np.stack(all_phis, axis=0)
    
    # Get phi for this disease in assigned signature across all batches and timepoints
    # Shape: (n_batches, T)
    phi_assigned = phi_stack[:, assigned_sig, disease_idx, :]
    
    # Flatten across timepoints to get all phi values across batches
    # Shape: (n_batches * T,)
    phi_values = phi_assigned.flatten()
    
    # Get pooled mean from master
    phi_master_assigned = master_phi[assigned_sig, disease_idx, :]  # (T,)
    phi_master_mean = np.mean(phi_master_assigned)
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Distribution (log scale)
    ax1 = axes[0]
    
    # Create histogram of phi values
    # Use log scale for y-axis as in S1.pdf
    counts, bins, patches = ax1.hist(phi_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_xlabel('φ (log hazard ratio)', fontsize=11)
    ax1.set_ylabel('Frequency (log scale)', fontsize=11)
    ax1.set_title(f'{disease_name}\nSignature {assigned_sig} - Distribution', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add vertical line for pooled mean
    ax1.axvline(phi_master_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Pooled mean: {phi_master_mean:.3f}')
    ax1.legend(loc='best', fontsize=9)
    
    # Panel B: Cumulative distribution
    ax2 = axes[1]
    
    # Sort phi values for cumulative distribution
    sorted_phi = np.sort(phi_values)
    cumulative = np.arange(1, len(sorted_phi) + 1) / len(sorted_phi)
    
    ax2.plot(sorted_phi, cumulative, linewidth=2, color='steelblue', label='Cumulative distribution')
    ax2.axvline(phi_master_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Pooled mean: {phi_master_mean:.3f}')
    ax2.set_xlabel('φ (log hazard ratio)', fontsize=11)
    ax2.set_ylabel('Cumulative probability', fontsize=11)
    ax2.set_title(f'Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def main():
    print("="*80)
    print("CREATING S1: ROBUSTNESS OF φ ESTIMATION ACROSS SUBSETS")
    print("="*80)
    
    # Load disease names
    print(f"\n{'='*80}")
    print("LOADING DISEASE NAMES")
    print(f"{'='*80}")
    disease_names = load_disease_names(DISEASE_NAMES_FILE)
    
    # Load phi from batches
    all_phis, clusters = load_phi_from_batches(BATCH_DIR, n_batches=N_BATCHES, clusters_file=CLUSTERS_FILE)
    
    # Load master phi
    master_phi = load_master_phi(MASTER_CHECKPOINT)
    
    # Verify shapes match
    if all_phis[0].shape != master_phi.shape:
        print(f"\n⚠️  Warning: Batch phi shape {all_phis[0].shape} != master phi shape {master_phi.shape}")
        print("  Using master shape for consistency")
        K, D, T = master_phi.shape
    else:
        K, D, T = all_phis[0].shape
    
    # Determine actual signature assignments for diseases
    print(f"\n{'='*80}")
    print("DETERMINING SIGNATURE ASSIGNMENTS")
    print(f"{'='*80}")
    
    diseases_to_plot = []
    for disease_idx, expected_sig, disease_name in DISEASES_TO_PLOT:
        if disease_idx >= D:
            print(f"  ⚠️  Disease index {disease_idx} out of range (max: {D-1})")
            continue
        
        # Get actual signature assignment from clusters
        if clusters is not None:
            assigned_sig = int(clusters[disease_idx])
        else:
            # Fallback: use expected signature or find max phi signature
            if expected_sig is not None:
                assigned_sig = expected_sig
            else:
                # Find signature with highest mean phi for this disease
                phi_master_disease = master_phi[:, disease_idx, :]
                mean_phi_per_sig = np.mean(phi_master_disease, axis=1)
                assigned_sig = int(np.argmax(mean_phi_per_sig))
        
        # Get disease name from loaded list
        if disease_names and disease_idx < len(disease_names):
            actual_name = disease_names[disease_idx]
        else:
            actual_name = disease_name
        
        diseases_to_plot.append((disease_idx, assigned_sig, actual_name))
        print(f"  Disease {disease_idx}: {actual_name} → Signature {assigned_sig}")
    
    # Create S1 figure
    print(f"\n{'='*80}")
    print("CREATING S1 FIGURE")
    print(f"{'='*80}")
    
    print(f"\nCreating S1 figure...")
    fig, se_values = create_s1_figure(all_phis, master_phi, clusters, disease_names, diseases_to_plot)
    
    # Save figure
    output_file = OUTPUT_DIR / "S1.pdf"
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved S1 figure to: {output_file}")
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Overall SE statistics
    mean_se_all = np.mean(se_values)
    median_se_all = np.median(se_values)
    p95_se_all = np.percentile(se_values, 95)
    p99_se_all = np.percentile(se_values, 99)
    
    print(f"\nOverall Standard Error Statistics (across all {len(se_values)} phi parameters):")
    print(f"  Mean SE: {mean_se_all:.6f}")
    print(f"  Median SE: {median_se_all:.6f}")
    print(f"  95th percentile: {p95_se_all:.6f}")
    print(f"  99th percentile: {p99_se_all:.6f}")
    
    # Per-disease statistics
    phi_stack = np.stack(all_phis, axis=0)  # (n_batches, K, D, T)
    
    print(f"\nPer-Disease Statistics:")
    for disease_idx, assigned_sig, disease_name in diseases_to_plot:
        phi_disease_sig = phi_stack[:, assigned_sig, disease_idx, :]  # (n_batches, T)
        
        # Compute SE across batches at each timepoint
        se_per_timepoint = np.std(phi_disease_sig, axis=0) / np.sqrt(len(all_phis))
        mean_se = np.mean(se_per_timepoint)
        max_se = np.max(se_per_timepoint)
        median_se = np.median(se_per_timepoint)
        
        print(f"\n  {disease_name} (Disease {disease_idx}, Signature {assigned_sig}):")
        print(f"    Mean SE across timepoints: {mean_se:.6f}")
        print(f"    Median SE: {median_se:.6f}")
        print(f"    Max SE: {max_se:.6f}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Main figure: {OUTPUT_DIR / 'S1.pdf'}")


if __name__ == '__main__':
    main()

