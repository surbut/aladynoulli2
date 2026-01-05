#!/usr/bin/env python3
"""
Verify exported parameters by recreating plots in the same style as plot_ukb_sigs.py

This script loads the exported parameters and creates signature plots to verify
they match the original plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import expit as sigmoid
import pandas as pd
import argparse

# Set style (matching plot_ukb_sigs.py)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def load_exported_parameters(export_dir):
    """Load exported parameters from the export directory."""
    export_dir = Path(export_dir)
    
    # Load phi
    phi_path = export_dir / 'phi_master_pooled.npy'
    phi = np.load(phi_path)
    print(f"✓ Loaded phi: {phi.shape}")
    
    # Load psi
    psi_path = export_dir / 'psi_master.npy'
    psi = np.load(psi_path)
    print(f"✓ Loaded psi: {psi.shape}")
    
    # Load disease names
    disease_names_path = export_dir / 'disease_names.csv'
    disease_names = None
    if disease_names_path.exists():
        disease_df = pd.read_csv(disease_names_path)
        if 'Disease_Name' in disease_df.columns:
            disease_names = disease_df['Disease_Name'].tolist()
        elif 'phenotype' in disease_df.columns:
            disease_names = disease_df['phenotype'].tolist()
        print(f"✓ Loaded {len(disease_names)} disease names")
    
    return phi, psi, disease_names


def plot_signatures_verification(phi, clusters, disease_names, output_path, cohort_name="UKB"):
    """
    Create signature plots in the same style as plot_ukb_sigs.py
    """
    K, D, T = phi.shape
    ages = np.arange(30, 30 + T)
    
    print(f"\nPlotting {K} signatures for {D} diseases...")
    
    # Group diseases by signature
    sig_to_diseases = {}
    for d in range(D):
        sig = int(clusters[d])
        if sig not in sig_to_diseases:
            sig_to_diseases[sig] = []
        sig_to_diseases[sig].append(d)
    
    # Create figure with subplots for each signature
    # Arrange in a grid: 5 columns, 5 rows (for 21 signatures)
    n_cols = 5
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()
    
    for sig in range(K):
        ax = axes[sig]
        diseases_in_sig = sig_to_diseases.get(sig, [])
        
        if len(diseases_in_sig) == 0:
            ax.text(0.5, 0.5, f'Signature {sig}\n(No diseases)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Signature {sig}', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Find top 5 diseases by maximum probability
        disease_max_probs = []
        for d in diseases_in_sig:
            prob_mean = sigmoid(phi[sig, d, :])
            max_prob = prob_mean.max()
            max_prob_age_idx = np.argmax(prob_mean)
            disease_name = disease_names[d] if disease_names and d < len(disease_names) else f'Disease {d}'
            disease_max_probs.append((d, max_prob, max_prob_age_idx, disease_name, prob_mean))
        
        # Sort by max probability and get top 5
        disease_max_probs.sort(key=lambda x: x[1], reverse=True)
        top_5_diseases = disease_max_probs[:5]
        other_diseases = disease_max_probs[5:]
        
        # Colors for top diseases
        top_colors = sns.color_palette("tab10", 5)
        
        # Plot top 5 diseases with labels
        for idx, (d, max_prob, max_age_idx, disease_name, prob_mean) in enumerate(top_5_diseases):
            color = top_colors[idx]
            ax.plot(ages, prob_mean, linewidth=2, alpha=0.8, color=color, 
                   label=f'{disease_name[:35]}')
            
            # Add text label at peak
            max_age = ages[max_age_idx]
            ax.text(max_age, max_prob, f'{idx+1}', 
                   fontsize=8, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8),
                   ha='center', va='bottom')
        
        # Plot other diseases in gray
        for d, max_prob, max_age_idx, disease_name, prob_mean in other_diseases:
            ax.plot(ages, prob_mean, linewidth=0.8, alpha=0.3, color='lightgray')
        
        ax.set_title(f'Signature {sig} (n={len(diseases_in_sig)} diseases)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Age (yr)', fontsize=9)
        ax.set_ylabel('Prob (disease | sig k, age)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([30, 81])
        ax.set_ylim([0, None])
        
        # Add legend for top 5 diseases
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=1)
    
    # Remove extra subplots
    for sig in range(K, len(axes)):
        axes[sig].axis('off')
    
    plt.suptitle(f'{cohort_name}: Disease Probabilities by Signature (from exported parameters)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved verification plot to: {output_path}")
    plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SIGNATURE SUMMARY")
    print(f"{'='*60}")
    for sig in sorted(sig_to_diseases.keys()):
        diseases = sig_to_diseases[sig]
        print(f"Signature {sig}: {len(diseases)} diseases")
        if len(diseases) > 0:
            # Show top 5 diseases by max probability
            max_probs = []
            for d in diseases:
                prob_mean = sigmoid(phi[sig, d, :])
                max_probs.append((d, prob_mean.max()))
            max_probs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Top diseases:")
            for d, max_prob in max_probs[:5]:
                name = disease_names[d] if disease_names and d < len(disease_names) else f'Disease {d}'
                print(f"    {name[:50]}: max prob = {max_prob:.6f}")


def load_clusters(clusters_path, cohort=None):
    """Load clusters from various file formats."""
    clusters_path = Path(clusters_path)
    
    # Auto-detect cohort-specific checkpoint paths if not provided
    if not clusters_path.exists() and cohort:
        if cohort.upper() == 'AOU':
            aou_checkpoint = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt')
            if aou_checkpoint.exists():
                clusters_path = aou_checkpoint
                print(f"  Auto-detected AOU checkpoint: {clusters_path.name}")
        elif cohort.upper() == 'MGB':
            mgb_checkpoint = Path('/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt')
            if mgb_checkpoint.exists():
                clusters_path = mgb_checkpoint
                print(f"  Auto-detected MGB checkpoint: {clusters_path.name}")
    
    if clusters_path.suffix == '.pt':
        # PyTorch file - could be a checkpoint or just clusters
        data = torch.load(clusters_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            # Checkpoint file - try common keys
            if 'clusters' in data:
                clusters = data['clusters']
            elif 'initial_clusters' in data:
                clusters = data['initial_clusters']
            else:
                raise ValueError(f"Could not find 'clusters' or 'initial_clusters' in {clusters_path}")
        else:
            clusters = data
        
        if torch.is_tensor(clusters):
            clusters = clusters.numpy()
        return clusters
    elif clusters_path.suffix == '.csv':
        # CSV file
        df = pd.read_csv(clusters_path)
        # Try common column names
        if 'clusters' in df.columns:
            clusters = df['clusters'].values
        elif 'cluster' in df.columns:
            clusters = df['cluster'].values
        else:
            # Assume first column
            clusters = df.iloc[:, 0].values
        return clusters
    else:
        raise ValueError(f"Unsupported clusters file format: {clusters_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description='Verify exported parameters by recreating plots')
    parser.add_argument('--export_dir', type=str, required=True,
                       help='Directory containing exported parameters')
    parser.add_argument('--clusters', type=str, required=True,
                       help='Path to clusters file (e.g., initial_clusters_400k.pt or checkpoint.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for verification plot (default: export_dir/verification_signatures.pdf)')
    parser.add_argument('--cohort', type=str, default='UKB',
                       help='Cohort name for plot title')
    
    args = parser.parse_args()
    
    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        raise ValueError(f"Export directory not found: {export_dir}")
    
    # Load exported parameters
    print(f"\n{'='*60}")
    print(f"LOADING EXPORTED PARAMETERS FROM: {export_dir}")
    print(f"{'='*60}")
    phi, psi, disease_names = load_exported_parameters(export_dir)
    
    # Detect cohort from export directory if not specified
    detected_cohort = args.cohort
    if not detected_cohort:
        export_str = str(export_dir).lower()
        if 'aou' in export_str:
            detected_cohort = 'AOU'
        elif 'mgb' in export_str:
            detected_cohort = 'MGB'
        elif 'ukb' in export_str:
            detected_cohort = 'UKB'
    
    # Load clusters
    print(f"\nLoading clusters from: {args.clusters}")
    clusters = load_clusters(args.clusters, cohort=detected_cohort)
    print(f"✓ Loaded clusters: {clusters.shape}")
    
    # Verify dimensions match
    K, D, T = phi.shape
    if len(clusters) != D:
        raise ValueError(f"Dimension mismatch: phi has {D} diseases but clusters has {len(clusters)}")
    
    # Create output path
    if args.output is None:
        output_path = export_dir / 'verification_signatures.pdf'
    else:
        output_path = Path(args.output)
    
    # Create verification plots
    print(f"\n{'='*60}")
    print("CREATING VERIFICATION PLOTS")
    print(f"{'='*60}")
    plot_signatures_verification(phi, clusters, disease_names, output_path, args.cohort)
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Verification plot saved to: {output_path}")
    print(f"\nCompare this plot with the original plot_ukb_sigs.py output to verify they match!")


if __name__ == '__main__':
    main()

