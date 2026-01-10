#!/usr/bin/env python3
"""
Compare gamma values from three sources:
1. censor_e_batchrun_vectorized_nolr (no lambda_reg, unshrunken)
2. censor_e_batchrun_vectorized (with lambda_reg, shrunken)
3. resultshighamp (old March version, likely unshrunken)

Generates PRS-signature plots for each source and saves them to separate directories.
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

from pathlib import Path
import argparse

# Import functions from generate_prs_signature_plots
from generate_prs_signature_plots import (
    load_prs_names,
    load_gamma_from_batches,
    load_gamma_from_old_structure,
    create_gamma_dataframe,
    plot_top_associations_bar,
    plot_significant_heatmap,
    plot_full_heatmap
)

def compare_gamma_sources(
    nolr_dir='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/',
    with_lr_dir='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/',
    old_march_dir='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/resultshighamp/results',
    output_base_dir='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig4/gamma_adventures',
    prs_names_csv='/Users/sarahurbut/aladynoulli2/prs_names.csv',
    n_signatures=21,
    n_top=30,
    fdr_threshold=0.05
):
    """Compare gamma from three sources and generate plots for each."""
    
    # Define sources
    # Note: Different versions have different checkpoint structures:
    # - Old March (resultshighamp): output_*_*/model.pt (old folder structure)
    # - With LR: enrollment_model_W0.0001_batch_*_*.pt (new batch structure)
    # - No LR: enrollment_model_VECTORIZED_W*_nolr_batch_*_*.pt (new batch structure with _nolr suffix)
    sources = {
        'nolr': {
            'name': 'No Lambda Reg (Unshrunken)',
            'dir': nolr_dir,
            'pattern': 'enrollment_model_VECTORIZED_W*_nolr_batch_*_*.pt',
            'method': 'batch',
            'description': 'New batches without lambda_reg penalty (unshrunken gamma)',
            'checkpoint_structure': 'new_batch_format'
        },
        'with_lr': {
            'name': 'With Lambda Reg (Shrunken)',
            'dir': with_lr_dir,
            'pattern': 'enrollment_model_W0.0001_batch_*_*.pt',
            'method': 'batch',
            'description': 'Batches with lambda_reg penalty (shrunken gamma)',
            'checkpoint_structure': 'new_batch_format'
        },
        'old_march': {
            'name': 'Old March Version',
            'dir': old_march_dir,
            'pattern': 'output_*_*/model.pt',
            'method': 'old_structure',
            'description': 'Old March version (likely unshrunken, no penalty, different model structure)',
            'checkpoint_structure': 'old_folder_structure'
        }
    }
    
    # Create base output directory
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Load PRS names
    print("="*80)
    print("LOADING PRS NAMES")
    print("="*80)
    prs_names = load_prs_names(prs_names_csv)
    print(f"Loaded {len(prs_names)} PRS names\n")
    
    # Load gamma from each source
    results = {}
    
    for source_key, source_info in sources.items():
        print("="*80)
        print(f"LOADING GAMMA: {source_info['name']}")
        print("="*80)
        print(f"Description: {source_info['description']}")
        print(f"Directory: {source_info['dir']}")
        print(f"Pattern: {source_info['pattern']}")
        
        gamma = None
        gamma_sem = None
        n_batches = 0
        
        if source_info['method'] == 'batch':
            result = load_gamma_from_batches(source_info['dir'], source_info['pattern'])
            if result:
                gamma, gamma_sem, n_batches = result
        elif source_info['method'] == 'old_structure':
            result = load_gamma_from_old_structure(source_info['dir'], source_info['pattern'])
            if result:
                gamma, gamma_sem, n_batches = result
        
        if gamma is None:
            print(f"\n⚠ WARNING: Could not load gamma from {source_info['name']}")
            print(f"  Skipping this source...\n")
            continue
        
        # Store results
        results[source_key] = {
            'gamma': gamma,
            'gamma_sem': gamma_sem,
            'n_batches': n_batches,
            'info': source_info
        }
        
        # Print summary statistics
        print(f"\n✓ Successfully loaded gamma from {source_info['name']}")
        print(f"  Shape: {gamma.shape}")
        print(f"  Mean |γ|: {abs(gamma).mean():.6f}")
        print(f"  Max |γ|: {abs(gamma).max():.6f}")
        print(f"  Std |γ|: {abs(gamma).std():.6f}")
        if gamma_sem is not None:
            print(f"  SEM available: Yes (mean={gamma_sem.mean():.6f})")
        else:
            print(f"  SEM available: No")
        print()
    
    if not results:
        print("ERROR: Could not load gamma from any source!")
        return
    
    # Generate plots for each source
    print("="*80)
    print("GENERATING PLOTS FOR EACH SOURCE")
    print("="*80)
    
    for source_key, result in results.items():
        source_info = result['info']
        gamma = result['gamma']
        gamma_sem = result['gamma_sem']
        
        print(f"\n{'='*80}")
        print(f"PROCESSING: {source_info['name']}")
        print(f"{'='*80}")
        
        # Create output directory for this source
        output_dir = output_base / f"prs_signatures_{source_key}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create gamma dataframe
        print("\nCreating gamma dataframe...")
        gamma_df = create_gamma_dataframe(gamma, prs_names, n_signatures, gamma_sem)
        print(f"  Created dataframe: {gamma_df.shape}")
        print(f"  Effect range: [{gamma_df['effect'].min():.6f}, {gamma_df['effect'].max():.6f}]")
        
        # Print statistics
        if gamma_sem is not None:
            valid_z = gamma_df['z_score'].dropna()
            if len(valid_z) > 0:
                print(f"\nZ-score Statistics:")
                print(f"  Z-score range: [{valid_z.min():.2f}, {valid_z.max():.2f}]")
                print(f"  Mean |z-score|: {valid_z.abs().mean():.2f}")
                print(f"  |z-score| > 2: {(valid_z.abs() > 2).sum()} / {len(valid_z)} ({(valid_z.abs() > 2).sum()/len(valid_z)*100:.1f}%)")
        
        if 'significant_fdr' in gamma_df.columns:
            n_sig = gamma_df['significant_fdr'].sum()
            print(f"\nSignificant associations (FDR < {fdr_threshold}): {n_sig} / {len(gamma_df)} ({100*n_sig/len(gamma_df):.2f}%)")
        
        # Generate plots
        print("\nGenerating plots...")
        
        # 1. Top associations bar plot
        print("  1. Top associations bar plot...")
        plot_top_associations_bar(gamma_df, output_dir, n_top, fdr_threshold)
        
        # 2. Significant associations heatmap
        print("  2. Significant associations heatmap...")
        plot_significant_heatmap(gamma_df, output_dir, fdr_threshold)
        
        # 3. Full heatmap
        print("  3. Full heatmap...")
        plot_full_heatmap(gamma_df, output_dir)
        
        # Save CSV
        csv_path = output_dir / 'gamma_associations.csv'
        gamma_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved gamma dataframe to: {csv_path}")
        print(f"✓ All plots saved to: {output_dir}\n")
    
    # Print comparison summary
    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for source_key, result in results.items():
        gamma = result['gamma']
        source_info = result['info']
        print(f"\n{source_info['name']}:")
        print(f"  Mean |γ|: {abs(gamma).mean():.6f}")
        print(f"  Max |γ|: {abs(gamma).max():.6f}")
        print(f"  Std |γ|: {abs(gamma).std():.6f}")
        print(f"  N batches: {result['n_batches']}")
    
    # Calculate ratios
    if 'nolr' in results and 'with_lr' in results:
        nolr_gamma = results['nolr']['gamma']
        with_lr_gamma = results['with_lr']['gamma']
        
        ratio_mean = abs(nolr_gamma).mean() / abs(with_lr_gamma).mean() if abs(with_lr_gamma).mean() > 0 else float('inf')
        ratio_max = abs(nolr_gamma).max() / abs(with_lr_gamma).max() if abs(with_lr_gamma).max() > 0 else float('inf')
        
        print(f"\n{'='*80}")
        print("SHRINKAGE COMPARISON (NoLR vs WithLR)")
        print(f"{'='*80}")
        print(f"  Mean |γ| ratio (NoLR/WithLR): {ratio_mean:.2f}x")
        print(f"  Max |γ| ratio (NoLR/WithLR): {ratio_max:.2f}x")
        print(f"  Shrinkage: {(1 - 1/ratio_mean)*100:.1f}%")
    
    if 'nolr' in results and 'old_march' in results:
        nolr_gamma = results['nolr']['gamma']
        old_gamma = results['old_march']['gamma']
        
        ratio_mean = abs(nolr_gamma).mean() / abs(old_gamma).mean() if abs(old_gamma).mean() > 0 else float('inf')
        ratio_max = abs(nolr_gamma).max() / abs(old_gamma).max() if abs(old_gamma).max() > 0 else float('inf')
        
        print(f"\n{'='*80}")
        print("COMPARISON (NoLR vs Old March)")
        print(f"{'='*80}")
        print(f"  Mean |γ| ratio (NoLR/OldMarch): {ratio_mean:.2f}x")
        print(f"  Max |γ| ratio (NoLR/OldMarch): {ratio_max:.2f}x")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll plots saved to: {output_base}")
    print(f"  - prs_signatures_nolr/")
    print(f"  - prs_signatures_with_lr/")
    print(f"  - prs_signatures_old_march/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare gamma from three sources')
    parser.add_argument('--nolr_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/',
                        help='Directory for _nolr batches (no lambda_reg)')
    parser.add_argument('--with_lr_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/',
                        help='Directory for batches with lambda_reg')
    parser.add_argument('--old_march_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/resultshighamp/results',
                        help='Directory for old March version (should point to the "results" subdirectory containing output_*_* folders)')
    parser.add_argument('--output_base_dir', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig4/gamma_adventures',
                        help='Base output directory')
    parser.add_argument('--prs_names_csv', type=str,
                        default='/Users/sarahurbut/aladynoulli2/prs_names.csv',
                        help='Path to PRS names CSV')
    parser.add_argument('--n_signatures', type=int, default=21,
                        help='Number of signatures')
    parser.add_argument('--n_top', type=int, default=30,
                        help='Number of top associations for bar plot')
    parser.add_argument('--fdr_threshold', type=float, default=0.05,
                        help='FDR threshold')
    
    args = parser.parse_args()
    
    compare_gamma_sources(
        nolr_dir=args.nolr_dir,
        with_lr_dir=args.with_lr_dir,
        old_march_dir=args.old_march_dir,
        output_base_dir=args.output_base_dir,
        prs_names_csv=args.prs_names_csv,
        n_signatures=args.n_signatures,
        n_top=args.n_top,
        fdr_threshold=args.fdr_threshold
    )

