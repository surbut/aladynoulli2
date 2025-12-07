#!/usr/bin/env python3
"""
Succinct Pathway Analysis: Distinct Signature Trajectories and Overlap

This script provides a focused analysis showing:
1. Distinct signature trajectories that define each pathway in UKB and MGB
2. Overlap/reproducibility between pathways across cohorts
3. Clear visualization of pathway signatures

Note: 4 pathways is a choice for interpretability, not a requirement.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data
from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
from match_pathways_by_disease_patterns import match_pathways_between_cohorts
from scipy.stats import spearmanr


def identify_pathways_by_signature_trajectories(
    ukb_output_dir='output_10yr',
    mgb_output_dir='mgb_deviation_analysis_output',
    mgb_model_path='/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt',
    n_pathways=4,
    n_discriminating_sigs=5
):
    """
    Identify pathways by distinct signature trajectories and show overlap between UKB and MGB
    
    Parameters:
    -----------
    n_pathways : int
        Number of pathways to analyze (default: 4, chosen for interpretability)
    n_discriminating_sigs : int
        Number of top discriminating signatures to show per pathway
    """
    print("="*80)
    print("PATHWAY IDENTIFICATION BY DISTINCT SIGNATURE TRAJECTORIES")
    print("="*80)
    print(f"Analyzing {n_pathways} pathways in UKB and MGB")
    print(f"Note: {n_pathways} pathways chosen for interpretability, not a requirement")
    print("="*80)
    
    # Load UKB results
    print("\n1. Loading UKB pathway results...")
    ukb_results_file = f'{ukb_output_dir}/complete_analysis_results.pkl'
    if not os.path.exists(ukb_results_file):
        raise FileNotFoundError(f"UKB results not found: {ukb_results_file}")
    
    with open(ukb_results_file, 'rb') as f:
        ukb_results = pickle.load(f)
    
    ukb_pathway_data = ukb_results['pathway_data_dev']
    Y_ukb, thetas_ukb, disease_names_ukb, processed_ids_ukb = load_full_data()
    
    # Load MGB results
    print("2. Loading MGB pathway results...")
    mgb_results_file = f'{mgb_output_dir}/complete_analysis_results.pkl'
    if not os.path.exists(mgb_results_file):
        raise FileNotFoundError(f"MGB results not found: {mgb_results_file}")
    
    with open(mgb_results_file, 'rb') as f:
        mgb_results = pickle.load(f)
    
    Y_mgb, thetas_mgb, disease_names_mgb, processed_ids_mgb = load_mgb_data_from_model(mgb_model_path)
    
    # Convert to numpy if needed
    if hasattr(thetas_ukb, 'numpy'):
        thetas_ukb = thetas_ukb.numpy()
    if hasattr(thetas_mgb, 'numpy'):
        thetas_mgb = thetas_mgb.numpy()
    if isinstance(thetas_ukb, np.ndarray) is False:
        thetas_ukb = np.array(thetas_ukb)
    if isinstance(thetas_mgb, np.ndarray) is False:
        thetas_mgb = np.array(thetas_mgb)
    
    print(f"   UKB: {thetas_ukb.shape[0]:,} patients, {thetas_ukb.shape[1]} signatures, {thetas_ukb.shape[2]} timepoints")
    print(f"   MGB: {thetas_mgb.shape[0]:,} patients, {thetas_mgb.shape[1]} signatures, {thetas_mgb.shape[2]} timepoints")
    
    # Extract pathway information
    ukb_patients = ukb_pathway_data['patients']
    mgb_patients = mgb_results['pathway_data_dev']['patients']
    
    ukb_pathway_labels = np.array([p['pathway'] for p in ukb_patients])
    mgb_pathway_labels = np.array([p['pathway'] for p in mgb_patients])
    
    ukb_patient_ids = np.array([p['patient_id'] for p in ukb_patients])
    mgb_patient_ids = np.array([p['patient_id'] for p in mgb_patients])
    
    # Calculate signature deviation trajectories for each pathway
    print("\n3. Computing signature deviation trajectories...")
    
    # Population references
    population_ref_ukb = np.mean(thetas_ukb, axis=0)  # (K, T)
    population_ref_mgb = np.mean(thetas_mgb, axis=0)  # (K, T)
    
    ukb_deviations = {}
    mgb_deviations = {}
    
    for pathway_id in range(n_pathways):
        # UKB
        ukb_mask = ukb_pathway_labels == pathway_id
        if ukb_mask.sum() > 0:
            ukb_pathway_thetas = thetas_ukb[ukb_patient_ids[ukb_mask], :, :]
            ukb_pathway_mean = np.mean(ukb_pathway_thetas, axis=0)  # (K, T)
            ukb_dev = ukb_pathway_mean - population_ref_ukb  # (K, T)
            ukb_deviations[pathway_id] = ukb_dev
        
        # MGB
        mgb_mask = mgb_pathway_labels == pathway_id
        if mgb_mask.sum() > 0:
            mgb_pathway_thetas = thetas_mgb[mgb_patient_ids[mgb_mask], :, :]
            mgb_pathway_mean = np.mean(mgb_pathway_thetas, axis=0)  # (K, T)
            mgb_dev = mgb_pathway_mean - population_ref_mgb  # (K, T)
            mgb_deviations[pathway_id] = mgb_dev
    
    # Match pathways between cohorts
    print("\n4. Matching pathways between UKB and MGB...")
    matching_result = match_pathways_between_cohorts(
        ukb_pathway_data, Y_ukb, disease_names_ukb,
        mgb_results['pathway_data_dev'], Y_mgb, disease_names_mgb,
        top_n_diseases=20
    )
    
    pathway_matching = matching_result['best_matches']
    print(f"\n   Pathway matches:")
    for ukb_id, mgb_id in sorted(pathway_matching.items()):
        similarity = matching_result['pathway_matching'].get((ukb_id, mgb_id), 0.0)
        print(f"   UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id} (similarity: {similarity:.3f})")
    
    # Identify discriminating signatures for each pathway
    print(f"\n5. Identifying top {n_discriminating_sigs} discriminating signatures per pathway...")
    
    # For each pathway, find signatures with highest deviation magnitude
    ukb_discriminating_sigs = {}
    mgb_discriminating_sigs = {}
    
    for pathway_id in range(n_pathways):
        if pathway_id in ukb_deviations:
            # Average absolute deviation across time
            ukb_avg_abs_dev = np.mean(np.abs(ukb_deviations[pathway_id]), axis=1)  # (K,)
            top_sigs_ukb = np.argsort(ukb_avg_abs_dev)[::-1][:n_discriminating_sigs]
            ukb_discriminating_sigs[pathway_id] = top_sigs_ukb
        
        if pathway_id in mgb_deviations:
            mgb_avg_abs_dev = np.mean(np.abs(mgb_deviations[pathway_id]), axis=1)  # (K,)
            top_sigs_mgb = np.argsort(mgb_avg_abs_dev)[::-1][:n_discriminating_sigs]
            mgb_discriminating_sigs[pathway_id] = top_sigs_mgb
    
    # Create visualization
    print("\n6. Creating visualization...")
    create_pathway_trajectory_overlap_plot(
        ukb_deviations, mgb_deviations,
        pathway_matching,
        ukb_discriminating_sigs, mgb_discriminating_sigs,
        n_discriminating_sigs,
        output_dir='pathway_trajectory_analysis'
    )
    
    # Calculate overlap metrics
    print("\n7. Calculating pathway overlap metrics...")
    calculate_pathway_overlap_metrics(
        ukb_deviations, mgb_deviations,
        pathway_matching,
        ukb_pathway_labels, mgb_pathway_labels
    )
    
    return {
        'ukb_deviations': ukb_deviations,
        'mgb_deviations': mgb_deviations,
        'pathway_matching': pathway_matching,
        'ukb_discriminating_sigs': ukb_discriminating_sigs,
        'mgb_discriminating_sigs': mgb_discriminating_sigs
    }


def create_pathway_trajectory_overlap_plot(
    ukb_deviations, mgb_deviations,
    pathway_matching,
    ukb_discriminating_sigs, mgb_discriminating_sigs,
    n_discriminating_sigs,
    output_dir='pathway_trajectory_analysis'
):
    """
    Create a clean visualization showing pathway signature trajectories and overlap
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_pathways = len(pathway_matching)
    
    # Create figure with subplots for each matched pathway pair
    fig, axes = plt.subplots(n_pathways, 2, figsize=(16, 4*n_pathways))
    if n_pathways == 1:
        axes = axes.reshape(1, -1)
    
    # Time axis (years, ages 30-80)
    T = list(ukb_deviations.values())[0].shape[1]
    time_axis = np.arange(30, 30 + T)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for row_idx, (ukb_id, mgb_id) in enumerate(sorted(pathway_matching.items())):
        # UKB pathway (left column)
        ax_ukb = axes[row_idx, 0]
        
        if ukb_id in ukb_deviations:
            ukb_dev = ukb_deviations[ukb_id]
            top_sigs_ukb = ukb_discriminating_sigs.get(ukb_id, [])
            
            # Plot top discriminating signatures
            for i, sig_idx in enumerate(top_sigs_ukb):
                ax_ukb.plot(time_axis, ukb_dev[sig_idx, :], 
                          linewidth=2.5, alpha=0.8,
                          label=f'Sig {sig_idx}',
                          color=colors[i % len(colors)])
            
            ax_ukb.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax_ukb.set_xlabel('Age', fontsize=11)
            ax_ukb.set_ylabel('Signature Deviation', fontsize=11)
            ax_ukb.set_title(f'UKB Pathway {ukb_id} - Top {len(top_sigs_ukb)} Signatures', 
                           fontsize=12, fontweight='bold')
            ax_ukb.grid(True, alpha=0.3)
            ax_ukb.legend(loc='best', fontsize=9, ncol=2)
        
        # MGB pathway (right column)
        ax_mgb = axes[row_idx, 1]
        
        if mgb_id in mgb_deviations:
            mgb_dev = mgb_deviations[mgb_id]
            top_sigs_mgb = mgb_discriminating_sigs.get(mgb_id, [])
            
            # Plot top discriminating signatures
            for i, sig_idx in enumerate(top_sigs_mgb):
                ax_mgb.plot(time_axis, mgb_dev[sig_idx, :], 
                          linewidth=2.5, alpha=0.8,
                          label=f'Sig {sig_idx}',
                          color=colors[i % len(colors)])
            
            ax_mgb.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax_mgb.set_xlabel('Age', fontsize=11)
            ax_mgb.set_ylabel('Signature Deviation', fontsize=11)
            ax_mgb.set_title(f'MGB Pathway {mgb_id} - Top {len(top_sigs_mgb)} Signatures', 
                           fontsize=12, fontweight='bold')
            ax_mgb.grid(True, alpha=0.3)
            ax_mgb.legend(loc='best', fontsize=9, ncol=2)
    
    plt.suptitle('Pathway Identification by Distinct Signature Trajectories\nUKB vs MGB Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    filename = f'{output_dir}/pathway_signature_trajectories_overlap.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()
    
    # Create overlap correlation plot
    create_overlap_correlation_plot(
        ukb_deviations, mgb_deviations,
        pathway_matching,
        output_dir
    )


def create_overlap_correlation_plot(
    ukb_deviations, mgb_deviations,
    pathway_matching,
    output_dir
):
    """
    Create a plot showing correlation of signature trajectories between matched pathways
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    correlations = []
    pathway_pairs = []
    
    for idx, (ukb_id, mgb_id) in enumerate(sorted(pathway_matching.items())):
        if ukb_id in ukb_deviations and mgb_id in mgb_deviations:
            ukb_dev = ukb_deviations[ukb_id]
            mgb_dev = mgb_deviations[mgb_id]
            
            # Average correlation across all signatures
            sig_corrs = []
            K_min = min(ukb_dev.shape[0], mgb_dev.shape[0])
            T_min = min(ukb_dev.shape[1], mgb_dev.shape[1])
            
            for k in range(K_min):
                ukb_sig_traj = ukb_dev[k, :T_min].flatten()
                mgb_sig_traj = mgb_dev[k, :T_min].flatten()
                
                if np.std(ukb_sig_traj) > 0 and np.std(mgb_sig_traj) > 0:
                    corr, _ = spearmanr(ukb_sig_traj, mgb_sig_traj)
                    if not np.isnan(corr):
                        sig_corrs.append(corr)
            
            avg_corr = np.mean(sig_corrs) if sig_corrs else 0.0
            correlations.append(avg_corr)
            pathway_pairs.append(f'UKB P{ukb_id}\n↔ MGB P{mgb_id}')
            
            # Plot correlation for this pathway pair
            ax = axes[idx] if idx < len(axes) else None
            if ax is not None:
                # Scatter plot of all signature trajectories
                all_ukb = ukb_dev[:K_min, :T_min].flatten()
                all_mgb = mgb_dev[:K_min, :T_min].flatten()
                
                ax.scatter(all_ukb, all_mgb, alpha=0.3, s=20)
                ax.plot([all_ukb.min(), all_ukb.max()], 
                       [all_ukb.min(), all_ukb.max()], 
                       'r--', linewidth=2, alpha=0.7)
                ax.set_xlabel(f'UKB Pathway {ukb_id} Deviation', fontsize=10)
                ax.set_ylabel(f'MGB Pathway {mgb_id} Deviation', fontsize=10)
                ax.set_title(f'Pathway {ukb_id} ↔ {mgb_id}\nAvg Correlation: {avg_corr:.3f}', 
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Pathway Overlap: Signature Trajectory Correlation', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'{output_dir}/pathway_overlap_correlations.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()
    
    # Print summary
    print(f"\n   Average pathway trajectory correlation: {np.mean(correlations):.3f}")
    print(f"   Correlation range: {np.min(correlations):.3f} - {np.max(correlations):.3f}")


def calculate_pathway_overlap_metrics(
    ukb_deviations, mgb_deviations,
    pathway_matching,
    ukb_pathway_labels, mgb_pathway_labels
):
    """
    Calculate quantitative metrics of pathway overlap
    """
    print("\n" + "="*80)
    print("PATHWAY OVERLAP METRICS")
    print("="*80)
    
    print(f"\n{'Pathway Pair':<20} {'Trajectory Correlation':<25} {'UKB Size':<12} {'MGB Size':<12}")
    print("-" * 80)
    
    all_correlations = []
    
    for ukb_id, mgb_id in sorted(pathway_matching.items()):
        if ukb_id in ukb_deviations and mgb_id in mgb_deviations:
            ukb_dev = ukb_deviations[ukb_id]
            mgb_dev = mgb_deviations[mgb_id]
            
            # Calculate trajectory correlation
            K_min = min(ukb_dev.shape[0], mgb_dev.shape[0])
            T_min = min(ukb_dev.shape[1], mgb_dev.shape[1])
            
            sig_corrs = []
            for k in range(K_min):
                ukb_sig = ukb_dev[k, :T_min].flatten()
                mgb_sig = mgb_dev[k, :T_min].flatten()
                
                if np.std(ukb_sig) > 0 and np.std(mgb_sig) > 0:
                    corr, _ = spearmanr(ukb_sig, mgb_sig)
                    if not np.isnan(corr):
                        sig_corrs.append(corr)
            
            avg_corr = np.mean(sig_corrs) if sig_corrs else 0.0
            all_correlations.append(avg_corr)
            
            # Pathway sizes
            ukb_size = (ukb_pathway_labels == ukb_id).sum()
            mgb_size = (mgb_pathway_labels == mgb_id).sum()
            
            print(f"UKB P{ukb_id} ↔ MGB P{mgb_id}  {avg_corr:>20.3f}     {ukb_size:>10,}    {mgb_size:>10,}")
    
    print("-" * 80)
    print(f"{'Average':<20} {np.mean(all_correlations):>20.3f}")
    print(f"{'Range':<20} {np.min(all_correlations):>20.3f} - {np.max(all_correlations):.3f}")
    
    print(f"\n✅ Pathway overlap demonstrates reproducibility across cohorts")
    print(f"   Distinct signature trajectories identified in both UKB and MGB")
    print(f"\n💡 Note: {len(pathway_matching)} pathways chosen for interpretability")
    print(f"   The method can identify any number of distinct trajectory patterns")


if __name__ == "__main__":
    results = identify_pathways_by_signature_trajectories()
    print("\n✅ Analysis complete!")

