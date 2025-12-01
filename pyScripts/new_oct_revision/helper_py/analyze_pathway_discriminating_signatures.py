#!/usr/bin/env python3
"""
Analyze Which Signatures Actually Discriminate Between Pathways

The issue: Signature 5-6 might be elevated in ALL pathways (general MI signature)
We need to find which signatures DIFFER between pathways, not just which are elevated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_pathway_discriminating_signatures(signature_analysis_results):
    """
    Find signatures that DIFFER between pathways, not just elevated in all
    
    Parameters:
    -----------
    signature_analysis_results : dict
        Results from transition analysis with group_signature_analysis
    
    Returns:
    --------
    dict with discriminating signatures for each pathway comparison
    """
    print("="*80)
    print("FINDING PATHWAY-DISCRIMINATING SIGNATURES")
    print("="*80)
    
    group_signature_analysis = signature_analysis_results['group_signature_analysis']
    group_names = list(group_signature_analysis.keys())
    
    # Get all signature deviations for each group
    all_deviations = {}
    for group_name in group_names:
        all_deviations[group_name] = group_signature_analysis[group_name]['mean_deviations']
    
    # Convert to array for easier comparison
    n_groups = len(group_names)
    n_sigs = len(all_deviations[group_names[0]])
    
    deviation_matrix = np.array([all_deviations[g] for g in group_names])  # (n_groups, n_sigs)
    
    print(f"\nAnalyzing {n_sigs} signatures across {n_groups} pathways")
    
    # Calculate variance across pathways for each signature
    # High variance = signature differs between pathways
    signature_variance = np.var(deviation_matrix, axis=0)
    
    # Rank signatures by how much they differ between pathways
    signature_discrimination = []
    for sig_idx in range(n_sigs):
        sig_deviations = deviation_matrix[:, sig_idx]
        variance = signature_variance[sig_idx]
        mean_abs_deviation = np.mean(np.abs(sig_deviations))
        max_deviation = np.max(np.abs(sig_deviations))
        min_deviation = np.min(np.abs(sig_deviations))
        range_deviation = max_deviation - min_deviation
        
        signature_discrimination.append({
            'signature_idx': sig_idx,
            'variance': variance,
            'mean_abs_deviation': mean_abs_deviation,
            'range': range_deviation,
            'deviations_by_group': {group_names[i]: sig_deviations[i] for i in range(n_groups)},
            'discrimination_score': variance  # Use variance as discrimination score
        })
    
    # Sort by discrimination score
    signature_discrimination.sort(key=lambda x: x['discrimination_score'], reverse=True)
    
    print("\nTop 10 signatures that DISCRIMINATE between pathways:")
    print(f"{'Rank':<6} {'Sig':<6} {'Variance':<12} {'Range':<12} {'Deviations by Group':<50}")
    print("-" * 90)
    
    for i, sig_info in enumerate(signature_discrimination[:10]):
        sig_idx = sig_info['signature_idx']
        variance = sig_info['variance']
        range_val = sig_info['range']
        deviations = sig_info['deviations_by_group']
        
        deviation_str = ", ".join([f"{g}: {d:+.3f}" for g, d in deviations.items()])
        print(f"{i+1:<6} {sig_idx:<6} {variance:<12.6f} {range_val:<12.6f} {deviation_str[:48]}")
    
    # Identify signatures that are similar across all pathways (likely general MI signature)
    print("\nSignatures that are SIMILAR across all pathways (likely general MI signature):")
    low_variance_sigs = [s for s in signature_discrimination if s['variance'] < 0.0001]
    print(f"Found {len(low_variance_sigs)} signatures with low variance (< 0.0001)")
    for sig_info in low_variance_sigs[:5]:
        sig_idx = sig_info['signature_idx']
        mean_dev = np.mean(list(sig_info['deviations_by_group'].values()))
        print(f"  Signature {sig_idx}: Mean deviation = {mean_dev:+.4f} (similar across all pathways)")
    
    return signature_discrimination


def create_discriminating_signatures_heatmap(ukb_results, mgb_results, 
                                           save_path='discriminating_signatures_heatmap.png'):
    """
    Create heatmap showing only signatures that DISCRIMINATE between pathways
    """
    print("\n2. CREATING DISCRIMINATING SIGNATURES HEATMAP")
    
    ukb_sig_analysis = ukb_results['signature_analysis']['group_signature_analysis']
    mgb_sig_analysis = mgb_results['signature_analysis']['group_signature_analysis']
    
    # Find discriminating signatures for UKB
    ukb_discriminating = find_pathway_discriminating_signatures(ukb_results['signature_analysis'])
    mgb_discriminating = find_pathway_discriminating_signatures(mgb_results['signature_analysis'])
    
    # Get top 10 discriminating signatures for each
    top_ukb_sigs = [s['signature_idx'] for s in ukb_discriminating[:10]]
    top_mgb_sigs = [s['signature_idx'] for s in mgb_discriminating[:10]]
    
    # Get common groups
    common_groups = sorted(set(ukb_sig_analysis.keys()) & set(mgb_sig_analysis.keys()))
    
    # Create heatmap with only discriminating signatures
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top row: UKB discriminating signatures
    ax1 = axes[0, 0]
    ukb_matrix = []
    for group_name in common_groups:
        deviations = ukb_sig_analysis[group_name]['mean_deviations']
        ukb_matrix.append([deviations[sig_idx] for sig_idx in top_ukb_sigs])
    ukb_matrix = np.array(ukb_matrix)
    
    im1 = ax1.imshow(ukb_matrix, cmap='RdBu_r', aspect='auto', 
                     vmin=-0.05, vmax=0.05, interpolation='nearest')
    ax1.set_xlabel('Top Discriminating Signatures')
    ax1.set_ylabel('Transition Group')
    ax1.set_title('UKB: Discriminating Signatures Only')
    ax1.set_yticks(range(len(common_groups)))
    ax1.set_yticklabels(common_groups)
    ax1.set_xticks(range(len(top_ukb_sigs)))
    ax1.set_xticklabels([f'Sig {idx}' for idx in top_ukb_sigs], rotation=45, ha='right')
    plt.colorbar(im1, ax=ax1, label='Deviation')
    
    # Top right: MGB discriminating signatures
    ax2 = axes[0, 1]
    mgb_matrix = []
    for group_name in common_groups:
        deviations = mgb_sig_analysis[group_name]['mean_deviations']
        mgb_matrix.append([deviations[sig_idx] for sig_idx in top_mgb_sigs])
    mgb_matrix = np.array(mgb_matrix)
    
    im2 = ax2.imshow(mgb_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-0.05, vmax=0.05, interpolation='nearest')
    ax2.set_xlabel('Top Discriminating Signatures')
    ax2.set_ylabel('Transition Group')
    ax2.set_title('MGB: Discriminating Signatures Only')
    ax2.set_yticks(range(len(common_groups)))
    ax2.set_yticklabels(common_groups)
    ax2.set_xticks(range(len(top_mgb_sigs)))
    ax2.set_xticklabels([f'Sig {idx}' for idx in top_mgb_sigs], rotation=45, ha='right')
    plt.colorbar(im2, ax=ax2, label='Deviation')
    
    # Bottom left: Difference between pathways (UKB)
    ax3 = axes[1, 0]
    # Calculate pairwise differences between pathways
    n_groups = len(common_groups)
    pathway_differences = np.zeros((n_groups, n_groups, len(top_ukb_sigs)))
    
    for i, group1 in enumerate(common_groups):
        for j, group2 in enumerate(common_groups):
            dev1 = [ukb_sig_analysis[group1]['mean_deviations'][sig_idx] for sig_idx in top_ukb_sigs]
            dev2 = [ukb_sig_analysis[group2]['mean_deviations'][sig_idx] for sig_idx in top_ukb_sigs]
            pathway_differences[i, j, :] = np.array(dev1) - np.array(dev2)
    
    # Show mean absolute difference for each signature
    mean_abs_diff = np.mean(np.abs(pathway_differences), axis=(0, 1))
    
    ax3.bar(range(len(top_ukb_sigs)), mean_abs_diff, alpha=0.7)
    ax3.set_xlabel('Signature Index')
    ax3.set_ylabel('Mean Absolute Difference Between Pathways')
    ax3.set_title('UKB: How Much Each Signature Differs Between Pathways')
    ax3.set_xticks(range(len(top_ukb_sigs)))
    ax3.set_xticklabels([f'Sig {idx}' for idx in top_ukb_sigs], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Same for MGB
    ax4 = axes[1, 1]
    pathway_differences_mgb = np.zeros((n_groups, n_groups, len(top_mgb_sigs)))
    
    for i, group1 in enumerate(common_groups):
        for j, group2 in enumerate(common_groups):
            dev1 = [mgb_sig_analysis[group1]['mean_deviations'][sig_idx] for sig_idx in top_mgb_sigs]
            dev2 = [mgb_sig_analysis[group2]['mean_deviations'][sig_idx] for sig_idx in top_mgb_sigs]
            pathway_differences_mgb[i, j, :] = np.array(dev1) - np.array(dev2)
    
    mean_abs_diff_mgb = np.mean(np.abs(pathway_differences_mgb), axis=(0, 1))
    
    ax4.bar(range(len(top_mgb_sigs)), mean_abs_diff_mgb, alpha=0.7, color='coral')
    ax4.set_xlabel('Signature Index')
    ax4.set_ylabel('Mean Absolute Difference Between Pathways')
    ax4.set_title('MGB: How Much Each Signature Differs Between Pathways')
    ax4.set_xticks(range(len(top_mgb_sigs)))
    ax4.set_xticklabels([f'Sig {idx}' for idx in top_mgb_sigs], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Pathway-Discriminating Signatures: UKB vs MGB\n(Excluding General MI Signature)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Figure saved to: {save_path}")
    plt.show()


def analyze_pathway_specific_signatures(ukb_results, mgb_results):
    """
    Analyze which signatures are pathway-specific vs general MI signature
    """
    print("\n" + "="*80)
    print("ANALYZING PATHWAY-SPECIFIC vs GENERAL SIGNATURES")
    print("="*80)
    
    ukb_sig_analysis = ukb_results['signature_analysis']['group_signature_analysis']
    mgb_sig_analysis = mgb_results['signature_analysis']['group_signature_analysis']
    
    common_groups = sorted(set(ukb_sig_analysis.keys()) & set(mgb_sig_analysis.keys()))
    
    # For each pathway, find signatures that are SPECIFIC to that pathway
    print("\n1. PATHWAY-SPECIFIC SIGNATURES")
    print("   (Signatures elevated in one pathway but not others)")
    
    for group_name in common_groups:
        print(f"\n   {group_name.upper()}:")
        
        # UKB pathway-specific signatures
        ukb_deviations = ukb_sig_analysis[group_name]['mean_deviations']
        other_groups = [g for g in common_groups if g != group_name]
        
        pathway_specific = []
        for sig_idx in range(len(ukb_deviations)):
            this_group_dev = ukb_deviations[sig_idx]
            other_groups_dev = [ukb_sig_analysis[g]['mean_deviations'][sig_idx] for g in other_groups]
            other_mean = np.mean(other_groups_dev)
            
            # Signature is pathway-specific if it's elevated in this pathway but not others
            if this_group_dev > 0.01 and this_group_dev > other_mean + 0.01:
                pathway_specific.append({
                    'signature_idx': sig_idx,
                    'this_pathway': this_group_dev,
                    'other_pathways': other_mean,
                    'difference': this_group_dev - other_mean
                })
        
        pathway_specific.sort(key=lambda x: x['difference'], reverse=True)
        
        print(f"   UKB pathway-specific signatures:")
        for sig_info in pathway_specific[:5]:
            print(f"     Signature {sig_info['signature_idx']:2d}: "
                  f"{sig_info['this_pathway']:+.4f} (this) vs "
                  f"{sig_info['other_pathways']:+.4f} (others), "
                  f"diff = {sig_info['difference']:+.4f}")
        
        # MGB pathway-specific signatures
        mgb_deviations = mgb_sig_analysis[group_name]['mean_deviations']
        pathway_specific_mgb = []
        for sig_idx in range(len(mgb_deviations)):
            this_group_dev = mgb_deviations[sig_idx]
            other_groups_dev = [mgb_sig_analysis[g]['mean_deviations'][sig_idx] for g in other_groups]
            other_mean = np.mean(other_groups_dev)
            
            if this_group_dev > 0.01 and this_group_dev > other_mean + 0.01:
                pathway_specific_mgb.append({
                    'signature_idx': sig_idx,
                    'this_pathway': this_group_dev,
                    'other_pathways': other_mean,
                    'difference': this_group_dev - other_mean
                })
        
        pathway_specific_mgb.sort(key=lambda x: x['difference'], reverse=True)
        
        print(f"   MGB pathway-specific signatures:")
        for sig_info in pathway_specific_mgb[:5]:
            print(f"     Signature {sig_info['signature_idx']:2d}: "
                  f"{sig_info['this_pathway']:+.4f} (this) vs "
                  f"{sig_info['other_pathways']:+.4f} (others), "
                  f"diff = {sig_info['difference']:+.4f}")
    
    print("\n2. GENERAL MI SIGNATURE")
    print("   (Signatures elevated in ALL pathways - not pathway-specific)")
    
    # Find signatures elevated in all pathways
    ukb_general = []
    for sig_idx in range(len(ukb_deviations)):
        all_group_deviations = [ukb_sig_analysis[g]['mean_deviations'][sig_idx] for g in common_groups]
        if all(d > 0.01 for d in all_group_deviations):
            ukb_general.append({
                'signature_idx': sig_idx,
                'mean_deviation': np.mean(all_group_deviations),
                'variance': np.var(all_group_deviations)
            })
    
    ukb_general.sort(key=lambda x: x['mean_deviation'], reverse=True)
    
    print(f"   UKB: {len(ukb_general)} signatures elevated in ALL pathways")
    for sig_info in ukb_general[:5]:
        print(f"     Signature {sig_info['signature_idx']:2d}: "
              f"Mean = {sig_info['mean_deviation']:+.4f}, "
              f"Variance = {sig_info['variance']:.6f}")
    
    mgb_general = []
    for sig_idx in range(len(mgb_deviations)):
        all_group_deviations = [mgb_sig_analysis[g]['mean_deviations'][sig_idx] for g in common_groups]
        if all(d > 0.01 for d in all_group_deviations):
            mgb_general.append({
                'signature_idx': sig_idx,
                'mean_deviation': np.mean(all_group_deviations),
                'variance': np.var(all_group_deviations)
            })
    
    mgb_general.sort(key=lambda x: x['mean_deviation'], reverse=True)
    
    print(f"   MGB: {len(mgb_general)} signatures elevated in ALL pathways")
    for sig_info in mgb_general[:5]:
        print(f"     Signature {sig_info['signature_idx']:2d}: "
              f"Mean = {sig_info['mean_deviation']:+.4f}, "
              f"Variance = {sig_info['variance']:.6f}")


if __name__ == "__main__":
    print("Pathway-Discriminating Signature Analysis")
    print("This script identifies which signatures actually differ between pathways,")
    print("rather than just being elevated in all pathways (general MI signature).")

