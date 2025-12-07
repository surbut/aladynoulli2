#!/usr/bin/env python3
"""
Fix Transition Heatmap Analysis

The issue: Signature 5-6 is elevated in ALL transition groups, making it look like
all pathways are the same. We need to:
1. Identify which signatures DISCRIMINATE between pathways (vary across pathways)
2. Separate "general MI signatures" (elevated in all) from "pathway-specific" (elevated in one)
3. Create visualizations that highlight pathway differences, not similarities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_signature_discrimination(signature_analysis_results):
    """
    Identify which signatures actually DISCRIMINATE between pathways
    
    Parameters:
    -----------
    signature_analysis_results : dict
        Results from transition analysis with group_signature_analysis
    
    Returns:
    --------
    dict with:
        - discriminating_signatures: Signatures that vary across pathways
        - general_signatures: Signatures elevated in all pathways
        - pathway_specific: Signatures elevated in specific pathways only
    """
    print("="*80)
    print("ANALYZING SIGNATURE DISCRIMINATION")
    print("="*80)
    
    group_signature_analysis = signature_analysis_results['group_signature_analysis']
    group_names = list(group_signature_analysis.keys())
    
    # Get all signature deviations for each group
    n_groups = len(group_names)
    n_sigs = len(group_signature_analysis[group_names[0]]['mean_deviations'])
    
    deviation_matrix = np.array([
        group_signature_analysis[g]['mean_deviations'] 
        for g in group_names
    ])  # Shape: (n_groups, n_sigs)
    
    print(f"\nAnalyzing {n_sigs} signatures across {n_groups} pathways")
    
    # Calculate variance across pathways for each signature
    # High variance = signature differs between pathways
    signature_variance = np.var(deviation_matrix, axis=0)
    signature_range = np.max(deviation_matrix, axis=0) - np.min(deviation_matrix, axis=0)
    
    # Identify general signatures (elevated in ALL pathways)
    # A signature is "general" if it's elevated (> threshold) in all pathways
    threshold = 0.01  # Minimum deviation to be considered "elevated"
    general_signatures = []
    pathway_specific_signatures = []
    
    for sig_idx in range(n_sigs):
        sig_deviations = deviation_matrix[:, sig_idx]
        
        # Check if elevated in all pathways
        is_elevated_all = np.all(sig_deviations > threshold)
        is_elevated_any = np.any(sig_deviations > threshold)
        
        # Check variance (how much it differs between pathways)
        variance = signature_variance[sig_idx]
        range_val = signature_range[sig_idx]
        
        sig_info = {
            'signature_idx': sig_idx,
            'variance': variance,
            'range': range_val,
            'mean_deviation': np.mean(sig_deviations),
            'max_deviation': np.max(sig_deviations),
            'min_deviation': np.min(sig_deviations),
            'deviations_by_group': {group_names[i]: sig_deviations[i] for i in range(n_groups)},
            'is_elevated_all': is_elevated_all,
            'is_elevated_any': is_elevated_any
        }
        
        if is_elevated_all:
            # General MI signature (elevated in all pathways)
            general_signatures.append(sig_info)
        elif is_elevated_any and variance > 0.0001:
            # Pathway-specific (elevated in some but not all, and varies)
            pathway_specific_signatures.append(sig_info)
    
    # Sort by variance (discrimination score)
    general_signatures.sort(key=lambda x: x['variance'], reverse=True)
    pathway_specific_signatures.sort(key=lambda x: x['variance'], reverse=True)
    
    print("\n1. GENERAL MI SIGNATURES (Elevated in ALL pathways):")
    print(f"   Found {len(general_signatures)} signatures")
    print(f"   {'Sig':<6} {'Variance':<12} {'Range':<12} {'Mean Dev':<12} {'Deviations'}")
    print("-" * 80)
    
    for sig_info in general_signatures[:10]:
        sig_idx = sig_info['signature_idx']
        variance = sig_info['variance']
        range_val = sig_info['range']
        mean_dev = sig_info['mean_deviation']
        deviations_str = ", ".join([f"{g}: {d:+.3f}" for g, d in sig_info['deviations_by_group'].items()])
        print(f"   {sig_idx:<6} {variance:<12.6f} {range_val:<12.6f} {mean_dev:<12.6f} {deviations_str[:50]}")
    
    print("\n2. PATHWAY-SPECIFIC SIGNATURES (Vary between pathways):")
    print(f"   Found {len(pathway_specific_signatures)} signatures")
    print(f"   {'Sig':<6} {'Variance':<12} {'Range':<12} {'Mean Dev':<12} {'Deviations'}")
    print("-" * 80)
    
    for sig_info in pathway_specific_signatures[:10]:
        sig_idx = sig_info['signature_idx']
        variance = sig_info['variance']
        range_val = sig_info['range']
        mean_dev = sig_info['mean_deviation']
        deviations_str = ", ".join([f"{g}: {d:+.3f}" for g, d in sig_info['deviations_by_group'].items()])
        print(f"   {sig_idx:<6} {variance:<12.6f} {range_val:<12.6f} {mean_dev:<12.6f} {deviations_str[:50]}")
    
    return {
        'general_signatures': general_signatures,
        'pathway_specific_signatures': pathway_specific_signatures,
        'all_signatures': general_signatures + pathway_specific_signatures,
        'deviation_matrix': deviation_matrix,
        'group_names': group_names
    }


def create_discriminating_heatmap(signature_analysis_results, 
                                  save_path='discriminating_signatures_heatmap.png',
                                  top_n=10):
    """
    Create heatmap showing only TOP DISCRIMINATING signatures
    (excludes general signatures that are similar across all pathways)
    """
    print("\n" + "="*80)
    print("CREATING DISCRIMINATING SIGNATURES HEATMAP")
    print("="*80)
    
    # Analyze discrimination
    discrimination_results = analyze_signature_discrimination(signature_analysis_results)
    
    # Get top discriminating signatures (by variance)
    top_discriminating = sorted(
        discrimination_results['all_signatures'],
        key=lambda x: x['variance'],
        reverse=True
    )[:top_n]
    
    top_sig_indices = [s['signature_idx'] for s in top_discriminating]
    
    print(f"\nUsing top {len(top_sig_indices)} discriminating signatures: {top_sig_indices}")
    
    # Get deviation matrix for these signatures only
    group_names = discrimination_results['group_names']
    deviation_matrix = discrimination_results['deviation_matrix']
    
    # Extract only top discriminating signatures
    top_deviations = deviation_matrix[:, top_sig_indices]  # (n_groups, top_n)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(top_sig_indices) * 0.8), max(6, len(group_names) * 1.5)))
    
    # Use centered colormap to show positive/negative deviations
    vmax = np.max(np.abs(top_deviations)) * 1.1
    vmin = -vmax
    
    im = ax.imshow(top_deviations, cmap='RdBu_r', aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    ax.set_xlabel('Top Discriminating Signatures', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transition Group', fontsize=12, fontweight='bold')
    ax.set_title(f'Signature Deviations: Top {top_n} Discriminating Signatures Only\n(Excludes General MI Signatures)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(top_sig_indices)))
    ax.set_xticklabels([f'Sig {idx}' for idx in top_sig_indices], rotation=45, ha='right')
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Deviation from Reference', shrink=0.8)
    
    # Add text annotations for values
    for i in range(len(group_names)):
        for j in range(len(top_sig_indices)):
            value = top_deviations[i, j]
            text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{value:+.3f}', ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved discriminating signatures heatmap to: {save_path}")
    plt.show()
    
    return discrimination_results


def create_pathway_specific_heatmap(signature_analysis_results,
                                    save_path='pathway_specific_signatures_heatmap.png'):
    """
    Create heatmap showing pathway-specific signatures (one per pathway)
    """
    print("\n" + "="*80)
    print("CREATING PATHWAY-SPECIFIC SIGNATURES HEATMAP")
    print("="*80)
    
    discrimination_results = analyze_signature_discrimination(signature_analysis_results)
    
    # For each pathway, find signatures that are MOST elevated in that pathway
    group_names = discrimination_results['group_names']
    deviation_matrix = discrimination_results['deviation_matrix']
    n_sigs = deviation_matrix.shape[1]
    
    pathway_specific_sigs = {}
    
    for group_idx, group_name in enumerate(group_names):
        # Find signatures where this pathway has the HIGHEST deviation
        group_deviations = deviation_matrix[group_idx, :]
        other_deviations = np.delete(deviation_matrix, group_idx, axis=0)
        other_max = np.max(other_deviations, axis=0)
        
        # Signature is pathway-specific if this pathway's deviation is > other pathways' max
        pathway_specific_mask = group_deviations > (other_max + 0.01)  # At least 0.01 higher
        pathway_specific_indices = np.where(pathway_specific_mask)[0]
        
        # Sort by how much more elevated this pathway is
        pathway_specific_scores = []
        for sig_idx in pathway_specific_indices:
            score = group_deviations[sig_idx] - other_max[sig_idx]
            pathway_specific_scores.append((sig_idx, score))
        
        pathway_specific_scores.sort(key=lambda x: x[1], reverse=True)
        pathway_specific_sigs[group_name] = pathway_specific_scores[:5]  # Top 5 per pathway
    
    # Create heatmap
    all_specific_sigs = set()
    for sigs in pathway_specific_sigs.values():
        all_specific_sigs.update([s[0] for s in sigs])
    
    all_specific_sigs = sorted(list(all_specific_sigs))
    
    # Create matrix
    specific_matrix = deviation_matrix[:, all_specific_sigs]
    
    fig, ax = plt.subplots(figsize=(max(12, len(all_specific_sigs) * 0.8), max(6, len(group_names) * 1.5)))
    
    vmax = np.max(np.abs(specific_matrix)) * 1.1
    vmin = -vmax
    
    im = ax.imshow(specific_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    ax.set_xlabel('Pathway-Specific Signatures', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transition Group', fontsize=12, fontweight='bold')
    ax.set_title('Pathway-Specific Signature Deviations\n(Signatures Most Elevated in Each Pathway)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(range(len(all_specific_sigs)))
    ax.set_xticklabels([f'Sig {idx}' for idx in all_specific_sigs], rotation=45, ha='right')
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    
    cbar = plt.colorbar(im, ax=ax, label='Deviation from Reference', shrink=0.8)
    
    # Highlight pathway-specific signatures
    for group_idx, group_name in enumerate(group_names):
        for sig_pos, sig_idx in enumerate(all_specific_sigs):
            if any(s[0] == sig_idx for s in pathway_specific_sigs[group_name]):
                # This signature is specific to this pathway - highlight it
                rect = plt.Rectangle((sig_pos - 0.5, group_idx - 0.5), 1, 1,
                                    fill=False, edgecolor='yellow', linewidth=3)
                ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved pathway-specific signatures heatmap to: {save_path}")
    plt.show()
    
    # Print summary
    print("\nPathway-Specific Signatures:")
    for group_name, sigs in pathway_specific_sigs.items():
        print(f"\n  {group_name}:")
        for sig_idx, score in sigs:
            print(f"    Signature {sig_idx}: +{score:.4f} more than other pathways")


def compare_ukb_mgb_discrimination(ukb_results, mgb_results,
                                   save_path='ukb_mgb_discrimination_comparison.png'):
    """
    Compare discriminating signatures between UKB and MGB
    """
    print("\n" + "="*80)
    print("COMPARING DISCRIMINATION: UKB vs MGB")
    print("="*80)
    
    ukb_disc = analyze_signature_discrimination(ukb_results['signature_analysis'])
    mgb_disc = analyze_signature_discrimination(mgb_results['signature_analysis'])
    
    # Get top 10 discriminating signatures for each
    ukb_top = sorted(ukb_disc['all_signatures'], key=lambda x: x['variance'], reverse=True)[:10]
    mgb_top = sorted(mgb_disc['all_signatures'], key=lambda x: x['variance'], reverse=True)[:10]
    
    print("\nTop 10 Discriminating Signatures:")
    print(f"{'Rank':<6} {'UKB Sig':<10} {'UKB Variance':<15} {'MGB Sig':<10} {'MGB Variance':<15}")
    print("-" * 70)
    
    for i in range(10):
        ukb_sig = ukb_top[i]['signature_idx'] if i < len(ukb_top) else None
        ukb_var = ukb_top[i]['variance'] if i < len(ukb_top) else None
        mgb_sig = mgb_top[i]['signature_idx'] if i < len(mgb_top) else None
        mgb_var = mgb_top[i]['variance'] if i < len(mgb_top) else None
        
        ukb_str = f"{ukb_sig} ({ukb_var:.6f})" if ukb_sig is not None else "N/A"
        mgb_str = f"{mgb_sig} ({mgb_var:.6f})" if mgb_sig is not None else "N/A"
        
        print(f"{i+1:<6} {ukb_str:<25} {mgb_str:<25}")
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # UKB top discriminating
    ukb_top_sigs = [s['signature_idx'] for s in ukb_top]
    ukb_top_vars = [s['variance'] for s in ukb_top]
    
    axes[0].bar(range(len(ukb_top_sigs)), ukb_top_vars, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Rank')
    axes[0].set_ylabel('Variance (Discrimination Score)')
    axes[0].set_title('UKB: Top 10 Discriminating Signatures')
    axes[0].set_xticks(range(len(ukb_top_sigs)))
    axes[0].set_xticklabels([f'Sig {idx}' for idx in ukb_top_sigs], rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # MGB top discriminating
    mgb_top_sigs = [s['signature_idx'] for s in mgb_top]
    mgb_top_vars = [s['variance'] for s in mgb_top]
    
    axes[1].bar(range(len(mgb_top_sigs)), mgb_top_vars, alpha=0.7, color='coral')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Variance (Discrimination Score)')
    axes[1].set_title('MGB: Top 10 Discriminating Signatures')
    axes[1].set_xticks(range(len(mgb_top_sigs)))
    axes[1].set_xticklabels([f'Sig {idx}' for idx in mgb_top_sigs], rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Signature Discrimination Comparison: UKB vs MGB', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison to: {save_path}")
    plt.show()
    
    return {
        'ukb_discrimination': ukb_disc,
        'mgb_discrimination': mgb_disc
    }


if __name__ == "__main__":
    print("Fix Transition Heatmap Analysis")
    print("This script identifies which signatures actually discriminate between pathways")
    print("and creates better visualizations that highlight pathway differences.")

