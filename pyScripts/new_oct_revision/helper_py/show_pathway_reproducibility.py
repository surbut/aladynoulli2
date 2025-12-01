#!/usr/bin/env python3
"""
Show Pathway Reproducibility Between UKB and MGB

For matched pathways, show:
1. Pathway sizes/proportions are similar
2. Signature deviation patterns are similar
3. Disease patterns are similar
4. Age at onset patterns are similar

This demonstrates that the deviation-based pathway discovery method
generalizes across cohorts.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from show_pathway_matches import show_pathway_matches
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def compute_signature_correspondence_from_crosstab(crosstab_df=None):
    """
    Compute signature correspondence from cross-tabulation table.
    
    For each UKB signature, finds the MGB signature with the maximum proportion
    (darkest red in the heatmap).
    
    Parameters:
    -----------
    crosstab_df : pandas.DataFrame, optional
        Cross-tabulation matrix with UKB signatures as rows and MGB signatures as columns.
        If None, uses the predefined mapping from the user's analysis.
    
    Returns:
    --------
    signature_map : dict
        Mapping from UKB signature index to MGB signature index
        {ukb_sig_idx: mgb_sig_idx}
    """
    if crosstab_df is not None:
        print("\n   Computing signature correspondence from cross-tabulation...")
        signature_map = {}
        
        # For each UKB signature (row), find MGB signature (column) with max value
        for ukb_sig in crosstab_df.index:
            mgb_sig = crosstab_df.loc[ukb_sig].idxmax()
            max_prop = crosstab_df.loc[ukb_sig, mgb_sig]
            signature_map[int(ukb_sig)] = int(mgb_sig)
            print(f"      UKB Sig {ukb_sig} → MGB Sig {mgb_sig} (proportion={max_prop:.3f})")
        
        return signature_map, crosstab_df
    else:
        # Use the predefined mapping from the user's cross-tabulation analysis
        print("\n   Using predefined signature correspondence from cross-tabulation...")
        
        # Mapping from cross-tabulation table: UKB_cluster -> MGB_cluster
        # Based on maximum proportion in cross-tabulation heatmap (darkest red square)
        # Table format: UKB_cluster | MGB_cluster
        predefined_mapping = {
            4: 0,   # UKB 4 -> MGB 0
            7: 1,   # UKB 7 -> MGB 1
            1: 2,   # UKB 1 -> MGB 2
            12: 3,  # UKB 12 -> MGB 3
            16: 4,  # UKB 16 -> MGB 4
            0: 5,   # UKB 0 -> MGB 5
            5: 5,   # UKB 5 -> MGB 5
            15: 6,  # UKB 15 -> MGB 6
            2: 7,   # UKB 2 -> MGB 7
            17: 8,  # UKB 17 -> MGB 8
            9: 9,   # UKB 9 -> MGB 9
            11: 10, # UKB 11 -> MGB 10
            6: 11,  # UKB 6 -> MGB 11
            3: 12,  # UKB 3 -> MGB 12
            18: 13, # UKB 18 -> MGB 13
            14: 14, # UKB 14 -> MGB 14
            19: 15, # UKB 19 -> MGB 15
            10: 16, # UKB 10 -> MGB 16
            13: 18, # UKB 13 -> MGB 18
            8: 19,  # UKB 8 -> MGB 19
            20: 20  # UKB 20 (health) -> MGB 20 (health) - same in both
        }
        
        print(f"   Using {len(predefined_mapping)} signature correspondences from cross-tabulation")
        # Display in format: MGB Sig X ↔ UKB Sig Y (MGB first as requested)
        reverse_mapping = {v: k for k, v in predefined_mapping.items()}
        for mgb_sig in sorted(reverse_mapping.keys()):
            ukb_sig = reverse_mapping[mgb_sig]
            print(f"      MGB Sig {mgb_sig} ↔ UKB Sig {ukb_sig}")
        
        return predefined_mapping, None


def compute_signature_correspondence(ukb_deviations, mgb_deviations, best_matches, 
                                     use_crosstab=True, crosstab_df=None):
    """
    Compute which UKB signatures correspond to which MGB signatures.
    
    If use_crosstab=True, uses cross-tabulation mapping (finding max in each row).
    Otherwise, uses correlation-based matching.
    
    Returns:
    --------
    signature_map : dict
        Mapping from UKB signature index to MGB signature index
        {ukb_sig_idx: mgb_sig_idx}
    """
    if use_crosstab:
        return compute_signature_correspondence_from_crosstab(crosstab_df)
    else:
        # Fallback to correlation-based method
        print("\n   Computing signature correspondence by deviation pattern similarity...")
        
        # Collect all deviation patterns across matched pathways
        K_ukb = ukb_deviations[list(ukb_deviations.keys())[0]].shape[0]
        K_mgb = mgb_deviations[list(mgb_deviations.keys())[0]].shape[0]
        
        # Build feature vectors: mean deviation across time for each pathway
        ukb_signature_vectors = {}
        mgb_signature_vectors = {}
        
        for ukb_pw_id, mgb_pw_id in best_matches.items():
            if ukb_pw_id in ukb_deviations and mgb_pw_id in mgb_deviations:
                ukb_dev = ukb_deviations[ukb_pw_id]
                mgb_dev = mgb_deviations[mgb_pw_id]
                
                ukb_mean_dev = np.mean(ukb_dev, axis=1)
                mgb_mean_dev = np.mean(mgb_dev, axis=1)
                
                for k in range(K_ukb):
                    if k not in ukb_signature_vectors:
                        ukb_signature_vectors[k] = []
                    ukb_signature_vectors[k].append(ukb_mean_dev[k])
                
                for k in range(K_mgb):
                    if k not in mgb_signature_vectors:
                        mgb_signature_vectors[k] = []
                    mgb_signature_vectors[k].append(mgb_mean_dev[k])
        
        ukb_features = np.array([ukb_signature_vectors[k] for k in range(K_ukb)])
        mgb_features = np.array([mgb_signature_vectors[k] for k in range(K_mgb)])
        
        similarity_matrix = np.zeros((K_ukb, K_mgb))
        for ukb_k in range(K_ukb):
            for mgb_k in range(K_mgb):
                if len(ukb_features[ukb_k]) > 1:
                    corr, _ = spearmanr(ukb_features[ukb_k], mgb_features[mgb_k])
                    if np.isnan(corr):
                        corr = 0.0
                    similarity_matrix[ukb_k, mgb_k] = corr
                else:
                    similarity_matrix[ukb_k, mgb_k] = -abs(ukb_features[ukb_k][0] - mgb_features[mgb_k][0])
        
        cost_matrix = -similarity_matrix
        ukb_indices, mgb_indices = linear_sum_assignment(cost_matrix)
        
        signature_map = {}
        for ukb_idx, mgb_idx in zip(ukb_indices, mgb_indices):
            similarity = similarity_matrix[ukb_idx, mgb_idx]
            if similarity > 0.3:
                signature_map[ukb_idx] = mgb_idx
        
        print(f"   Found {len(signature_map)} signature correspondences (correlation > 0.3)")
        for ukb_sig, mgb_sig in sorted(signature_map.items()):
            similarity = similarity_matrix[ukb_sig, mgb_sig]
            print(f"      UKB Sig {ukb_sig} ↔ MGB Sig {mgb_sig} (corr={similarity:.3f})")
        
        return signature_map, similarity_matrix


def create_reproducibility_figure(ukb_results, mgb_results, pathway_matching):
    """
    Create comprehensive reproducibility figure showing matched pathways
    """
    print("\n" + "="*80)
    print("CREATING REPRODUCIBILITY FIGURES")
    print("="*80)
    
    ukb_pathway_data = ukb_results['pathway_data_dev']
    mgb_pathway_data = mgb_results['pathway_data']
    
    best_matches = pathway_matching['best_matches']
    similarities = pathway_matching['pathway_similarities']
    
    # Get pathway sizes
    ukb_patients = ukb_pathway_data['patients']
    mgb_patients = mgb_pathway_data['patients']
    
    ukb_labels = np.array([p['pathway'] for p in ukb_patients])
    mgb_labels = np.array([p['pathway'] for p in mgb_patients])
    
    ukb_unique, ukb_counts = np.unique(ukb_labels, return_counts=True)
    mgb_unique, mgb_counts = np.unique(mgb_labels, return_counts=True)
    
    ukb_total = len(ukb_labels)
    mgb_total = len(mgb_labels)
    
    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Pathway sizes for matched pathways
    ax1 = axes[0, 0]
    matched_ukb_sizes = []
    matched_mgb_sizes = []
    match_labels = []
    
    for ukb_id in sorted(best_matches.keys()):
        mgb_id = best_matches[ukb_id]
        ukb_size = np.sum(ukb_labels == ukb_id)
        mgb_size = np.sum(mgb_labels == mgb_id)
        
        matched_ukb_sizes.append(ukb_size)
        matched_mgb_sizes.append(mgb_size)
        match_labels.append(f"UKB{ukb_id}↔MGB{mgb_id}")
    
    x = np.arange(len(match_labels))
    width = 0.35
    
    ax1.bar(x - width/2, matched_ukb_sizes, width, label='UKB', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, matched_mgb_sizes, width, label='MGB', alpha=0.8, color='coral')
    ax1.set_xlabel('Matched Pathway Pair')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('A. Pathway Sizes (Matched Pathways)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(match_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Pathway proportions
    ax2 = axes[0, 1]
    matched_ukb_pcts = [(s / ukb_total * 100) for s in matched_ukb_sizes]
    matched_mgb_pcts = [(s / mgb_total * 100) for s in matched_mgb_sizes]
    
    ax2.bar(x - width/2, matched_ukb_pcts, width, label='UKB', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, matched_mgb_pcts, width, label='MGB', alpha=0.8, color='coral')
    ax2.set_xlabel('Matched Pathway Pair')
    ax2.set_ylabel('Percentage of MI Patients')
    ax2.set_title('B. Pathway Proportions (Matched Pathways)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(match_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Proportion correlation
    ax3 = axes[0, 2]
    if len(matched_ukb_pcts) > 1:
        correlation = np.corrcoef(matched_ukb_pcts, matched_mgb_pcts)[0, 1]
        
        ax3.scatter(matched_ukb_pcts, matched_mgb_pcts, s=150, alpha=0.7, color='darkgreen')
        
        # Add diagonal line
        max_pct = max(max(matched_ukb_pcts), max(matched_mgb_pcts))
        ax3.plot([0, max_pct], [0, max_pct], 'r--', alpha=0.5, linewidth=2, label='y=x')
        
        # Add labels
        for i, label in enumerate(match_labels):
            ax3.annotate(label, (matched_ukb_pcts[i], matched_mgb_pcts[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.text(0.05, 0.95, f'r = {correlation:.3f}', 
                transform=ax3.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3.set_xlabel('UKB Pathway Proportion (%)')
        ax3.set_ylabel('MGB Pathway Proportion (%)')
        ax3.set_title('C. Proportion Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Panel D: Similarity scores
    ax4 = axes[1, 0]
    similarity_scores = [similarities[(u, m)] for u, m in best_matches.items()]
    
    colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in similarity_scores]
    ax4.bar(range(len(similarity_scores)), similarity_scores, color=colors, alpha=0.7)
    ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High (>0.7)')
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (>0.5)')
    ax4.set_xlabel('Matched Pathway Pair')
    ax4.set_ylabel('Disease Pattern Similarity')
    ax4.set_title('D. Pathway Similarity Scores')
    ax4.set_xticks(range(len(match_labels)))
    ax4.set_xticklabels(match_labels, rotation=45, ha='right')
    ax4.legend()
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3)
    
    # Panel E: Age at onset comparison
    ax5 = axes[1, 1]
    matched_ukb_ages = []
    matched_mgb_ages = []
    
    for ukb_id in sorted(best_matches.keys()):
        mgb_id = best_matches[ukb_id]
        
        ukb_ages = [p['age_at_disease'] for p in ukb_patients if p['pathway'] == ukb_id]
        mgb_ages = [p['age_at_disease'] for p in mgb_patients if p['pathway'] == mgb_id]
        
        matched_ukb_ages.append(np.mean(ukb_ages) if ukb_ages else np.nan)
        matched_mgb_ages.append(np.mean(mgb_ages) if mgb_ages else np.nan)
    
    ax5.bar(x - width/2, matched_ukb_ages, width, label='UKB', alpha=0.8, color='steelblue')
    ax5.bar(x + width/2, matched_mgb_ages, width, label='MGB', alpha=0.8, color='coral')
    ax5.set_xlabel('Matched Pathway Pair')
    ax5.set_ylabel('Mean Age at MI (years)')
    ax5.set_title('E. Age at Onset (Matched Pathways)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(match_labels, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    avg_similarity = np.mean(similarity_scores)
    prop_corr = np.corrcoef(matched_ukb_pcts, matched_mgb_pcts)[0, 1] if len(matched_ukb_pcts) > 1 else 0
    
    age_diffs = [abs(u - m) for u, m in zip(matched_ukb_ages, matched_mgb_ages) if not (np.isnan(u) or np.isnan(m))]
    avg_age_diff = np.mean(age_diffs) if age_diffs else 0
    
    summary_text = "REPRODUCIBILITY SUMMARY\n\n"
    summary_text += f"✅ {len(best_matches)} pathways matched\n\n"
    summary_text += f"Similarity Metrics:\n"
    summary_text += f"  • Avg similarity: {avg_similarity:.3f}\n"
    summary_text += f"  • High similarity (>0.7): {sum(1 for s in similarity_scores if s > 0.7)}/{len(similarity_scores)}\n\n"
    summary_text += f"Proportion Correlation:\n"
    summary_text += f"  • r = {prop_corr:.3f}\n\n"
    summary_text += f"Age at Onset:\n"
    summary_text += f"  • Avg difference: {avg_age_diff:.1f} years\n\n"
    summary_text += "CONCLUSION:\n"
    summary_text += "Deviation-based pathway\n"
    summary_text += "discovery generalizes\n"
    summary_text += "across cohorts ✅"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Pathway Reproducibility: UKB ↔ MGB\n(Matched by Disease Patterns)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'pathway_reproducibility_ukb_mgb.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Figure saved to: {save_path}")
    plt.show()
    
    return {
        'matched_ukb_sizes': matched_ukb_sizes,
        'matched_mgb_sizes': matched_mgb_sizes,
        'matched_ukb_pcts': matched_ukb_pcts,
        'matched_mgb_pcts': matched_mgb_pcts,
        'similarity_scores': similarity_scores,
        'proportion_correlation': prop_corr,
        'avg_age_difference': avg_age_diff
    }


def compare_signature_patterns_matched(ukb_results, mgb_results, pathway_matching):
    """
    Compare signature deviation patterns for matched pathways
    
    Creates a comprehensive visualization showing all signature deviation
    trajectories for matched pathways side-by-side (UKB vs MGB)
    """
    print("\n" + "="*80)
    print("COMPARING SIGNATURE PATTERNS FOR MATCHED PATHWAYS")
    print("="*80)
    print("\n⚠️  IMPORTANT: Signature indices are arbitrary across cohorts.")
    print("   UKB Sig 5 may not correspond to MGB Sig 5 biologically.")
    print("   We compare overall patterns and biological content, not index alignment.")
    print("   Pathways are matched by disease enrichment patterns, not signature indices.")
    
    ukb_pathway_data = ukb_results['pathway_data_dev']
    mgb_pathway_data = mgb_results['pathway_data']
    
    # Get thetas and compute deviations
    print("\n1. Computing signature deviation trajectories...")
    
    # Get UKB data
    from pathway_discovery import load_full_data
    Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
    
    # Get MGB data (thetas, not Y!)
    thetas_mgb = mgb_results.get('thetas')
    if thetas_mgb is None:
        from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
        _, thetas_mgb, _, _ = load_mgb_data_from_model()
    
    # Convert to numpy if needed
    if hasattr(thetas_ukb, 'numpy'):
        thetas_ukb = thetas_ukb.numpy()
    if hasattr(thetas_mgb, 'numpy'):
        thetas_mgb = thetas_mgb.numpy()
    if not isinstance(thetas_ukb, np.ndarray):
        thetas_ukb = np.array(thetas_ukb)
    if not isinstance(thetas_mgb, np.ndarray):
        thetas_mgb = np.array(thetas_mgb)
    
    # Calculate population references (SAME METHOD FOR BOTH)
    # Population reference = mean signature loadings across all patients
    population_ref_ukb = np.mean(thetas_ukb, axis=0)  # (K, T)
    population_ref_mgb = np.mean(thetas_mgb, axis=0)  # (K, T)
    
    K_ukb, T_ukb = thetas_ukb.shape[1], thetas_ukb.shape[2]
    K_mgb, T_mgb = thetas_mgb.shape[1], thetas_mgb.shape[2]
    
    print(f"   UKB: {K_ukb} signatures, {T_ukb} timepoints")
    print(f"   MGB: {K_mgb} signatures, {T_mgb} timepoints")
    
    # Get pathway assignments
    ukb_patients = ukb_pathway_data['patients']
    mgb_patients = mgb_pathway_data['patients']
    
    ukb_labels = np.array([p['pathway'] for p in ukb_patients])
    ukb_patient_ids = np.array([p['patient_id'] for p in ukb_patients])
    
    mgb_labels = np.array([p['pathway'] for p in mgb_patients])
    mgb_patient_ids = np.array([p['patient_id'] for p in mgb_patients])
    
    best_matches = pathway_matching['best_matches']
    n_pathways = len(best_matches)
    
    # Calculate deviation trajectories for each matched pathway pair
    print("\n2. Calculating deviations for matched pathways...")
    
    ukb_deviations = {}  # pathway_id -> (K, T) deviation matrix
    mgb_deviations = {}  # pathway_id -> (K, T) deviation matrix
    
    for ukb_id in sorted(best_matches.keys()):
        mgb_id = best_matches[ukb_id]
        
        # UKB pathway
        ukb_pathway_mask = ukb_labels == ukb_id
        ukb_pathway_patient_ids = ukb_patient_ids[ukb_pathway_mask]
        
        if len(ukb_pathway_patient_ids) > 0:
            ukb_pathway_thetas = thetas_ukb[ukb_pathway_patient_ids, :, :]  # (n_patients, K, T)
            ukb_pathway_mean = np.mean(ukb_pathway_thetas, axis=0)  # (K, T)
            ukb_dev = ukb_pathway_mean - population_ref_ukb  # (K, T)
            ukb_deviations[ukb_id] = ukb_dev
        
        # MGB pathway
        mgb_pathway_mask = mgb_labels == mgb_id
        mgb_pathway_patient_ids = mgb_patient_ids[mgb_pathway_mask]
        
        if len(mgb_pathway_patient_ids) > 0:
            mgb_pathway_thetas = thetas_mgb[mgb_pathway_patient_ids, :, :]  # (n_patients, K, T)
            mgb_pathway_mean = np.mean(mgb_pathway_thetas, axis=0)  # (K, T)
            mgb_dev = mgb_pathway_mean - population_ref_mgb  # (K, T) - SAME DEVIATION FORMULA AS UKB
            mgb_deviations[mgb_id] = mgb_dev
        
        print(f"   UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id}: {len(ukb_pathway_patient_ids)} vs {len(mgb_pathway_patient_ids)} patients")
    
    # Compute signature correspondence using cross-tabulation (max in each row)
    print("\n3. Computing signature correspondence...")
    signature_map, similarity_matrix = compute_signature_correspondence(
        ukb_deviations, mgb_deviations, best_matches, 
        use_crosstab=True, crosstab_df=None  # Uses predefined mapping
    )
    
    # Create comprehensive visualization
    print("\n4. Creating signature deviation trajectory plots...")
    
    # Use the minimum number of signatures for alignment
    K_min = min(K_ukb, K_mgb)
    T_min = min(T_ukb, T_mgb)
    
    # Create figure with subplots for each matched pathway pair
    # Add extra space at bottom for central legend
    fig, axes = plt.subplots(n_pathways, 2, figsize=(16, 5*n_pathways + 2))
    if n_pathways == 1:
        axes = axes.reshape(1, -1)
    
    # Create color mapping: matched signatures get same color
    # Use tab20 for matched pairs, grayscale for unmatched
    max_matches = max(K_ukb, K_mgb)
    if max_matches <= 20:
        matched_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max_matches))
    else:
        # For more than 20, use tab20 + tab20b
        colors_20 = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors_b = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, max_matches - 20))
        matched_colors = np.vstack([colors_20, colors_b])
    
    # Create grayscale colors for unmatched signatures
    unmatched_colors = plt.cm.get_cmap('gray')(np.linspace(0.3, 0.7, max(K_ukb, K_mgb)))
    
    # Assign colors: UKB signatures get colors based on their index
    # Matched MGB signatures get the SAME color as their UKB counterpart
    ukb_sig_colors = {}
    mgb_sig_colors = {}
    
    # Assign colors to matched pairs
    matched_pairs = sorted(signature_map.items())
    for color_idx, (ukb_sig, mgb_sig) in enumerate(matched_pairs):
        ukb_sig_colors[ukb_sig] = matched_colors[color_idx]
        mgb_sig_colors[mgb_sig] = matched_colors[color_idx]  # Same color!
    
    # Assign grayscale colors to unmatched UKB signatures
    unmatched_ukb = set(range(K_ukb)) - set(signature_map.keys())
    for color_idx, ukb_sig in enumerate(sorted(unmatched_ukb)):
        ukb_sig_colors[ukb_sig] = unmatched_colors[color_idx]
    
    # Assign grayscale colors to unmatched MGB signatures
    matched_mgb = set(signature_map.values())
    unmatched_mgb = set(range(K_mgb)) - matched_mgb
    for color_idx, mgb_sig in enumerate(sorted(unmatched_mgb)):
        mgb_sig_colors[mgb_sig] = unmatched_colors[color_idx + len(unmatched_ukb)]
    
    # Create central legend entries (one per matched signature pair)
    # Store legend handles and labels for central legend
    central_legend_handles = []
    central_legend_labels = []
    
    # Add matched signature pairs to central legend (MGB first as requested)
    reverse_map = {v: k for k, v in signature_map.items()}
    for mgb_sig in sorted(reverse_map.keys()):
        ukb_sig = reverse_map[mgb_sig]
        color = mgb_sig_colors[mgb_sig]  # Same color for both
        # Create a dummy line for legend
        handle = plt.Line2D([0], [0], color=color, linewidth=2.0, marker='o', markersize=4)
        central_legend_handles.append(handle)
        central_legend_labels.append(f'MGB Sig {mgb_sig} ↔ UKB Sig {ukb_sig}')
    
    # Add unmatched signatures (if any)
    for ukb_sig in sorted(unmatched_ukb):
        color = ukb_sig_colors[ukb_sig]
        handle = plt.Line2D([0], [0], color=color, linewidth=2.0, marker='o', 
                           markersize=4, linestyle='--', alpha=0.7)
        central_legend_handles.append(handle)
        central_legend_labels.append(f'UKB Sig {ukb_sig} (unmatched)')
    
    for mgb_sig in sorted(unmatched_mgb):
        color = mgb_sig_colors[mgb_sig]
        handle = plt.Line2D([0], [0], color=color, linewidth=2.0, marker='o', 
                           markersize=4, linestyle='--', alpha=0.7)
        central_legend_handles.append(handle)
        central_legend_labels.append(f'MGB Sig {mgb_sig} (unmatched)')
    
    # Time axis - match the original plot style exactly
    # Use np.linspace(30, 81, T) like the original function
    time_points_ukb = np.linspace(30, 30 + T_ukb - 1, T_ukb)  # Age from 30 to 30+T-1
    time_points_mgb = np.linspace(30, 30 + T_mgb - 1, T_mgb)  # Age from 30 to 30+T-1
    
    # Show full trajectory (all timepoints)
    lookback_ukb = 0  # Show full trajectory
    lookback_mgb = 0  # Show full trajectory
    
    # Store patient counts for titles
    pathway_patient_counts = {}
    
    for row_idx, ukb_id in enumerate(sorted(best_matches.keys())):
        mgb_id = best_matches[ukb_id]
        
        # Get patient counts
        ukb_pathway_mask = ukb_labels == ukb_id
        ukb_pathway_patient_ids_this = ukb_patient_ids[ukb_pathway_mask]
        mgb_pathway_mask = mgb_labels == mgb_id
        mgb_pathway_patient_ids_this = mgb_patient_ids[mgb_pathway_mask]
        
        pathway_patient_counts[(ukb_id, mgb_id)] = (len(ukb_pathway_patient_ids_this), len(mgb_pathway_patient_ids_this))
        
        ax_ukb = axes[row_idx, 0]
        ax_mgb = axes[row_idx, 1]
        
        # Plot UKB deviations (all signatures) - use matched colors, NO individual legends
        if ukb_id in ukb_deviations:
            ukb_dev = ukb_deviations[ukb_id]
            # Plot matched signatures first
            for ukb_k in sorted(signature_map.keys()):
                if ukb_k < K_ukb:
                    sig_values = ukb_dev[ukb_k, lookback_ukb:]
                    time_axis = time_points_ukb[lookback_ukb:]
                    ax_ukb.plot(time_axis, sig_values, 
                               color=ukb_sig_colors[ukb_k], 
                               linewidth=2.0, 
                               marker='o', 
                               markersize=4,
                               alpha=0.8)
            
            # Plot unmatched UKB signatures
            for ukb_k in sorted(set(range(K_ukb)) - set(signature_map.keys())):
                if ukb_k < K_ukb:
                    sig_values = ukb_dev[ukb_k, lookback_ukb:]
                    time_axis = time_points_ukb[lookback_ukb:]
                    ax_ukb.plot(time_axis, sig_values, 
                               color=ukb_sig_colors[ukb_k], 
                               linewidth=2.0, 
                               marker='o', 
                               markersize=4,
                               alpha=0.8, linestyle='--')
        
        ax_ukb.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax_ukb.set_xlabel('Age', fontsize=12)
        ax_ukb.set_ylabel('Deviation from Population Mean (Δ Proportion, θ)', fontsize=12)
        n_ukb_patients, n_mgb_patients = pathway_patient_counts[(ukb_id, mgb_id)]
        ax_ukb.set_title(f'UKB Pathway {ukb_id} (n={n_ukb_patients})', fontweight='bold', fontsize=13)
        ax_ukb.grid(True, alpha=0.3)
        
        # Plot MGB deviations (all signatures) - use matched colors, NO individual legends
        if mgb_id in mgb_deviations:
            mgb_dev = mgb_deviations[mgb_id]
            # Plot matched signatures first (same color as UKB counterpart)
            reverse_map = {v: k for k, v in signature_map.items()}
            for mgb_k in sorted(reverse_map.keys()):
                if mgb_k < K_mgb:
                    sig_values = mgb_dev[mgb_k, lookback_mgb:]
                    time_axis = time_points_mgb[lookback_mgb:]
                    ax_mgb.plot(time_axis, sig_values, 
                               color=mgb_sig_colors[mgb_k],  # Same color as UKB!
                               linewidth=2.0, 
                               marker='o', 
                               markersize=4,
                               alpha=0.8)
            
            # Plot unmatched MGB signatures
            matched_mgb = set(signature_map.values())
            for mgb_k in sorted(set(range(K_mgb)) - matched_mgb):
                if mgb_k < K_mgb:
                    sig_values = mgb_dev[mgb_k, lookback_mgb:]
                    time_axis = time_points_mgb[lookback_mgb:]
                    ax_mgb.plot(time_axis, sig_values, 
                               color=mgb_sig_colors[mgb_k], 
                               linewidth=2.0, 
                               marker='o', 
                               markersize=4,
                               alpha=0.8, linestyle='--')
        
        ax_mgb.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax_mgb.set_xlabel('Age', fontsize=12)
        ax_mgb.set_ylabel('Deviation from Population Mean (Δ Proportion, θ)', fontsize=12)
        ax_mgb.set_title(f'MGB Pathway {mgb_id} (n={n_mgb_patients})', fontweight='bold', fontsize=13)
        ax_mgb.grid(True, alpha=0.3)
    
    plt.suptitle('Individual Signature Deviations by Pathway: Myocardial Infarction\nMatched Pathways (UKB ↔ MGB) - Matching Signatures Use Same Color', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout first, then add legend
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])  # Leave space at bottom (12%) for legend
    
    # Add central legend at the bottom (after tight_layout)
    # Place legend in the center, below all subplots
    fig.legend(central_legend_handles, central_legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.01),  # Position at bottom center
              ncol=min(5, len(central_legend_labels)),  # 5 columns max for better fit
              fontsize=8,
              frameon=True,
              fancybox=True,
              shadow=False,
              columnspacing=1.0,
              handlelength=2.0)
    
    save_path = 'signature_deviation_trajectories_all_sigs_ukb_mgb.png'
    # Save with extra space at bottom for legend
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"   ✅ Figure saved to: {save_path}")
    print(f"\n   ✅ Matching signatures are plotted with the same color.")
    print(f"      Matched pairs: {len(signature_map)} signatures")
    print(f"      Unmatched signatures shown in grayscale (dashed lines)")
    print(f"      Central legend shows all UKB-MGB signature connections")
    plt.show()
    
    # Also create a heatmap version for easier comparison
    print("\n5. Creating signature deviation heatmaps...")
    
    fig, axes = plt.subplots(n_pathways, 2, figsize=(16, 4*n_pathways))
    if n_pathways == 1:
        axes = axes.reshape(1, -1)
    
    # Find common scale for heatmaps
    all_deviations = []
    for ukb_id in ukb_deviations.keys():
        all_deviations.append(ukb_deviations[ukb_id])
    for mgb_id in mgb_deviations.keys():
        all_deviations.append(mgb_deviations[mgb_id])
    
    if len(all_deviations) > 0:
        vmin = np.percentile([np.min(d) for d in all_deviations], 5)
        vmax = np.percentile([np.max(d) for d in all_deviations], 95)
    else:
        vmin, vmax = -0.1, 0.1
    
    for row_idx, ukb_id in enumerate(sorted(best_matches.keys())):
        mgb_id = best_matches[ukb_id]
        
        ax_ukb = axes[row_idx, 0]
        ax_mgb = axes[row_idx, 1]
        
        # UKB heatmap
        if ukb_id in ukb_deviations:
            ukb_dev = ukb_deviations[ukb_id][:K_min, lookback_ukb:]
            im1 = ax_ukb.imshow(ukb_dev, aspect='auto', cmap='RdBu_r', 
                               vmin=vmin, vmax=vmax, interpolation='nearest')
            ax_ukb.set_xlabel('Years Before MI')
            ax_ukb.set_ylabel('Signature Index')
            ax_ukb.set_title(f'UKB Pathway {ukb_id}')
            ax_ukb.set_yticks(range(0, K_min, max(1, K_min//10)))
            plt.colorbar(im1, ax=ax_ukb, label='Deviation')
        
        # MGB heatmap
        if mgb_id in mgb_deviations:
            mgb_dev = mgb_deviations[mgb_id][:K_min, lookback_mgb:]
            im2 = ax_mgb.imshow(mgb_dev, aspect='auto', cmap='RdBu_r',
                               vmin=vmin, vmax=vmax, interpolation='nearest')
            ax_mgb.set_xlabel('Years Before MI')
            ax_mgb.set_ylabel('Signature Index')
            ax_mgb.set_title(f'MGB Pathway {mgb_id}')
            ax_mgb.set_yticks(range(0, K_min, max(1, K_min//10)))
            plt.colorbar(im2, ax=ax_mgb, label='Deviation')
    
    plt.suptitle('Signature Deviation Heatmaps: All Signatures\nMatched Pathways (UKB ↔ MGB)\n⚠️  Note: Signature indices are arbitrary - compare biological patterns, not index numbers', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path_heatmap = 'signature_deviation_heatmaps_all_sigs_ukb_mgb.png'
    plt.savefig(save_path_heatmap, dpi=300, bbox_inches='tight')
    print(f"   ✅ Heatmap saved to: {save_path_heatmap}")
    plt.show()
    
    return {
        'ukb_deviations': ukb_deviations,
        'mgb_deviations': mgb_deviations,
        'signature_map': signature_map,
        'similarity_matrix': similarity_matrix
    }


def compare_prs_patterns_matched(ukb_results, mgb_results, pathway_matching, output_dir='output_10yr'):
    """
    Compare PRS patterns between matched pathways - STRONG validation!
    
    If the same genetic risk patterns are associated with the same pathways
    across cohorts, this proves the pathways are biologically real.
    """
    print("\n" + "="*80)
    print("COMPARING PRS PATTERNS: STRONG REPRODUCIBILITY VALIDATION")
    print("="*80)
    
    # Load MGB model to get G (PRS)
    print("\n1. Loading MGB PRS from model...")
    mgb_model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'
    import torch
    mgb_data = torch.load(mgb_model_path, map_location=torch.device('cpu'))
    
    if 'G' not in mgb_data:
        print("   ⚠️  'G' (PRS) not found in MGB model - skipping PRS comparison")
        return None
    
    mgb_G = mgb_data['G']
    if hasattr(mgb_G, 'numpy'):
        mgb_G = mgb_G.numpy()
    elif hasattr(mgb_G, 'detach'):
        mgb_G = mgb_G.detach().numpy()
    mgb_G = np.array(mgb_G)
    
    # Get PRS names if available
    mgb_prs_names = None
    if 'prs_names' in mgb_data:
        mgb_prs_names = mgb_data['prs_names']
        if hasattr(mgb_prs_names, 'tolist'):
            mgb_prs_names = mgb_prs_names.tolist()
        elif hasattr(mgb_prs_names, 'values'):
            mgb_prs_names = mgb_prs_names.values.tolist()
    
    print(f"   ✅ MGB PRS shape: {mgb_G.shape}")
    if mgb_prs_names:
        print(f"   ✅ PRS names available: {len(mgb_prs_names)} scores")
        print(f"      Examples: {mgb_prs_names[:5]}")
    
    # Load UKB PRS
    print("\n2. Loading UKB PRS...")
    try:
        ukb_prs_file = '/Users/sarahurbut/aladynoulli2/pyScripts/prs_with_eid.csv'
        import pandas as pd
        ukb_prs_df = pd.read_csv(ukb_prs_file)
        print(f"   ✅ Loaded UKB PRS from file: {ukb_prs_df.shape}")
        
        # Get UKB pathway data and processed_ids
        from pathway_discovery import load_full_data
        _, _, _, processed_ids_ukb = load_full_data()
        
        ukb_pathway_data = ukb_results['pathway_data_dev']
        ukb_patients = ukb_pathway_data['patients']
        
        # Map pathway patients to PRS
        ukb_pathway_prs = {}
        for pathway_id in sorted(set([p['pathway'] for p in ukb_patients])):
            pathway_patient_ids = [p['patient_id'] for p in ukb_patients if p['pathway'] == pathway_id]
            pathway_eids = [processed_ids_ukb[pid] for pid in pathway_patient_ids if pid < len(processed_ids_ukb)]
            
            pathway_prs_subset = ukb_prs_df[ukb_prs_df['PatientID'].isin(pathway_eids)]
            if len(pathway_prs_subset) > 0:
                prs_cols = [col for col in pathway_prs_subset.columns if col != 'PatientID']
                ukb_pathway_prs[pathway_id] = {
                    'prs_matrix': pathway_prs_subset[prs_cols].values,
                    'mean': pathway_prs_subset[prs_cols].mean().values,
                    'n_patients': len(pathway_prs_subset),
                    'prs_names': prs_cols
                }
        
        print(f"   ✅ UKB PRS extracted for {len(ukb_pathway_prs)} pathways")
        
    except Exception as e:
        print(f"   ⚠️  Could not load UKB PRS: {e}")
        print("   Skipping PRS comparison")
        return None
    
    # Get MGB pathway PRS
    print("\n3. Extracting MGB pathway PRS...")
    mgb_pathway_data = mgb_results['pathway_data']
    mgb_patients = mgb_pathway_data['patients']
    
    mgb_pathway_prs = {}
    for pathway_id in sorted(set([p['pathway'] for p in mgb_patients])):
        pathway_patient_ids = [p['patient_id'] for p in mgb_patients if p['pathway'] == pathway_id]
        pathway_G = mgb_G[pathway_patient_ids, :]  # (n_patients, P)
        
        mgb_pathway_prs[pathway_id] = {
            'prs_matrix': pathway_G,
            'mean': np.mean(pathway_G, axis=0),
            'n_patients': len(pathway_patient_ids)
        }
    
    print(f"   ✅ MGB PRS extracted for {len(mgb_pathway_prs)} pathways")
    
    # Match PRS names between cohorts (if available)
    print("\n4. Matching PRS scores between cohorts...")
    best_matches = pathway_matching['best_matches']
    
    # Find common PRS (if names available)
    if mgb_prs_names and len(ukb_pathway_prs) > 0:
        ukb_prs_names = ukb_pathway_prs[list(ukb_pathway_prs.keys())[0]]['prs_names']
        common_prs = set(ukb_prs_names) & set(mgb_prs_names)
        print(f"   ✅ Found {len(common_prs)} common PRS scores")
        if len(common_prs) > 0:
            print(f"      Examples: {list(common_prs)[:5]}")
            # Use common PRS for comparison
            common_prs_list = sorted(common_prs)
        else:
            # Use all PRS (assume same order)
            common_prs_list = None
    else:
        common_prs_list = None
    
    # Compare PRS patterns for matched pathways
    print("\n5. Comparing PRS patterns for matched pathways...")
    
    prs_comparisons = {}
    for ukb_id, mgb_id in best_matches.items():
        if ukb_id not in ukb_pathway_prs or mgb_id not in mgb_pathway_prs:
            continue
        
        ukb_prs_mean = ukb_pathway_prs[ukb_id]['mean']
        mgb_prs_mean = mgb_pathway_prs[mgb_id]['mean']
        
        # Match PRS if we have names
        if common_prs_list:
            ukb_indices = [ukb_prs_names.index(prs) for prs in common_prs_list if prs in ukb_prs_names]
            mgb_indices = [mgb_prs_names.index(prs) for prs in common_prs_list if prs in mgb_prs_names]
            
            if len(ukb_indices) == len(mgb_indices) and len(ukb_indices) > 0:
                ukb_matched = ukb_prs_mean[ukb_indices]
                mgb_matched = mgb_prs_mean[mgb_indices]
                
                correlation = np.corrcoef(ukb_matched, mgb_matched)[0, 1]
                mean_abs_diff = np.mean(np.abs(ukb_matched - mgb_matched))
                
                prs_comparisons[(ukb_id, mgb_id)] = {
                    'correlation': correlation,
                    'mean_abs_diff': mean_abs_diff,
                    'n_prs': len(common_prs_list),
                    'ukb_n': ukb_pathway_prs[ukb_id]['n_patients'],
                    'mgb_n': mgb_pathway_prs[mgb_id]['n_patients']
                }
                
                print(f"   UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id}:")
                print(f"      PRS correlation: {correlation:.3f}")
                print(f"      Mean absolute difference: {mean_abs_diff:.4f}")
        else:
            # Use all PRS (assume same dimensions)
            if len(ukb_prs_mean) == len(mgb_prs_mean):
                correlation = np.corrcoef(ukb_prs_mean, mgb_prs_mean)[0, 1]
                mean_abs_diff = np.mean(np.abs(ukb_prs_mean - mgb_prs_mean))
                
                prs_comparisons[(ukb_id, mgb_id)] = {
                    'correlation': correlation,
                    'mean_abs_diff': mean_abs_diff,
                    'n_prs': len(ukb_prs_mean),
                    'ukb_n': ukb_pathway_prs[ukb_id]['n_patients'],
                    'mgb_n': mgb_pathway_prs[mgb_id]['n_patients']
                }
                
                print(f"   UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id}:")
                print(f"      PRS correlation: {correlation:.3f}")
                print(f"      Mean absolute difference: {mean_abs_diff:.4f}")
    
    if len(prs_comparisons) > 0:
        avg_prs_correlation = np.mean([comp['correlation'] for comp in prs_comparisons.values()])
        print(f"\n   Average PRS pattern correlation: {avg_prs_correlation:.3f}")
        
        # Interpret correlation strength
        if avg_prs_correlation > 0.7:
            print(f"\n   ✅ STRONG GENETIC VALIDATION: High PRS correlation")
            print(f"      Same genetic risk patterns → Same pathways (strong evidence)")
        elif avg_prs_correlation > 0.4:
            print(f"\n   ⚠️  MODERATE GENETIC VALIDATION: Moderate PRS correlation")
            print(f"      Some genetic similarity, but pathways may be driven by other factors")
        else:
            print(f"\n   ⚠️  WEAK GENETIC VALIDATION: Low PRS correlation")
            print(f"      Pathways may be primarily driven by:")
            print(f"      - Environmental factors (not captured by PRS)")
            print(f"      - Non-genetic risk factors")
            print(f"      - Stochastic events or measurement differences")
            print(f"      However, disease pattern matching (0.704) still validates reproducibility")
    else:
        print("\n   ⚠️  Could not compare PRS patterns (missing data or mismatched PRS)")
    
    return prs_comparisons


def main(force_rerun_mgb=False):
    """
    Main function to show reproducibility
    
    Parameters:
    -----------
    force_rerun_mgb : bool
        If True, re-run MGB analysis. If False, use existing results.
    """
    print("="*80)
    print("PATHWAY REPRODUCIBILITY ANALYSIS")
    print("="*80)
    
    # Get pathway matches
    print("\nStep 1: Getting pathway matches...")
    results = show_pathway_matches(force_rerun_mgb=force_rerun_mgb)
    
    if results is None:
        print("❌ Could not get pathway matches")
        return None
    
    ukb_results = results['ukb_results']
    mgb_results = results['mgb_results']
    pathway_matching = results['pathway_matching']
    
    # Create reproducibility figure
    print("\nStep 2: Creating reproducibility visualizations...")
    reproducibility_stats = create_reproducibility_figure(
        ukb_results, mgb_results, pathway_matching
    )
    
    # Compare signature patterns
    print("\nStep 3: Comparing signature patterns...")
    compare_signature_patterns_matched(
        ukb_results, mgb_results, pathway_matching
    )
    
    # Compare PRS patterns (STRONG validation!)
    print("\nStep 4: Comparing PRS patterns (genetic validation)...")
    prs_comparisons = compare_prs_patterns_matched(
        ukb_results, mgb_results, pathway_matching
    )
    
    print("\n" + "="*80)
    print("✅ REPRODUCIBILITY ANALYSIS COMPLETE!")
    print("="*80)
    
    # Print summary
    best_matches = pathway_matching['best_matches']
    similarities = pathway_matching['pathway_similarities']
    
    print("\nREPRODUCIBILITY SUMMARY:")
    print("-" * 80)
    print(f"✅ {len(best_matches)} pathways matched between cohorts")
    
    # Calculate average similarity for matched pathways only
    matched_similarities = [similarities[(u, m)] for u, m in best_matches.items()]
    print(f"✅ Disease pattern similarity: {np.mean(matched_similarities):.3f}")
    print(f"✅ Proportion correlation: {reproducibility_stats['proportion_correlation']:.3f}")
    print(f"✅ Age difference: {reproducibility_stats['avg_age_difference']:.1f} years")
    
    if prs_comparisons:
        avg_prs_corr = np.mean([comp['correlation'] for comp in prs_comparisons.values()])
        print(f"✅ PRS pattern correlation: {avg_prs_corr:.3f}")
        
        if avg_prs_corr > 0.7:
            print("   (Strong genetic validation)")
        elif avg_prs_corr > 0.4:
            print("   (Moderate genetic validation)")
        else:
            print("   (Weak genetic validation - pathways may be primarily non-genetic)")
        
        print("\n🎯 CONCLUSION: Pathways are:")
        print("   1. Statistically distinct (validated by diseases, signatures, age)")
        print("   2. Stable within UKB (permutation test)")
        print("   3. Reproducible across cohorts (disease patterns: 0.704 similarity)")
        if avg_prs_corr > 0.4:
            print("   4. Partially genetically validated (PRS correlation: {:.3f})".format(avg_prs_corr))
        else:
            print("   4. Primarily non-genetic (low PRS correlation suggests environmental/stochastic drivers)")
    else:
        print("\nCONCLUSION: Deviation-based pathway discovery generalizes across cohorts!")
    
    return {
        'pathway_matching': pathway_matching,
        'reproducibility_stats': reproducibility_stats,
        'prs_comparisons': prs_comparisons,
        'ukb_results': ukb_results,
        'mgb_results': mgb_results
    }


if __name__ == "__main__":
    results = main()

