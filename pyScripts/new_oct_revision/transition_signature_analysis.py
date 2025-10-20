"""
Transition-Based Signature Analysis

This script analyzes signature patterns for specific disease transitions,
addressing the "patient morphs" narrative by looking at how different
diseases lead to the same outcome through different biological pathways.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import torch

def find_disease_transitions(Y, disease_names, target_disease_name, 
                           transition_diseases, processed_ids=None):
    """
    Find patients who had specific disease transitions before target disease
    
    Parameters:
    -----------
    Y : torch.Tensor
        Binary disease matrix (patients x diseases x time)
    disease_names : list
        List of disease names
    target_disease_name : str
        Name of target disease (e.g., "myocardial infarction")
    transition_diseases : list
        List of diseases to look for before target (e.g., ["rheumatoid arthritis", "diabetes"])
    processed_ids : array, optional
        Patient IDs for mapping
    
    Returns:
    --------
    dict : Dictionary with transition groups and their patients
    """
    print(f"=== FINDING DISEASE TRANSITIONS TO {target_disease_name.upper()} ===")
    
    # Find target disease index
    target_idx = None
    for i, name in enumerate(disease_names):
        if target_disease_name.lower() in name.lower():
            target_idx = i
            break
    
    if target_idx is None:
        print(f"Could not find target disease: {target_disease_name}")
        return None
    
    print(f"Target disease: {disease_names[target_idx]} (index {target_idx})")
    
    # Find transition disease indices
    transition_indices = {}
    for transition_disease in transition_diseases:
        for i, name in enumerate(disease_names):
            if transition_disease.lower() in name.lower():
                transition_indices[transition_disease] = i
                print(f"Found transition disease: {name} (index {i})")
                break
        if transition_disease not in transition_indices:
            print(f"⚠️  Could not find transition disease: {transition_disease}")
    
    # Find patients with target disease
    target_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, target_idx, :].sum() > 0:
            # Find age at first occurrence of target disease
            first_occurrence = torch.where(Y[patient_id, target_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_target = first_occurrence.min().item() + 30
                target_patients.append({
                    'patient_id': patient_id,
                    'age_at_target': age_at_target
                })
    
    print(f"Found {len(target_patients)} patients with {target_disease_name}")
    
    # For each patient, check which transition diseases they had BEFORE target
    transition_groups = {disease: [] for disease in transition_diseases}
    transition_groups['no_transition'] = []  # Patients with target but no transition diseases
    
    for patient_info in target_patients:
        patient_id = patient_info['patient_id']
        age_at_target = patient_info['age_at_target']
        
        # Check each transition disease
        has_transition = False
        for transition_disease, transition_idx in transition_indices.items():
            # Check if patient had this transition disease BEFORE target
            target_time_idx = age_at_target - 30
            if target_time_idx > 0 and Y[patient_id, transition_idx, :target_time_idx].sum() > 0:
                # Find age at transition disease
                transition_occurrence = torch.where(Y[patient_id, transition_idx, :target_time_idx] > 0)[0]
                if len(transition_occurrence) > 0:
                    age_at_transition = transition_occurrence.min().item() + 30
                    
                    transition_groups[transition_disease].append({
                        'patient_id': patient_id,
                        'age_at_target': age_at_target,
                        'age_at_transition': age_at_transition,
                        'transition_disease': transition_disease
                    })
                    has_transition = True
        
        # If no transition diseases, add to no_transition group
        if not has_transition:
            transition_groups['no_transition'].append({
                'patient_id': patient_id,
                'age_at_target': age_at_target,
                'age_at_transition': None,
                'transition_disease': None
            })
    
    # Print summary
    print(f"\n=== TRANSITION GROUP SUMMARY ===")
    for group_name, patients in transition_groups.items():
        print(f"{group_name}: {len(patients)} patients")
    
    return {
        'target_disease': target_disease_name,
        'target_idx': target_idx,
        'transition_groups': transition_groups,
        'transition_indices': transition_indices
    }


def analyze_signature_patterns_by_transition(transition_data, thetas, disease_names):
    """
    Analyze signature patterns for each transition group - NORMALIZED BY REFERENCE
    """
    print(f"\n=== ANALYZING SIGNATURE PATTERNS BY TRANSITION (REFERENCE-NORMALIZED) ===")
    
    transition_groups = transition_data['transition_groups']
    
    # Calculate population reference (average signature trajectory across all patients)
    print("Computing population reference for normalization...")
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    print(f"Population reference shape: {population_reference.shape}")
    
    # Calculate signature trajectories for each transition group
    group_signature_analysis = {}
    
    for group_name, patients in transition_groups.items():
        if len(patients) == 0:
            continue
            
        print(f"\nAnalyzing {group_name} ({len(patients)} patients)...")
        
        # Get signature trajectories for patients in this group
        group_deviations = []
        valid_patients = []
        
        for patient_info in patients:
            patient_id = patient_info['patient_id']
            age_at_target = patient_info['age_at_target']
            
            # Get signature trajectory
            theta_patient = thetas[patient_id, :, :]  # Shape: (K, T)
            
            # Get pre-disease period (5 years before target)
            target_time_idx = age_at_target - 30
            lookback_idx = max(0, target_time_idx - 5)
            
            if target_time_idx > 5:
                # Get pre-disease trajectory for this patient
                pre_disease_traj = theta_patient[:, lookback_idx:target_time_idx]  # Shape: (K, 5)
                
                # Get corresponding population reference for same time window
                ref_traj = population_reference[:, lookback_idx:target_time_idx]  # Shape: (K, 5)
                
                # Calculate DEVIATION from reference (averaged over 5 years)
                deviation = pre_disease_traj - ref_traj  # Shape: (K, 5)
                avg_deviation = np.mean(deviation, axis=1)  # Average over time: Shape (K,)
                
                group_deviations.append(avg_deviation)
                valid_patients.append(patient_info)
        
        if group_deviations:
            group_deviations = np.array(group_deviations)  # Shape: (n_patients, K)
            
            # Calculate statistics on DEVIATIONS (not raw loadings)
            mean_deviations = np.mean(group_deviations, axis=0)
            std_deviations = np.std(group_deviations, axis=0)
            
            # Find most elevated signatures (by deviation from reference)
            signature_elevations = []
            for sig_idx in range(len(mean_deviations)):
                deviation = mean_deviations[sig_idx]
                signature_elevations.append({
                    'signature_idx': sig_idx,
                    'mean_deviation': deviation,
                    'std_deviation': std_deviations[sig_idx],
                    'elevation_score': abs(deviation)  # Use absolute deviation for ranking
                })
            
            # Sort by absolute deviation (most different from reference)
            signature_elevations.sort(key=lambda x: x['elevation_score'], reverse=True)
            
            group_signature_analysis[group_name] = {
                'n_patients': len(valid_patients),
                'mean_deviations': mean_deviations,
                'std_deviations': std_deviations,
                'top_signatures': signature_elevations[:10],  # Top 10
                'deviations': group_deviations
            }
            
            print(f"  Top 5 signatures (by deviation from reference):")
            for i, sig_info in enumerate(signature_elevations[:5]):
                deviation = sig_info['mean_deviation']
                direction = "↑" if deviation > 0 else "↓"
                print(f"    {i+1}. Signature {sig_info['signature_idx']}: {deviation:+.4f} ± {sig_info['std_deviation']:.4f} {direction}")
    
    return group_signature_analysis


def visualize_transition_signature_patterns(transition_data, signature_analysis):
    """
    Create visualizations comparing signature patterns across transition groups
    """
    print(f"\n=== CREATING TRANSITION SIGNATURE VISUALIZATIONS ===")
    
    transition_groups = transition_data['transition_groups']
    group_names = [name for name, patients in transition_groups.items() if len(patients) > 0]
    
    if len(group_names) < 2:
        print("Need at least 2 transition groups for comparison")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Signature Patterns by Disease Transition to {transition_data["target_disease"].title()}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Mean signature deviations by group
    ax1 = axes[0, 0]
    
    K = len(signature_analysis[group_names[0]]['mean_deviations'])
    x_pos = np.arange(K)
    width = 0.8 / len(group_names)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))
    
    for i, group_name in enumerate(group_names):
        if group_name in signature_analysis:
            mean_deviations = signature_analysis[group_name]['mean_deviations']
            ax1.bar(x_pos + i * width, mean_deviations, width, 
                   label=group_name, color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Signature Index')
    ax1.set_ylabel('Mean Deviation from Reference')
    ax1.set_title('Mean Signature Deviations by Transition Group')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top signatures heatmap
    ax2 = axes[0, 1]
    
    # Create matrix of top signatures for each group
    top_sigs_matrix = []
    group_labels = []
    
    for group_name in group_names:
        if group_name in signature_analysis:
            top_sigs = signature_analysis[group_name]['top_signatures'][:5]  # Top 5
            sig_values = [sig['mean_deviation'] for sig in top_sigs]
            top_sigs_matrix.append(sig_values)
            group_labels.append(group_name)
    
    if top_sigs_matrix:
        top_sigs_matrix = np.array(top_sigs_matrix)
        im = ax2.imshow(top_sigs_matrix, cmap='RdYlBu_r', aspect='auto')
        ax2.set_yticks(range(len(group_labels)))
        ax2.set_yticklabels(group_labels)
    ax2.set_xlabel('Rank (Top 5 Signatures)')
    ax2.set_title('Top Signature Deviations by Group')
    plt.colorbar(im, ax=ax2, label='Mean Deviation')
    
    # 3. Signature elevation comparison
    ax3 = axes[1, 0]
    
    # Compare specific signatures across groups
    signature_comparison = defaultdict(list)
    group_names_clean = []
    
    for group_name in group_names:
        if group_name in signature_analysis:
            group_names_clean.append(group_name)
            mean_deviations = signature_analysis[group_name]['mean_deviations']
            for sig_idx in range(min(K, 10)):  # First 10 signatures
                signature_comparison[sig_idx].append(mean_deviations[sig_idx])
    
    # Plot signature comparisons
    x_pos = np.arange(len(group_names_clean))
    for sig_idx in range(min(K, 5)):  # First 5 signatures
        values = signature_comparison[sig_idx]
        ax3.plot(x_pos, values, marker='o', linewidth=2, label=f'Sig {sig_idx}')
    
    ax3.set_xlabel('Transition Group')
    ax3.set_ylabel('Mean Signature Deviation')
    ax3.set_title('Signature Deviation Comparison Across Groups')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(group_names_clean, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Group size and diversity
    ax4 = axes[1, 1]
    
    group_sizes = []
    group_diversities = []  # Standard deviation as proxy for diversity
    
    for group_name in group_names:
        if group_name in signature_analysis:
            n_patients = signature_analysis[group_name]['n_patients']
            std_deviations = signature_analysis[group_name]['std_deviations']
            diversity = np.mean(std_deviations)  # Average standard deviation
            
            group_sizes.append(n_patients)
            group_diversities.append(diversity)
    
    # Create scatter plot
    scatter = ax4.scatter(group_sizes, group_diversities, 
                         c=colors[:len(group_sizes)], s=100, alpha=0.7)
    
    # Add labels
    for i, group_name in enumerate(group_names_clean):
        ax4.annotate(group_name, (group_sizes[i], group_diversities[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Number of Patients')
    ax4.set_ylabel('Signature Diversity (Mean Std)')
    ax4.set_title('Group Size vs. Signature Diversity')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/transition_signature_analysis.png', 
                dpi=150, bbox_inches='tight')
    print('✅ Saved transition signature analysis plot')
    
    return fig


def run_transition_analysis(target_disease, transition_diseases, Y, thetas, disease_names, processed_ids=None):
    """
    Run complete transition-based signature analysis
    """
    print(f"="*80)
    print(f"TRANSITION SIGNATURE ANALYSIS: {target_disease.upper()}")
    print(f"="*80)
    
    # Step 1: Find disease transitions
    transition_data = find_disease_transitions(
        Y, disease_names, target_disease, transition_diseases, processed_ids
    )
    
    if transition_data is None:
        return None
    
    # Step 2: Analyze signature patterns
    signature_analysis = analyze_signature_patterns_by_transition(
        transition_data, thetas, disease_names
    )
    
    # Step 3: Create visualizations
    fig = visualize_transition_signature_patterns(transition_data, signature_analysis)
    
    return {
        'transition_data': transition_data,
        'signature_analysis': signature_analysis,
        'figure': fig
    }


if __name__ == "__main__":
    print("Transition Signature Analysis Script")
    print("Analyzes signature patterns for specific disease transitions")
