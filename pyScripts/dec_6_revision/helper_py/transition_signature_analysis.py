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


def analyze_signature_patterns_by_transition(transition_data, thetas, disease_names, window_years=10):
    """
    Analyze signature patterns for each transition group - PER-TIMEPOINT DEVIATIONS
    For each timepoint, calculate mean signature loading per group, then deviation from reference
    """
    print(f"\n=== ANALYZING SIGNATURE PATTERNS BY TRANSITION (PER-TIMEPOINT) ===")
    
    transition_groups = transition_data['transition_groups']
    K, T = thetas.shape[1], thetas.shape[2]
    
    # Calculate population reference (sig_refs in R)
    print("Computing population reference (sig_refs)...")
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    print(f"Population reference shape: {population_reference.shape}")
    
    # Initialize arrays like R: time_diff_by_cluster[cluster, sig, time]
    n_groups = len([g for g in transition_groups.values() if len(g) > 0])
    time_diff_by_cluster = np.full((n_groups, K, T), np.nan)
    time_means_by_cluster_array = np.full((n_groups, K, T), np.nan)
    
    # Create group mapping
    group_names = []
    group_to_idx = {}
    idx = 0
    for group_name, patients in transition_groups.items():
        if len(patients) > 0:
            group_names.append(group_name)
            group_to_idx[group_name] = idx
            idx += 1
    
    print(f"Processing {len(group_names)} transition groups...")
    
    # For each timepoint t (like R loop) - BUT FOCUS ON PRE-DISEASE PERIOD
    for t in range(T):
        # Get signature loadings for all patients at time t
        time_spec_theta = thetas[:, :, t]  # Shape: (N, K) - all patients, all signatures at time t
        
        # Calculate mean signature loading per group at time t
        for group_name, patients in transition_groups.items():
            if len(patients) == 0:
                continue
                
            group_idx = group_to_idx[group_name]
            
            # Filter patients to only those who have the target disease at age 30+t
            # AND focus on the pre-disease period (before their target disease onset)
            valid_patients = []
            for patient_info in patients:
                patient_id = patient_info['patient_id']
                age_at_target = patient_info['age_at_target']
                target_time_idx = age_at_target - 30
                
                # Only include this patient if:
                # 1. Current time t is before their target disease onset
                # 2. We have enough pre-disease history (at least window_years before)
                # FIXED: Make the filtering much less restrictive - just check if t < target_time_idx
                if t < target_time_idx:
                    valid_patients.append(patient_id)
            
            if len(valid_patients) > 0:
                # Get signature loadings for valid patients in this group at time t
                group_theta_t = time_spec_theta[valid_patients, :]  # Shape: (n_valid_patients, K)
                
                # Calculate mean across valid patients in this group
                time_means_by_cluster_array[group_idx, :, t] = np.mean(group_theta_t, axis=0)
                
                # Calculate deviation from reference: sweep(..., 2, sig_refs[, t], "-")
                time_diff_by_cluster[group_idx, :, t] = time_means_by_cluster_array[group_idx, :, t] - population_reference[:, t]
    
    # DEBUG: Print some statistics about the filtering
    print(f"\nDEBUG: Filtering statistics:")
    for group_name, patients in transition_groups.items():
        if len(patients) == 0:
            continue
        group_idx = group_to_idx[group_name]
        non_nan_count = np.sum(~np.isnan(time_diff_by_cluster[group_idx, 0, :]))
        print(f"  {group_name}: {non_nan_count}/{T} timepoints have valid data")
        
        # Additional debugging for first few patients
        if group_name == "rheumatoid arthritis" and len(patients) > 0:
            print(f"    Sample patient ages at target: {[p['age_at_target'] for p in patients[:3]]}")
            print(f"    Sample target time indices: {[(p['age_at_target'] - 30) for p in patients[:3]]}")
    
    # Create results structure
    group_signature_analysis = {}
    for group_name, patients in transition_groups.items():
        if len(patients) == 0:
            continue
            
        group_idx = group_to_idx[group_name]
        
        # Get the full deviation trajectory for this group
        group_deviation_traj = time_diff_by_cluster[group_idx, :, :]  # Shape: (K, T)
        
        # Calculate summary statistics - handle NaN values properly
        mean_deviations = np.nanmean(group_deviation_traj, axis=1)  # Average over time, ignoring NaNs
        std_deviations = np.nanstd(group_deviation_traj, axis=1)    # Std over time, ignoring NaNs
        
        # Find most elevated signatures
        signature_elevations = []
        for sig_idx in range(K):
            deviation = mean_deviations[sig_idx]
            signature_elevations.append({
                'signature_idx': sig_idx,
                'mean_deviation': deviation,
                'std_deviation': std_deviations[sig_idx],
                'elevation_score': abs(deviation)
            })
        
        signature_elevations.sort(key=lambda x: x['elevation_score'], reverse=True)
        
        group_signature_analysis[group_name] = {
            'n_patients': len(patients),
            'mean_deviations': mean_deviations,
            'std_deviations': std_deviations,
            'top_signatures': signature_elevations[:10],
            'deviation_trajectory': group_deviation_traj,  # Full K x T trajectory
            'group_idx': group_idx
        }
        
        print(f"\n{group_name} ({len(patients)} patients):")
        print(f"  Top 5 signatures (by deviation from reference):")
        for i, sig_info in enumerate(signature_elevations[:5]):
            deviation = sig_info['mean_deviation']
            direction = "↑" if deviation > 0 else "↓"
            print(f"    {i+1}. Signature {sig_info['signature_idx']}: {deviation:+.4f} ± {sig_info['std_deviation']:.4f} {direction}")
    
    return {
        'group_signature_analysis': group_signature_analysis,
        'time_diff_by_cluster': time_diff_by_cluster,
        'time_means_by_cluster_array': time_means_by_cluster_array,
        'population_reference': population_reference,
        'group_names': group_names,
        'group_to_idx': group_to_idx
    }


def visualize_transition_signature_patterns(transition_data, signature_analysis):
    """
    Create visualizations comparing signature patterns across transition groups
    MATCHING R STACKED AREA PLOT LOGIC
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    print(f"\n=== CREATING TRANSITION SIGNATURE VISUALIZATIONS (R-STYLE) ===")
    
    # Extract data
    group_signature_analysis = signature_analysis['group_signature_analysis']
    time_diff_by_cluster = signature_analysis['time_diff_by_cluster']
    group_names = signature_analysis['group_names']
    group_to_idx = signature_analysis['group_to_idx']
    
    if len(group_names) == 0:
        print("No groups to visualize")
        return
    
    K, T = time_diff_by_cluster.shape[1], time_diff_by_cluster.shape[2]
    
    # Create the main stacked area plot (like R)
    fig, axes = plt.subplots(len(group_names), 1, figsize=(14, 4*len(group_names)))
    if len(group_names) == 1:
        axes = [axes]
    
    fig.suptitle('Signature Deviations from Reference by Transition Group', fontsize=16, fontweight='bold')
    
    # Colors for signatures - use tab20 + tab20b for distinct colors
    if K <= 20:
        sig_colors = plt.cm.tab20(np.linspace(0, 1, K))
    else:
        # For 21 signatures, use tab20 + tab20b
        colors_20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_b = plt.cm.tab20b(np.linspace(0, 1, 20))
        sig_colors = np.vstack([colors_20, colors_b[0:1]])  # Take first color from tab20b for 21st
        sig_colors = sig_colors[:K]  # In case K > 21
    
    for i, group_name in enumerate(group_names):
        ax = axes[i]
        group_idx = group_to_idx[group_name]
        n_patients = group_signature_analysis[group_name]['n_patients']
        
        # Get deviation trajectory for this group: Shape (K, T)
        group_deviations = time_diff_by_cluster[group_idx, :, :]
        
        # Create stacked area plot
        time_points = np.arange(T) + 30  # Age 30 to 30+T
        
        # Stack signatures (like R geom_area with position="stack")
        cumulative = np.zeros(T)
        
        for sig_idx in range(K):
            sig_values = group_deviations[sig_idx, :]
            ax.fill_between(time_points, cumulative, cumulative + sig_values, 
                           color=sig_colors[sig_idx], alpha=0.8, 
                           label=f'Sig {sig_idx}')
            cumulative += sig_values
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        
        ax.set_title(f'{group_name} (n={n_patients})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Age')
        ax.set_ylabel('Signature Deviation from Reference')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for first subplot to avoid clutter
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.show()
    
    # Create summary comparison plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot mean deviations for each group
    x_pos = np.arange(len(group_names))
    width = 0.8 / K
    
    for sig_idx in range(min(K, 10)):  # First 10 signatures
        sig_deviations = []
        for group_name in group_names:
            mean_dev = group_signature_analysis[group_name]['mean_deviations'][sig_idx]
            sig_deviations.append(mean_dev)
        
        ax.bar(x_pos + sig_idx * width, sig_deviations, width, 
               label=f'Sig {sig_idx}', alpha=0.8)
    
    ax.set_xlabel('Transition Group')
    ax.set_ylabel('Mean Signature Deviation')
    ax.set_title('Mean Signature Deviations by Group')
    ax.set_xticks(x_pos + width * (K-1) / 2)
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Created R-style visualizations for {len(group_names)} transition groups")


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
