"""
Plot signature deviations for transition vs non-transition patients

Compare signature trajectories for patients who had a disease (e.g., RA) and:
- DID develop the target disease (e.g., MI) - transition group
- DID NOT develop the target disease - control group
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_transition_vs_nontransition_deviations_fixed(
    transition_disease_name, 
    target_disease_name,
    Y, 
    thetas, 
    disease_names,
    years_before=10,
    years_after=0,  # Years AFTER transition disease to plot (0 = only use years_before)
    age_tolerance=5,  # Match within ±5 years of age at diagnosis
    min_followup=5,   # Minimum follow-up time after diagnosis
    save_plots=True
):
    """
    Plot signature deviations comparing patients who had transition disease:
    - WITH subsequent target disease (transition)
    - WITHOUT subsequent target disease (no transition)
    
    Can plot BEFORE transition disease (years_before) or AFTER (years_after)
    
    FIXED: Explicit age matching at transition disease diagnosis + sufficient follow-up
    """
    
    print(f"\n{'='*80}")
    print(f"COMPARING SIGNATURE DEVIATIONS (AGE-MATCHED + FOLLOW-UP):")
    print(f"  Transition disease: {transition_disease_name}")
    print(f"  Target disease: {target_disease_name}")
    print(f"  Age tolerance: ±{age_tolerance} years")
    print(f"  Min follow-up: {min_followup} years")
    print(f"{'='*80}\n")
    
    # Find disease indices
    transition_idx = None
    target_idx = None
    
    for i, name in enumerate(disease_names):
        if transition_disease_name.lower() in name.lower():
            transition_idx = i
            print(f"Found transition disease: {name} (index {i})")
        if target_disease_name.lower() in name.lower():
            target_idx = i
            print(f"Found target disease: {name} (index {i})")
    
    if transition_idx is None or target_idx is None:
        print("Could not find both diseases")
        return None
    
    # Calculate population reference
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    
    # Find patients with transition disease
    transition_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, transition_idx, :].sum() > 0:
            # Find age at first occurrence of transition disease
            first_occurrence = torch.where(Y[patient_id, transition_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_transition = first_occurrence.min().item() + 30
                transition_patients.append({
                    'patient_id': patient_id,
                    'age_at_transition': age_at_transition
                })
    
    print(f"\nFound {len(transition_patients)} patients with {transition_disease_name}")
    
    # Split into two groups: those who developed target disease and those who didn't
    developed_target = []
    no_target = []
    
    for patient_info in transition_patients:
        patient_id = patient_info['patient_id']
        age_at_transition = patient_info['age_at_transition']
        
        # Check if they developed target disease AFTER transition disease
        transition_time_idx = age_at_transition - 30
        if transition_time_idx < Y.shape[2]:
            # Check for target disease after transition
            if Y[patient_id, target_idx, transition_time_idx:].sum() > 0:
                # They developed target disease
                target_occurrence = torch.where(Y[patient_id, target_idx, transition_time_idx:] > 0)[0]
                if len(target_occurrence) > 0:
                    age_at_target = transition_time_idx + target_occurrence.min().item() + 30
                    developed_target.append({
                        'patient_id': patient_id,
                        'age_at_transition': age_at_transition,
                        'age_at_target': age_at_target
                    })
            else:
                # They did NOT develop target disease
                no_target.append({
                    'patient_id': patient_id,
                    'age_at_transition': age_at_transition,
                    'age_at_target': None
                })
    
    print(f"\nInitial patient groups:")
    print(f"  With {target_disease_name}: {len(developed_target)} patients")
    print(f"  Without {target_disease_name}: {len(no_target)} patients")
    
    if len(developed_target) == 0 or len(no_target) == 0:
        print("Not enough patients in both groups")
        return None
    
    # AGE MATCHING: Match patients on age at transition disease diagnosis
    print(f"\n=== AGE MATCHING ===")
    
    # Get age distribution of patients who developed target
    target_ages = [p['age_at_transition'] for p in developed_target]
    print(f"Age at {transition_disease_name} for target group: {np.min(target_ages):.1f} - {np.max(target_ages):.1f}")
    
    # For each patient who developed target, find age-matched controls
    matched_pairs = []
    
    for target_patient in developed_target:
        target_age = target_patient['age_at_transition']
        
        # Find age-matched controls (within ±age_tolerance years)
        age_matched_controls = []
        for control_patient in no_target:
            control_age = control_patient['age_at_transition']
            if abs(target_age - control_age) <= age_tolerance:
                age_matched_controls.append(control_patient)
        
        if len(age_matched_controls) > 0:
            # Randomly select one control (or could take closest age)
            import random
            selected_control = random.choice(age_matched_controls)
            matched_pairs.append({
                'target': target_patient,
                'control': selected_control
            })
    
    print(f"Found {len(matched_pairs)} age-matched pairs")
    
    if len(matched_pairs) == 0:
        print("No age-matched pairs found. Try increasing age_tolerance.")
        return None
    
    # FOLLOW-UP CHECK: Ensure controls have sufficient follow-up time
    print(f"\n=== FOLLOW-UP CHECK ===")
    
    valid_pairs = []
    for pair in matched_pairs:
        target_patient = pair['target']
        control_patient = pair['control']
        
        # Calculate follow-up time for control (time from transition to end of observation)
        control_age_at_transition = control_patient['age_at_transition']
        max_observation_age = 30 + Y.shape[2] - 1  # Last time point
        followup_time = max_observation_age - control_age_at_transition
        
        if followup_time >= min_followup:
            valid_pairs.append(pair)
        else:
            print(f"Control patient {control_patient['patient_id']}: only {followup_time:.1f} years follow-up (excluded)")
    
    print(f"After follow-up check: {len(valid_pairs)} valid pairs")
    
    if len(valid_pairs) == 0:
        print("No valid pairs after follow-up check. Try reducing min_followup.")
        return None
    
    # Calculate signature deviations for matched pairs
    K, T = thetas.shape[1], thetas.shape[2]
    
    with_target_deviations = []
    without_target_deviations = []
    
    for pair in valid_pairs:
        target_patient = pair['target']
        control_patient = pair['control']
        
        # Both patients analyzed at same age relative to their transition disease diagnosis
        for patient_info in [target_patient, control_patient]:
            patient_id = patient_info['patient_id']
            age_at_transition = patient_info['age_at_transition']
            transition_time_idx = age_at_transition - 30
            
            # Use years_before window FROM transition disease diagnosis (plotting forward from diagnosis)
            start_idx = transition_time_idx
            end_idx = min(transition_time_idx + years_before, T)
            
            if end_idx - start_idx == years_before:
                # Get signature trajectory for this window
                patient_traj = thetas[patient_id, :, start_idx:end_idx]  # Shape: (K, W)
                ref_traj = population_reference[:, start_idx:end_idx]
                
                deviation = patient_traj - ref_traj
                
                if patient_info == target_patient:
                    with_target_deviations.append(deviation)
                else:
                    without_target_deviations.append(deviation)
    
    if len(with_target_deviations) == 0 or len(without_target_deviations) == 0:
        print("Could not calculate deviations for both groups")
        return None
    
    # Calculate average deviations
    with_target_avg = np.mean(with_target_deviations, axis=0)  # Shape: (K, W)
    without_target_avg = np.mean(without_target_deviations, axis=0)  # Shape: (K, W)
    
    print(f"\nCalculated deviations (AGE-MATCHED + FOLLOW-UP):")
    print(f"  With target: {len(with_target_deviations)} patients")
    print(f"  Without target: {len(without_target_deviations)} patients")
    print(f"  Both groups analyzed: {years_before} years AFTER {transition_disease_name} diagnosis")
    
    # Print age distribution for verification
    target_ages_final = [p['target']['age_at_transition'] for p in valid_pairs]
    control_ages_final = [p['control']['age_at_transition'] for p in valid_pairs]
    print(f"  Target group age: {np.mean(target_ages_final):.1f} ± {np.std(target_ages_final):.1f}")
    print(f"  Control group age: {np.mean(control_ages_final):.1f} ± {np.std(control_ages_final):.1f}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Time points (years relative to transition disease diagnosis)
    time_points = np.arange(0, years_before)
    
    # Define colors for signatures - use tab20 + tab20b for all signatures
    from matplotlib import cm
    colors_20 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    colors_b = cm.get_cmap('tab20b')(np.linspace(0, 1, 20))
    sig_colors = np.vstack([colors_20, colors_b[0:1]])  # Take first color from tab20b for 21st
    
    # Plot 1: Patients who developed target disease
    ax = axes[0]
    bottom_pos = np.zeros(years_before)
    bottom_neg = np.zeros(years_before)
    
    for sig_idx in range(K):
        values = with_target_avg[sig_idx, :]
        pos_values = np.maximum(values, 0)
        neg_values = np.minimum(values, 0)
        
        ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                       label=f'Sig {sig_idx}',
                       color=sig_colors[sig_idx], alpha=0.8)
        ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                       color=sig_colors[sig_idx], alpha=0.5)
        
        bottom_pos += pos_values
        bottom_neg += neg_values
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f'{transition_disease_name} → {target_disease_name}\n(n={len(with_target_deviations)} patients)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Years After {transition_disease_name} Diagnosis')
    ax.set_ylabel('Signature Deviation from Population')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Patients who did NOT develop target disease
    ax = axes[1]
    bottom_pos = np.zeros(years_before)
    bottom_neg = np.zeros(years_before)
    
    for sig_idx in range(K):
        values = without_target_avg[sig_idx, :]
        pos_values = np.maximum(values, 0)
        neg_values = np.minimum(values, 0)
        
        ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                       label=f'Sig {sig_idx}',
                       color=sig_colors[sig_idx], alpha=0.8)
        ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                       color=sig_colors[sig_idx], alpha=0.5)
        
        bottom_pos += pos_values
        bottom_neg += neg_values
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f'{transition_disease_name} (no {target_disease_name})\n(n={len(without_target_deviations)} patients)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Years After {transition_disease_name} Diagnosis')
    ax.set_ylabel('Signature Deviation from Population')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Set same y-limits for comparison
    all_values = np.concatenate([with_target_avg.flatten(), without_target_avg.flatten()])
    max_dev = np.max(np.abs(all_values))
    for ax in axes:
        ax.set_ylim(-max_dev, max_dev)
    
    fig.suptitle(f'Signature Deviations: {transition_disease_name} Patients (AGE-MATCHED + FOLLOW-UP)\nComparing Those Who Did vs. Did Not Develop {target_disease_name}',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'transition_deviations_age_matched_{transition_disease_name.replace(" ", "_")}_to_{target_disease_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'")
    
    plt.show()
    
    return {
        'with_target_deviations': with_target_deviations,
        'without_target_deviations': without_target_deviations,
        'with_target_avg': with_target_avg,
        'without_target_avg': without_target_avg,
        'matched_pairs': valid_pairs,
        'age_stats': {
            'target_mean': np.mean(target_ages_final),
            'target_std': np.std(target_ages_final),
            'control_mean': np.mean(control_ages_final),
            'control_std': np.std(control_ages_final)
        },
        'figure': fig
    }

    
def plot_transition_vs_nontransition_deviations(
    transition_disease_name, 
    target_disease_name,
    Y, 
    thetas, 
    disease_names,
    years_before=10,
    save_plots=True
):
    """
    Plot signature deviations comparing patients who had transition disease:
    - WITH subsequent target disease (transition)
    - WITHOUT subsequent target disease (no transition)
    
    Parameters:
    -----------
    transition_disease_name : str
        Name of transition disease (e.g., "rheumatoid arthritis")
    target_disease_name : str
        Name of target disease (e.g., "myocardial infarction")
    Y : torch.Tensor
        Binary disease matrix (patients x diseases x time)
    thetas : np.ndarray
        Signature loadings (patients x signatures x time)
    disease_names : list
        List of disease names
    years_before : int
        How many years before target to analyze
    save_plots : bool
        Whether to save plots
    """
    
    print(f"\n{'='*80}")
    print(f"COMPARING SIGNATURE DEVIATIONS:")
    print(f"  Transition disease: {transition_disease_name}")
    print(f"  Target disease: {target_disease_name}")
    print(f"{'='*80}\n")
    
    # Find disease indices
    transition_idx = None
    target_idx = None
    
    for i, name in enumerate(disease_names):
        if transition_disease_name.lower() in name.lower():
            transition_idx = i
            print(f"Found transition disease: {name} (index {i})")
        if target_disease_name.lower() in name.lower():
            target_idx = i
            print(f"Found target disease: {name} (index {i})")
    
    if transition_idx is None or target_idx is None:
        print("Could not find both diseases")
        return None
    
    # Calculate population reference
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    print(f"\nPopulation reference shape: {population_reference.shape}")
    
    # Find patients with transition disease
    transition_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, transition_idx, :].sum() > 0:
            # Find age at first occurrence of transition disease
            first_occurrence = torch.where(Y[patient_id, transition_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_transition = first_occurrence.min().item() + 30
                transition_patients.append({
                    'patient_id': patient_id,
                    'age_at_transition': age_at_transition
                })
    
    print(f"\nFound {len(transition_patients)} patients with {transition_disease_name}")
    
    # Split into two groups: those who developed target disease and those who didn't
    developed_target = []
    no_target = []
    
    for patient_info in transition_patients:
        patient_id = patient_info['patient_id']
        age_at_transition = patient_info['age_at_transition']
        
        # Check if they developed target disease AFTER transition disease
        transition_time_idx = age_at_transition - 30
        if transition_time_idx < Y.shape[2]:
            # Check for target disease after transition
            if Y[patient_id, target_idx, transition_time_idx:].sum() > 0:
                # They developed target disease
                target_occurrence = torch.where(Y[patient_id, target_idx, transition_time_idx:] > 0)[0]
                if len(target_occurrence) > 0:
                    age_at_target = transition_time_idx + target_occurrence.min().item() + 30
                    developed_target.append({
                        'patient_id': patient_id,
                        'age_at_transition': age_at_transition,
                        'age_at_target': age_at_target
                    })
            else:
                # They did NOT develop target disease
                no_target.append({
                    'patient_id': patient_id,
                    'age_at_transition': age_at_transition,
                    'age_at_target': None
                })
    
    print(f"\nPatient groups:")
    print(f"  With {target_disease_name}: {len(developed_target)} patients")
    print(f"  Without {target_disease_name}: {len(no_target)} patients")
    
    if len(developed_target) == 0 or len(no_target) == 0:
        print("Not enough patients in both groups")
        return None
    
    # Calculate signature deviations for each group
    K, T = thetas.shape[1], thetas.shape[2]
    
    # Group 1: Developed target (use years before target as reference point)
    with_target_deviations = []
    for patient_info in developed_target:
        patient_id = patient_info['patient_id']
        age_at_target = patient_info['age_at_target']
        target_time_idx = age_at_target - 30
        lookback_idx = max(0, target_time_idx - years_before)
        
        if target_time_idx > years_before:
            # Get signature trajectory for this window
            patient_traj = thetas[patient_id, :, lookback_idx:target_time_idx]  # Shape: (K, W)
            ref_traj = population_reference[:, lookback_idx:target_time_idx]
            
            if patient_traj.shape[1] == years_before:
                deviation = patient_traj - ref_traj
                with_target_deviations.append(deviation)
    
    # Group 2: Did NOT develop target (use same time window relative to transition)
    without_target_deviations = []
    for patient_info in no_target:
        patient_id = patient_info['patient_id']
        age_at_transition = patient_info['age_at_transition']
        transition_time_idx = age_at_transition - 30
        
        # Use years_before window after transition (similar to group 1)
        end_idx = min(transition_time_idx + years_before, T)
        start_idx = transition_time_idx
        
        if end_idx - start_idx == years_before:
            patient_traj = thetas[patient_id, :, start_idx:end_idx]  # Shape: (K, W)
            ref_traj = population_reference[:, start_idx:end_idx]
            
            deviation = patient_traj - ref_traj
            without_target_deviations.append(deviation)
    
    if len(with_target_deviations) == 0 or len(without_target_deviations) == 0:
        print("Could not calculate deviations for both groups")
        return None
    
    # Calculate average deviations
    with_target_avg = np.mean(with_target_deviations, axis=0)  # Shape: (K, W)
    without_target_avg = np.mean(without_target_deviations, axis=0)  # Shape: (K, W)
    
    print(f"\nCalculated deviations:")
    print(f"  With target: {len(with_target_deviations)} patients")
    print(f"  Without target: {len(without_target_deviations)} patients")
    print(f"  Deviation shape: {with_target_avg.shape}")
    
    # Create plots (like plot_signature_deviations_over_time.py)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Time points (years relative to event/observation)
    time_points = np.arange(-years_before, 0)
    
    # Define colors for signatures
    sig_colors = plt.cm.viridis(np.linspace(0, 1, K))
    
    # Plot 1: Patients who developed target disease
    ax = axes[0]
    bottom_pos = np.zeros(years_before)
    bottom_neg = np.zeros(years_before)
    
    for sig_idx in range(K):
        values = with_target_avg[sig_idx, :]
        pos_values = np.maximum(values, 0)
        neg_values = np.minimum(values, 0)
        
        ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                       label=f'Sig {sig_idx}' if sig_idx < 10 else '',
                       color=sig_colors[sig_idx], alpha=0.8)
        ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                       color=sig_colors[sig_idx], alpha=0.5)
        
        bottom_pos += pos_values
        bottom_neg += neg_values
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f'{transition_disease_name} → {target_disease_name}\n(n={len(with_target_deviations)} patients)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Years Before {target_disease_name}')
    ax.set_ylabel('Signature Deviation from Population')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Patients who did NOT develop target disease
    ax = axes[1]
    bottom_pos = np.zeros(years_before)
    bottom_neg = np.zeros(years_before)
    
    for sig_idx in range(K):
        values = without_target_avg[sig_idx, :]
        pos_values = np.maximum(values, 0)
        neg_values = np.minimum(values, 0)
        
        ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                       label=f'Sig {sig_idx}' if sig_idx < 10 else '',
                       color=sig_colors[sig_idx], alpha=0.8)
        ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                       color=sig_colors[sig_idx], alpha=0.5)
        
        bottom_pos += pos_values
        bottom_neg += neg_values
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f'{transition_disease_name} (no {target_disease_name})\n(n={len(without_target_deviations)} patients)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Years After {transition_disease_name}')
    ax.set_ylabel('Signature Deviation from Population')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Set same y-limits for comparison
    all_values = np.concatenate([with_target_avg.flatten(), without_target_avg.flatten()])
    max_dev = np.max(np.abs(all_values))
    for ax in axes:
        ax.set_ylim(-max_dev, max_dev)
    
    fig.suptitle(f'Signature Deviations: {transition_disease_name} Patients\nComparing Those Who Did vs. Did Not Develop {target_disease_name}',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'transition_deviations_{transition_disease_name.replace(" ", "_")}_to_{target_disease_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'")
    
    plt.show()
    
    return {
        'with_target_deviations': with_target_deviations,
        'without_target_deviations': without_target_deviations,
        'with_target_avg': with_target_avg,
        'without_target_avg': without_target_avg,
        'figure': fig
    }


def plot_bc_to_mi_progression(
    transition_disease_name,
    target_disease_name,
    Y,
    thetas,
    disease_names,
    years_before=10,
    age_tolerance=5,
    min_followup=5,
    save_plots=True
):
    """
    Compare MI patients with BC history vs MI patients without BC history.
    
    Matching: Age at MI diagnosis
    Plotting: Years before MI (same time window for both groups)
    
    Parameters:
    -----------
    transition_disease_name : str
        Name of precursor disease (e.g., 'Breast cancer')
    target_disease_name : str
        Name of target disease (e.g., 'myocardial infarction')
    Y : torch.Tensor
        Binary disease matrix
    thetas : np.ndarray
        Signature loadings
    disease_names : list
        Disease names
    years_before : int
        Years before MI to plot
    age_tolerance : int
        Age matching tolerance
    min_followup : int
        Minimum follow-up time
    save_plots : bool
        Whether to save plots
    """
    
    print(f"\n{'='*80}")
    print(f"BC PROGRESSION ANALYSIS (MATCHED ON AGE AT BC DIAGNOSIS)")
    print(f"  Precursor: {transition_disease_name}")
    print(f"  Target: {target_disease_name}")
    print(f"  Plotting: {years_before} years leading up to MI")
    print(f"  Age tolerance: ±{age_tolerance} years")
    print(f"{'='*80}\n")
    
    # Find disease indices
    transition_idx = None
    target_idx = None
    
    for i, name in enumerate(disease_names):
        if transition_disease_name.lower() in name.lower():
            transition_idx = i
            print(f"Found transition disease: {name} (index {i})")
        if target_disease_name.lower() in name.lower():
            target_idx = i
            print(f"Found target disease: {name} (index {i})")
    
    if transition_idx is None or target_idx is None:
        print("Could not find both diseases")
        return None
    
    # Calculate population reference
    population_reference = np.mean(thetas, axis=0)
    print(f"Population reference shape: {population_reference.shape}")
    
    # Find BC patients who develop MI
    progressors = []
    for patient_id in range(Y.shape[0]):
        # Check if patient has BC
        bc_occurrences = torch.where(Y[patient_id, transition_idx, :] > 0)[0]
        if len(bc_occurrences) == 0:
            continue
        bc_age = bc_occurrences[0].item() + 30
        
        # Check if patient has MI
        mi_occurrences = torch.where(Y[patient_id, target_idx, :] > 0)[0]
        if len(mi_occurrences) == 0:
            continue
        mi_age = mi_occurrences[0].item() + 30
        
        # Must have MI after BC
        if mi_age > bc_age:
            progressors.append({
                'patient_id': patient_id,
                'age_at_bc': bc_age,
                'age_at_mi': mi_age
            })
    
    print(f"Found {len(progressors)} BC patients who develop MI")
    
    # Find BC patients who DON'T develop MI
    non_progressors = []
    for patient_id in range(Y.shape[0]):
        # Check if patient has BC
        bc_occurrences = torch.where(Y[patient_id, transition_idx, :] > 0)[0]
        if len(bc_occurrences) == 0:
            continue
        bc_age = bc_occurrences[0].item() + 30
        
        # Check patient does NOT have MI
        mi_occurrences = torch.where(Y[patient_id, target_idx, :] > 0)[0]
        if len(mi_occurrences) > 0:
            continue
        
        # Require minimum follow-up from BC
        max_observation_age = 30 + Y.shape[2] - 1
        followup_time = max_observation_age - bc_age
        if followup_time >= min_followup:
            non_progressors.append({
                'patient_id': patient_id,
                'age_at_bc': bc_age
            })
    
    print(f"Found {len(non_progressors)} BC patients who DON'T develop MI")
    
    if len(progressors) == 0 or len(non_progressors) == 0:
        print("Not enough patients in one or both groups")
        return None
    
    # AGE MATCHING: Match non-progressors to progressors on age at BC diagnosis
    print(f"\n=== AGE MATCHING AT BC DIAGNOSIS ===")
    matched_pairs = []
    
    for prog in progressors:
        bc_age_prog = prog['age_at_bc']
        
        # Find age-matched non-progressors
        age_matched_np = []
        for non_prog_candidate in non_progressors:
            if abs(non_prog_candidate['age_at_bc'] - bc_age_prog) <= age_tolerance:
                age_matched_np.append(non_prog_candidate)
        
        if len(age_matched_np) > 0:
            import random
            random.seed(42)  # For reproducibility
            selected_np = random.choice(age_matched_np)
            matched_pairs.append({
                'progressor': prog,
                'non_progressor': selected_np
            })
    
    print(f"Found {len(matched_pairs)} age-matched pairs")
    
    if len(matched_pairs) == 0:
        print("No matched pairs found. Try increasing age_tolerance.")
        return None
    
    # Analyze signature trajectories
    progressor_deviations = []
    non_progressor_deviations = []
    
    # Calculate average time from BC to MI for equivalent window
    avg_bc_to_mi_time = int(np.mean([p['age_at_mi'] - p['age_at_bc'] for p in progressors]))
    print(f"Average time from BC to MI: {avg_bc_to_mi_time} years")
    
    # Use years_before for trajectory length
    min_trajectory_length = years_before
    
    for pair in matched_pairs:
        prog = pair['progressor']
        non_prog = pair['non_progressor']
        
        # Progressor: Plot years before MI
        prog_patient_id = prog['patient_id']
        prog_mi_age = prog['age_at_mi']
        prog_mi_idx = prog_mi_age - 30
        
        # Take years_before years before MI
        start_idx = max(0, prog_mi_idx - min_trajectory_length)
        end_idx = prog_mi_idx
        
        if end_idx - start_idx == min_trajectory_length:
            prog_traj = thetas[prog_patient_id, :, start_idx:end_idx]
            prog_ref = population_reference[:, start_idx:end_idx]
            prog_dev = prog_traj - prog_ref
            progressor_deviations.append(prog_dev)
        
        # Non-progressor: Plot years before "fake MI" (BC_age + avg_BC_to_MI_time)
        non_prog_patient_id = non_prog['patient_id']
        non_prog_bc_age = non_prog['age_at_bc']
        
        # Calculate "fake MI" age for non-progressor
        fake_mi_age = non_prog_bc_age + avg_bc_to_mi_time
        fake_mi_idx = fake_mi_age - 30
        
        # Plot years_before years before this fake MI
        start_idx_np = max(0, fake_mi_idx - min_trajectory_length)
        end_idx_np = fake_mi_idx
        
        # Handle edge cases where fake MI might be beyond data bounds
        if end_idx_np > start_idx_np:
            # Cap end_idx to data bounds
            end_idx_np = min(end_idx_np, Y.shape[2])
            
            # Adjust start_idx to ensure we get at least some data
            if start_idx_np < 0:
                start_idx_np = 0
            
            # Only proceed if we have enough data
            if end_idx_np > start_idx_np and end_idx_np - start_idx_np >= min_trajectory_length:
                non_prog_traj = thetas[non_prog_patient_id, :, start_idx_np:end_idx_np]
                non_prog_ref = population_reference[:, start_idx_np:end_idx_np]
                non_prog_dev = non_prog_traj - non_prog_ref
                
                # Ensure we have exactly min_trajectory_length by taking the last N years
                if non_prog_dev.shape[1] >= min_trajectory_length:
                    non_prog_dev = non_prog_dev[:, -min_trajectory_length:]
                    non_progressor_deviations.append(non_prog_dev)
    
    if len(progressor_deviations) == 0 or len(non_progressor_deviations) == 0:
        print("Could not calculate deviations for both groups")
        return None
    
    # All trajectories have the same length (years_before)
    min_len = min_trajectory_length
    
    print(f"\nAnalyzing {min_len} years")
    print(f"Progressors (BC→MI): {len(progressor_deviations)} patients")
    print(f"Non-progressors (BC only): {len(non_progressor_deviations)} patients")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Time points - years before MI (from -max to -1) for both groups
    # For example: if min_len=10, shows [-10, -9, -8, ..., -2, -1] (years before MI)
    time_points = np.arange(-min_len, 0)
    
    # Use same color scheme
    from matplotlib import cm
    colors_20 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    colors_b = cm.get_cmap('tab20b')(np.linspace(0, 1, 20))
    sig_colors = np.vstack([colors_20, colors_b[0:1]])
    
    # Plot progressors (BC→MI): years before MI
    prog_avg = np.mean(progressor_deviations, axis=0)
    cumulative = np.zeros(min_len)
    
    for sig_idx in range(prog_avg.shape[0]):
        sig_values = prog_avg[sig_idx, :]
        axes[0].fill_between(time_points, cumulative, cumulative + sig_values,
                       color=sig_colors[sig_idx], alpha=0.85,
                       label=f'Sig {sig_idx}', edgecolor='white', linewidth=0.3)
        cumulative += sig_values
    
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    axes[0].set_title(f'{transition_disease_name} → {target_disease_name}\n(n={len(progressor_deviations)} patients)',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Years Before MI')
    axes[0].set_ylabel('Signature Deviation')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # Plot non-progressors (BC only): years after BC
    np_avg = np.mean(non_progressor_deviations, axis=0)
    cumulative = np.zeros(min_len)
    
    for sig_idx in range(np_avg.shape[0]):
        sig_values = np_avg[sig_idx, :]
        axes[1].fill_between(time_points, cumulative, cumulative + sig_values,
                       color=sig_colors[sig_idx], alpha=0.85,
                       label=f'Sig {sig_idx}', edgecolor='white', linewidth=0.3)
        cumulative += sig_values
    
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    axes[1].set_title(f'{transition_disease_name} (no {target_disease_name})\n(n={len(non_progressor_deviations)} patients)',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Years Before MI (Equivalent Window)')
    axes[1].set_ylabel('Signature Deviation')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    # Set same y-axis limits
    max_dev = max(np.abs(prog_avg).max(), np.abs(np_avg).max())
    axes[0].set_ylim(-max_dev, max_dev)
    axes[1].set_ylim(-max_dev, max_dev)
    
    fig.suptitle(f'Signature Patterns: BC Patients (MATCHED ON AGE AT BC)\nComparing Those Who Do vs. Don\'t Develop MI',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'bc_progression_{transition_disease_name.replace(" ", "_")}_matched_on_bc_age.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'")
    
    plt.show()
    
    return {
        'progressor_deviations': progressor_deviations,
        'non_progressor_deviations': non_progressor_deviations,
        'matched_pairs': matched_pairs,
        'prog_avg': prog_avg,  # Average deviations for progressors (signatures x time)
        'np_avg': np_avg,      # Average deviations for non-progressors
        'time_points': time_points  # Time axis
    }


