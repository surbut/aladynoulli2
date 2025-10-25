"""
Plot MI Patients With vs Without Precursor Disease

This script compares signature deviations for MI patients who had a specific precursor disease
vs MI patients who did NOT have that precursor disease. This is the complementary analysis
to plot_transition_deviations.py - same endpoint (MI) but different precursor histories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict

def plot_mi_with_vs_without_precursor(transition_disease_name, target_disease_name, 
                                     Y, thetas, disease_names, years_before=10, 
                                     age_tolerance=5, min_followup=5, save_plots=True):
    """
    Compare signature deviations for MI patients with vs without a specific precursor disease
    
    FIXED: Now includes age matching at MI diagnosis to ensure fair comparison
    
    Parameters:
    -----------
    transition_disease_name : str
        Name of the precursor disease (e.g., 'rheumatoid arthritis')
    target_disease_name : str
        Name of the target disease (e.g., 'myocardial infarction')
    Y : torch.Tensor
        Binary disease matrix (patients x diseases x time)
    thetas : np.array
        Signature loadings (patients x signatures x time)
    disease_names : list
        List of disease names
    years_before : int
        Number of years before MI to analyze
    age_tolerance : int
        Age matching tolerance in years (±age_tolerance)
    min_followup : int
        Minimum follow-up time after MI for controls
    save_plots : bool
        Whether to save plots to files
        
    Returns:
    --------
    dict : Analysis results with age-matched groups
    """
    
    print(f"COMPARING MI PATIENTS WITH vs WITHOUT {transition_disease_name.upper()}")
    print(f"Target disease: {target_disease_name}")
    print(f"Precursor disease: {transition_disease_name}")
    print("="*80)
    
    # Find disease indices
    target_idx = None
    transition_idx = None
    
    for i, name in enumerate(disease_names):
        if target_disease_name.lower() in name.lower():
            target_idx = i
            break
    
    for i, name in enumerate(disease_names):
        if transition_disease_name.lower() in name.lower():
            transition_idx = i
            break
    
    if target_idx is None:
        print(f"Could not find target disease: {target_disease_name}")
        return None
        
    if transition_idx is None:
        print(f"Could not find transition disease: {transition_disease_name}")
        return None
    
    print(f"Found target disease: {disease_names[target_idx]} (index {target_idx})")
    print(f"Found transition disease: {disease_names[transition_idx]} (index {transition_idx})")
    
    # Calculate population reference
    population_reference = np.mean(thetas, axis=0)
    print(f"Population reference shape: {population_reference.shape}")
    
    # Find all patients with target disease (MI)
    mi_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, target_idx, :].sum() > 0:
            first_occurrence = torch.where(Y[patient_id, target_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_mi = first_occurrence.min().item() + 30
                mi_patients.append({
                    'patient_id': patient_id,
                    'age_at_mi': age_at_mi
                })
    
    print(f"Found {len(mi_patients)} patients with {target_disease_name}")
    
    # Split MI patients into two groups
    mi_with_precursor = []
    mi_without_precursor = []
    
    for patient_info in mi_patients:
        patient_id = patient_info['patient_id']
        age_at_mi = patient_info['age_at_mi']
        mi_time_idx = age_at_mi - 30
        
        # Check if patient had precursor disease BEFORE MI
        if mi_time_idx > 0 and Y[patient_id, transition_idx, :mi_time_idx].sum() > 0:
            mi_with_precursor.append(patient_info)
        else:
            mi_without_precursor.append(patient_info)
    
    print(f"\nInitial MI patient groups:")
    print(f"  With {transition_disease_name}: {len(mi_with_precursor)} patients")
    print(f"  Without {transition_disease_name}: {len(mi_without_precursor)} patients")
    
    if len(mi_with_precursor) == 0 or len(mi_without_precursor) == 0:
        print("❌ Not enough patients in one or both groups")
        return None
    
    # AGE MATCHING: Match patients on age at MI diagnosis
    print(f"\n=== AGE MATCHING AT MI DIAGNOSIS ===")
    print(f"Age tolerance: ±{age_tolerance} years")
    print(f"Min follow-up: {min_followup} years")
    
    # Get age distribution of patients with precursor
    precursor_ages = [p['age_at_mi'] for p in mi_with_precursor]
    print(f"Age at MI for precursor group: {np.min(precursor_ages):.1f} - {np.max(precursor_ages):.1f}")
    
    # For each patient with precursor, find age-matched controls without precursor
    matched_pairs = []
    
    for precursor_patient in mi_with_precursor:
        precursor_age = precursor_patient['age_at_mi']
        
        # Find age-matched controls (within ±age_tolerance years)
        age_matched_controls = []
        for control_patient in mi_without_precursor:
            control_age = control_patient['age_at_mi']
            if abs(precursor_age - control_age) <= age_tolerance:
                age_matched_controls.append(control_patient)
        
        if len(age_matched_controls) > 0:
            # Randomly select one control (or could take closest age)
            import random
            selected_control = random.choice(age_matched_controls)
            matched_pairs.append({
                'precursor': precursor_patient,
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
        precursor_patient = pair['precursor']
        control_patient = pair['control']
        
        # Calculate follow-up time for control (time from MI to end of observation)
        control_age_at_mi = control_patient['age_at_mi']
        max_observation_age = 30 + Y.shape[2] - 1  # Last time point
        followup_time = max_observation_age - control_age_at_mi
        
        if followup_time >= min_followup:
            valid_pairs.append(pair)
        else:
            print(f"Control patient {control_patient['patient_id']}: only {followup_time:.1f} years follow-up (excluded)")
    
    print(f"After follow-up check: {len(valid_pairs)} valid pairs")
    
    if len(valid_pairs) == 0:
        print("No valid pairs after follow-up check. Try reducing min_followup.")
        return None
    
    # Extract age-matched groups
    mi_with_precursor_matched = [pair['precursor'] for pair in valid_pairs]
    mi_without_precursor_matched = [pair['control'] for pair in valid_pairs]
    
    print(f"\nFinal age-matched groups:")
    print(f"  With {transition_disease_name}: {len(mi_with_precursor_matched)} patients")
    print(f"  Without {transition_disease_name}: {len(mi_without_precursor_matched)} patients")
    
    # Print age distribution for verification
    precursor_ages_final = [p['age_at_mi'] for p in mi_with_precursor_matched]
    control_ages_final = [p['age_at_mi'] for p in mi_without_precursor_matched]
    print(f"  Precursor group age: {np.mean(precursor_ages_final):.1f} ± {np.std(precursor_ages_final):.1f}")
    print(f"  Control group age: {np.mean(control_ages_final):.1f} ± {np.std(control_ages_final):.1f}")
    
    # Analyze signature patterns for both groups
    def analyze_mi_group(patients, group_name):
        print(f"\nAnalyzing {group_name}...")
        
        # Collect signature trajectories aligned to MI timing
        trajectories = []
        
        for patient_info in patients:
            patient_id = patient_info['patient_id']
            age_at_mi = patient_info['age_at_mi']
            mi_time_idx = age_at_mi - 30
            
            # Get the years_before window before MI
            start_time = max(0, mi_time_idx - years_before)
            end_time = mi_time_idx
            
            if end_time > start_time:
                # Get signature trajectory for this patient
                patient_trajectory = thetas[patient_id, :, start_time:end_time]
                trajectories.append(patient_trajectory)
        
        print(f"  Collected {len(trajectories)} valid trajectories")
        
        if len(trajectories) == 0:
            return None
        
        # Align trajectories to the same length (take minimum length)
        min_length = min(traj.shape[1] for traj in trajectories)
        aligned_trajectories = []
        
        for traj in trajectories:
            if traj.shape[1] >= min_length:
                # Take the last min_length timepoints (closest to MI)
                aligned_traj = traj[:, -min_length:]
                aligned_trajectories.append(aligned_traj)
        
        if len(aligned_trajectories) == 0:
            return None
        
        # Calculate mean trajectory and deviations
        aligned_trajectories = np.array(aligned_trajectories)
        mean_trajectory = np.mean(aligned_trajectories, axis=0)
        
        # Get corresponding population reference
        ref_start = max(0, thetas.shape[2] - min_length)
        ref_end = thetas.shape[2]
        population_ref_window = population_reference[:, ref_start:ref_end]
        
        # Calculate deviations
        deviations = mean_trajectory - population_ref_window
        
        return {
            'mean_trajectory': mean_trajectory,
            'deviations': deviations,
            'n_patients': len(aligned_trajectories),
            'min_length': min_length
        }
    
    # Analyze both age-matched groups
    with_precursor_results = analyze_mi_group(mi_with_precursor_matched, f"MI with {transition_disease_name} (age-matched)")
    without_precursor_results = analyze_mi_group(mi_without_precursor_matched, f"MI without {transition_disease_name} (age-matched)")
    
    if with_precursor_results is None or without_precursor_results is None:
        print("❌ Could not analyze one or both groups")
        return None
    
    # Create visualization
    print(f"\nCreating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'MI Patients: With vs Without {transition_disease_name.title()} History (AGE-MATCHED)', 
                 fontsize=16, fontweight='bold')
    
    # Colors for signatures - use all 21 signatures
    sig_colors = plt.cm.viridis(np.linspace(0, 1, 21))
    
    # Plot both groups
    for i, (results, group_name) in enumerate([(with_precursor_results, f"MI with {transition_disease_name}"), 
                                               (without_precursor_results, f"MI without {transition_disease_name}")]):
        ax = axes[i]
        
        deviations = results['deviations']
        n_patients = results['n_patients']
        min_length = results['min_length']
        
        # Create time axis
        time_points = np.arange(-min_length, 0)
        
        # Create stacked area plot
        cumulative_pos = np.zeros(min_length)
        cumulative_neg = np.zeros(min_length)
        
        for sig_idx in range(deviations.shape[0]):  # All signatures
            sig_values = deviations[sig_idx, :]
            pos_values = np.maximum(sig_values, 0)
            neg_values = np.minimum(sig_values, 0)
            
            # Positive deviations (above zero)
            ax.fill_between(time_points, cumulative_pos, cumulative_pos + pos_values,
                           color=sig_colors[sig_idx], alpha=0.7, 
                           label=f'Sig {sig_idx}' if sig_idx < 10 else '')
            
            # Negative deviations (below zero)
            ax.fill_between(time_points, cumulative_neg, cumulative_neg + neg_values,
                           color=sig_colors[sig_idx], alpha=0.7)
            
            cumulative_pos += pos_values
            cumulative_neg += neg_values
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        ax.set_title(f'{group_name.title()}\n(n={n_patients} patients)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Years Before Myocardial Infarction')
        ax.set_ylabel('Signature Deviation from Population')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'mi_with_vs_without_{transition_disease_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS (AGE-MATCHED) ===")
    print(f"MI patients with {transition_disease_name}: {with_precursor_results['n_patients']}")
    print(f"MI patients without {transition_disease_name}: {without_precursor_results['n_patients']}")
    
    # Calculate and print mean deviations for each group
    print(f"\nMean signature deviations:")
    print(f"MI with {transition_disease_name}:")
    mean_dev_with = np.mean(with_precursor_results['deviations'], axis=1)
    for sig_idx in range(len(mean_dev_with)):
        print(f"  Sig {sig_idx}: {mean_dev_with[sig_idx]:+.4f}")
    
    print(f"\nMI without {transition_disease_name}:")
    mean_dev_without = np.mean(without_precursor_results['deviations'], axis=1)
    for sig_idx in range(len(mean_dev_without)):
        print(f"  Sig {sig_idx}: {mean_dev_without[sig_idx]:+.4f}")
    
    # Calculate differences between groups
    print(f"\nDifference (with - without):")
    diff = mean_dev_with - mean_dev_without
    for sig_idx in range(len(diff)):
        direction = "↑" if diff[sig_idx] > 0 else "↓"
        print(f"  Sig {sig_idx}: {diff[sig_idx]:+.4f} {direction}")
    
    return {
        'with_precursor': with_precursor_results,
        'without_precursor': without_precursor_results,
        'transition_disease': transition_disease_name,
        'target_disease': target_disease_name,
        'matched_pairs': valid_pairs,
        'age_stats': {
            'precursor_mean': np.mean(precursor_ages_final),
            'precursor_std': np.std(precursor_ages_final),
            'control_mean': np.mean(control_ages_final),
            'control_std': np.std(control_ages_final)
        }
    }


def run_mi_comparison_analysis():
    """
    Run MI comparison analysis for different precursor diseases
    """
    # Load data
    print("Loading data...")
    
    # Load Y matrix
    Y = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
    print(f"Loaded Y: {Y.shape}")
    
    # Load thetas (using NEW model with PCs)
    thetas = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt').numpy()
    print(f"Loaded thetas: {thetas.shape}")
    
    # Subset Y to match thetas
    Y = Y[:400000, :, :]
    print(f"Subset Y to match thetas: {Y.shape}")
    
    # Load disease names
    disease_names = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/disease_names.csv')['disease_name'].tolist()
    print(f"Loaded {len(disease_names)} diseases")
    
    # Test with rheumatoid arthritis
    results = plot_mi_with_vs_without_precursor(
        transition_disease_name='rheumatoid arthritis',
        target_disease_name='myocardial infarction',
        Y=Y,
        thetas=thetas,
        disease_names=disease_names,
        years_before=10,
        save_plots=True
    )
    
    return results


if __name__ == "__main__":
    results = run_mi_comparison_analysis()
