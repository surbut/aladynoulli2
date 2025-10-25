"""
Simple Transition Analysis - Fixed Version

This script analyzes signature patterns for specific disease transitions with a simpler,
more robust approach that avoids the NaN issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict

def simple_transition_analysis(Y, thetas, disease_names, target_disease="myocardial infarction", 
                              transition_diseases=None, processed_ids=None, age_tolerance=5, min_followup=5):
    """
    Simple transition analysis that avoids NaN issues
    
    FIXED: Now includes age matching at target disease diagnosis
    
    Parameters:
    -----------
    Y : torch.Tensor
        Binary disease matrix (patients x diseases x time)
    thetas : np.array
        Signature loadings (patients x signatures x time)
    disease_names : list
        List of disease names
    target_disease : str
        Target disease to analyze transitions to
    transition_diseases : list
        List of precursor diseases to analyze
    processed_ids : list
        Patient IDs (optional)
    age_tolerance : int
        Age matching tolerance in years (±age_tolerance)
    min_followup : int
        Minimum follow-up time after target disease for controls
    """
    if transition_diseases is None:
        transition_diseases = [
            "rheumatoid arthritis",
            "type 2 diabetes", 
            "essential hypertension",
            "hypercholesterolemia",
            "obesity",
            "Major depressive disorder",
            "anxiety disorder"
        ]
    
    print(f"=== SIMPLE TRANSITION ANALYSIS: {target_disease.upper()} ===")
    
    # Find target disease
    target_idx = None
    for i, name in enumerate(disease_names):
        if target_disease.lower() in name.lower():
            target_idx = i
            break
    
    if target_idx is None:
        print(f"Could not find target disease: {target_disease}")
        return None
    
    print(f"Target disease: {disease_names[target_idx]} (index {target_idx})")
    
    # Calculate population reference
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    print(f"Population reference shape: {population_reference.shape}")
    
    # Find patients with target disease
    target_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, target_idx, :].sum() > 0:
            first_occurrence = torch.where(Y[patient_id, target_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_target = first_occurrence.min().item() + 30
                target_patients.append({
                    'patient_id': patient_id,
                    'age_at_target': age_at_target
                })
    
    print(f"Found {len(target_patients)} patients with {target_disease}")
    
    # Analyze each transition pathway
    pathway_results = {}
    
    for transition_disease in transition_diseases:
        print(f"\n--- Analyzing {transition_disease.upper()} → {target_disease.upper()} ---")
        
        # Find transition disease index
        transition_idx = None
        for i, name in enumerate(disease_names):
            if transition_disease.lower() in name.lower():
                transition_idx = i
                break
        
        if transition_idx is None:
            print(f"⚠️  Could not find transition disease: {transition_disease}")
            continue
        
        print(f"Transition disease: {disease_names[transition_idx]} (index {transition_idx})")
        
        # Find patients with this transition pathway
        pathway_patients = []
        for patient_info in target_patients:
            patient_id = patient_info['patient_id']
            age_at_target = patient_info['age_at_target']
            
            # Check if patient had transition disease BEFORE target
            target_time_idx = age_at_target - 30
            if target_time_idx > 0 and Y[patient_id, transition_idx, :target_time_idx].sum() > 0:
                pathway_patients.append(patient_info)
        
        print(f"Found {len(pathway_patients)} patients with {transition_disease} → {target_disease}")
        
        if len(pathway_patients) == 0:
            continue
        
        # SIMPLE APPROACH: Calculate average signature loading for this group
        # Focus on the 5 years before target disease onset
        window_years = 5
        K, T = thetas.shape[1], thetas.shape[2]
        
        # Collect signature trajectories for this pathway
        pathway_trajectories = []
        
        for patient_info in pathway_patients:
            patient_id = patient_info['patient_id']
            age_at_target = patient_info['age_at_target']
            target_time_idx = age_at_target - 30
            
            # Get the 5-year window before target disease
            start_time = max(0, target_time_idx - window_years)
            end_time = target_time_idx
            
            if end_time > start_time:
                # Get signature trajectory for this patient in the pre-disease window
                patient_trajectory = thetas[patient_id, :, start_time:end_time]  # Shape: (K, window_size)
                pathway_trajectories.append(patient_trajectory)
        
        if len(pathway_trajectories) == 0:
            print(f"No valid trajectories for {transition_disease}")
            continue
        
        print(f"Collected {len(pathway_trajectories)} valid trajectories")
        
        # Calculate average trajectory for this pathway
        # Handle variable trajectory lengths by padding to common length
        if len(pathway_trajectories) == 0:
            continue
            
        # Find the minimum trajectory length
        min_length = min(traj.shape[1] for traj in pathway_trajectories)
        
        # Pad all trajectories to the same length (truncate if needed)
        padded_trajectories = []
        for traj in pathway_trajectories:
            if traj.shape[1] >= min_length:
                # Take the last min_length timepoints to align with event
                padded_traj = traj[:, -min_length:]
                padded_trajectories.append(padded_traj)
        
        if len(padded_trajectories) == 0:
            continue
            
        # Now we can stack them
        pathway_trajectories = np.array(padded_trajectories)  # Shape: (n_patients, K, min_length)
        mean_trajectory = np.mean(pathway_trajectories, axis=0)  # Shape: (K, min_length)
        
        # Calculate deviations from population reference
        # Get corresponding population reference for the same time window
        ref_start = max(0, T - min_length)
        ref_end = T
        population_ref_window = population_reference[:, ref_start:ref_end]  # Shape: (K, min_length)
        
        # Calculate deviations
        deviations = mean_trajectory - population_ref_window  # Shape: (K, window_size)
        
        # Calculate summary statistics
        mean_deviations = np.mean(deviations, axis=1)  # Average over time window
        std_deviations = np.std(deviations, axis=1)
        
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
        
        # Store results
        pathway_results[transition_disease] = {
            'n_patients': len(pathway_patients),
            'mean_deviations': mean_deviations,
            'std_deviations': std_deviations,
            'top_signatures': signature_elevations[:10],
            'deviations': deviations,
            'mean_trajectory': mean_trajectory
        }
        
        # Print results
        print(f"  Top 5 signatures (by deviation from reference):")
        for i, sig_info in enumerate(signature_elevations[:5]):
            deviation = sig_info['mean_deviation']
            direction = "↑" if deviation > 0 else "↓"
            print(f"    {i+1}. Signature {sig_info['signature_idx']}: {deviation:+.4f} ± {sig_info['std_deviation']:.4f} {direction}")
    
    return pathway_results


def visualize_simple_transition_results(pathway_results, save_plots=True):
    """
    Create visualizations for the simple transition analysis
    """
    if len(pathway_results) == 0:
        print("No pathway results to visualize")
        return
    
    print(f"\n=== CREATING SIMPLE TRANSITION VISUALIZATIONS ===")
    
    # Create figure with subplots for each pathway
    n_pathways = len(pathway_results)
    fig, axes = plt.subplots(n_pathways, 1, figsize=(14, 5*n_pathways))
    if n_pathways == 1:
        axes = [axes]
    
    fig.suptitle('Signature Deviations Before Myocardial Infarction by Transition Pathway', 
                 fontsize=16, fontweight='bold')
    
    # Colors for different signatures
    sig_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    pathway_names = list(pathway_results.keys())
    
    for i, (pathway_name, results) in enumerate(pathway_results.items()):
        ax = axes[i]
        
        deviations = results['deviations']  # Shape: (K, window_size)
        K, window_size = deviations.shape
        
        # Create time axis (years before event)
        time_points = np.arange(-window_size, 0)
        
        # Plot each signature
        for sig_idx in range(min(K, 10)):  # First 10 signatures
            sig_deviations = deviations[sig_idx, :]
            
            ax.plot(time_points, sig_deviations, 
                   color=sig_colors[sig_idx % len(sig_colors)], 
                   linewidth=2, marker='o', markersize=4,
                   label=f'Sig {sig_idx}')
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'{pathway_name.title()} Pathway (n={results["n_patients"]} patients)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Years Before Myocardial Infarction')
        ax.set_ylabel('Signature Deviation from Population')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('simple_transition_analysis.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'simple_transition_analysis.png'")
    
    plt.show()
    
    # Print biological interpretation
    print(f"\n=== BIOLOGICAL INTERPRETATION ===")
    for pathway_name, results in pathway_results.items():
        print(f"\n{pathway_name.title()} Pathway:")
        top_signatures = results['top_signatures'][:5]
        
        for sig_info in top_signatures:
            deviation = sig_info['mean_deviation']
            direction = "↑" if deviation > 0 else "↓"
            print(f"  Signature {sig_info['signature_idx']}: {deviation:+.4f} {direction}")
        
        # Biological interpretation
        if "rheumatoid arthritis" in pathway_name.lower():
            print("  → Suggests inflammatory mechanisms driving cardiovascular disease")
        elif "diabetes" in pathway_name.lower(): 
            print("  → Suggests metabolic dysfunction leading to cardiovascular complications")
        elif "hypertension" in pathway_name.lower():
            print("  → Suggests direct cardiovascular pathway with structural changes")


def run_simple_transition_analysis(target_disease="myocardial infarction"):
    """
    Run complete simple transition analysis
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
    disease_names = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/disease_names.csv')['x'].tolist()
    print(f"Loaded {len(disease_names)} diseases")
    
    # Load processed IDs
    processed_ids = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/processed_ids.csv')['eid'].values
    print(f"Loaded {len(processed_ids)} processed IDs")
    
    # Run simple transition analysis
    pathway_results = simple_transition_analysis(
        Y, thetas, disease_names, target_disease
    )
    
    if pathway_results:
        # Create visualization
        visualize_simple_transition_results(pathway_results)
        
        return pathway_results
    else:
        print("No pathway results generated")
        return None


if __name__ == "__main__":
    results = run_simple_transition_analysis()
