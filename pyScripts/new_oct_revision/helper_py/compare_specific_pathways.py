"""
Compare Specific Disease Transition Pathways

This script compares specific biologically meaningful pathways to the same disease,
focusing on signature-specific biological interpretation rather than just statistical clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict

def compare_specific_pathways(Y, thetas, disease_names, processed_ids=None, 
                            target_disease="myocardial infarction", 
                            pathway_pairs=None):
    """
    Compare specific disease transition pathways with biological interpretation
    
    Parameters:
    -----------
    Y : torch.Tensor
        Binary disease matrix (patients x diseases x time)
    thetas : np.array
        Signature loadings (patients x signatures x time)
    disease_names : list
        List of disease names
    processed_ids : array, optional
        Patient IDs for mapping
    target_disease : str
        Target disease to analyze
    pathway_pairs : list of tuples
        List of (transition_disease, pathway_name, expected_signatures) tuples
        
    Returns:
    --------
    dict : Analysis results with biological interpretations
    """
    
    if pathway_pairs is None:
        # Define biologically meaningful pathways
        pathway_pairs = [
            ("rheumatoid arthritis", "Inflammatory", [5, 7, 0]),  # Coronary, Pain/Mood, Cardiac
            ("type 2 diabetes", "Metabolic", [15, 5, 0]),        # Diabetes, Coronary, Cardiac  
            ("essential hypertension", "Cardiovascular", [0, 5, 12])  # Cardiac, Coronary, Other CVD
        ]
    
    print(f"=== COMPARING SPECIFIC BIOLOGICAL PATHWAYS TO {target_disease.upper()} ===")
    
    # Signature interpretations (from your documentation)
    signature_interpretations = {
        0: "Cardiac Structure & Rhythm Disorders",
        5: "Coronary Atherosclerosis", 
        7: "Chronic Pain, Mood & Metabolic",
        12: "Other Cardiovascular Conditions",
        15: "Diabetes & Complications"
    }
    
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
    
    # Analyze each pathway
    pathway_results = {}
    
    for transition_disease, pathway_name, expected_sigs in pathway_pairs:
        print(f"\n--- {pathway_name} PATHWAY: {transition_disease.upper()} → {target_disease.upper()} ---")
        
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
                transition_occurrence = torch.where(Y[patient_id, transition_idx, :target_time_idx] > 0)[0]
                if len(transition_occurrence) > 0:
                    age_at_transition = transition_occurrence.min().item() + 30
                    pathway_patients.append({
                        'patient_id': patient_id,
                        'age_at_target': age_at_target,
                        'age_at_transition': age_at_transition
                    })
        
        print(f"Found {len(pathway_patients)} patients with {pathway_name} pathway")
        
        if len(pathway_patients) == 0:
            continue
        
        # Analyze signature patterns for this pathway
        pathway_analysis = analyze_pathway_signatures(
            pathway_patients, thetas, population_reference, 
            expected_sigs, signature_interpretations, window_years=5
        )
        
        pathway_results[pathway_name] = {
            'transition_disease': transition_disease,
            'n_patients': len(pathway_patients),
            'expected_signatures': expected_sigs,
            'analysis': pathway_analysis
        }
    
    return pathway_results


def analyze_pathway_signatures(pathway_patients, thetas, population_reference, 
                              expected_sigs, signature_interpretations, window_years=5):
    """
    Analyze signature patterns for a specific pathway with biological interpretation
    """
    K, T = thetas.shape[1], thetas.shape[2]
    
    # Focus on expected signatures
    sig_indices = expected_sigs
    n_sigs = len(sig_indices)
    
    # Calculate deviations in pre-disease window
    deviations = np.zeros((n_sigs, T))
    n_valid_patients = np.zeros(T)
    
    for t in range(T):
        valid_patients = []
        
        for patient_info in pathway_patients:
            patient_id = patient_info['patient_id']
            age_at_target = patient_info['age_at_target']
            target_time_idx = age_at_target - 30
            
            # Only include if current time is in pre-disease window
            if t < target_time_idx and t >= max(0, target_time_idx - window_years):
                valid_patients.append(patient_id)
        
        if len(valid_patients) > 0:
            # Calculate mean signature loading for this timepoint
            patient_thetas = thetas[valid_patients, :, t]  # Shape: (n_patients, K)
            mean_theta_t = np.mean(patient_thetas, axis=0)  # Shape: (K,)
            
            # Calculate deviations from reference for expected signatures
            for i, sig_idx in enumerate(sig_indices):
                deviations[i, t] = mean_theta_t[sig_idx] - population_reference[sig_idx, t]
            
            n_valid_patients[t] = len(valid_patients)
    
    # Calculate summary statistics
    summary_stats = {}
    for i, sig_idx in enumerate(sig_indices):
        sig_name = signature_interpretations.get(sig_idx, f"Signature {sig_idx}")
        sig_deviations = deviations[i, :]
        
        # Average deviation over the pre-disease window
        mean_dev = np.nanmean(sig_deviations)
        max_dev = np.nanmax(np.abs(sig_deviations))
        
        summary_stats[sig_idx] = {
            'name': sig_name,
            'mean_deviation': mean_dev,
            'max_deviation': max_dev,
            'deviations_over_time': sig_deviations
        }
    
    return {
        'deviations': deviations,
        'signature_indices': sig_indices,
        'summary_stats': summary_stats,
        'n_valid_patients': n_valid_patients
    }


def visualize_pathway_comparison(pathway_results, save_plots=True):
    """
    Create side-by-side comparison of different biological pathways
    """
    if len(pathway_results) == 0:
        print("No pathway results to visualize")
        return
    
    print(f"\n=== CREATING PATHWAY COMPARISON VISUALIZATION ===")
    
    # Create figure with subplots for each pathway
    n_pathways = len(pathway_results)
    fig, axes = plt.subplots(n_pathways, 1, figsize=(14, 5*n_pathways))
    if n_pathways == 1:
        axes = [axes]
    
    fig.suptitle('Biological Pathway Comparison: Signature Deviations Before Myocardial Infarction', 
                 fontsize=16, fontweight='bold')
    
    # Colors for different signatures
    sig_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    pathway_names = list(pathway_results.keys())
    
    for i, (pathway_name, results) in enumerate(pathway_results.items()):
        ax = axes[i]
        analysis = results['analysis']
        
        deviations = analysis['deviations']
        sig_indices = analysis['signature_indices']
        summary_stats = analysis['summary_stats']
        
        # Create time axis (years before event)
        T = deviations.shape[1]
        time_points = np.arange(-5, 0, 5/T)[:T]
        
        # Plot each signature
        for j, sig_idx in enumerate(sig_indices):
            sig_name = summary_stats[sig_idx]['name']
            sig_deviations = deviations[j, :]
            
            ax.plot(time_points, sig_deviations, 
                   color=sig_colors[j % len(sig_colors)], 
                   linewidth=2, marker='o', markersize=4,
                   label=f'{sig_name}')
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'{pathway_name} Pathway (n={results["n_patients"]} patients)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Years Before Myocardial Infarction')
        ax.set_ylabel('Signature Deviation from Population')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('pathway_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'pathway_comparison.png'")
    
    plt.show()
    
    # Print biological interpretation
    print(f"\n=== BIOLOGICAL INTERPRETATION ===")
    for pathway_name, results in pathway_results.items():
        print(f"\n{pathway_name} Pathway:")
        summary_stats = results['analysis']['summary_stats']
        
        for sig_idx, stats in summary_stats.items():
            mean_dev = stats['mean_deviation']
            direction = "↑" if mean_dev > 0 else "↓"
            print(f"  {stats['name']}: {mean_dev:+.4f} {direction}")
        
        # Biological interpretation
        if pathway_name == "Inflammatory":
            print("  → Suggests inflammatory mechanisms driving cardiovascular disease")
        elif pathway_name == "Metabolic": 
            print("  → Suggests metabolic dysfunction leading to cardiovascular complications")
        elif pathway_name == "Cardiovascular":
            print("  → Suggests direct cardiovascular pathway with structural changes")


def run_pathway_comparison(target_disease="myocardial infarction"):
    """
    Run complete pathway comparison analysis
    """
    # Load data (same as your other scripts)
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
    
    # Run pathway comparison
    pathway_results = compare_specific_pathways(
        Y, thetas, disease_names, processed_ids, target_disease
    )
    
    if pathway_results:
        # Create visualization
        visualize_pathway_comparison(pathway_results)
        
        return pathway_results
    else:
        print("No pathway results generated")
        return None


if __name__ == "__main__":
    results = run_pathway_comparison()
