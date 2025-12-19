#!/usr/bin/env python3
"""
Create Supplementary Figure S6: Individual Patient Trajectories

This script generates S6 showing individual patient trajectories with signature loadings,
disease timelines, and disease probabilities for selected example patients.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add path to import plot_patient_timeline
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/pythonscripts')
from plot_patient_timeline_v3 import plot_patient_timeline


def find_good_example_patients(
    Y_path: str,
    theta_path: str,
    disease_names_path: str,
    min_diseases: int = 3,
    min_time_spread: int = 10,
    top_n: int = 20
) -> list:
    """
    Find good example patients for trajectory visualization.
    
    Criteria:
    - Multiple diseases (at least min_diseases)
    - Diseases spread over time (min_time_spread years)
    - Diverse signature involvement
    
    Args:
        Y_path: Path to Y tensor (N, D, T)
        theta_path: Path to theta tensor (N, K, T)
        disease_names_path: Path to disease names CSV
        min_diseases: Minimum number of diseases
        min_time_spread: Minimum time spread in years
        top_n: Number of top candidates to return
        
    Returns:
        List of patient indices sorted by quality score
    """
    print("="*80)
    print("FINDING GOOD EXAMPLE PATIENTS")
    print("="*80)
    
    # Load Y
    Y = torch.load(Y_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(Y):
        Y_np = Y.numpy()
    else:
        Y_np = Y
    
    # Load theta
    theta_full = torch.load(theta_path, map_location='cpu', weights_only=False)
    if isinstance(theta_full, dict):
        if 'theta' in theta_full:
            theta = theta_full['theta']
        elif 'thetas' in theta_full:
            theta = theta_full['thetas']
        else:
            theta = list(theta_full.values())[0]
    else:
        theta = theta_full
    
    if torch.is_tensor(theta):
        theta = theta.numpy()
    
    N, D, T = Y_np.shape
    print(f"Loaded data: N={N}, D={D}, T={T}")
    
    # Score patients
    patient_scores = []
    
    for n in range(N):
        # Count diseases
        n_diseases = np.sum(np.any(Y_np[n, :, :] > 0.5, axis=1))
        
        if n_diseases < min_diseases:
            continue
        
        # Find diagnosis times
        diagnosis_times = []
        for d in range(D):
            event_times = np.where(Y_np[n, d, :] > 0.5)[0]
            if len(event_times) > 0:
                diagnosis_times.extend(event_times.tolist())
        
        if len(diagnosis_times) == 0:
            continue
        
        # Calculate time spread
        time_spread = max(diagnosis_times) - min(diagnosis_times)
        if time_spread < min_time_spread:
            continue
        
        # Calculate signature diversity (how many signatures are active)
        if theta.shape[0] > n:
            patient_theta = theta[n, :, :]  # (K, T)
            avg_theta = patient_theta.mean(axis=1)  # (K,)
            n_active_sigs = np.sum(avg_theta > 0.05)  # Signatures with >5% average loading
        else:
            n_active_sigs = 0
        
        # Score: prioritize more diseases, longer time spread, more signature diversity
        score = (n_diseases * 10) + (time_spread * 2) + (n_active_sigs * 5)
        
        patient_scores.append((score, n, n_diseases, time_spread, n_active_sigs))
    
    # Sort by score
    patient_scores.sort(reverse=True, key=lambda x: x[0])
    
    print(f"\nFound {len(patient_scores)} patients meeting criteria")
    print(f"\nTop {min(top_n, len(patient_scores))} candidates:")
    print(f"{'Rank':<6} {'Patient':<10} {'Score':<8} {'Diseases':<10} {'Time Spread':<12} {'Active Sigs':<12}")
    print("-" * 70)
    
    for i, (score, n, n_diseases, time_spread, n_active_sigs) in enumerate(patient_scores[:top_n]):
        print(f"{i+1:<6} {n:<10} {score:<8.1f} {n_diseases:<10} {time_spread:<12} {n_active_sigs:<12}")
    
    return [n for _, n, _, _, _ in patient_scores[:top_n]]


def create_S6_figure(
    patient_indices: list = None,
    output_path: str = None,
    theta_path: str = '/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective_correctE.pt',
    checkpoint_path: str = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt',
    pi_path: str = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/pi_fullmode_400k.pt',
    Y_path: str = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt',
    initial_clusters_path: str = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt',
    disease_names_path: str = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv',
    auto_select: bool = True,
    n_patients: int = 3
):
    """
    Create S6 figure with individual patient trajectories.
    
    Args:
        patient_indices: List of patient indices to plot. If None and auto_select=True, will find good examples.
        output_path: Path to save S6 figure. If None, uses default S6 location.
        theta_path: Path to theta file
        checkpoint_path: Path to checkpoint file
        pi_path: Path to pi predictions file
        Y_path: Path to Y tensor
        initial_clusters_path: Path to initial clusters
        disease_names_path: Path to disease names CSV
        auto_select: If True, automatically find good example patients
        n_patients: Number of patients to include if auto_selecting
    """
    print("="*80)
    print("CREATING S6: INDIVIDUAL PATIENT TRAJECTORIES")
    print("="*80)
    
    if output_path is None:
        output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s6/S6.pdf'
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-select patients if needed
    if patient_indices is None and auto_select:
        print("\nAuto-selecting good example patients...")
        patient_indices = find_good_example_patients(
            Y_path=Y_path,
            theta_path=theta_path,
            disease_names_path=disease_names_path,
            min_diseases=3,
            min_time_spread=10,
            top_n=n_patients
        )
        print(f"\nSelected patients: {patient_indices}")
    elif patient_indices is None:
        # Default examples (you can customize these)
        patient_indices = [148745, 50000, 100000]  # Example indices
        print(f"\nUsing default patient indices: {patient_indices}")
    
    # Limit to n_patients
    patient_indices = patient_indices[:n_patients]
    
    # Create individual plots for each patient
    # For S6, we might want to create a multi-panel figure or individual plots
    # Let's create individual high-quality plots that can be combined
    
    print(f"\nGenerating trajectories for {len(patient_indices)} patients...")
    
    individual_figs = []
    for i, patient_idx in enumerate(patient_indices):
        print(f"\n  Patient {i+1}/{len(patient_indices)}: Index {patient_idx}")
        
        # Create individual patient plot
        individual_output = output_dir / f'S6_patient_{patient_idx}.pdf'
        
        fig = plot_patient_timeline(
            patient_idx=patient_idx,
            theta_path=theta_path,
            checkpoint_path=checkpoint_path,
            pi_path=pi_path,
            Y_path=Y_path,
            initial_clusters_path=initial_clusters_path,
            disease_names_path=disease_names_path,
            output_path=str(individual_output),
            figsize=(20, 14)
        )
        
        individual_figs.append(fig)
        print(f"    ✓ Saved to: {individual_output}")
    
    # For the main S6 figure, we could create a combined multi-panel figure
    # showing 2-3 patients side-by-side, or use one representative patient
    # For now, let's use the first (best) patient as the main S6 figure
    
    if len(patient_indices) > 0:
        print(f"\nCreating main S6 figure using patient {patient_indices[0]}...")
        main_fig = plot_patient_timeline(
            patient_idx=patient_indices[0],
            theta_path=theta_path,
            checkpoint_path=checkpoint_path,
            pi_path=pi_path,
            Y_path=Y_path,
            initial_clusters_path=initial_clusters_path,
            disease_names_path=disease_names_path,
            output_path=output_path,
            figsize=(20, 14)
        )
        print(f"✓ Main S6 figure saved to: {output_path}")
    
    print("\n" + "="*80)
    print("S6 GENERATION COMPLETE")
    print("="*80)
    print(f"Main figure: {output_path}")
    print(f"Individual patient figures saved to: {output_dir}")
    
    return individual_figs


def main():
    """Main function to generate S6."""
    # You can specify patient indices here, or let it auto-select
    # Example: patient_indices = [148745, 50000, 100000]
    
    create_S6_figure(
        patient_indices=None,  # Auto-select good examples
        auto_select=True,
        n_patients=3
    )


if __name__ == '__main__':
    main()

