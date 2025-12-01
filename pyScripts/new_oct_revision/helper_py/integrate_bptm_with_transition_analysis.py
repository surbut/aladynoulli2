"""
Integration example: Using BPTM with existing transition analysis

This script shows how to integrate the Bayesian Pathway Transition Model
with the existing transition analysis pipeline.
"""

import numpy as np
import torch
import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from bayesian_pathway_transition_model import BayesianPathwayTransitionModel
from plot_transition_deviations import plot_bc_to_mi_progression
from pathway_discovery import load_full_data


def integrate_bptm_with_transition_analysis(
    transition_disease_name='Rheumatoid arthritis',
    target_disease_name='myocardial infarction',
    years_before=10,
    age_tolerance=5,
    min_followup=5
):
    """
    Integrate BPTM with existing transition analysis workflow.
    
    This demonstrates:
    1. Running existing transition analysis
    2. Extracting signature deviations
    3. Fitting BPTM model
    4. Making probabilistic predictions
    """
    
    print("="*80)
    print("INTEGRATING BPTM WITH TRANSITION ANALYSIS")
    print("="*80)
    
    # ============================================================================
    # STEP 1: Load data and run existing transition analysis
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: Running existing transition analysis")
    print("="*80)
    
    Y, thetas, disease_names, _ = load_full_data()
    
    # Convert to numpy if needed
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    if isinstance(thetas, torch.Tensor):
        thetas = thetas.numpy()
    
    print(f"✅ Data loaded:")
    print(f"   Y shape: {Y.shape}")
    print(f"   Thetas shape: {thetas.shape}")
    print(f"   Diseases: {len(disease_names)}")
    
    # Run existing transition analysis
    transition_results = plot_bc_to_mi_progression(
        transition_disease_name=transition_disease_name,
        target_disease_name=target_disease_name,
        Y=Y,
        thetas=thetas,
        disease_names=disease_names,
        years_before=years_before,
        age_tolerance=age_tolerance,
        min_followup=min_followup,
        save_plots=False  # Don't save plots for this example
    )
    
    if transition_results is None:
        print("❌ Transition analysis failed")
        return None
    
    print("✅ Transition analysis complete")
    
    # ============================================================================
    # STEP 2: Extract signature deviations and pathway information
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: Extracting signature deviations")
    print("="*80)
    
    N, K, T = thetas.shape
    
    # Compute population reference
    population_reference = np.mean(thetas, axis=0)  # (K, T)
    
    # Get transition patients from results
    progressor_deviations = transition_results.get('progressor_deviations', [])
    non_progressor_deviations = transition_results.get('non_progressor_deviations', [])
    
    print(f"✅ Extracted deviations:")
    print(f"   Progressors: {len(progressor_deviations)}")
    print(f"   Non-progressors: {len(non_progressor_deviations)}")
    
    # ============================================================================
    # STEP 3: Initialize BPTM model
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 3: Initializing BPTM model")
    print("="*80)
    
    model = BayesianPathwayTransitionModel(
        K=K,
        T=T,
        P=0,  # No genetic factors for now
        lookback_window=years_before,
        n_pathways=4,  # Could be determined from pathway discovery
        use_pathway_effects=False  # Start simple
    )
    
    print("✅ BPTM model initialized")
    
    # ============================================================================
    # STEP 4: Fit model (placeholder - would use full MCMC)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 4: Fitting BPTM model")
    print("="*80)
    
    # For demonstration, we'll manually set some parameters based on the data
    # In full implementation, this would use MCMC
    
    # Set population reference
    model.mu_k_t = population_reference
    
    # Initialize parameters based on empirical patterns
    # (In full MCMC, these would be sampled from posterior)
    
    # Baseline transition rate
    n_progressors = len(progressor_deviations)
    n_non_progressors = len(non_progressor_deviations)
    
    if n_non_progressors > 0:
        baseline_logit = np.log(n_progressors / n_non_progressors)
    else:
        baseline_logit = 0.0
    
    model.alpha[transition_disease_name] = baseline_logit
    model.beta[transition_disease_name] = 0.0  # No time trend initially
    
    # Initialize signature effects (would be learned via MCMC)
    # For now, use small random values
    model.gamma = np.random.normal(0, 0.1, size=K)
    model.omega = np.random.normal(0, 0.05, size=(K, years_before))
    
    print("✅ Model parameters initialized")
    print(f"   Baseline logit: {baseline_logit:.3f}")
    print(f"   Signature effects shape: {model.gamma.shape}")
    
    # ============================================================================
    # STEP 5: Make predictions for example patients
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 5: Making predictions")
    print("="*80)
    
    # Example: Predict for a progressor patient
    if len(progressor_deviations) > 0:
        example_patient = progressor_deviations[0]
        patient_id = example_patient.get('patient_id')
        t_precursor = example_patient.get('t_precursor')
        
        # Get patient's signature deviations
        patient_deviations = thetas[patient_id, :, :]  # (K, T)
        
        # Predict transition probability trajectory
        trajectory = model.predict_transition_trajectories(
            patient_deviations,
            t_precursor,
            precursor_disease=transition_disease_name,
            max_horizon=10
        )
        
        print(f"\n✅ Example prediction for patient {patient_id}:")
        print(f"   Precursor at timepoint: {t_precursor}")
        print(f"   Transition probability trajectory (next 10 timepoints):")
        for i, prob in enumerate(trajectory):
            print(f"     t+{i+1}: {prob:.4f}")
    
    # ============================================================================
    # STEP 6: Compare progressors vs non-progressors
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 6: Comparing predictions for progressors vs non-progressors")
    print("="*80)
    
    # Average predictions for each group
    progressor_predictions = []
    non_progressor_predictions = []
    
    # Progressors
    for patient_data in progressor_deviations[:10]:  # Sample first 10
        patient_id = patient_data.get('patient_id')
        t_precursor = patient_data.get('t_precursor')
        
        if patient_id < N and t_precursor < T:
            patient_deviations = thetas[patient_id, :, :]
            trajectory = model.predict_transition_trajectories(
                patient_deviations,
                t_precursor,
                precursor_disease=transition_disease_name,
                max_horizon=5
            )
            progressor_predictions.append(trajectory)
    
    # Non-progressors
    for patient_data in non_progressor_deviations[:10]:  # Sample first 10
        patient_id = patient_data.get('patient_id')
        t_precursor = patient_data.get('t_precursor')
        
        if patient_id < N and t_precursor < T:
            patient_deviations = thetas[patient_id, :, :]
            trajectory = model.predict_transition_trajectories(
                patient_deviations,
                t_precursor,
                precursor_disease=transition_disease_name,
                max_horizon=5
            )
            non_progressor_predictions.append(trajectory)
    
    if len(progressor_predictions) > 0 and len(non_progressor_predictions) > 0:
        avg_progressor = np.mean(progressor_predictions, axis=0)
        avg_non_progressor = np.mean(non_progressor_predictions, axis=0)
        
        print(f"\n✅ Average predictions:")
        print(f"   Progressors (n={len(progressor_predictions)}):")
        for i, prob in enumerate(avg_progressor):
            print(f"     t+{i+1}: {prob:.4f}")
        print(f"   Non-progressors (n={len(non_progressor_predictions)}):")
        for i, prob in enumerate(avg_non_progressor):
            print(f"     t+{i+1}: {prob:.4f}")
    
    print("\n" + "="*80)
    print("✅ INTEGRATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Implement full MCMC sampling for parameter estimation")
    print("2. Add pathway-specific effects")
    print("3. Validate predictive performance")
    print("4. Compare with existing descriptive methods")
    
    return {
        'model': model,
        'transition_results': transition_results,
        'progressor_predictions': progressor_predictions if 'progressor_predictions' in locals() else [],
        'non_progressor_predictions': non_progressor_predictions if 'non_progressor_predictions' in locals() else []
    }


if __name__ == "__main__":
    results = integrate_bptm_with_transition_analysis(
        transition_disease_name='Rheumatoid arthritis',
        target_disease_name='myocardial infarction',
        years_before=10,
        age_tolerance=5,
        min_followup=5
    )
    
    print("\n✅ Integration example complete!")


