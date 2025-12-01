"""
Calculate π (disease probabilities) for each MI pathway

π is calculated as: π = einsum('nkt,kdt->ndt', theta, phi_prob) * kappa

For each pathway, we want to see what diseases they're most likely to develop (highest π values)
"""

import torch
import numpy as np
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data

def calculate_pathway_pi(target_disease="myocardial infarction", output_dir="output_10yr"):
    """
    Calculate π (disease probabilities) for each pathway
    
    Returns average disease probabilities by pathway
    """
    
    # Load data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    N_total = thetas.shape[0]
    print(f"Loaded {N_total} patients")
    
    # Check if pi values already exist
    pi_save_path = f'{output_dir}/all_pi_values.npy'
    import os
    if os.path.exists(pi_save_path):
        print(f"\nFound existing pi values at {pi_save_path}")
        print("Loading instead of recomputing...")
        pi_all = np.load(pi_save_path)
        print(f"Loaded π for {pi_all.shape[0]} patients")
    else:
        # Need to compute pi
        pi_all = None
    
    # Load pathway results
    import pickle
    results_file = f'{output_dir}/complete_analysis_results.pkl'
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    pathway_data = results['pathway_data_dev']
    patients = pathway_data['patients']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    
    # Calculate pi for all patients by processing each 10K batch
    # We have 40 batches: 0-10K, 10K-20K, ..., 390K-400K
    batch_size = 10000
    n_batches = N_total // batch_size
    K, D, T = thetas.shape[1], len(disease_names), thetas.shape[2]
    
    if pi_all is not None:
        print("Using pre-computed π values")
    else:
        # Build pi_all incrementally, loading and clearing each batch
        print(f"\nProcessing {n_batches} batches to calculate π...")
        print("Loading each batch, computing pi, then clearing from memory")
        pi_all_list = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            
            print(f"\n{'='*60}")
            print(f"Processing batch {batch_idx}/{n_batches}: patients {start_idx}-{end_idx}")
            print(f"{'='*60}")
            
            model_path = f'/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_{start_idx}_{end_idx}.pt'
            
            try:
                # Load model
                model_dict = torch.load(model_path, weights_only=False)['model_state_dict']
                
                # Extract parameters
                phi = model_dict['phi']
                kappa = model_dict['kappa']
                
                if isinstance(phi, torch.Tensor):
                    phi = phi.detach().numpy()
                if isinstance(kappa, torch.Tensor):
                    kappa = kappa.item()
                
                # Calculate phi_prob
                phi_prob = 1.0 / (1.0 + np.exp(-phi))  # [K, D, T]
                
                # Get thetas for this batch
                batch_thetas = thetas[start_idx:end_idx, :, :]  # [batch_size, K, T]
                
                # Calculate pi for this batch using vectorized operations
                batch_pi = np.zeros((batch_thetas.shape[0], D, T))
                
                for d in range(D):
                    for t in range(T):
                        # [N] = [N, K] @ [K] * scalar
                        batch_pi[:, d, t] = (batch_thetas[:, :, t] @ phi_prob[:, d, t]) * kappa
                
                pi_all_list.append(batch_pi)
                print(f"  ✓ Calculated pi for batch {batch_idx} (shape: {batch_pi.shape})")
                
                # Clear model from memory
                del model_dict, phi, kappa, phi_prob, batch_thetas, batch_pi
                import gc
                gc.collect()
                
            except FileNotFoundError:
                print(f"  ✗ Model file not found for batch {batch_idx}, skipping...")
        
        # Concatenate all batches (after ALL loops complete)
        print(f"\n{'='*80}")
        print(f"SUMMARY: Processed {len(pi_all_list)} batches out of {n_batches}")
        print(f"{'='*80}")
        
        if len(pi_all_list) > 0:
            shapes = [arr.shape for arr in pi_all_list[:5]]
            print(f"First 5 batch shapes: {shapes}")
            total_patients = sum(arr.shape[0] for arr in pi_all_list)
            print(f"Total patients in all batches: {total_patients}")
        
        if len(pi_all_list) == 0:
            print("ERROR: No batches were successfully processed!")
            raise ValueError("Failed to process any batches")
        
        pi_all = np.concatenate(pi_all_list, axis=0)  # [N_total, D, T]
        print(f"\n✓ Concatenated: Final pi_all shape = {pi_all.shape}")
        
        # Save pi_all to disk for future use
        np.save(pi_save_path, pi_all)
        print(f"✓ Saved π values to: {pi_save_path}")
    
    print(f"\n{'='*80}")
    print("CALCULATING π (DISEASE PROBABILITIES) FOR EACH PATHWAY")
    print(f"{'='*80}\n")
    
    pathway_pi_results = {}
    
    for pathway_id in range(n_pathways):
        print(f"\n{'='*60}")
        print(f"PATHWAY {pathway_id}")
        print('='*60)
        
        # Get patients in this pathway
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        pathway_patient_ids = [p['patient_id'] for p in pathway_patients]
        n_pathway_patients = len(pathway_patient_ids)
        
        print(f"Number of patients: {n_pathway_patients}")
        
        # Get pi for this pathway from pre-computed pi_all
        pathway_pi = pi_all[pathway_patient_ids, :, :]  # [N_pathway, D, T]
        
        # Get π at the time of MI (age of disease onset) for each patient in this pathway
        pi_at_event = np.zeros((n_pathway_patients, D))  # [N_pathway, D]
        
        for i, patient_info in enumerate(pathway_patients):
            patient_id = patient_info['patient_id']
            age_at_mi = patient_info['age_at_disease']
            mi_time_idx = age_at_mi - 30  # Convert age to time index
            
            if 0 <= mi_time_idx < T:
                # Get π at the time of MI event for this patient
                pi_at_event[i, :] = pathway_pi[i, :, mi_time_idx]  # [D] - π for all diseases at MI time
        
        # Average across patients in this pathway at the time of their MI
        pi_pathway_at_mi = np.mean(pi_at_event, axis=0)  # [D] - average π at MI time
        
        # For comparison, compute population baseline π at the average age of MI for this pathway
        avg_mi_age = np.mean([p['age_at_disease'] for p in pathway_patients])
        avg_mi_time_idx = int(avg_mi_age - 30)
        
        if 0 <= avg_mi_time_idx < T:
            # Get population average π at this time
            population_pi_at_time = np.mean(pi_all[:, :, avg_mi_time_idx], axis=0)  # [D]
        else:
            population_pi_at_time = np.mean(pi_all.reshape(-1, D), axis=0)  # Average over all
        
        # Get top diseases by probability at time of MI
        disease_probs = pi_pathway_at_mi  # [D] - π at MI time
        disease_deviation = disease_probs - population_pi_at_time  # Deviation from population
        disease_hazard_ratio = np.divide(disease_probs, population_pi_at_time + 1e-8)  # HR = π_pathway / π_pop
        
        # Sort and get top diseases by hazard ratio (most elevated vs population)
        top_disease_indices = np.argsort(disease_hazard_ratio)[::-1][:10]
        
        print(f"\nTop diseases by hazard ratio at time of MI:")
        print(f"Average MI age: {avg_mi_age:.1f} years")
        for idx in top_disease_indices:
            disease_name = disease_names[idx]
            prob = disease_probs[idx]
            pop_prob = population_pi_at_time[idx]
            hr = disease_hazard_ratio[idx]
            dev = disease_deviation[idx]
            print(f"  {disease_name:50s}: π = {prob:.4f}, pop = {pop_prob:.4f}, HR = {hr:.2f}x, Δ = {dev:+.4f}")
        
        # Store results
        pathway_pi_results[pathway_id] = {
            'n_patients': n_pathway_patients,
            'avg_mi_age': avg_mi_age,
            'disease_probs_at_mi': disease_probs,
            'population_probs_at_time': population_pi_at_time,
            'disease_deviation': disease_deviation,
            'disease_hazard_ratio': disease_hazard_ratio,
            'top_diseases': [(disease_names[idx], disease_probs[idx], population_pi_at_time[idx], 
                            disease_hazard_ratio[idx], disease_deviation[idx])
                            for idx in top_disease_indices]
        }
        
        # Also calculate for MI specifically
        mi_idx = next(i for i, name in enumerate(disease_names) if 'myocardial infarction' in name.lower())
        mi_prob = disease_probs[mi_idx]
        mi_pop_prob = population_pi_at_time[mi_idx]
        mi_hr = disease_hazard_ratio[mi_idx]
        mi_dev = disease_deviation[mi_idx]
        print(f"\nMyocardial Infarction:")
        print(f"  Pathway π: {mi_prob:.4f}")
        print(f"  Population π: {mi_pop_prob:.4f}")
        print(f"  Hazard Ratio: {mi_hr:.2f}x")
        print(f"  Deviation: {mi_dev:+.4f}")
    
    return pathway_pi_results

if __name__ == "__main__":
    results = calculate_pathway_pi()
