import numpy as np
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import gc
import matplotlib.pyplot as plt


def calculate_pathway_pi():
    """
    Calculate disease probabilities (π) for MI patients grouped by pathways.
    
    This function:
    1. Loads or computes π values for all patients across all time points
    2. Extracts π at the time of MI for each patient
    3. Groups patients by pathway and compares to population baseline
    4. Performs statistical tests and saves results
    """
    
    # ========================================================================
    # SETUP & CONFIGURATION
    # ========================================================================
    
    results_dir = Path("results")  # Adjust to your actual results directory
    pi_save_path = results_dir / "pi_all_patients.npy"
    
    # Load your data (adjust these to match your actual data structures)
    # Assuming you have:
    # - patients: list of dicts with 'patient_id', 'pathway', 'age_at_disease'
    # - disease_names: list of disease names
    # - D: number of diseases
    # - T: number of time points
    # - n_batches: number of patient batches
    
    # Example placeholders - replace with your actual data loading
    # patients = load_patients()  # Your function to load patient data
    # disease_names = load_disease_names()  # Your function to load disease names
    # D = len(disease_names)
    # T = 81  # Age 30-110
    # n_batches = get_number_of_batches()
    
    # For this example, using placeholder variables
    # You'll need to replace these with your actual data
    print("Note: Update the data loading section with your actual data sources")
    
    # ========================================================================
    # LOAD OR COMPUTE π VALUES
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("LOADING π VALUES FOR ALL PATIENTS")
    print(f"{'='*80}\n")
    
    if pi_save_path.exists():
        print(f"✓ Loading pre-computed π from: {pi_save_path}")
        pi_all = np.load(pi_save_path)
        print(f"✓ Loaded: pi_all shape = {pi_all.shape}")
    else:
        print(f"Computing π values from scratch...")
        print(f"This may take a while...\n")
        
        pi_all_list = []
        
        for batch_idx in range(n_batches):
            try:
                print(f"Processing batch {batch_idx + 1}/{n_batches}...")
                
                # Load model for this batch
                model_path = results_dir / f"model_batch_{batch_idx}.pkl"
                # model_dict = load_model(model_path)  # Your function to load model
                
                # Extract parameters
                # phi = model_dict['phi']  # [N_batch, K]
                # kappa = model_dict['kappa']  # [K, D, T]
                
                # Compute π = softmax(φκ)
                # logits = np.einsum('nk,kdt->ndt', phi, kappa)  # [N_batch, D, T]
                # phi_prob = softmax(logits, axis=1)  # [N_batch, D, T]
                
                # pi_all_list.append(phi_prob)
                
                # Clear memory
                # del model_dict, phi, kappa, phi_prob
                # gc.collect()
                
            except FileNotFoundError:
                print(f"  ✗ Model file not found for batch {batch_idx}, skipping...")
        
        # Concatenate all batches
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
        
        # Save π to disk for future use
        np.save(pi_save_path, pi_all)
        print(f"✓ Saved π values to: {pi_save_path}")
    
    # ========================================================================
    # CREATE PATIENT ID TO INDEX MAPPING (IMPORTANT FIX!)
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("CREATING PATIENT INDEX MAPPING")
    print(f"{'='*80}\n")
    
    patient_id_to_idx = {p['patient_id']: idx for idx, p in enumerate(patients)}
    print(f"✓ Created mapping for {len(patient_id_to_idx)} patients")
    
    # Verify mapping
    sample_ids = list(patient_id_to_idx.keys())[:5]
    print(f"Sample mappings: {[(pid, patient_id_to_idx[pid]) for pid in sample_ids]}")
    
    # ========================================================================
    # FIND MI DISEASE INDEX
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("IDENTIFYING MYOCARDIAL INFARCTION DISEASE")
    print(f"{'='*80}\n")
    
    mi_indices = [i for i, name in enumerate(disease_names) 
                  if 'myocardial infarction' in name.lower() or 'mi' == name.lower()]
    
    if not mi_indices:
        print("WARNING: MI disease not found in disease names!")
        print("Searching for alternative terms...")
        mi_indices = [i for i, name in enumerate(disease_names) 
                     if 'heart attack' in name.lower() or 'acute coronary' in name.lower()]
    
    if mi_indices:
        mi_idx = mi_indices[0]
        print(f"✓ Found MI at index {mi_idx}: '{disease_names[mi_idx]}'")
    else:
        print("✗ Could not find MI disease index")
        mi_idx = None
    
    # ========================================================================
    # ANALYZE EACH PATHWAY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("CALCULATING π (DISEASE PROBABILITIES) FOR EACH PATHWAY")
    print(f"{'='*80}\n")
    
    # Get unique pathways
    n_pathways = len(set(p['pathway'] for p in patients))
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
        
        if n_pathway_patients == 0:
            print("⚠ No patients in this pathway, skipping...")
            continue
        
        # FIXED: Use proper index mapping
        pathway_indices = [patient_id_to_idx[pid] for pid in pathway_patient_ids]
        pathway_pi = pi_all[pathway_indices, :, :]  # [N_pathway, D, T]
        
        print(f"✓ Extracted π for pathway: shape = {pathway_pi.shape}")
        
        # ====================================================================
        # EXTRACT π AT TIME OF MI FOR EACH PATIENT
        # ====================================================================
        
        pi_at_event = np.zeros((n_pathway_patients, D))  # [N_pathway, D]
        valid_patients = []
        
        for i, patient_info in enumerate(pathway_patients):
            patient_id = patient_info['patient_id']
            age_at_mi = patient_info['age_at_disease']
            mi_time_idx = age_at_mi - 30  # Convert age to time index (assuming age 30-110)
            
            if 0 <= mi_time_idx < T:
                # Get π at the time of MI event for this patient
                pi_at_event[i, :] = pathway_pi[i, :, mi_time_idx]  # [D]
                valid_patients.append(i)
            else:
                print(f"  ⚠ Patient {patient_id}: age {age_at_mi} out of range")
        
        if len(valid_patients) == 0:
            print("✗ No valid patients with MI times in range")
            continue
        
        # Use only valid patients for analysis
        pi_at_event = pi_at_event[valid_patients, :]
        valid_pathway_patients = [pathway_patients[i] for i in valid_patients]
        n_valid = len(valid_patients)
        
        print(f"✓ Valid patients with MI in time range: {n_valid}/{n_pathway_patients}")
        
        # ====================================================================
        # COMPUTE PATHWAY STATISTICS
        # ====================================================================
        
        # Average π across patients at the time of their MI
        pi_pathway_at_mi = np.mean(pi_at_event, axis=0)  # [D]
        pi_pathway_std = np.std(pi_at_event, axis=0)  # [D]
        
        # Age statistics
        ages_at_mi = [p['age_at_disease'] for p in valid_pathway_patients]
        avg_mi_age = np.mean(ages_at_mi)
        std_mi_age = np.std(ages_at_mi)
        min_mi_age = np.min(ages_at_mi)
        max_mi_age = np.max(ages_at_mi)
        
        print(f"\nAge at MI statistics:")
        print(f"  Mean: {avg_mi_age:.1f} ± {std_mi_age:.1f} years")
        print(f"  Range: {min_mi_age:.0f} - {max_mi_age:.0f} years")
        print(f"  Median: {np.median(ages_at_mi):.1f} years")
        
        # ====================================================================
        # COMPUTE POPULATION BASELINE
        # ====================================================================
        
        avg_mi_time_idx = int(avg_mi_age - 30)
        
        if 0 <= avg_mi_time_idx < T:
            # Get population average π at this time
            population_pi_at_time = np.mean(pi_all[:, :, avg_mi_time_idx], axis=0)  # [D]
        else:
            # Fallback: average over all times
            population_pi_at_time = np.mean(pi_all.reshape(-1, D), axis=0)  # [D]
        
        # ====================================================================
        # CALCULATE METRICS
        # ====================================================================
        
        disease_probs = pi_pathway_at_mi  # [D]
        disease_deviation = disease_probs - population_pi_at_time  # Absolute difference
        disease_hazard_ratio = np.divide(disease_probs, population_pi_at_time + 1e-8)  # HR
        
        # ====================================================================
        # STATISTICAL SIGNIFICANCE TESTING
        # ====================================================================
        
        print(f"\nPerforming statistical tests...")
        
        # T-test: pathway π vs population π for each disease
        t_stats = np.zeros(D)
        p_values = np.zeros(D)
        
        for d in range(D):
            t_stat, p_val = stats.ttest_1samp(pi_at_event[:, d], population_pi_at_time[d])
            t_stats[d] = t_stat
            p_values[d] = p_val
        
        # Bootstrap confidence intervals
        print(f"Computing 95% confidence intervals (1000 bootstrap samples)...")
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(n_valid, n_valid, replace=True)
            bootstrap_means.append(np.mean(pi_at_event[sample_indices, :], axis=0))
        
        bootstrap_means = np.array(bootstrap_means)  # [n_bootstrap, D]
        ci_lower = np.percentile(bootstrap_means, 2.5, axis=0)  # [D]
        ci_upper = np.percentile(bootstrap_means, 97.5, axis=0)  # [D]
        
        # ====================================================================
        # IDENTIFY TOP DISEASES
        # ====================================================================
        
        # Sort by hazard ratio (most elevated vs population)
        top_disease_indices_hr = np.argsort(disease_hazard_ratio)[::-1][:10]
        
        # Also sort by absolute deviation
        top_disease_indices_dev = np.argsort(disease_deviation)[::-1][:10]
        
        # Sort by statistical significance
        top_disease_indices_pval = np.argsort(p_values)[:10]
        
        print(f"\n{'='*60}")
        print(f"TOP DISEASES BY HAZARD RATIO AT TIME OF MI")
        print(f"{'='*60}")
        
        for rank, idx in enumerate(top_disease_indices_hr, 1):
            disease_name = disease_names[idx]
            prob = disease_probs[idx]
            pop_prob = population_pi_at_time[idx]
            hr = disease_hazard_ratio[idx]
            dev = disease_deviation[idx]
            ci_low = ci_lower[idx]
            ci_high = ci_upper[idx]
            p_val = p_values[idx]
            
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            print(f"{rank:2d}. {disease_name:45s}")
            print(f"    Pathway π: {prob:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"    Population π: {pop_prob:.4f}")
            print(f"    HR: {hr:.2f}x | Δ: {dev:+.4f} | p: {p_val:.4f} {sig_marker}")
        
        print(f"\n{'='*60}")
        print(f"TOP DISEASES BY ABSOLUTE DEVIATION")
        print(f"{'='*60}")
        
        for rank, idx in enumerate(top_disease_indices_dev, 1):
            disease_name = disease_names[idx]
            prob = disease_probs[idx]
            pop_prob = population_pi_at_time[idx]
            hr = disease_hazard_ratio[idx]
            dev = disease_deviation[idx]
            p_val = p_values[idx]
            
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            print(f"{rank:2d}. {disease_name:45s}: Δ = {dev:+.4f} (HR: {hr:.2f}x, p: {p_val:.4f} {sig_marker})")
        
        # ====================================================================
        # MI-SPECIFIC ANALYSIS
        # ====================================================================
        
        if mi_idx is not None:
            print(f"\n{'='*60}")
            print(f"MYOCARDIAL INFARCTION ANALYSIS")
            print(f"{'='*60}")
            
            mi_prob = disease_probs[mi_idx]
            mi_pop_prob = population_pi_at_time[mi_idx]
            mi_hr = disease_hazard_ratio[mi_idx]
            mi_dev = disease_deviation[mi_idx]
            mi_ci_low = ci_lower[mi_idx]
            mi_ci_high = ci_upper[mi_idx]
            mi_pval = p_values[mi_idx]
            
            print(f"Pathway π:    {mi_prob:.4f} [{mi_ci_low:.4f}, {mi_ci_high:.4f}]")
            print(f"Population π: {mi_pop_prob:.4f}")
            print(f"Hazard Ratio: {mi_hr:.2f}x")
            print(f"Deviation:    {mi_dev:+.4f}")
            print(f"P-value:      {mi_pval:.4f}")
            
            if mi_pval < 0.05:
                print(f"✓ Significantly elevated (p < 0.05)")
            else:
                print(f"⚠ Not significantly different from population")
        
        # ====================================================================
        # π DISTRIBUTION SUMMARY
        # ====================================================================
        
        print(f"\n{'='*60}")
        print(f"π DISTRIBUTION AT MI")
        print(f"{'='*60}")
        
        print(f"Overall statistics:")
        print(f"  Mean:   {np.mean(pi_at_event):.4f}")
        print(f"  Median: {np.median(pi_at_event):.4f}")
        print(f"  Std:    {np.std(pi_at_event):.4f}")
        print(f"  Min:    {np.min(pi_at_event):.4f}")
        print(f"  Max:    {np.max(pi_at_event):.4f}")
        
        # ====================================================================
        # STORE RESULTS
        # ====================================================================
        
        pathway_pi_results[pathway_id] = {
            'n_patients': n_pathway_patients,
            'n_valid_patients': n_valid,
            'avg_mi_age': float(avg_mi_age),
            'std_mi_age': float(std_mi_age),
            'age_range': (float(min_mi_age), float(max_mi_age)),
            'disease_probs_at_mi': disease_probs,
            'disease_probs_std': pi_pathway_std,
            'population_probs_at_time': population_pi_at_time,
            'disease_deviation': disease_deviation,
            'disease_hazard_ratio': disease_hazard_ratio,
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'top_diseases_by_hr': [
                (disease_names[idx], float(disease_probs[idx]), 
                 float(population_pi_at_time[idx]), float(disease_hazard_ratio[idx]),
                 float(disease_deviation[idx]), float(p_values[idx]))
                for idx in top_disease_indices_hr
            ],
            'top_diseases_by_deviation': [
                (disease_names[idx], float(disease_probs[idx]), 
                 float(population_pi_at_time[idx]), float(disease_hazard_ratio[idx]),
                 float(disease_deviation[idx]), float(p_values[idx]))
                for idx in top_disease_indices_dev
            ]
        }
        
        if mi_idx is not None:
            pathway_pi_results[pathway_id]['mi_specific'] = {
                'pi': float(mi_prob),
                'population_pi': float(mi_pop_prob),
                'hazard_ratio': float(mi_hr),
                'deviation': float(mi_dev),
                'ci': (float(mi_ci_low), float(mi_ci_high)),
                'p_value': float(mi_pval)
            }
    
    # ========================================================================
    # CROSS-PATHWAY COMPARISON
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("CROSS-PATHWAY COMPARISON")
    print(f"{'='*80}\n")
    
    pathway_ids = sorted(pathway_pi_results.keys())
    
    print(f"Summary across all pathways:")
    print(f"{'-'*60}")
    print(f"{'Pathway':<10} {'N Patients':<12} {'Valid':<8} {'Avg Age':<12} {'Age Range'}")
    print(f"{'-'*60}")
    
    for pid in pathway_ids:
        data = pathway_pi_results[pid]
        age_range = f"{data['age_range'][0]:.0f}-{data['age_range'][1]:.0f}"
        print(f"{pid:<10} {data['n_patients']:<12} {data['n_valid_patients']:<8} "
              f"{data['avg_mi_age']:>6.1f} ± {data['std_mi_age']:<4.1f}  {age_range}")
    
    # MI-specific comparison across pathways
    if mi_idx is not None and all('mi_specific' in pathway_pi_results[pid] for pid in pathway_ids):
        print(f"\n{'='*60}")
        print(f"MYOCARDIAL INFARCTION π ACROSS PATHWAYS")
        print(f"{'='*60}")
        print(f"{'Pathway':<10} {'π':<12} {'95% CI':<25} {'HR':<8} {'p-value'}")
        print(f"{'-'*60}")
        
        for pid in pathway_ids:
            mi_data = pathway_pi_results[pid]['mi_specific']
            pi_val = mi_data['pi']
            ci = mi_data['ci']
            hr = mi_data['hazard_ratio']
            pval = mi_data['p_value']
            
            sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            
            print(f"{pid:<10} {pi_val:<12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  {hr:<8.2f} {pval:.4f} {sig_marker}")
    
    # Identify pathway with highest/lowest MI risk
    if mi_idx is not None and pathway_ids:
        mi_hrs = {pid: pathway_pi_results[pid]['mi_specific']['hazard_ratio'] 
                  for pid in pathway_ids if 'mi_specific' in pathway_pi_results[pid]}
        
        if mi_hrs:
            highest_risk_pathway = max(mi_hrs, key=mi_hrs.get)
            lowest_risk_pathway = min(mi_hrs, key=mi_hrs.get)
            
            print(f"\n{'='*60}")
            print(f"Highest MI risk: Pathway {highest_risk_pathway} (HR = {mi_hrs[highest_risk_pathway]:.2f}x)")
            print(f"Lowest MI risk:  Pathway {lowest_risk_pathway} (HR = {mi_hrs[lowest_risk_pathway]:.2f}x)")
            print(f"Ratio: {mi_hrs[highest_risk_pathway] / mi_hrs[lowest_risk_pathway]:.2f}x difference")
    
    # Statistical comparison between pathways
    print(f"\n{'='*60}")
    print(f"PAIRWISE PATHWAY COMPARISONS")
    print(f"{'='*60}")
    
    if len(pathway_ids) > 1:
        print(f"\nComparing MI π between pathways (t-tests):")
        
        # Get π values for each pathway
        pathway_pi_values = {}
        for pid in pathway_ids:
            pathway_patients_for_comparison = [p for p in patients if p['pathway'] == pid]
            pathway_indices_for_comparison = [patient_id_to_idx[p['patient_id']] 
                                             for p in pathway_patients_for_comparison]
            
            pi_at_mi = []
            for i, p in enumerate(pathway_patients_for_comparison):
                age_at_mi = p['age_at_disease']
                mi_time_idx = age_at_mi - 30
                if 0 <= mi_time_idx < T and mi_idx is not None:
                    idx = pathway_indices_for_comparison[i]
                    pi_at_mi.append(pi_all[idx, mi_idx, mi_time_idx])
            
            pathway_pi_values[pid] = np.array(pi_at_mi)
        
        # Pairwise comparisons
        for i, pid1 in enumerate(pathway_ids):
            for pid2 in pathway_ids[i+1:]:
                if len(pathway_pi_values[pid1]) > 1 and len(pathway_pi_values[pid2]) > 1:
                    t_stat, p_val = stats.ttest_ind(pathway_pi_values[pid1], 
                                                     pathway_pi_values[pid2])
                    
                    mean1 = np.mean(pathway_pi_values[pid1])
                    mean2 = np.mean(pathway_pi_values[pid2])
                    
                    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    
                    print(f"  Pathway {pid1} vs {pid2}: "
                          f"π₁={mean1:.4f}, π₂={mean2:.4f}, "
                          f"p={p_val:.4f} {sig_marker}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save detailed results to JSON
    results_file = results_dir / "pathway_pi_analysis_results.json"
    
    results_to_save = {
        'metadata': {
            'n_pathways': len(pathway_ids),
            'total_patients': len(patients),
            'mi_disease_index': mi_idx,
            'mi_disease_name': disease_names[mi_idx] if mi_idx is not None else None,
            'time_points': T,
            'n_diseases': D
        },
        'pathways': {}
    }
    
    for pathway_id, data in pathway_pi_results.items():
        results_to_save['pathways'][f'pathway_{pathway_id}'] = {
            'n_patients': int(data['n_patients']),
            'n_valid_patients': int(data['n_valid_patients']),
            'avg_mi_age': float(data['avg_mi_age']),
            'std_mi_age': float(data['std_mi_age']),
            'age_range': [float(data['age_range'][0]), float(data['age_range'][1])],
            'top_diseases_by_hazard_ratio': [
                {
                    'name': name,
                    'pi': float(pi_val),
                    'population_pi': float(pop_pi),
                    'hazard_ratio': float(hr),
                    'deviation': float(dev),
                    'p_value': float(pval)
                }
                for name, pi_val, pop_pi, hr, dev, pval in data['top_diseases_by_hr']
            ],
            'top_diseases_by_deviation': [
                {
                    'name': name,
                    'pi': float(pi_val),
                    'population_pi': float(pop_pi),
                    'hazard_ratio': float(hr),
                    'deviation': float(dev),
                    'p_value': float(pval)
                }
                for name, pi_val, pop_pi, hr, dev, pval in data['top_diseases_by_deviation']
            ]
        }
        
        if 'mi_specific' in data:
            results_to_save['pathways'][f'pathway_{pathway_id}']['mi_specific'] = data['mi_specific']
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✓ Saved detailed results to: {results_file}")
    
    # Save summary table as CSV
    summary_file = results_dir / "pathway_summary.csv"
    
    with open(summary_file, 'w') as f:
        # Header
        f.write("pathway,n_patients,n_valid,avg_age,std_age,min_age,max_age")
        if mi_idx is not None:
            f.write(",mi_pi,mi_pop_pi,mi_hr,mi_pvalue\n")
        else:
            f.write("\n")
        
        # Data
        for pid in pathway_ids:
            data = pathway_pi_results[pid]
            f.write(f"{pid},{data['n_patients']},{data['n_valid_patients']},"
                   f"{data['avg_mi_age']:.2f},{data['std_mi_age']:.2f},"
                   f"{data['age_range'][0]:.0f},{data['age_range'][1]:.0f}")
            
            if mi_idx is not None and 'mi_specific' in data:
                mi = data['mi_specific']
                f.write(f",{mi['pi']:.6f},{mi['population_pi']:.6f},"
                       f"{mi['hazard_ratio']:.4f},{mi['p_value']:.6f}\n")
            else:
                f.write("\n")
    
    print(f"✓ Saved summary table to: {summary_file}")
    
    # Save top diseases for each pathway
    for pathway_id in pathway_ids:
        top_diseases_file = results_dir / f"pathway_{pathway_id}_top_diseases.csv"
        
        with open(top_diseases_file, 'w') as f:
            f.write("rank,disease,pi,population_pi,hazard_ratio,deviation,p_value\n")
            
            for rank, (name, pi_val, pop_pi, hr, dev, pval) in enumerate(
                pathway_pi_results[pathway_id]['top_diseases_by_hr'], 1):
                f.write(f"{rank},{name},{pi_val:.6f},{pop_pi:.6f},{hr:.4f},{dev:.6f},{pval:.6f}\n")
        
        print(f"✓ Saved Pathway {pathway_id} top diseases to: {top_diseases_file}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")
    
    return pathway_pi_results


def plot_mi_pi_trajectories_comparative(pi_all: np.ndarray,
                                        patients_pathway_low: List[dict],
                                        patients_pathway_high: List[dict],
                                        patients_pop_low: List[dict],
                                        patients_pop_high: List[dict],
                                        disease_names: List[str],
                                        out_dir: str,
                                        years_before: int = 10,
                                        label_pw_low: str = "Pathway 1 Low PRS",
                                        label_pw_high: str = "Pathway 1 High PRS",
                                        label_pop_low: str = "Pop Low PRS (no MI)",
                                        label_pop_high: str = "Pop High PRS (no MI)") -> None:
    """
    Plot MI π trajectories comparing:
    1. Pathway 1 patients (low vs high CAD PRS)
    2. Population non-MI patients matched by CAD PRS
    
    Tests whether genetic risk alone predicts MI risk, or if Pathway 1 has additional biology.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    mi_indices = [i for i, name in enumerate(disease_names) if 'myocardial infarction' in name.lower()]
    if not mi_indices:
        print("MI disease not found; skipping.")
        return
    mi_idx = mi_indices[0]
    
    T = pi_all.shape[2]
    
    def extract_trajs(patient_list, is_mi_pathway=True):
        """Extract aligned trajectories for a patient list."""
        trajs, absages = [], []
        for p in patient_list:
            pid = p['patient_id']
            if is_mi_pathway:
                t_mi = int(p['age_at_disease'] - 30)
            else:
                # Non-MI: pick a "reference age" (e.g., 65) for consistency
                t_mi = int(p.get('age_at_disease', 65) - 30)
            if t_mi <= 0 or t_mi > T-1:
                continue
            start = t_mi - years_before
            end = t_mi
            if start < 0:
                continue
            traj = pi_all[pid, mi_idx, start:end]
            if traj.shape[0] == years_before:
                trajs.append(traj)
                absages.append((start, end))
        return np.array(trajs), absages
    
    # Extract trajectories
    trajs_pw_low, ages_pw_low = extract_trajs(patients_pathway_low, is_mi_pathway=True)
    trajs_pw_high, ages_pw_high = extract_trajs(patients_pathway_high, is_mi_pathway=True)
    trajs_pop_low, ages_pop_low = extract_trajs(patients_pop_low, is_mi_pathway=False)
    trajs_pop_high, ages_pop_high = extract_trajs(patients_pop_high, is_mi_pathway=False)
    
    # Compute means
    means = {
        'pw_low': trajs_pw_low.mean(axis=0),
        'pw_high': trajs_pw_high.mean(axis=0),
        'pop_low': trajs_pop_low.mean(axis=0),
        'pop_high': trajs_pop_high.mean(axis=0)
    }
    
    # Align population baselines for each group
    pop_mi_by_t = pi_all[:, mi_idx, :].mean(axis=0)
    
    def align_pop(absages):
        abs_t_lists = [[] for _ in range(years_before)]
        for (start, end) in absages:
            for i in range(years_before):
                abs_t_lists[i].append(start + i)
        return np.array([np.mean(pop_mi_by_t[ts]) if len(ts) > 0 else np.nan for ts in abs_t_lists])
    
    baselines = {
        'pw_low': align_pop(ages_pw_low),
        'pw_high': align_pop(ages_pw_high),
        'pop_low': align_pop(ages_pop_low),
        'pop_high': align_pop(ages_pop_high)
    }
    
    rel_axis = np.arange(-years_before, 0)
    
    # Create figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: All trajectories
    ax1 = axes[0, 0]
    ax1.plot(rel_axis, means['pw_low'], label=label_pw_low, color='#4C78A8', linewidth=2.5)
    ax1.plot(rel_axis, means['pw_high'], label=label_pw_high, color='#F58518', linewidth=2.5)
    ax1.plot(rel_axis, means['pop_low'], label=label_pop_low, color='#4C78A8', linewidth=2, linestyle=':')
    ax1.plot(rel_axis, means['pop_high'], label=label_pop_high, color='#F58518', linewidth=2, linestyle=':')
    ax1.set_xlabel('Years before MI')
    ax1.set_ylabel('π (MI)')
    ax1.set_title('All Trajectories: Pathway 1 vs Population (PRS Stratified)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # Top-right: Δ vs population
    ax2 = axes[0, 1]
    delta_pw_low = means['pw_low'] - baselines['pw_low']
    delta_pw_high = means['pw_high'] - baselines['pw_high']
    delta_pop_low = means['pop_low'] - baselines['pop_low']
    delta_pop_high = means['pop_high'] - baselines['pop_high']
    ax2.plot(rel_axis, delta_pw_low, label=label_pw_low + ' Δ', color='#4C78A8', linewidth=2.5)
    ax2.plot(rel_axis, delta_pw_high, label=label_pw_high + ' Δ', color='#F58518', linewidth=2.5)
    ax2.plot(rel_axis, delta_pop_low, label=label_pop_low + ' Δ', color='#4C78A8', linewidth=2, linestyle=':')
    ax2.plot(rel_axis, delta_pop_high, label=label_pop_high + ' Δ', color='#F58518', linewidth=2, linestyle=':')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Years before MI')
    ax2.set_ylabel('Δ π (MI)')
    ax2.set_title('Δ Trajectories: Pathway 1 vs Population (by PRS)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    # Bottom-left: Low PRS comparison
    ax3 = axes[1, 0]
    ax3.plot(rel_axis, means['pw_low'], label=label_pw_low, color='#4C78A8', linewidth=2.5)
    ax3.plot(rel_axis, means['pop_low'], label=label_pop_low, color='#4C78A8', linewidth=2, linestyle='--')
    ax3.set_xlabel('Years before MI')
    ax3.set_ylabel('π (MI)')
    ax3.set_title('Low CAD PRS: Pathway 1 vs Population')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)
    
    # Bottom-right: High PRS comparison
    ax4 = axes[1, 1]
    ax4.plot(rel_axis, means['pw_high'], label=label_pw_high, color='#F58518', linewidth=2.5)
    ax4.plot(rel_axis, means['pop_high'], label=label_pop_high, color='#F58518', linewidth=2, linestyle='--')
    ax4.set_xlabel('Years before MI')
    ax4.set_ylabel('π (MI)')
    ax4.set_title('High CAD PRS: Pathway 1 vs Population')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'mi_pi_trajectories_pathway_vs_population_by_prs.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative plot: {out_path}")

def plot_mi_pi_trajectories_side_by_side(pi_all: np.ndarray,
                                         patients_low: List[dict],
                                         patients_high: List[dict],
                                         disease_names: List[str],
                                         out_dir: str,
                                         years_before: int = 10,
                                         label_low: str = "Low PRS",
                                         label_high: str = "High PRS") -> None:
    """
    Plot MI π trajectories side-by-side: low vs high CAD PRS within Pathway 1.
    Creates both overlays and delta plots on one figure.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    mi_indices = [i for i, name in enumerate(disease_names) if 'myocardial infarction' in name.lower()]
    if not mi_indices:
        print("MI disease not found; skipping.")
        return
    mi_idx = mi_indices[0]
    
    T = pi_all.shape[2]
    pop_mi_by_t = pi_all[:, mi_idx, :].mean(axis=0)  # [T]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Process low PRS
    trajs_low = []
    absages_low = []
    for p in patients_low:
        pid = p['patient_id']
        t_mi = int(p['age_at_disease'] - 30)
        if t_mi <= 0 or t_mi > T-1:
            continue
        start = t_mi - years_before
        end = t_mi
        if start < 0:
            continue
        traj = pi_all[pid, mi_idx, start:end]
        if traj.shape[0] == years_before:
            trajs_low.append(traj)
            absages_low.append((start, end))
    
    # Process high PRS
    trajs_high = []
    absages_high = []
    for p in patients_high:
        pid = p['patient_id']
        t_mi = int(p['age_at_disease'] - 30)
        if t_mi <= 0 or t_mi > T-1:
            continue
        start = t_mi - years_before
        end = t_mi
        if start < 0:
            continue
        traj = pi_all[pid, mi_idx, start:end]
        if traj.shape[0] == years_before:
            trajs_high.append(traj)
            absages_high.append((start, end))
    
    trajs_low = np.array(trajs_low)  # [N, years_before]
    trajs_high = np.array(trajs_high)
    pw_low_mean = trajs_low.mean(axis=0)
    pw_high_mean = trajs_high.mean(axis=0)
    
    # Align pop baselines
    def align_pop(absages):
        abs_t_lists = [[] for _ in range(years_before)]
        for (start, end) in absages:
            for i in range(years_before):
                abs_t_lists[i].append(start + i)
        return np.array([np.mean(pop_mi_by_t[ts]) if len(ts) > 0 else np.nan for ts in abs_t_lists])
    
    pop_low = align_pop(absages_low)
    pop_high = align_pop(absages_high)
    
    rel_axis = np.arange(-years_before, 0)
    
    # Left: overlay
    ax1.plot(rel_axis, pw_low_mean, label=label_low, color='#4C78A8', linewidth=2.5)
    ax1.plot(rel_axis, pw_high_mean, label=label_high, color='#F58518', linewidth=2.5)
    ax1.plot(rel_axis, (pop_low + pop_high) / 2, label='Pop (age-matched)', color='gray', linewidth=2, linestyle='--')
    ax1.set_xlabel('Years before MI')
    ax1.set_ylabel('π (MI)')
    ax1.set_title(f'MI π Trajectory: {label_low} vs {label_high}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Right: delta
    delta_low = pw_low_mean - pop_low
    delta_high = pw_high_mean - pop_high
    ax2.plot(rel_axis, delta_low, label=f'{label_low} Δ', color='#4C78A8', linewidth=2.5)
    ax2.plot(rel_axis, delta_high, label=f'{label_high} Δ', color='#F58518', linewidth=2.5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Years before MI')
    ax2.set_ylabel('Δ π (MI)')
    ax2.set_title(f'Δ(π_MI): {label_low} vs {label_high}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'mi_pi_trajectories_prs_stratified.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved side-by-side plot: {out_path}")

def plot_mi_pi_trajectories(pi_all: np.ndarray,
                            patients: List[dict],
                            disease_names: List[str],
                            out_dir: str,
                            years_before: int = 10) -> None:
    """
    Plot MI π trajectories (pathway vs population) for the window [-years_before, -1] before MI.
    Saves two figures:
      - mi_pi_trajectories.png (pathway vs population)
      - mi_pi_delta.png (pathway minus population)
    """
    # Identify MI index
    mi_indices = [i for i, name in enumerate(disease_names) if 'myocardial infarction' in name.lower()]
    if not mi_indices:
        print("MI disease not found in names; skipping trajectory plotting.")
        return
    mi_idx = mi_indices[0]

    T = pi_all.shape[2]
    # Support arbitrary pathway labels (e.g., only pathway=1 in a subset).
    unique_labels = sorted(set(p['pathway'] for p in patients))
    n_pathways = len(unique_labels)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    # Precompute population MI baseline once (fast)
    pop_mi_by_t = pi_all[:, mi_idx, :].mean(axis=0)  # [T]

    # Collect aligned trajectories per pathway
    # For each patient with valid window, extract π_MI[tMI-years_before : tMI]
    per_pathway_trajs = {pw: [] for pw in range(n_pathways)}
    per_pathway_absages = {pw: [] for pw in range(n_pathways)}

    for p in patients:
        pid = p['patient_id']
        t_mi = int(p['age_at_disease'] - 30)
        if t_mi <= 0 or t_mi > T-1:
            continue
        start = t_mi - years_before
        end = t_mi
        if start < 0:
            continue  # insufficient history

        traj = pi_all[pid, mi_idx, start:end]  # length = years_before
        if traj.shape[0] != years_before:
            continue

        pw_idx = label_to_idx[p['pathway']]
        per_pathway_trajs[pw_idx].append(traj)
        per_pathway_absages[pw_idx].append((start, end))

    # Compute population baseline π at corresponding absolute ages and align to relative axis
    # For each relative year r in [-years_before, -1], we take mean over all patients' absolute ages at r
    rel_axis = np.arange(-years_before, 0)
    fig, axes = plt.subplots(n_pathways, 1, figsize=(10, 3.5*n_pathways), squeeze=False)
    fig_delta, axes_delta = plt.subplots(n_pathways, 1, figsize=(10, 3.5*n_pathways), squeeze=False)

    for pw in range(n_pathways):
        ax = axes[pw][0]
        axd = axes_delta[pw][0]
        if len(per_pathway_trajs[pw]) == 0:
            ax.set_title(f'Pathway {pw}: no valid trajectories')
            ax.axis('off')
            axd.set_title(f'Pathway {pw}: no valid trajectories')
            axd.axis('off')
            continue

        trajs = np.array(per_pathway_trajs[pw])  # [N_pw, years_before]
        # pathway mean
        pw_mean = trajs.mean(axis=0)

        # population baseline aligned by absolute ages of each patient/time
        # Build for each relative idx r -> list of absolute t's across patients
        abs_t_lists = [[] for _ in range(years_before)]
        for (start, end) in per_pathway_absages[pw]:
            for i in range(years_before):  # i maps to relative rel_axis[i]
                abs_t_lists[i].append(start + i)
        # Use precomputed population MI series and average only across needed absolute ages
        pop_mean_aligned = np.array([np.mean(pop_mi_by_t[ts]) if len(ts) > 0 else np.nan
                                     for ts in abs_t_lists])

        # Plot pathway vs population
        ax.plot(rel_axis, pw_mean, label='Pathway (π_MI)', color='#4C78A8', linewidth=2)
        ax.plot(rel_axis, pop_mean_aligned, label='Population (age-matched π_MI)',
                color='#F58518', linewidth=2, linestyle='--')
        ax.set_title(f'Pathway {pw}: MI π trajectory ({years_before} years prior)')
        ax.set_xlabel('Years before MI')
        ax.set_ylabel('π (MI)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Δ plot
        delta = pw_mean - pop_mean_aligned
        axd.plot(rel_axis, delta, color='#54A24B', linewidth=2)
        axd.axhline(0, color='black', linestyle='--', alpha=0.5)
        axd.set_title(f'Pathway {pw}: Δ(π_MI) vs population')
        axd.set_xlabel('Years before MI')
        axd.set_ylabel('Δ π (MI)')
        axd.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = Path(out_dir) / 'mi_pi_trajectories.png'
    fig.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.tight_layout()
    out2 = Path(out_dir) / 'mi_pi_delta.png'
    fig_delta.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig_delta)
    print(f"Saved MI π trajectory plots to: {out1} and {out2}")


if __name__ == "__main__":
    # Run the analysis
    results = calculate_pathway_pi()
    
    print("\nResults object contains:")
    print(f"  - {len(results)} pathways analyzed")
    print(f"  - Keys: {list(results.keys())}")
    print("\nAccess results via: results[pathway_id]['key']")
    print("Available keys:", list(results[list(results.keys())[0]].keys()) if results else "None")

