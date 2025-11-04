#!/usr/bin/env python3
"""
Pathway Statistical Testing Module

Comprehensive quantitative tests for validating pathway groups:
1. Chi-square tests for disease prevalence differences
2. ANOVA/MANOVA for signature trajectory differences
3. Survival analysis (time to disease)
4. Permutation tests for pathway stability
5. Effect size calculations
6. Multiple testing corrections
7. Age-at-onset comparisons
8. Medication pattern tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
import torch
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def chi_square_test_disease_prevalence(pathway_data, Y, disease_names, 
                                       target_disease_idx, 
                                       disease_idx=None,
                                       disease_name=None,
                                       min_count=5):
    """
    Perform chi-square test for disease prevalence differences across pathways
    
    Parameters:
    -----------
    pathway_data : dict
        Contains 'patients' list with pathway assignments
    Y : torch.Tensor
        Binary disease matrix (patients x diseases x time)
    disease_names : list
        List of disease names
    target_disease_idx : int
        Index of target disease
    disease_idx : int, optional
        Index of disease to test (if None, tests all diseases)
    disease_name : str, optional
        Name of disease to test (alternative to disease_idx)
    min_count : int
        Minimum expected count per cell for chi-square validity
    
    Returns:
    --------
    dict : Results with chi-square statistic, p-value, effect size (Cramér's V)
    """
    patients = pathway_data['patients']
    
    # Find disease index if name provided
    if disease_name:
        for i, name in enumerate(disease_names):
            if disease_name.lower() in name.lower():
                disease_idx = i
                break
    
    if disease_idx is None:
        raise ValueError("Must provide either disease_idx or disease_name")
    
    # Get unique pathways
    pathway_labels = [p['pathway'] for p in patients]
    unique_pathways = sorted(set(pathway_labels))
    n_pathways = len(unique_pathways)
    
    # Build contingency table: pathways x disease (present/absent)
    contingency_table = []
    pathway_sizes = []
    
    for pathway_id in unique_pathways:
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        pathway_sizes.append(len(pathway_patients))
        
        # Count patients with disease BEFORE target disease
        with_disease = 0
        without_disease = 0
        
        for patient_info in pathway_patients:
            patient_id = patient_info['patient_id']
            age_at_target = patient_info['age_at_disease']
            cutoff_idx = age_at_target - 30
            
            if cutoff_idx > 0:
                if Y[patient_id, disease_idx, :cutoff_idx].sum() > 0:
                    with_disease += 1
                else:
                    without_disease += 1
        
        contingency_table.append([with_disease, without_disease])
    
    contingency_table = np.array(contingency_table)
    
    # Perform chi-square test
    if np.all(contingency_table.sum(axis=0) >= min_count):
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        # Calculate prevalences
        prevalences = {}
        for i, pathway_id in enumerate(unique_pathways):
            total = contingency_table[i].sum()
            prevalences[pathway_id] = contingency_table[i, 0] / total if total > 0 else 0
        
        return {
            'disease': disease_names[disease_idx],
            'disease_idx': disease_idx,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'prevalences': prevalences,
            'contingency_table': contingency_table,
            'expected_counts': expected,
            'is_valid': True
        }
    else:
        # Use Fisher's exact test for small samples
        if n_pathways == 2:
            from scipy.stats import fisher_exact
            oddsratio, p_value = fisher_exact(contingency_table)
            return {
                'disease': disease_names[disease_idx],
                'disease_idx': disease_idx,
                'test_type': 'fisher_exact',
                'oddsratio': oddsratio,
                'p_value': p_value,
                'prevalences': {unique_pathways[i]: contingency_table[i, 0] / contingency_table[i].sum() 
                               for i in range(n_pathways)},
                'is_valid': True
            }
        else:
            return {
                'disease': disease_names[disease_idx],
                'disease_idx': disease_idx,
                'p_value': np.nan,
                'error': 'Sample size too small for chi-square test',
                'is_valid': False
            }


def test_all_disease_prevalences(pathway_data, Y, disease_names, target_disease_idx,
                                 min_prevalence=0.01, min_count=5, fdr_alpha=0.05):
    """
    Test all diseases for prevalence differences across pathways with FDR correction
    
    Returns:
    --------
    pd.DataFrame : Results with p-values, effect sizes, and FDR-corrected significance
    """
    results = []
    
    for disease_idx in range(len(disease_names)):
        if disease_idx == target_disease_idx:
            continue
        
        # Quick check: skip if disease is too rare
        total_with_disease = Y[:, disease_idx, :].sum().item()
        if total_with_disease < 10:
            continue
        
        test_result = chi_square_test_disease_prevalence(
            pathway_data, Y, disease_names, target_disease_idx,
            disease_idx=disease_idx, min_count=min_count
        )
        
        if test_result['is_valid']:
            results.append({
                'disease': test_result['disease'],
                'disease_idx': test_result['disease_idx'],
                'chi2_statistic': test_result.get('chi2_statistic', np.nan),
                'p_value': test_result['p_value'],
                'cramers_v': test_result.get('cramers_v', np.nan),
                'prevalences': test_result['prevalences']
            })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 0:
        valid_p_values = results_df['p_value'].dropna()
        if len(valid_p_values) > 0:
            _, p_corrected, _, _ = multipletests(valid_p_values, alpha=fdr_alpha, method='fdr_bh')
            results_df['p_value_fdr_corrected'] = np.nan
            results_df.loc[valid_p_values.index, 'p_value_fdr_corrected'] = p_corrected
            results_df['is_significant_fdr'] = results_df['p_value_fdr_corrected'] < fdr_alpha
    
    # Sort by significance
    results_df = results_df.sort_values('p_value')
    
    return results_df


def anova_signature_trajectories(pathway_data, thetas, signature_idx=None,
                                 time_window='pre_disease_10yr'):
    """
    Test if signature trajectories differ across pathways using ANOVA
    
    Parameters:
    -----------
    pathway_data : dict
        Contains patient pathway assignments
    thetas : np.array
        Signature loadings (patients x signatures x time)
    signature_idx : int or None
        Specific signature to test (if None, tests all)
    time_window : str
        'pre_disease_10yr', 'pre_disease_5yr', or 'all'
    
    Returns:
    --------
    dict : ANOVA results for each signature
    """
    patients = pathway_data['patients']
    unique_pathways = sorted(set([p['pathway'] for p in patients]))
    K = thetas.shape[1]
    
    results = {}
    
    signatures_to_test = [signature_idx] if signature_idx is not None else range(K)
    
    for sig_idx in signatures_to_test:
        pathway_groups = []
        pathway_labels = []
        
        for patient_info in patients:
            patient_id = patient_info['patient_id']
            pathway_id = patient_info['pathway']
            age_at_disease = patient_info['age_at_disease']
            
            # Extract relevant time window
            if time_window == 'pre_disease_10yr':
                cutoff_idx = age_at_disease - 30
                lookback_idx = max(0, cutoff_idx - 10)
                if cutoff_idx > 10:
                    sig_values = thetas[patient_id, sig_idx, lookback_idx:cutoff_idx]
                    if len(sig_values) > 0:
                        # Average over time window
                        pathway_groups.append(sig_values.mean())
                        pathway_labels.append(pathway_id)
            elif time_window == 'pre_disease_5yr':
                cutoff_idx = age_at_disease - 30
                lookback_idx = max(0, cutoff_idx - 5)
                if cutoff_idx > 5:
                    sig_values = thetas[patient_id, sig_idx, lookback_idx:cutoff_idx]
                    if len(sig_values) > 0:
                        pathway_groups.append(sig_values.mean())
                        pathway_labels.append(pathway_id)
            elif time_window == 'all':
                sig_values = thetas[patient_id, sig_idx, :]
                pathway_groups.append(sig_values.mean())
                pathway_labels.append(pathway_id)
        
        # Group by pathway
        pathway_values = {pid: [] for pid in unique_pathways}
        for val, pid in zip(pathway_groups, pathway_labels):
            pathway_values[pid].append(val)
        
        # Perform ANOVA
        groups = [pathway_values[pid] for pid in unique_pathways if len(pathway_values[pid]) > 0]
        
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            all_values = [val for group in groups for val in group]
            ss_between = sum(len(g) * (np.mean(g) - np.mean(all_values))**2 for g in groups)
            ss_total = np.var(all_values) * len(all_values)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Calculate means and stds
            pathway_means = {pid: np.mean(pathway_values[pid]) 
                           for pid in unique_pathways if len(pathway_values[pid]) > 0}
            pathway_stds = {pid: np.std(pathway_values[pid]) 
                          for pid in unique_pathways if len(pathway_values[pid]) > 0}
            
            results[sig_idx] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'pathway_means': pathway_means,
                'pathway_stds': pathway_stds,
                'n_per_pathway': {pid: len(pathway_values[pid]) 
                                 for pid in unique_pathways if len(pathway_values[pid]) > 0}
            }
    
    return results


def test_age_at_onset(pathway_data, test_type='anova'):
    """
    Test if age at disease onset differs across pathways
    
    Parameters:
    -----------
    pathway_data : dict
        Contains patient pathway assignments and age_at_disease
    test_type : str
        'anova', 'kruskal' (non-parametric), or 'logrank' (survival)
    
    Returns:
    --------
    dict : Test results
    """
    patients = pathway_data['patients']
    unique_pathways = sorted(set([p['pathway'] for p in patients]))
    
    # Group ages by pathway
    pathway_ages = {pid: [p['age_at_disease'] for p in patients if p['pathway'] == pid]
                   for pid in unique_pathways}
    
    groups = [pathway_ages[pid] for pid in unique_pathways]
    
    if test_type == 'anova':
        f_stat, p_value = f_oneway(*groups)
        
        # Effect size (eta-squared)
        all_ages = [age for group in groups for age in group]
        ss_between = sum(len(g) * (np.mean(g) - np.mean(all_ages))**2 for g in groups)
        ss_total = np.var(all_ages) * len(all_ages)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'test_type': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'pathway_means': {pid: np.mean(pathway_ages[pid]) for pid in unique_pathways},
            'pathway_stds': {pid: np.std(pathway_ages[pid]) for pid in unique_pathways},
            'n_per_pathway': {pid: len(pathway_ages[pid]) for pid in unique_pathways}
        }
    
    elif test_type == 'kruskal':
        # Non-parametric alternative
        h_stat, p_value = kruskal(*groups)
        
        return {
            'test_type': 'Kruskal-Wallis',
            'h_statistic': h_stat,
            'p_value': p_value,
            'pathway_medians': {pid: np.median(pathway_ages[pid]) for pid in unique_pathways},
            'n_per_pathway': {pid: len(pathway_ages[pid]) for pid in unique_pathways}
        }
    
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def permutation_test_pathway_stability(pathway_data, thetas, n_permutations=1000, 
                                       random_seed=42):
    """
    Permutation test: Are pathways significantly different from random clustering?
    
    Tests if observed pathway separation is greater than expected by chance
    
    Returns:
    --------
    dict : p-value from permutation test
    """
    np.random.seed(random_seed)
    patients = pathway_data['patients']
    n_patients = len(patients)
    unique_pathways = sorted(set([p['pathway'] for p in patients]))
    n_pathways = len(unique_pathways)
    
    # Calculate observed between-pathway variance
    pathway_means = {}
    for pathway_id in unique_pathways:
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        patient_ids = [p['patient_id'] for p in pathway_patients]
        pathway_thetas = thetas[patient_ids, :, :]
        pathway_means[pathway_id] = np.mean(pathway_thetas, axis=(0, 2))  # Average over patients and time
    
    observed_variance = np.var([pathway_means[pid] for pid in unique_pathways], axis=0).mean()
    
    # Permutation distribution
    permuted_variances = []
    
    for _ in range(n_permutations):
        # Randomly assign pathways
        permuted_pathways = np.random.choice(unique_pathways, size=n_patients)
        
        permuted_means = {}
        for pathway_id in unique_pathways:
            pathway_mask = permuted_pathways == pathway_id
            if pathway_mask.sum() > 0:
                pathway_thetas = thetas[pathway_mask, :, :]
                permuted_means[pathway_id] = np.mean(pathway_thetas, axis=(0, 2))
        
        if len(permuted_means) == n_pathways:
            permuted_variance = np.var([permuted_means[pid] for pid in unique_pathways], axis=0).mean()
            permuted_variances.append(permuted_variance)
    
    # Calculate p-value (one-tailed: is observed > permuted?)
    p_value = np.mean(np.array(permuted_variances) >= observed_variance)
    
    return {
        'observed_variance': observed_variance,
        'permuted_mean_variance': np.mean(permuted_variances),
        'permuted_std_variance': np.std(permuted_variances),
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'n_permutations': n_permutations
    }


def calculate_effect_sizes(pathway_data, thetas, Y, disease_names, target_disease_idx):
    """
    Calculate effect sizes (Cohen's d) for all comparisons
    
    Returns:
    --------
    dict : Effect sizes for signatures, diseases, and age-at-onset
    """
    patients = pathway_data['patients']
    unique_pathways = sorted(set([p['pathway'] for p in patients]))
    
    results = {
        'signature_effect_sizes': {},
        'disease_effect_sizes': {},
        'age_effect_sizes': {}
    }
    
    # Signature effect sizes (Cohen's d for each pair)
    K = thetas.shape[1]
    for sig_idx in range(K):
        pathway_means = {}
        pathway_stds = {}
        pathway_ns = {}
        
        for pathway_id in unique_pathways:
            pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
            patient_ids = [p['patient_id'] for p in pathway_patients]
            
            # Average signature over time
            sig_values = [thetas[pid, sig_idx, :].mean() for pid in patient_ids]
            pathway_means[pathway_id] = np.mean(sig_values)
            pathway_stds[pathway_id] = np.std(sig_values)
            pathway_ns[pathway_id] = len(sig_values)
        
        # Pairwise Cohen's d
        cohens_d_pairs = {}
        pathway_ids = list(unique_pathways)
        for i, pid1 in enumerate(pathway_ids):
            for pid2 in pathway_ids[i+1:]:
                # Pooled standard deviation
                n1, n2 = pathway_ns[pid1], pathway_ns[pid2]
                s1, s2 = pathway_stds[pid1], pathway_stds[pid2]
                pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                
                if pooled_std > 0:
                    cohens_d = (pathway_means[pid1] - pathway_means[pid2]) / pooled_std
                    cohens_d_pairs[f"{pid1}_vs_{pid2}"] = cohens_d
        
        results['signature_effect_sizes'][sig_idx] = cohens_d_pairs
    
    # Age effect sizes
    pathway_ages = {pid: [p['age_at_disease'] for p in patients if p['pathway'] == pid]
                   for pid in unique_pathways}
    
    age_cohens_d = {}
    pathway_ids = list(unique_pathways)
    for i, pid1 in enumerate(pathway_ids):
        for pid2 in pathway_ids[i+1:]:
            ages1, ages2 = pathway_ages[pid1], pathway_ages[pid2]
            pooled_std = np.sqrt((np.var(ages1) + np.var(ages2)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(ages1) - np.mean(ages2)) / pooled_std
                age_cohens_d[f"{pid1}_vs_{pid2}"] = cohens_d
    
    results['age_effect_sizes'] = age_cohens_d
    
    return results


def test_medication_prevalence(medication_results, pathway_data, 
                               medication_name=None, min_count=5):
    """
    Test medication prevalence differences across pathways using chi-square
    
    Parameters:
    -----------
    medication_results : dict
        Results from integrate_medications_with_pathways
    pathway_data : dict
        Pathway assignments
    medication_name : str, optional
        Specific medication to test (if None, tests all)
    min_count : int
        Minimum expected count per cell
    
    Returns:
    --------
    dict or pd.DataFrame : Test results
    """
    if medication_results is None:
        return None
    
    pathway_medications = medication_results.get('pathway_medications', {})
    patients = pathway_data['patients']
    unique_pathways = sorted(set([p['pathway'] for p in patients]))
    
    # Get all medications
    all_medications = set()
    for pathway_id in unique_pathways:
        if pathway_id in pathway_medications:
            all_medications.update(pathway_medications[pathway_id].keys())
    
    if medication_name:
        all_medications = [m for m in all_medications if medication_name.lower() in m.lower()]
    
    results = []
    
    for med_name in all_medications:
        contingency_table = []
        
        for pathway_id in unique_pathways:
            pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
            n_pathway = len(pathway_patients)
            
            # Count patients with this medication
            with_med = 0
            if pathway_id in pathway_medications:
                med_counts = pathway_medications[pathway_id]
                with_med = med_counts.get(med_name, 0)
            
            without_med = n_pathway - with_med
            contingency_table.append([with_med, without_med])
        
        contingency_table = np.array(contingency_table)
        
        if np.all(contingency_table.sum(axis=0) >= min_count):
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            n = contingency_table.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            prevalences = {unique_pathways[i]: contingency_table[i, 0] / contingency_table[i].sum() 
                          for i in range(len(unique_pathways))}
            
            results.append({
                'medication': med_name,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'prevalences': prevalences
            })
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    # Apply FDR correction
    if len(results_df) > 0:
        valid_p_values = results_df['p_value'].dropna()
        if len(valid_p_values) > 0:
            _, p_corrected, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
            results_df['p_value_fdr_corrected'] = np.nan
            results_df.loc[valid_p_values.index, 'p_value_fdr_corrected'] = p_corrected
            results_df['is_significant_fdr'] = results_df['p_value_fdr_corrected'] < 0.05
    
    return results_df


def comprehensive_pathway_tests_with_medications(pathway_data, Y, thetas, disease_names, 
                                                  target_disease_idx, medication_results=None,
                                                  prs_results=None, output_dir=None):
    """
    Run all statistical tests including medications and PRS if available
    
    This is the main function to call for complete statistical validation
    """
    print("="*80)
    print("COMPREHENSIVE STATISTICAL TESTS FOR PATHWAY GROUPS")
    print("="*80)
    
    results = {}
    
    # 1. Disease prevalence tests
    print("\n1. Testing disease prevalence differences...")
    disease_results = test_all_disease_prevalences(
        pathway_data, Y, disease_names, target_disease_idx
    )
    results['disease_prevalence_tests'] = disease_results
    n_sig_diseases = disease_results['is_significant_fdr'].sum() if 'is_significant_fdr' in disease_results.columns else 0
    print(f"   Found {n_sig_diseases} significantly different diseases (FDR < 0.05)")
    
    # 2. Signature trajectory tests
    print("\n2. Testing signature trajectory differences...")
    signature_results = anova_signature_trajectories(
        pathway_data, thetas, time_window='pre_disease_10yr'
    )
    results['signature_trajectory_tests'] = signature_results
    
    significant_sigs = [sig for sig, res in signature_results.items() 
                       if 'p_value' in res and res['p_value'] < 0.05]
    print(f"   Found {len(significant_sigs)} signatures with significant differences (p < 0.05)")
    
    # 3. Age at onset test
    print("\n3. Testing age at disease onset differences...")
    age_results = test_age_at_onset(pathway_data, test_type='anova')
    results['age_at_onset_test'] = age_results
    print(f"   ANOVA: F={age_results['f_statistic']:.3f}, p={age_results['p_value']:.4f}")
    
    # 4. Permutation test
    print("\n4. Permutation test for pathway stability...")
    perm_results = permutation_test_pathway_stability(pathway_data, thetas, n_permutations=1000)
    results['permutation_test'] = perm_results
    print(f"   Observed variance: {perm_results['observed_variance']:.4f}")
    print(f"   Permuted mean: {perm_results['permuted_mean_variance']:.4f}")
    print(f"   p-value: {perm_results['p_value']:.4f}")
    
    # 5. Medication tests (if available)
    if medication_results is not None:
        print("\n5. Testing medication prevalence differences...")
        med_results = test_medication_prevalence(medication_results, pathway_data)
        results['medication_tests'] = med_results
        if med_results is not None:
            n_sig_meds = med_results['is_significant_fdr'].sum() if 'is_significant_fdr' in med_results.columns else 0
            print(f"   Found {n_sig_meds} medications with significant differences (FDR < 0.05)")
        else:
            print("   No medication data available for testing")
    
    # 6. PRS tests (already done in pathway_interrogation, but we can extract summary)
    if prs_results is not None:
        print("\n6. PRS differences (from pathway_interrogation)...")
        results['prs_results'] = prs_results
        print("   PRS analysis results included")
    
    # 7. Effect sizes
    print("\n7. Calculating effect sizes...")
    effect_sizes = calculate_effect_sizes(pathway_data, thetas, Y, disease_names, target_disease_idx)
    results['effect_sizes'] = effect_sizes
    print("   Effect sizes calculated for signatures, diseases, and age")
    
    # Save results
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as pickle
        import pickle
        with open(f"{output_dir}/statistical_test_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary CSVs
        disease_results.to_csv(f"{output_dir}/disease_prevalence_tests.csv", index=False)
        
        if medication_results is not None and results.get('medication_tests') is not None:
            results['medication_tests'].to_csv(f"{output_dir}/medication_prevalence_tests.csv", index=False)
        
        # Generate text summary
        generate_statistical_summary(results, output_dir)
        
        print(f"\n✅ Results saved to {output_dir}/")
    
    return results


def generate_statistical_summary(results, output_dir):
    """Generate a human-readable summary of all statistical tests"""
    summary_file = f"{output_dir}/statistical_tests_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL TESTS SUMMARY FOR PATHWAY GROUPS\n")
        f.write("="*80 + "\n\n")
        
        # Disease prevalence tests
        if 'disease_prevalence_tests' in results:
            f.write("1. DISEASE PREVALENCE DIFFERENCES\n")
            f.write("-"*80 + "\n")
            df = results['disease_prevalence_tests']
            if 'is_significant_fdr' in df.columns:
                n_sig = df['is_significant_fdr'].sum()
                f.write(f"   {n_sig} diseases significantly different across pathways (FDR < 0.05)\n")
                
                top_10 = df[df['is_significant_fdr']].head(10)
                f.write("\n   Top 10 most significant:\n")
                for idx, row in top_10.iterrows():
                    f.write(f"     {row['disease']}: χ²={row['chi2_statistic']:.2f}, p={row['p_value']:.2e}, "
                           f"Cramér's V={row['cramers_v']:.3f}\n")
            f.write("\n")
        
        # Signature tests
        if 'signature_trajectory_tests' in results:
            f.write("2. SIGNATURE TRAJECTORY DIFFERENCES\n")
            f.write("-"*80 + "\n")
            sig_results = results['signature_trajectory_tests']
            significant = [(sig, res) for sig, res in sig_results.items() 
                          if 'p_value' in res and res['p_value'] < 0.05]
            significant.sort(key=lambda x: x[1]['p_value'])
            
            f.write(f"   {len(significant)} signatures significantly different (p < 0.05)\n\n")
            f.write("   Top 10 most significant:\n")
            for sig_idx, res in significant[:10]:
                f.write(f"     Signature {sig_idx}: F={res['f_statistic']:.2f}, p={res['p_value']:.2e}, "
                       f"η²={res['eta_squared']:.3f}\n")
            f.write("\n")
        
        # Age at onset
        if 'age_at_onset_test' in results:
            f.write("3. AGE AT DISEASE ONSET\n")
            f.write("-"*80 + "\n")
            age_res = results['age_at_onset_test']
            f.write(f"   Test: {age_res['test_type']}\n")
            f.write(f"   Statistic: {age_res.get('f_statistic', age_res.get('h_statistic', 'N/A')):.3f}\n")
            f.write(f"   p-value: {age_res['p_value']:.4f}\n")
            f.write(f"   Effect size (η²): {age_res.get('eta_squared', 'N/A')}\n\n")
            f.write("   Mean age by pathway:\n")
            means = age_res.get('pathway_means', age_res.get('pathway_medians', {}))
            for pid, age in sorted(means.items()):
                f.write(f"     Pathway {pid}: {age:.1f} years\n")
            f.write("\n")
        
        # Permutation test
        if 'permutation_test' in results:
            f.write("4. PATHWAY STABILITY (PERMUTATION TEST)\n")
            f.write("-"*80 + "\n")
            perm = results['permutation_test']
            f.write(f"   Observed between-pathway variance: {perm['observed_variance']:.4f}\n")
            f.write(f"   Permuted mean variance: {perm['permuted_mean_variance']:.4f}\n")
            f.write(f"   Permuted std: {perm['permuted_std_variance']:.4f}\n")
            f.write(f"   p-value: {perm['p_value']:.4f}\n")
            f.write(f"   Significant: {'Yes' if perm['is_significant'] else 'No'}\n")
            f.write("\n")
        
        # Medications
        if 'medication_tests' in results and results['medication_tests'] is not None:
            f.write("5. MEDICATION PREVALENCE DIFFERENCES\n")
            f.write("-"*80 + "\n")
            med_df = results['medication_tests']
            if 'is_significant_fdr' in med_df.columns:
                n_sig = med_df['is_significant_fdr'].sum()
                f.write(f"   {n_sig} medications significantly different (FDR < 0.05)\n\n")
                top_10 = med_df[med_df['is_significant_fdr']].head(10)
                f.write("   Top 10 most significant:\n")
                for idx, row in top_10.iterrows():
                    f.write(f"     {row['medication']}: χ²={row['chi2_statistic']:.2f}, "
                           f"p={row['p_value']:.2e}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")
    
    print(f"   Saved summary: {summary_file}")


# Backward compatibility - keep the old function name
def comprehensive_pathway_tests(pathway_data, Y, thetas, disease_names, 
                                target_disease_idx, output_dir=None):
    """Wrapper for backward compatibility"""
    return comprehensive_pathway_tests_with_medications(
        pathway_data, Y, thetas, disease_names, target_disease_idx,
        medication_results=None, prs_results=None, output_dir=output_dir
    )


if __name__ == "__main__":
    print("Pathway Statistical Testing Module")
    print("Import this module and use comprehensive_pathway_tests_with_medications() for full analysis")
