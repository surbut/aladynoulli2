#!/usr/bin/env python3
"""
Compare 5-year vs 10-year Deviation Analyses

This script runs the deviation-from-reference pathway discovery with both
5-year and 10-year lookback windows to see how the temporal window affects
pathway identification.
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

import numpy as np
import torch
from pathway_discovery import load_full_data, discover_disease_pathways
from pathway_interrogation import interrogate_disease_pathways
import matplotlib.pyplot as plt

def run_deviation_analysis_with_window(target_disease, n_pathways, lookback_years):
    """
    Run deviation analysis with specified lookback window
    
    Parameters:
    - target_disease: Disease name
    - n_pathways: Number of pathways
    - lookback_years: Years to look back (5 or 10)
    """
    print(f"\n{'='*80}")
    print(f"DEVIATION ANALYSIS: {lookback_years}-YEAR LOOKBACK")
    print(f"{'='*80}\n")
    
    # Load data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Modify pathway_discovery to use specified lookback
    # We'll need to temporarily modify the lookback parameter
    # For now, let's create a wrapper that does the clustering with different windows
    
    # Find target disease
    target_disease_idx = None
    for i, name in enumerate(disease_names):
        if target_disease.lower() in name.lower():
            target_disease_idx = i
            break
    
    if target_disease_idx is None:
        print(f"Could not find disease: {target_disease}")
        return None
    
    print(f"Target disease: {disease_names[target_disease_idx]} (index {target_disease_idx})")
    
    # Find patients who developed this disease
    target_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, target_disease_idx, :].sum() > 0:
            first_occurrence = torch.where(Y[patient_id, target_disease_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_disease = first_occurrence.min().item() + 30
                target_patients.append({
                    'patient_id': patient_id,
                    'age_at_disease': age_at_disease
                })
    
    print(f"Found {len(target_patients)} patients who developed {target_disease}")
    
    # Get signature trajectories
    N, K, T = thetas.shape
    patient_trajectories = []
    
    for patient_info in target_patients:
        patient_id = patient_info['patient_id']
        trajectory = thetas[patient_id, :, :]  # Shape: (K, T)
        patient_info['trajectory'] = trajectory
        patient_trajectories.append(patient_info)
    
    # Calculate population reference
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    
    # Calculate deviations with specified lookback
    trajectory_features = []
    valid_patients = []
    
    for patient_info in patient_trajectories:
        trajectory = patient_info['trajectory']
        age_at_disease = patient_info['age_at_disease']
        
        cutoff_idx = age_at_disease - 30  # Time index at disease onset
        lookback_idx = max(0, cutoff_idx - lookback_years)
        
        if cutoff_idx > lookback_years:
            pre_disease_traj = trajectory[:, lookback_idx:cutoff_idx]
            ref_traj = population_reference[:, lookback_idx:cutoff_idx]
            
            deviation_per_timepoint = pre_disease_traj - ref_traj
            features = deviation_per_timepoint.flatten()
            
            trajectory_features.append(features)
            valid_patients.append(patient_info)
    
    trajectory_features = np.array(trajectory_features)
    print(f"Created {trajectory_features.shape[1]} features ({K * lookback_years} from {K} signatures × {lookback_years} years)")
    print(f"Kept {len(valid_patients)} patients with sufficient pre-disease history")
    
    # Cluster patients
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(trajectory_features)
    
    clusterer = KMeans(n_clusters=n_pathways, random_state=42, n_init=10)
    pathway_labels = clusterer.fit_predict(features_scaled)
    
    print(f"\nDiscovered {n_pathways} pathways:")
    unique_labels, counts = np.unique(pathway_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Pathway {label}: {count} patients ({count/len(valid_patients)*100:.1f}%)")
    
    # Assign pathway labels
    for i, patient_info in enumerate(valid_patients):
        patient_info['pathway'] = pathway_labels[i]
    
    return {
        'target_disease': target_disease,
        'target_disease_idx': target_disease_idx,
        'patients': valid_patients,
        'pathway_labels': pathway_labels,
        'lookback_years': lookback_years,
        'population_reference': population_reference
    }

def compare_pathways(pathway_5yr, pathway_10yr):
    """
    Compare pathways from 5-year vs 10-year analyses
    """
    print(f"\n{'='*80}")
    print("COMPARING 5-YEAR vs 10-YEAR PATHWAYS")
    print(f"{'='*80}\n")
    
    # Load Y and disease names for disease pattern analysis
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Analyze disease patterns for each
    for window, pathway_data in [('5-year', pathway_5yr), ('10-year', pathway_10yr)]:
        print(f"\n{window.upper()} ANALYSIS:")
        print("-" * 80)
        
        patients = pathway_data['patients']
        pathway_labels = np.array([p['pathway'] for p in patients])
        target_disease_idx = pathway_data['target_disease_idx']
        
        unique_labels = np.unique(pathway_labels)
        
        for pathway_id in unique_labels:
            pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
            
            # Count diseases before MI
            pathway_diseases = {}
            for disease_idx in range(Y.shape[1]):
                if disease_idx != target_disease_idx:
                    disease_count = 0
                    for patient_info in pathway_patients:
                        patient_id = patient_info['patient_id']
                        age_at_target = patient_info['age_at_disease']
                        cutoff_idx = age_at_target - 30
                        
                        if cutoff_idx > 0:
                            if Y[patient_id, disease_idx, :cutoff_idx].sum() > 0:
                                disease_count += 1
                    
                    if disease_count > 0:
                        pathway_diseases[disease_names[disease_idx]] = disease_count
            
            # Sort and print top diseases
            pathway_diseases = dict(sorted(pathway_diseases.items(), key=lambda x: x[1], reverse=True))
            
            print(f"\nPathway {pathway_id}: {len(pathway_patients)} patients")
            for i, (disease, count) in enumerate(list(pathway_diseases.items())[:5]):
                pct = count / len(pathway_patients) * 100
                print(f"  {i+1}. {disease[:50]:50s}: {count:4d} ({pct:5.1f}%)")

def analyze_biological_insights(pathway_5yr, pathway_10yr, comparison_df):
    """
    Compare which analysis gives more biologically meaningful pathways
    by looking at disease prevalence patterns
    """
    print(f"\n{'='*80}")
    print("BIOLOGICAL INSIGHTS: COMPARING PATHWAY STRUCTURES")
    print(f"{'='*80}\n")
    
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Analyze each analysis
    for window, pathway_data in [('5-year', pathway_5yr), ('10-year', pathway_10yr)]:
        print(f"\n{'='*60}")
        print(f"{window.upper()} PATHWAY STRUCTURE")
        print('='*60)
        
        patients = pathway_data['patients']
        target_disease_idx = pathway_data['target_disease_idx']
        
        unique_labels = np.unique([p['pathway'] for p in patients])
        
        # Calculate disease prevalences for each pathway
        pathway_disease_prevs = {}
        
        for pathway_id in unique_labels:
            pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
            pathway_disease_prevs[pathway_id] = {}
            
            for disease_idx in range(Y.shape[1]):
                if disease_idx != target_disease_idx:
                    disease_count = 0
                    for patient_info in pathway_patients:
                        patient_id = patient_info['patient_id']
                        age_at_target = patient_info['age_at_disease']
                        cutoff_idx = age_at_target - 30
                        
                        if cutoff_idx > 0:
                            if Y[patient_id, disease_idx, :cutoff_idx].sum() > 0:
                                disease_count += 1
                    
                    if disease_count > 0:
                        pathway_disease_prevs[pathway_id][disease_names[disease_idx]] = {
                            'count': disease_count,
                            'prevalence': disease_count / len(pathway_patients)
                        }
        
        # Find diseases that differentiate pathways the most
        print("\nTop 10 diseases that best differentiate pathways (by variance in prevalence):")
        print("-" * 60)
        
        # Get all diseases
        all_diseases = set()
        for pathway_id in pathway_disease_prevs:
            all_diseases.update(pathway_disease_prevs[pathway_id].keys())
        
        # Calculate variance in prevalence across pathways for each disease
        disease_variance = {}
        for disease in all_diseases:
            prevalences = []
            for pathway_id in unique_labels:
                if disease in pathway_disease_prevs[pathway_id]:
                    prevalences.append(pathway_disease_prevs[pathway_id][disease]['prevalence'])
                else:
                    prevalences.append(0.0)
            disease_variance[disease] = np.var(prevalences)
        
        # Sort by variance
        top_diseases = sorted(disease_variance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (disease, variance) in enumerate(top_diseases):
            print(f"\n{i+1}. {disease}")
            for pathway_id in sorted(unique_labels):
                if disease in pathway_disease_prevs[pathway_id]:
                    count = pathway_disease_prevs[pathway_id][disease]['count']
                    prev = pathway_disease_prevs[pathway_id][disease]['prevalence']
                    print(f"   Pathway {pathway_id}: {count:4d} patients ({prev*100:5.1f}%)")
                else:
                    print(f"   Pathway {pathway_id}:    0 patients ( 0.0%)")
        
        # Calculate pathway separation score
        # Higher variance = better separation between pathways
        total_variance = sum(disease_variance.values())
        mean_variance = np.mean(list(disease_variance.values()))
        print(f"\n{'='*60}")
        print(f"PATHWAY SEPARATION METRICS:")
        print(f"  Total variance across all diseases: {total_variance:.3f}")
        print(f"  Mean variance per disease: {mean_variance:.3f}")
        print(f"  Median variance: {np.median(list(disease_variance.values())):.3f}")
        
        if window == '5-year':
            pathway_5yr_metrics = {
                'total_variance': total_variance,
                'mean_variance': mean_variance,
                'median_variance': np.median(list(disease_variance.values()))
            }
        else:
            pathway_10yr_metrics = {
                'total_variance': total_variance,
                'mean_variance': mean_variance,
                'median_variance': np.median(list(disease_variance.values()))
            }
    
    # Final comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print("\nSeparation Metrics:")
    print(f"5-year analysis:")
    print(f"  Total variance: {pathway_5yr_metrics['total_variance']:.3f}")
    print(f"  Mean variance:  {pathway_5yr_metrics['mean_variance']:.3f}")
    print(f"  Median variance: {pathway_5yr_metrics['median_variance']:.3f}")
    print(f"\n10-year analysis:")
    print(f"  Total variance: {pathway_10yr_metrics['total_variance']:.3f}")
    print(f"  Mean variance:  {pathway_10yr_metrics['mean_variance']:.3f}")
    print(f"  Median variance: {pathway_10yr_metrics['median_variance']:.3f}")
    
    winner = '10-year' if pathway_10yr_metrics['mean_variance'] > pathway_5yr_metrics['mean_variance'] else '5-year'
    print(f"\n→ {winner.upper()} analysis provides better pathway separation (higher disease variance)")


def main():
    """Run comparison"""
    target_disease = "myocardial infarction"
    n_pathways = 4
    
    # Run 5-year analysis
    pathway_5yr = run_deviation_analysis_with_window(target_disease, n_pathways, lookback_years=5)
    
    # Run 10-year analysis
    pathway_10yr = run_deviation_analysis_with_window(target_disease, n_pathways, lookback_years=10)
    
    # Compare results
    if pathway_5yr and pathway_10yr:
        compare_pathways(pathway_5yr, pathway_10yr)
    
    return pathway_5yr, pathway_10yr

if __name__ == "__main__":
    pathway_5yr, pathway_10yr = main()
