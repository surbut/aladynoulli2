#!/usr/bin/env python3
"""
Complete Pathway Analysis - 10-Year Deviation from Reference

This script performs a complete pathway analysis using the 10-year deviation-from-reference
method, including:
- Pathway discovery
- Visualizations (stacked area plots, signature deviation plots)
- Most variable signatures analysis
- Most distinctive diseases
- Medication differences
- Complete summary report

Outputs all results to saved files.
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Import our modules
from pathway_discovery import load_full_data

def discover_pathways_10yr(target_disease, n_pathways=4):
    """
    Discover pathways using 10-year deviation from reference method
    
    Returns pathway_data dictionary with all patient and pathway information
    """
    print("="*80)
    print(f"PATHWAY DISCOVERY: {target_disease.upper()}")
    print(f"Method: 10-Year Deviation from Reference")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Find target disease
    target_disease_idx = None
    for i, name in enumerate(disease_names):
        if target_disease.lower() in name.lower():
            target_disease_idx = i
            target_disease_name = name
            break
    
    if target_disease_idx is None:
        raise ValueError(f"Could not find disease: {target_disease}")
    
    print(f"Target disease: {target_disease_name} (index {target_disease_idx})")
    
    # Find patients who developed this disease
    print("Finding patients...")
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
    
    print(f"Found {len(target_patients)} patients who developed {target_disease_name}")
    
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
    
    # Calculate deviations with 10-year lookback
    print("\nCalculating deviations from reference...")
    trajectory_features = []
    valid_patients = []
    
    for patient_info in patient_trajectories:
        trajectory = patient_info['trajectory']
        age_at_disease = patient_info['age_at_disease']
        
        cutoff_idx = age_at_disease - 30  # Time index at disease onset
        lookback_years = 10
        lookback_idx = max(0, cutoff_idx - lookback_years)
        
        if cutoff_idx > lookback_years:
            pre_disease_traj = trajectory[:, lookback_idx:cutoff_idx]
            ref_traj = population_reference[:, lookback_idx:cutoff_idx]
            
            deviation_per_timepoint = pre_disease_traj - ref_traj
            features = deviation_per_timepoint.flatten()
            
            trajectory_features.append(features)
            valid_patients.append(patient_info)
    
    trajectory_features = np.array(trajectory_features)
    print(f"Created {trajectory_features.shape[1]} features ({K * 10} from {K} signatures × 10 years)")
    print(f"Kept {len(valid_patients)} patients with sufficient pre-disease history")
    
    # Cluster patients
    print("\nClustering patients...")
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
        'target_disease': target_disease_name,
        'target_disease_idx': target_disease_idx,
        'patients': valid_patients,
        'pathway_labels': pathway_labels,
        'population_reference': population_reference,
        'disease_names': disease_names
    }

def analyze_most_variable_signatures(pathway_data, thetas, K, T):
    """
    Identify which signatures are most variable across pathways
    """
    print("\n" + "="*80)
    print("MOST VARIABLE SIGNATURES ANALYSIS")
    print("="*80)
    
    patients = pathway_data['patients']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    
    # Get patient indices for each pathway
    pathway_patients = {}
    for pathway_id in range(n_pathways):
        pathway_patients[pathway_id] = [i for i, p in enumerate(patients) if p['pathway'] == pathway_id]
    
    # Calculate mean signature values for each pathway (5 years before disease)
    signature_means_by_pathway = np.zeros((n_pathways, K, T))
    
    for pathway_id in range(n_pathways):
        patient_indices = pathway_patients[pathway_id]
        if len(patient_indices) > 0:
            pathway_thetas = thetas[patient_indices, :, :]
            signature_means_by_pathway[pathway_id] = np.mean(pathway_thetas, axis=0)
    
    # Calculate variance across pathways for each signature
    # Look at the 10 years before disease onset (age 50-60 range typically)
    signature_variance = []
    for k in range(K):
        sig_values = signature_means_by_pathway[:, k, :]
        variance = np.var(sig_values, axis=0).mean()  # Mean variance across time
        signature_variance.append(variance)
    
    # Sort and report top signatures
    top_sigs = sorted(enumerate(signature_variance), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most variable signatures across pathways:")
    print("-"*60)
    for i, (sig_idx, variance) in enumerate(top_sigs[:10]):
        print(f"{i+1}. Signature {sig_idx}: Variance = {variance:.6f}")
    
    return [sig_idx for sig_idx, _ in top_sigs[:10]]

def analyze_distinctive_diseases(pathway_data, Y, disease_names):
    """
    Find diseases that most distinguish pathways
    """
    print("\n" + "="*80)
    print("MOST DISTINCTIVE DISEASES")
    print("="*80)
    
    patients = pathway_data['patients']
    target_disease_idx = pathway_data['target_disease_idx']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    
    # Calculate disease prevalences for each pathway
    pathway_disease_prevs = {}
    
    for pathway_id in range(n_pathways):
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
                    prevalence = disease_count / len(pathway_patients)
                    pathway_disease_prevs[pathway_id][disease_names[disease_idx]] = {
                        'count': disease_count,
                        'prevalence': prevalence
                    }
    
    # Find diseases that differentiate pathways the most
    all_diseases = set()
    for pathway_id in pathway_disease_prevs:
        all_diseases.update(pathway_disease_prevs[pathway_id].keys())
    
    # Calculate variance in prevalence across pathways for each disease
    disease_variance = {}
    for disease in all_diseases:
        prevalences = []
        for pathway_id in range(n_pathways):
            if disease in pathway_disease_prevs[pathway_id]:
                prevalences.append(pathway_disease_prevs[pathway_id][disease]['prevalence'])
            else:
                prevalences.append(0.0)
        disease_variance[disease] = np.var(prevalences)
    
    # Sort by variance
    top_diseases = sorted(disease_variance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("\nTop 15 diseases that best differentiate pathways:")
    print("-"*80)
    
    pathway_ids = list(range(n_pathways))
    
    for i, (disease, variance) in enumerate(top_diseases):
        print(f"\n{i+1}. {disease}")
        for pathway_id in pathway_ids:
            if disease in pathway_disease_prevs[pathway_id]:
                count = pathway_disease_prevs[pathway_id][disease]['count']
                prev = pathway_disease_prevs[pathway_id][disease]['prevalence']
                print(f"   Pathway {pathway_id}: {count:4d} patients ({prev*100:5.1f}%)")
            else:
                print(f"   Pathway {pathway_id}:    0 patients ( 0.0%)")
    
    return top_diseases, pathway_disease_prevs

def create_stacked_deviation_plot(pathway_data, thetas, output_dir):
    """
    Create stacked area plot showing signature deviations from reference
    """
    print("\n" + "="*80)
    print("CREATING STACKED DEVIATION PLOT")
    print("="*80)
    
    patients = pathway_data['patients']
    target_disease = pathway_data['target_disease']
    population_reference = pathway_data['population_reference']
    
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    K, T = thetas.shape[1], thetas.shape[2]
    
    # Get patient indices for each pathway
    pathway_patients_indices = {}
    for pathway_id in range(n_pathways):
        pathway_patients_indices[pathway_id] = [i for i, p in enumerate(patients) if p['pathway'] == pathway_id]
    
    # Calculate average deviation for each pathway
    time_diff_by_cluster = np.zeros((n_pathways, K, T))
    
    for pathway_id in range(n_pathways):
        patient_indices = pathway_patients_indices[pathway_id]
        # Get global patient IDs from the patient_info dictionaries
        global_indices = [patients[i]['patient_id'] for i in patient_indices]
        pathway_thetas = thetas[global_indices, :, :]
        
        for k in range(K):
            for t in range(T):
                pathway_mean = np.mean(pathway_thetas[:, k, t])
                time_diff_by_cluster[pathway_id, k, t] = pathway_mean - population_reference[k, t]
    
    # Create plot
    if K <= 20:
        sig_colors = cm.get_cmap('tab20')(np.linspace(0, 1, K))
    else:
        sig_colors = cm.get_cmap('tab20')(np.linspace(0, 1, K))
    
    fig, axes = plt.subplots(n_pathways, 1, figsize=(14, 5*n_pathways))
    if n_pathways == 1:
        axes = [axes]
    
    fig.suptitle(f'Signature Deviations from Reference: {target_disease}\n(10-Year Lookback Analysis)', 
                 fontsize=18, fontweight='bold')
    
    time_points = np.linspace(30, 81, T)
    
    for i in range(n_pathways):
        ax = axes[i]
        n_patients = len(pathway_patients_indices[i])
        pathway_deviations = time_diff_by_cluster[i, :, :]
        
        cumulative = np.zeros(T)
        
        for sig_idx in range(K):
            sig_values = pathway_deviations[sig_idx, :]
            ax.fill_between(time_points, cumulative, cumulative + sig_values, 
                           color=sig_colors[sig_idx], alpha=0.95, label=f'Sig {sig_idx}')
            cumulative += sig_values
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_title(f'Pathway {i} (n={n_patients})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Signature Deviation from Reference', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    filename = f'{output_dir}/stacked_deviation_{target_disease.replace(" ", "_")}_10yr.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()

def save_summary_report(pathway_data, top_diseases, most_variable_sigs, output_dir):
    """
    Save a comprehensive summary report
    """
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)
    
    target_disease = pathway_data['target_disease']
    patients = pathway_data['patients']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    
    report_path = f'{output_dir}/pathway_analysis_summary_10yr.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"PATHWAY ANALYSIS SUMMARY: {target_disease.upper()}\n")
        f.write("Analysis Method: 10-Year Deviation from Reference\n")
        f.write("="*80 + "\n\n")
        
        f.write("PATHWAY SIZES:\n")
        f.write("-"*80 + "\n")
        unique_labels, counts = np.unique([p['pathway'] for p in patients], return_counts=True)
        for label, count in zip(unique_labels, counts):
            pct = count / len(patients) * 100
            f.write(f"Pathway {label}: {count:5d} patients ({pct:5.1f}%)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("MOST DISTINCTIVE DISEASES (Top 15)\n")
        f.write("="*80 + "\n")
        for i, (disease, variance) in enumerate(top_diseases):
            f.write(f"{i+1}. {disease} (variance: {variance:.4f})\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("MOST VARIABLE SIGNATURES (Top 10)\n")
        f.write("="*80 + "\n")
        for i, sig_idx in enumerate(most_variable_sigs):
            f.write(f"{i+1}. Signature {sig_idx}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("="*80 + "\n")
    
    print(f"\nSaved: {report_path}")

def main(target_disease, n_pathways=4, output_dir='pathway_analysis_output'):
    """
    Run complete pathway analysis
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Discover pathways
    pathway_data = discover_pathways_10yr(target_disease, n_pathways)
    
    # Analyze most variable signatures
    K, T = thetas.shape[1], thetas.shape[2]
    most_variable_sigs = analyze_most_variable_signatures(pathway_data, thetas, K, T)
    
    # Analyze distinctive diseases
    top_diseases, pathway_disease_prevs = analyze_distinctive_diseases(pathway_data, Y, disease_names)
    
    # Create visualizations
    create_stacked_deviation_plot(pathway_data, thetas, output_dir)
    
    # Save summary report
    save_summary_report(pathway_data, top_diseases, most_variable_sigs, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput saved to: {output_dir}/")
    
    return pathway_data, top_diseases, most_variable_sigs

if __name__ == "__main__":
    # Run analysis for myocardial infarction
    pathway_data, top_diseases, most_variable_sigs = main("myocardial infarction", n_pathways=4)
