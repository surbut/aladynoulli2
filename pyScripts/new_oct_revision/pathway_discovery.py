#!/usr/bin/env python3
"""
Pathway Discovery Script
Discovers different pathways patients take to reach the same disease outcome
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

def load_full_data():
    """Load the full dataset for pathway discovery"""
    print("Loading full dataset...")
    
    # Load full Y matrix
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
    print(f"Loaded Y (full): {Y_full.shape}")
    
    # Load full thetas (using NEW model with PCs)
    thetas = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt').numpy()
    print(f"Loaded thetas: {thetas.shape}")
    
    # Load processed IDs - these are the actual eids for the first 400K patients
    # Patient index i in our analysis corresponds to processed_ids[i]
    processed_ids_df = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/processed_ids.csv')
    processed_ids = processed_ids_df['eid'].values[:400000]  # First 400K eids
    print(f"Loaded {len(processed_ids)} processed IDs")
    
    # Subset Y to match the patients we have thetas for (first 400K rows)
    # Y doesn't contain eids - the first 400K rows just correspond to the first 400K processed_ids
    Y = Y_full[:400000, :, :]
    print(f"Subset Y to first 400K patients: {Y.shape}")
    
    # Load disease names
    disease_names_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv')
    disease_names = disease_names_df['x'].tolist()
    
    print(f"Loaded {len(disease_names)} diseases")
    print(f"Total patients with complete data: {Y.shape[0]}")
    
    # Store processed IDs for later use in medication mapping
    return Y, thetas, disease_names, processed_ids

def discover_disease_pathways(target_disease_name, Y, thetas, disease_names, n_pathways=4, method='average_loading'):
    """
    Discover the different pathways patients take to reach the same disease outcome
    
    Parameters:
    - target_disease_name: Name of disease to analyze
    - Y: Full binary event matrix
    - thetas: Full signature loadings array
    - disease_names: List of disease names
    - n_pathways: Number of pathways to discover
    - method: 'average_loading', 'trajectory_similarity', or 'deviation_from_reference'
    """
    print(f"=== DISCOVERING PATHWAYS TO {target_disease_name.upper()} ===")
    print(f"Method: {method}")
    
    # Find the target disease
    target_disease_idx = None
    for i, name in enumerate(disease_names):
        if target_disease_name.lower() in name.lower():
            target_disease_idx = i
            break
    
    if target_disease_idx is None:
        print(f"Could not find disease: {target_disease_name}")
        return None
    
    print(f"Found target disease: {disease_names[target_disease_idx]} (index {target_disease_idx})")
    
    # Find all patients who developed this disease
    target_patients = []
    for patient_id in range(Y.shape[0]):
        if Y[patient_id, target_disease_idx, :].sum() > 0:
            # Find age at first occurrence
            first_occurrence = torch.where(Y[patient_id, target_disease_idx, :] > 0)[0]
            if len(first_occurrence) > 0:
                age_at_disease = first_occurrence.min().item() + 30
                target_patients.append({
                    'patient_id': patient_id,
                    'age_at_disease': age_at_disease
                })
    
    print(f"Found {len(target_patients)} patients who developed {target_disease_name}")
    
    if len(target_patients) < 50:
        print("Not enough patients for pathway analysis")
        return None
    
    # Get signature trajectories for these patients
    N, K, T = thetas.shape
    patient_trajectories = []
    
    for patient_info in target_patients:
        patient_id = patient_info['patient_id']
        age_at_disease = patient_info['age_at_disease']
        
        # Get signature trajectory
        theta_patient = thetas[patient_id, :, :]  # [K, T]
        
        patient_trajectories.append({
            'patient_id': patient_id,
            'age_at_disease': age_at_disease,
            'trajectory': theta_patient
        })
    
    # Create trajectory features for clustering - FOCUS ON PRE-DISEASE PERIOD
    print(f"\nCreating trajectory features for pathway discovery...")
    print(f"Method: {method}")
    
    if method == 'average_loading':
        # Method 1: Average signature loading BEFORE disease (to find different paths to disease)
        trajectory_features = []
        valid_patients = []
        
        for patient_info in patient_trajectories:
            trajectory = patient_info['trajectory']
            age_at_disease = patient_info['age_at_disease']
            
            # Get pre-disease trajectory (up to 5 years before disease)
            cutoff_idx = age_at_disease - 30  # Time index at disease onset
            lookback_idx = max(0, cutoff_idx - 5)  # Look back 5 years
            
            if cutoff_idx > 5:  # Need at least 5 years of pre-disease history
                pre_disease_traj = trajectory[:, lookback_idx:cutoff_idx]
                
                # Calculate average signature loading in the 5 years BEFORE disease
                avg_loadings = np.mean(pre_disease_traj, axis=1)  # Average over time dimension
                trajectory_features.append(avg_loadings)
                valid_patients.append(patient_info)
        
        trajectory_features = np.array(trajectory_features)
        patient_trajectories = valid_patients  # Update to only valid patients
        print(f"Created {trajectory_features.shape[1]} features per patient (average loading 5 years PRE-disease)")
        print(f"Kept {len(valid_patients)} patients with sufficient pre-disease history")
        
    elif method == 'trajectory_similarity':
        # Method 2: Detailed trajectory features BEFORE disease (dynamics + variance)
        trajectory_features = []
        valid_patients = []
        
        for patient_info in patient_trajectories:
            trajectory = patient_info['trajectory']
            age_at_disease = patient_info['age_at_disease']
            
            # Get pre-disease trajectory (5-10 years before disease)
            cutoff_idx = age_at_disease - 30
            lookback_10yr = max(0, cutoff_idx - 10)
            lookback_5yr = max(0, cutoff_idx - 5)
            
            if cutoff_idx > 5:  # Need at least 5 years of pre-disease history
                features = []
                
                # 1. Average loading in 5 years before disease
                recent_pre_disease = trajectory[:, lookback_5yr:cutoff_idx]
                avg_recent = np.mean(recent_pre_disease, axis=1)
                features.extend(avg_recent)
                
                # 2. Slope (10 years before vs 5 years before)
                if cutoff_idx > 10:
                    early_pre_disease = trajectory[:, lookback_10yr:lookback_5yr]
                    avg_early = np.mean(early_pre_disease, axis=1)
                    slope = avg_recent - avg_early
                    features.extend(slope)
                else:
                    features.extend(np.zeros(K))
                
                # 3. Variance in pre-disease period (instability)
                pre_var = np.var(recent_pre_disease, axis=1)
                features.extend(pre_var)
                
                # 4. Peak signatures in pre-disease period
                peak_sigs = np.max(recent_pre_disease, axis=1)
                features.extend(peak_sigs)
                
                # 5. Age at disease onset (normalized)
                features.append(age_at_disease / 100.0)
                
                trajectory_features.append(features)
                valid_patients.append(patient_info)
        
        trajectory_features = np.array(trajectory_features)
        patient_trajectories = valid_patients
        print(f"Created {trajectory_features.shape[1]} trajectory features per patient (PRE-disease dynamics)")
        print(f"Kept {len(valid_patients)} patients with sufficient pre-disease history")
    
    elif method == 'deviation_from_reference':
        # Method 3: DEVIATION from population reference BEFORE disease
        # This addresses the concern that age-matched patterns might dominate
        # We want to find what makes pathways DIFFERENT from typical age-related changes
        
        print(f"\n--- COMPUTING POPULATION REFERENCE FOR DEVIATION-BASED CLUSTERING ---")
        
        # Calculate population reference (average signature trajectory across all patients)
        print(f"Computing population-level signature reference from all {N} patients...")
        population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
        print(f"Population reference shape: {population_reference.shape}")
        
        trajectory_features = []
        valid_patients = []
        
        for patient_info in patient_trajectories:
            trajectory = patient_info['trajectory']
            age_at_disease = patient_info['age_at_disease']
            
            # Get pre-disease trajectory (5 years before disease)
            cutoff_idx = age_at_disease - 30  # Time index at disease onset
            lookback_idx = max(0, cutoff_idx - 5)  # Look back 5 years
            
            if cutoff_idx > 5:  # Need at least 5 years of pre-disease history
                # Get pre-disease trajectory for this patient
                pre_disease_traj = trajectory[:, lookback_idx:cutoff_idx]  # Shape: (K, 5)
                
                # Get corresponding population reference for same time window
                ref_traj = population_reference[:, lookback_idx:cutoff_idx]  # Shape: (K, 5)
                
                # Calculate DEVIATION from reference PER TIMEPOINT (like digital twinning)
                # This preserves temporal information and matches per timepoint
                deviation_per_timepoint = pre_disease_traj - ref_traj  # Shape: (K, 5)
                
                # Flatten to get per-timepoint deviations as features
                # This gives us K*5 features: deviation for each signature at each timepoint
                features = deviation_per_timepoint.flatten()  # Shape: (K*5,)
                
                trajectory_features.append(features)
                valid_patients.append(patient_info)
        
        trajectory_features = np.array(trajectory_features)
        patient_trajectories = valid_patients
        print(f"Created {trajectory_features.shape[1]} features per patient (DEVIATION from reference)")
        print(f"  - {K*5} features: deviation per signature per timepoint (K signatures × 5 timepoints)")
        print(f"Kept {len(valid_patients)} patients with sufficient pre-disease history")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(trajectory_features)
    
    # Cluster patients by their trajectories
    clusterer = KMeans(n_clusters=n_pathways, random_state=42, n_init=10)
    pathway_labels = clusterer.fit_predict(features_scaled)
    
    print(f"\nDiscovered {n_pathways} pathways to {target_disease_name}:")
    unique_labels, counts = np.unique(pathway_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Pathway {label}: {count} patients ({count/len(target_patients)*100:.1f}%)")
    
    # Assign pathway labels to patients
    for i, patient_info in enumerate(patient_trajectories):
        patient_info['pathway'] = pathway_labels[i]
    
    return {
        'target_disease': target_disease_name,
        'target_disease_idx': target_disease_idx,
        'patients': patient_trajectories,
        'pathway_labels': pathway_labels,
        'trajectory_features': trajectory_features,
        'features_scaled': features_scaled,
        'method': method
    }

def compare_clustering_methods(target_disease_name, Y, thetas, disease_names, n_pathways=4):
    """Compare all three clustering methods for the same disease"""
    print(f"=== COMPARING CLUSTERING METHODS FOR {target_disease_name.upper()} ===")
    
    # Method 1: Average loading (raw signature values)
    print("\n1. Clustering by Average Signature Loading:")
    pathway_data_avg = discover_disease_pathways(
        target_disease_name, Y, thetas, disease_names, n_pathways, method='average_loading'
    )
    
    # Method 2: Trajectory similarity (dynamics + variance)
    print("\n2. Clustering by Trajectory Similarity:")
    pathway_data_traj = discover_disease_pathways(
        target_disease_name, Y, thetas, disease_names, n_pathways, method='trajectory_similarity'
    )
    
    # Method 3: Deviation from reference (age-independent differences)
    print("\n3. Clustering by Deviation from Population Reference:")
    pathway_data_dev = discover_disease_pathways(
        target_disease_name, Y, thetas, disease_names, n_pathways, method='deviation_from_reference'
    )
    
    return pathway_data_avg, pathway_data_traj, pathway_data_dev

if __name__ == "__main__":
    # Load full data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Test with myocardial infarction
    print("\n" + "="*80)
    print("TESTING PATHWAY DISCOVERY")
    print("="*80)
    
    # Compare both methods
    pathway_data_avg, pathway_data_traj = compare_clustering_methods(
        "myocardial infarction", Y, thetas, disease_names, n_pathways=4
    )
    
    print("\n✅ Pathway discovery analysis complete!")
