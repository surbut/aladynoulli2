#!/usr/bin/env python3
"""
Analyze how predictions and signature loadings change across age offsets (t0-t9).

For patients with specific precursor diseases, track:
1. How their predictions change across offsets 0-9
2. Which signatures/clusters are most impacted
3. Which precursor diseases drive which signature changes

This extends the age offset analysis to understand model learning dynamics.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')

def get_major_diseases():
    """Get major disease groups mapping"""
    major_diseases = {
        'ASCVD': [
            'Myocardial infarction',
            'Coronary atherosclerosis',
            'Other acute and subacute forms of ischemic heart disease',
            'Unstable angina (intermediate coronary syndrome)',
            'Angina pectoris',
            'Other chronic ischemic heart disease, unspecified'
        ],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': [
            'Colon cancer',
            'Malignant neoplasm of rectum, rectosigmoid junction, and anus',
            'Cancer of bronchus; lung',
            'Breast cancer [female]',
            'Malignant neoplasm of female breast',
            'Cancer of prostate',
            'Malignant neoplasm of bladder',
            'Secondary malignant neoplasm',
            'Secondary malignancy of lymph nodes',
            'Secondary malignancy of respiratory organs',
            'Secondary malignant neoplasm of digestive systems',
            'Secondary malignant neoplasm of liver',
            'Secondary malignancy of bone'
        ],
        'Stroke': [
            'Cerebral artery occlusion, with cerebral infarction',
            'Cerebral ischemia'
        ],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': [
            'Chronic airway obstruction',
            'Emphysema',
            'Obstructive chronic bronchitis'
        ],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': [
            'Iron deficiency anemias, unspecified or not due to blood loss',
            'Other anemias'
        ],
        'Colorectal_Cancer': [
            'Colon cancer',
            'Malignant neoplasm of rectum, rectosigmoid junction, and anus'
        ],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
        'Prostate_Cancer': ['Cancer of prostate'],
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': [
            'Secondary malignant neoplasm',
            'Secondary malignancy of lymph nodes',
            'Secondary malignancy of respiratory organs',
            'Secondary malignant neoplasm of digestive systems',
            'Secondary malignant neoplasm of liver',
            'Secondary malignancy of bone'
        ],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Hypothyroidism NOS', 'Hyperthyroidism NOS'],
        'Osteoarthritis': ['Osteoarthritis NOS']
    }
    return major_diseases

def load_essentials():
    """Load model essentials."""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def load_cluster_assignments():
    """Load cluster assignments for diseases."""
    possible_paths = ['/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                clusters = torch.load(path, weights_only=False)
                if isinstance(clusters, torch.Tensor):
                    print(f"  ✓ Loaded clusters as tensor from: {path}")
                    result = clusters.numpy()
                    print(f"  Cluster shape: {result.shape}")
                    return result
                elif isinstance(clusters, np.ndarray):
                    print(f"  ✓ Loaded clusters as numpy array from: {path}")
                    print(f"  Cluster shape: {clusters.shape}")
                    return clusters
                elif isinstance(clusters, dict):
                    print(f"  Found dict with keys: {list(clusters.keys())}")
                    if 'clusters' in clusters:
                        print(f"  ✓ Loaded clusters from dict key 'clusters' in: {path}")
                        result = clusters['clusters'].numpy() if isinstance(clusters['clusters'], torch.Tensor) else clusters['clusters']
                        print(f"  Cluster shape: {result.shape}")
                        return result
                    elif 'initial_clusters' in clusters:
                        print(f"  ✓ Loaded clusters from dict key 'initial_clusters' in: {path}")
                        result = clusters['initial_clusters'].numpy() if isinstance(clusters['initial_clusters'], torch.Tensor) else clusters['initial_clusters']
                        print(f"  Cluster shape: {result.shape}")
                        return result
                    else:
                        print(f"  ⚠️  Unknown cluster format. Keys: {list(clusters.keys())}")
                        print(f"  Available keys don't match expected 'clusters' or 'initial_clusters'")
                else:
                    print(f"  ⚠️  Loaded object is unexpected type: {type(clusters)}")
            except Exception as e:
                print(f"  ⚠️  Error loading {path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"  ⚠️  Path does not exist: {path}")
    
    print(f"  ⚠️  Could not load clusters from any of {len(possible_paths)} paths")
    return None

def analyze_patient_prediction_changes(pi_batches, Y_batch, E_batch, disease_names, 
                                      precursor_diseases, target_disease_name='ASCVD',
                                      start_idx=0, end_idx=10000):
    """
    Analyze how predictions change across offsets for patients with specific precursors.
    
    Parameters:
    -----------
    pi_batches : list of tensors
        Pi tensors for offsets 0-9, shape [n_patients, n_diseases, n_timepoints]
    Y_batch : tensor
        Disease outcomes, shape [n_patients, n_diseases, n_timepoints]
    E_batch : tensor
        Event times, shape [n_patients, n_diseases]
    disease_names : list
        List of disease names
    precursor_diseases : list
        List of precursor disease names to analyze
    target_disease_name : str
        Target disease to predict (e.g., 'ASCVD')
    """
    
    # Get target disease indices
    major_diseases = get_major_diseases()
    if target_disease_name not in major_diseases:
        raise ValueError(f"Target disease {target_disease_name} not found in major diseases")
    
    target_disease_list = major_diseases[target_disease_name]
    target_indices = [i for i, name in enumerate(disease_names) if name in target_disease_list]
    
    if len(target_indices) == 0:
        raise ValueError(f"No diseases found for {target_disease_name}")
    
    # Get precursor disease indices
    precursor_indices = {}
    for precursor_name in precursor_diseases:
        for i, name in enumerate(disease_names):
            if precursor_name.lower() in name.lower():
                precursor_indices[precursor_name] = i
                break
    
    print(f"\nFound {len(precursor_indices)} precursor diseases:")
    for name, idx in precursor_indices.items():
        print(f"  {name}: index {idx}")
    
    n_patients = pi_batches[0].shape[0]
    n_offsets = len(pi_batches)
    
    # For each patient, track predictions across offsets
    patient_analysis = []
    
    for patient_idx in range(n_patients):
        # Get enrollment age
        enroll_age = E_batch[patient_idx, 0].item() if E_batch[patient_idx, 0] > 0 else 30
        t_enroll = int(enroll_age - 30)
        
        if t_enroll < 0 or t_enroll >= pi_batches[0].shape[2]:
            continue
        
        # Check which precursors this patient has (before enrollment)
        has_precursors = {}
        for precursor_name, precursor_idx in precursor_indices.items():
            if t_enroll > 0:
                has_precursors[precursor_name] = Y_batch[patient_idx, precursor_idx, :t_enroll].sum() > 0
            else:
                has_precursors[precursor_name] = False
        
        # Get predictions across all offsets
        predictions_by_offset = []
        for offset_idx, pi_offset in enumerate(pi_batches):
            # Prediction time is t_enroll + offset_idx
            pred_time = t_enroll + offset_idx
            
            if pred_time >= pi_offset.shape[2]:
                predictions_by_offset.append(np.nan)
                continue
            
            # Get combined risk for target disease group
            pi_diseases = pi_offset[patient_idx, target_indices, pred_time].numpy()
            risk = 1 - np.prod(1 - pi_diseases)
            predictions_by_offset.append(risk)
        
        # Calculate prediction change (offset 9 - offset 0)
        if not np.isnan(predictions_by_offset[0]) and not np.isnan(predictions_by_offset[-1]):
            prediction_change = predictions_by_offset[-1] - predictions_by_offset[0]
        else:
            prediction_change = np.nan
        
        # Check if patient had target event in year after enrollment
        if t_enroll + 2 <= Y_batch.shape[2]:
            had_event = Y_batch[patient_idx, target_indices, t_enroll:t_enroll+2].sum().item() > 0
        else:
            had_event = False
        
        patient_analysis.append({
            'patient_idx': start_idx + patient_idx,
            'enroll_age': enroll_age,
            'prediction_offset_0': predictions_by_offset[0],
            'prediction_offset_9': predictions_by_offset[-1],
            'prediction_change': prediction_change,
            'had_event': had_event,
            **has_precursors  # Add all precursor flags
        })
    
    return pd.DataFrame(patient_analysis)

def load_model_checkpoints(pi_base_dir, start_idx, end_idx, max_offset=9):
    """Load model checkpoints and extract lambda (patient-specific parameters) for each offset."""
    model_filename_pattern = 'model_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun.pt'
    
    lambda_by_offset = []
    
    for k in range(max_offset + 1):
        model_filename = model_filename_pattern.format(k=k, start=start_idx, end=end_idx)
        model_path = pi_base_dir / model_filename
        
        if not model_path.exists():
            print(f"⚠️  Model checkpoint not found for offset {k}: {model_path}")
            lambda_by_offset.append(None)
            continue
        
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Try different possible keys for lambda (patient-specific parameters)
            lambda_val = None
            if isinstance(checkpoint, dict):
                # First check model_state_dict['lambda_'] (most common location)
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                    if 'lambda_' in model_state_dict:
                        lambda_val = model_state_dict['lambda_']
                        print(f"    Found lambda in model_state_dict['lambda_']")
                    elif '_lambda' in model_state_dict:
                        lambda_val = model_state_dict['_lambda']
                        print(f"    Found lambda in model_state_dict['_lambda']")
                # Also check other possible locations
                if lambda_val is None:
                    if 'lambda' in checkpoint:
                        lambda_val = checkpoint['lambda']
                    elif 'lambda_' in checkpoint:
                        lambda_val = checkpoint['lambda_']
                elif 'model' in checkpoint:
                    model = checkpoint['model']
                    if hasattr(model, 'lambda'):
                        lambda_val = getattr(model, 'lambda')  # Use getattr because lambda is a reserved keyword
                    elif hasattr(model, 'lambda_'):
                        lambda_val = getattr(model, 'lambda_')
                elif 'state_dict' in checkpoint:
                    # Look for lambda in state_dict
                    state_dict = checkpoint['state_dict']
                    if 'lambda_' in state_dict:
                        lambda_val = state_dict['lambda_']
                    elif '_lambda' in state_dict:
                        lambda_val = state_dict['_lambda']
                    else:
                        for key in state_dict.keys():
                            if 'lambda' in key.lower() and 'phi' not in key.lower():
                                lambda_val = state_dict[key]
                                break
            
            lambda_by_offset.append(lambda_val)
            if lambda_val is not None:
                print(f"  ✓ Loaded lambda for offset {k}: shape {lambda_val.shape if isinstance(lambda_val, torch.Tensor) else 'unknown'}")
            else:
                print(f"  ⚠️  Could not extract lambda from offset {k} checkpoint")
                if isinstance(checkpoint, dict):
                    print(f"    Available keys: {list(checkpoint.keys())}")
                    if 'model_state_dict' in checkpoint:
                        print(f"    model_state_dict keys (first 10): {list(checkpoint['model_state_dict'].keys())[:10]}")
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint for offset {k}: {e}")
            lambda_by_offset.append(None)
    
    return lambda_by_offset

def analyze_signature_changes(pi_batches, Y_batch, E_batch, disease_names, 
                              cluster_assignments, precursor_diseases, target_disease_name='ASCVD',
                              lambda_by_offset=None, start_idx=0, end_idx=10000):
    """
    Analyze which signatures/clusters change most for patients with specific precursors.
    
    If lambda_by_offset is provided, analyzes lambda (patient-specific parameter) changes.
    Otherwise, uses cluster assignments to track changes.
    """
    
    # Get target disease indices
    major_diseases = get_major_diseases()
    target_disease_list = major_diseases[target_disease_name]
    target_indices = [i for i, name in enumerate(disease_names) if name in target_disease_list]
    
    # Get precursor disease indices
    precursor_indices = {}
    for precursor_name in precursor_diseases:
        for i, name in enumerate(disease_names):
            if precursor_name.lower() in name.lower():
                precursor_indices[precursor_name] = i
                break
    
    n_patients = pi_batches[0].shape[0]
    
    # If we have lambda, analyze patient-specific parameter changes
    if lambda_by_offset is not None and any(l is not None for l in lambda_by_offset):
        print("\nAnalyzing lambda (patient-specific parameter) changes across offsets...")
        return analyze_lambda_changes(lambda_by_offset, precursor_indices, target_indices, 
                                     Y_batch, E_batch, disease_names, cluster_assignments)
    
    # Otherwise, use cluster-based analysis
    if cluster_assignments is None:
        print("⚠️  Cluster assignments not available and lambda not loaded, skipping signature analysis")
        return None
    
    print("\nAnalyzing cluster-based changes (lambda not available, using cluster assignments)...")
    
    # Get target disease indices
    major_diseases = get_major_diseases()
    target_disease_list = major_diseases[target_disease_name]
    target_indices = [i for i, name in enumerate(disease_names) if name in target_disease_list]
    
    # Get precursor disease indices
    precursor_indices = {}
    for precursor_name in precursor_diseases:
        for i, name in enumerate(disease_names):
            if precursor_name.lower() in name.lower():
                precursor_indices[precursor_name] = i
                break
    
    n_patients = pi_batches[0].shape[0]
    n_offsets = len(pi_batches)
    
    # For each signature/cluster, track changes
    signature_changes_by_precursor = defaultdict(lambda: defaultdict(list))
    
    for patient_idx in range(n_patients):
        enroll_age = E_batch[patient_idx, 0].item() if E_batch[patient_idx, 0] > 0 else 30
        t_enroll = int(enroll_age - 30)
        
        if t_enroll < 0:
            continue
        
        # Check which precursors this patient has
        for precursor_name, precursor_idx in precursor_indices.items():
            if t_enroll > 0:
                has_precursor = Y_batch[patient_idx, precursor_idx, :t_enroll].sum() > 0
            else:
                has_precursor = False
            
            if not has_precursor:
                continue
            
            # For each disease in target group, track signature changes
            for target_idx in target_indices:
                cluster_id = cluster_assignments[target_idx] if target_idx < len(cluster_assignments) else -1
                
                if cluster_id == -1:
                    continue
                
                # Get predictions at offset 0 and offset 9
                pred_time_0 = t_enroll
                pred_time_9 = min(t_enroll + 9, pi_batches[0].shape[2] - 1)
                
                if pred_time_0 < pi_batches[0].shape[2] and pred_time_9 < pi_batches[-1].shape[2]:
                    pi_0 = pi_batches[0][patient_idx, target_idx, pred_time_0].item()
                    pi_9 = pi_batches[-1][patient_idx, target_idx, pred_time_9].item()
                    
                    change = pi_9 - pi_0
                    signature_changes_by_precursor[precursor_name][f'Cluster_{int(cluster_id)}'].append(change)
    
    # Summarize signature changes
    signature_summary = []
    for precursor_name, cluster_changes in signature_changes_by_precursor.items():
        for cluster_name, changes in cluster_changes.items():
            if len(changes) > 0:
                signature_summary.append({
                    'Precursor': precursor_name,
                    'Cluster': cluster_name,
                    'N_patients': len(changes),
                    'Mean_change': np.mean(changes),
                    'Median_change': np.median(changes),
                    'Std_change': np.std(changes)
                })
    
    return pd.DataFrame(signature_summary)

def analyze_lambda_changes(lambda_by_offset, precursor_indices, target_indices,
                          Y_batch, E_batch, disease_names, cluster_assignments):
    """
    Analyze how lambda (patient-specific parameters) changes across offsets for patients with precursors.
    
    lambda_by_offset: list of lambda tensors, one per offset
    Lambda should be shape [n_patients, n_signatures] or [n_patients, n_diseases, n_signatures]
    """
    
    n_offsets = len(lambda_by_offset)
    n_patients = Y_batch.shape[0]
    
    # Find first non-None lambda to determine shape
    lambda_0 = None
    for l in lambda_by_offset:
        if l is not None:
            lambda_0 = l
            break
    
    if lambda_0 is None:
        return None
    
    # Determine lambda shape
    lambda_shape = lambda_0.shape
    print(f"  Lambda shape: {lambda_shape}")
    
    # Lambda could be:
    # - [n_patients, n_signatures] - patient-specific signature loadings (no time dimension)
    # - [n_patients, n_signatures, n_timepoints] - patient-signature-time (most common)
    # - [n_patients, n_diseases, n_signatures] - patient-disease-signature
    
    if len(lambda_shape) == 2:
        if lambda_shape[0] == n_patients:
            # [n_patients, n_signatures]
            n_signatures = lambda_shape[1]
            lambda_per_disease = False
            lambda_has_time = False
        else:
            print(f"⚠️  Unexpected lambda shape: {lambda_shape}")
            return None
    elif len(lambda_shape) == 3:
        if lambda_shape[0] == n_patients:
            # Could be [n_patients, n_signatures, n_timepoints] or [n_patients, n_diseases, n_signatures]
            # Check: if middle dimension is small (< 50), likely signatures; if large, likely diseases
            if lambda_shape[1] < 50:  # Likely signatures (21 signatures)
                # [n_patients, n_signatures, n_timepoints]
                n_signatures = lambda_shape[1]
                n_timepoints = lambda_shape[2]
                lambda_per_disease = False
                lambda_has_time = True
            else:  # Likely diseases
                # [n_patients, n_diseases, n_signatures]
                n_signatures = lambda_shape[2]
                lambda_per_disease = True
                lambda_has_time = False
        else:
            print(f"⚠️  Unexpected lambda shape: {lambda_shape}")
            return None
    else:
        print(f"⚠️  Unexpected lambda shape: {lambda_shape}")
        return None
    
    print(f"  n_signatures: {n_signatures}, lambda_per_disease: {lambda_per_disease}, lambda_has_time: {lambda_has_time}")
    
    # For each precursor, track lambda changes by signature
    signature_changes_by_precursor = defaultdict(lambda: defaultdict(list))
    
    patients_with_precursors_count = 0
    
    for patient_idx in range(n_patients):
        enroll_age = E_batch[patient_idx, 0].item() if E_batch[patient_idx, 0] > 0 else 30
        t_enroll = int(enroll_age - 30)
        
        if t_enroll < 0:
            continue
        
        # Check which precursors this patient has
        for precursor_name, precursor_idx in precursor_indices.items():
            if t_enroll > 0:
                has_precursor = Y_batch[patient_idx, precursor_idx, :t_enroll].sum() > 0
            else:
                has_precursor = False
            
            if not has_precursor:
                continue
            
            patients_with_precursors_count += 1
            
            # Get lambda values for this patient across offsets
            # For each offset, extract lambda at the appropriate timepoint
            lambda_values_by_offset = []
            for offset_idx, lambda_offset in enumerate(lambda_by_offset):
                if lambda_offset is None:
                    lambda_values_by_offset.append(None)
                    continue
                
                # Determine timepoint: at offset k, we're predicting at t_enroll + k
                pred_time = t_enroll + offset_idx
                
                if lambda_has_time:
                    # Lambda is [n_patients, n_signatures, n_timepoints]
                    if pred_time < lambda_offset.shape[2]:
                        lambda_patient = lambda_offset[patient_idx, :, pred_time]
                    else:
                        lambda_values_by_offset.append(None)
                        continue
                elif lambda_per_disease:
                    # Lambda is [n_patients, n_diseases, n_signatures]
                    # For target diseases, average across diseases in group
                    lambda_patient = lambda_offset[patient_idx, target_indices, :].mean(dim=0)
                else:
                    # Lambda is [n_patients, n_signatures] (no time dimension)
                    lambda_patient = lambda_offset[patient_idx, :]
                
                lambda_values_by_offset.append(lambda_patient.cpu().numpy() if isinstance(lambda_patient, torch.Tensor) else lambda_patient)
            
            # Calculate change for each signature (offset 9 - offset 0)
            if lambda_values_by_offset[0] is not None and lambda_values_by_offset[-1] is not None:
                lambda_change = lambda_values_by_offset[-1] - lambda_values_by_offset[0]
                
                # Track changes for each signature
                for sig_idx in range(n_signatures):
                    signature_changes_by_precursor[precursor_name][f'Signature_{sig_idx}'].append(lambda_change[sig_idx])
    
    print(f"  Found {patients_with_precursors_count} patient-precursor combinations")
    
    # Summarize lambda changes by signature
    signature_summary = []
    for precursor_name, sig_changes in signature_changes_by_precursor.items():
        for sig_name, changes in sig_changes.items():
            if len(changes) > 0:
                signature_summary.append({
                    'Precursor': precursor_name,
                    'Signature': sig_name,
                    'N_patients': len(changes),
                    'Mean_change': np.mean(changes),
                    'Median_change': np.median(changes),
                    'Std_change': np.std(changes),
                    'Abs_mean_change': np.mean(np.abs(changes))
                })
    
    print(f"  Generated {len(signature_summary)} signature change records")
    
    return pd.DataFrame(signature_summary)

def main():
    parser = argparse.ArgumentParser(description='Analyze age offset signature changes')
    parser.add_argument('--approach', type=str, default='pooled_retrospective',
                       help='Approach name')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start patient index')
    parser.add_argument('--end_idx', type=int, default=10000,
                       help='End patient index')
    parser.add_argument('--target_disease', type=str, default='ASCVD',
                       help='Target disease to analyze')
    parser.add_argument('--precursors', type=str, nargs='+',
                       default=['Hypercholesterolemia', 'Essential hypertension', 'Type 2 diabetes'],
                       help='Precursor diseases to analyze')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANALYZING AGE OFFSET SIGNATURE CHANGES")
    print("="*80)
    print(f"\nApproach: {args.approach}")
    print(f"Batch: {args.start_idx}-{args.end_idx}")
    print(f"Target disease: {args.target_disease}")
    print(f"Precursor diseases: {args.precursors}")
    
    # Load essentials
    print("\nLoading essentials...")
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load cluster assignments
    print("Loading cluster assignments...")
    cluster_assignments = load_cluster_assignments()
    if cluster_assignments is not None:
        print(f"✓ Loaded clusters: {len(cluster_assignments)} diseases, {len(np.unique(cluster_assignments))} clusters")
    else:
        print("⚠️  Could not load cluster assignments")
    
    # Load data batch
    print(f"\nLoading data batch {args.start_idx}-{args.end_idx}...")
    Y_batch = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_batch = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    
    # Subset to batch
    Y_batch = Y_batch[args.start_idx:args.end_idx]
    E_batch = E_batch[args.start_idx:args.end_idx]
    
    # Load pi batches and model checkpoints for offsets 0-9
    print("\nLoading pi batches for offsets 0-9...")
    pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/age_offset_files')
    pi_filename_pattern = 'pi_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun.pt'
    
    pi_batches = []
    for k in range(10):  # Offsets 0-9
        pi_filename = pi_filename_pattern.format(k=k, start=args.start_idx, end=args.end_idx)
        pi_path = pi_base_dir / pi_filename
        
        if not pi_path.exists():
            print(f"⚠️  Pi file not found for offset {k}: {pi_path}")
            continue
        
        print(f"  Loading offset {k}...")
        pi_batch = torch.load(pi_path, weights_only=False)
        pi_batches.append(pi_batch)
    
    if len(pi_batches) == 0:
        raise ValueError("No pi batches loaded!")
    
    print(f"✓ Loaded {len(pi_batches)} pi batches")
    
    # Load model checkpoints to extract lambda (patient-specific parameters)
    print("\nLoading model checkpoints to extract lambda (patient-specific parameters)...")
    lambda_by_offset = load_model_checkpoints(pi_base_dir, args.start_idx, args.end_idx, max_offset=9)
    
    n_lambda_loaded = sum(1 for l in lambda_by_offset if l is not None)
    if n_lambda_loaded > 0:
        print(f"✓ Loaded lambda from {n_lambda_loaded}/{len(lambda_by_offset)} model checkpoints")
    else:
        print("⚠️  Could not load lambda from model checkpoints, will use cluster-based analysis")
    
    # Analyze patient-level prediction changes
    print("\n" + "="*80)
    print("ANALYZING PATIENT-LEVEL PREDICTION CHANGES")
    print("="*80)
    
    patient_df = analyze_patient_prediction_changes(
        pi_batches, Y_batch, E_batch, disease_names,
        args.precursors, args.target_disease,
        args.start_idx, args.end_idx
    )
    
    print(f"\nAnalyzed {len(patient_df)} patients")
    
    # Summary by precursor
    print("\n" + "="*80)
    print("PREDICTION CHANGES BY PRECURSOR DISEASE")
    print("="*80)
    
    for precursor in args.precursors:
        if precursor in patient_df.columns:
            precursor_patients = patient_df[patient_df[precursor] == True]
            if len(precursor_patients) > 0:
                mean_change = precursor_patients['prediction_change'].mean()
                median_change = precursor_patients['prediction_change'].median()
                event_rate = precursor_patients['had_event'].mean() * 100
                
                print(f"\n{precursor}:")
                print(f"  N patients: {len(precursor_patients)}")
                print(f"  Mean prediction change (offset 9 - 0): {mean_change:.4f}")
                print(f"  Median prediction change: {median_change:.4f}")
                print(f"  Event rate (year 0-1): {event_rate:.1f}%")
    
    # Analyze signature changes
    if cluster_assignments is not None:
        print("\n" + "="*80)
        print("ANALYZING SIGNATURE/CLUSTER CHANGES")
        print("="*80)
        
        signature_df = analyze_signature_changes(
            pi_batches, Y_batch, E_batch, disease_names,
            cluster_assignments, args.precursors, args.target_disease,
            lambda_by_offset=lambda_by_offset,
            start_idx=args.start_idx, end_idx=args.end_idx
        )
        
        if signature_df is not None and len(signature_df) > 0:
            print("\nSignature changes by precursor:")
            print(signature_df.to_string(index=False))
            
            # Save
            output_dir = Path('results/analysis')
            output_dir.mkdir(parents=True, exist_ok=True)
            signature_df.to_csv(output_dir / f'signature_changes_age_offset_{args.target_disease}.csv', index=False)
            print(f"\n✓ Saved signature changes to: {output_dir / f'signature_changes_age_offset_{args.target_disease}.csv'}")
        elif signature_df is not None:
            print(f"⚠️  Signature analysis returned empty DataFrame (0 rows)")
        else:
            print(f"⚠️  Signature analysis returned None")
    
    # Save patient-level analysis
    output_dir = Path('results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(output_dir / f'patient_prediction_changes_age_offset_{args.target_disease}.csv', index=False)
    print(f"\n✓ Saved patient-level analysis to: {output_dir / f'patient_prediction_changes_age_offset_{args.target_disease}.csv'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

