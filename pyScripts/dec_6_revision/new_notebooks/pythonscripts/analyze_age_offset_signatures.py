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
                                      start_idx=0, end_idx=10000, cluster_assignments=None):
    """
    Analyze how predictions change across offsets for patients with specific precursors.
    Accounts for signature/cluster membership to distinguish signature-related precursors.
    
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
    cluster_assignments : numpy array, optional
        Cluster assignments for diseases, shape [n_diseases]
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
    
    # Find target signature (which signature do ASCVD diseases belong to?)
    target_signature = None
    precursor_signature_membership = {}
    
    if cluster_assignments is not None and len(cluster_assignments) == len(disease_names):
        # Find the signature that target diseases belong to (most common)
        target_signatures = [cluster_assignments[idx] for idx in target_indices if idx < len(cluster_assignments)]
        if len(target_signatures) > 0:
            # Use most common signature among target diseases
            from collections import Counter
            target_signature = Counter(target_signatures).most_common(1)[0][0]
            print(f"\n✓ Target diseases belong to Signature {target_signature}")
        
        # Check which precursors belong to the same signature
        for precursor_name, precursor_idx in precursor_indices.items():
            if precursor_idx < len(cluster_assignments):
                precursor_sig = cluster_assignments[precursor_idx]
                precursor_signature_membership[precursor_name] = (precursor_sig == target_signature)
                if precursor_sig == target_signature:
                    print(f"  ✓ {precursor_name} belongs to Signature {target_signature} (signature-related)")
                else:
                    print(f"  - {precursor_name} belongs to Signature {precursor_sig} (not signature-related)")
    else:
        print(f"\n⚠️  Cluster assignments not available or wrong size. Skipping signature analysis.")
        # Default: assume all precursors are signature-related (conservative)
        for precursor_name in precursor_indices.keys():
            precursor_signature_membership[precursor_name] = True
    
    n_patients = pi_batches[0].shape[0]
    n_offsets = len(pi_batches)
    
    # For each patient, track predictions across offsets
    patient_analysis = []
    
    for patient_idx in range(n_patients):
        # Get enrollment time (E already contains age - 30)
        t_enroll = int(E_batch[patient_idx, 0].item()) if E_batch[patient_idx, 0] > 0 else 0
        
        if t_enroll < 0 or t_enroll >= pi_batches[0].shape[2]:
            continue
        
        # Check which precursors this patient has (before enrollment)
        has_precursors = {}
        for precursor_name, precursor_idx in precursor_indices.items():
            if t_enroll > 0:
                has_precursors[precursor_name] = Y_batch[patient_idx, precursor_idx, :t_enroll].sum() > 0
            else:
                has_precursors[precursor_name] = False
        
        # Compare predictions at SAME timepoint (t_enroll+9) from models with different washout periods
        # This matches Sasha's question: Predict for age 50 using data from 0-49 vs 0-45
        target_prediction_time = t_enroll + 9  # Predict 9 years in the future (e.g., age 50)
        
        # Model 1: Trained to t_enroll (9-year washout) - equivalent to "data from age 0-49"
        # Model 2: Trained to t_enroll+5 (4-year washout) - equivalent to "data from age 0-45"
        # Both predict at t_enroll+9 (age 50)
        
        # Get prediction from model trained to t_enroll (offset 0, 9-year washout)
        pred_time = target_prediction_time
        pred_offset_0 = np.nan
        if pred_time < pi_batches[0].shape[2]:
            # Check if patient had event before prediction time
            had_event_before_pred = False
            if pred_time > t_enroll and pred_time <= Y_batch.shape[2]:
                had_event_before_pred = Y_batch[patient_idx, target_indices, t_enroll:pred_time].sum().item() > 0
            
            if not had_event_before_pred:
                pi_diseases = pi_batches[0][patient_idx, target_indices, pred_time].numpy()
                pred_offset_0 = 1 - np.prod(1 - pi_diseases)
        
        # Get prediction from model trained to t_enroll+5 (offset 5, 4-year washout)
        pred_offset_5 = np.nan
        if len(pi_batches) > 5 and pred_time < pi_batches[5].shape[2]:
            # Check if patient had event before prediction time
            had_event_before_pred = False
            if pred_time > t_enroll and pred_time <= Y_batch.shape[2]:
                had_event_before_pred = Y_batch[patient_idx, target_indices, t_enroll:pred_time].sum().item() > 0
            
            if not had_event_before_pred:
                pi_diseases = pi_batches[5][patient_idx, target_indices, pred_time].numpy()
                pred_offset_5 = 1 - np.prod(1 - pi_diseases)
        
        # Calculate prediction change (4yr washout vs 9yr washout)
        if not np.isnan(pred_offset_0) and not np.isnan(pred_offset_5):
            prediction_change_4yr = pred_offset_5 - pred_offset_0  # 4yr washout - 9yr washout
        else:
            prediction_change_4yr = np.nan
        
        # Categorize patients based on what happened in washout period (t_enroll+5 to t_enroll+9)
        # This is the period between the two training cutoffs
        washout_start = t_enroll + 5
        washout_end = t_enroll + 9
        
        had_outcome_in_washout = False
        had_precursor_in_washout = {}
        
        if washout_end <= Y_batch.shape[2]:
            # Check for outcome (ASCVD) in washout period
            had_outcome_in_washout = Y_batch[patient_idx, target_indices, washout_start:washout_end].sum().item() > 0
            
            # Check for precursors in washout period
            for precursor_name, precursor_idx in precursor_indices.items():
                had_precursor_in_washout[precursor_name] = Y_batch[patient_idx, precursor_idx, washout_start:washout_end].sum().item() > 0
        
        # Categorize: Conservative washout (had outcome) vs Accurate washout (had signature-related precursor, no outcome)
        # Only precursors in the same signature as target disease count as "accurate washout"
        had_signature_related_precursor = False
        had_unrelated_precursor = False
        
        for precursor_name, had_precursor in had_precursor_in_washout.items():
            if had_precursor:
                if precursor_signature_membership.get(precursor_name, False):
                    had_signature_related_precursor = True
                else:
                    had_unrelated_precursor = True
        
        washout_category = 'neither'
        if had_outcome_in_washout:
            washout_category = 'conservative'  # Had real outcome condition
        elif had_signature_related_precursor:
            washout_category = 'accurate'  # Had signature-related pre-clinical condition, no outcome
        elif had_unrelated_precursor:
            washout_category = 'unrelated'  # Had precursor but not signature-related
        
        # For backward compatibility, also track early events
        if t_enroll + 2 <= Y_batch.shape[2]:
            had_event_early = Y_batch[patient_idx, target_indices, t_enroll:t_enroll+2].sum().item() > 0
        else:
            had_event_early = False
        
        had_event = had_event_early
        
        # Calculate enroll_age from t_enroll (E already contains age - 30, so enroll_age = t_enroll + 30)
        enroll_age = t_enroll + 30
        
        # Track signature-related precursors in washout
        had_sig_related_precursor_in_washout = False
        for precursor_name, had_precursor in had_precursor_in_washout.items():
            if had_precursor and precursor_signature_membership.get(precursor_name, False):
                had_sig_related_precursor_in_washout = True
                break
        
        patient_analysis.append({
            'patient_idx': start_idx + patient_idx,
            'enroll_age': enroll_age,
            'prediction_offset_0': pred_offset_0,  # Model trained to t_enroll (9yr washout), predicts at t_enroll+9
            'prediction_offset_5': pred_offset_5,  # Model trained to t_enroll+5 (4yr washout), predicts at t_enroll+9
            'prediction_change_4yr': prediction_change_4yr,  # Change: 4yr washout - 9yr washout (both predict at t_enroll+9)
            'washout_category': washout_category,  # 'conservative', 'accurate', 'unrelated', or 'neither'
            'had_outcome_in_washout': had_outcome_in_washout,  # Had ASCVD event in washout period (t_enroll+5 to t_enroll+9)
            'had_sig_related_precursor_in_washout': had_sig_related_precursor_in_washout,  # Had signature-related precursor in washout
            'had_event': had_event,  # Had event in first 2 years (for backward compatibility)
            **has_precursors,  # Add all precursor flags (before enrollment)
            **{f'had_{k}_in_washout': v for k, v in had_precursor_in_washout.items()}  # Add precursor flags for washout period
        })
    
    return pd.DataFrame(patient_analysis)

def load_model_checkpoints(pi_base_dir, start_idx, end_idx, max_offset=9):
    """Load model checkpoints and extract lambda (patient-specific parameters) for each offset."""
    model_filename_pattern = 'model_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun_pooledall.pt'
    
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
        # E already contains age - 30, so use it directly
        t_enroll = int(E_batch[patient_idx, 0].item()) if E_batch[patient_idx, 0] > 0 else 0
        
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
    precursor_check_debug = {name: 0 for name in precursor_indices.keys()}
    
    print(f"  Checking {n_patients} patients for precursors: {list(precursor_indices.keys())}")
    print(f"  Precursor indices: {precursor_indices}")
    print(f"  Y_batch shape: {Y_batch.shape}")
    
    t_enroll_stats = []
    for patient_idx in range(n_patients):
        # E already contains age - 30, so use it directly
        t_enroll = int(E_batch[patient_idx, 0].item()) if E_batch[patient_idx, 0] > 0 else 0
        t_enroll_stats.append(t_enroll)
    
    print(f"  t_enroll stats: min={min(t_enroll_stats)}, max={max(t_enroll_stats)}, mean={np.mean(t_enroll_stats):.1f}")
    print(f"  Patients with t_enroll > 0: {sum(1 for t in t_enroll_stats if t > 0)}")
    print(f"  Patients with t_enroll <= 0: {sum(1 for t in t_enroll_stats if t <= 0)}")
    
    # Quick check: do any patients have these precursors at ANY time?
    for precursor_name, precursor_idx in precursor_indices.items():
        if precursor_idx < Y_batch.shape[1]:
            any_time_count = (Y_batch[:, precursor_idx, :].sum(dim=1) > 0).sum().item()
            print(f"  Patients with {precursor_name} at ANY time: {any_time_count}")
    
    for patient_idx in range(n_patients):
        # E already contains age - 30, so use it directly
        t_enroll = int(E_batch[patient_idx, 0].item()) if E_batch[patient_idx, 0] > 0 else 0
        
        if t_enroll < 0:
            continue
        
        # Analyze ALL patients (not just those with precursors before enrollment)
        # We want to check if they developed precursors NEWLY in the washout period
        # This captures the true pre-clinical signal
        
        # Target timepoint: t_enroll + 9 (future prediction)
        target_timepoint = t_enroll + 9
        washout_start_4yr = t_enroll + 5  # 4-year washout period
        washout_start_9yr = t_enroll  # 9-year washout period
        
        # Check if target timepoint is valid (within lambda tensor bounds)
        if target_timepoint >= lambda_by_offset[0].shape[2] if lambda_by_offset[0] is not None else True:
            continue  # Skip if target timepoint is out of bounds
        
        # Get lambda values for this patient across offsets
        lambda_values_by_offset = []
        for offset_idx, lambda_offset in enumerate(lambda_by_offset):
            if lambda_offset is None:
                lambda_values_by_offset.append(None)
                continue
            
            # Extract lambda at the SAME future timepoint (t_enroll+9) from all models
            pred_time = target_timepoint
            
            if lambda_has_time:
                if pred_time < lambda_offset.shape[2]:
                    lambda_patient = lambda_offset[patient_idx, :, pred_time]
                else:
                    lambda_values_by_offset.append(None)
                    continue
            elif lambda_per_disease:
                lambda_patient = lambda_offset[patient_idx, target_indices, :].mean(dim=0)
            else:
                lambda_patient = lambda_offset[patient_idx, :]
            
            lambda_values_by_offset.append(lambda_patient.cpu().numpy() if isinstance(lambda_patient, torch.Tensor) else lambda_patient)
        
        # Check for outcome events in washout periods
        had_outcome_event_4yr = False
        had_outcome_event_9yr = False
        if target_timepoint <= Y_batch.shape[2]:
            if washout_start_4yr < target_timepoint:
                had_outcome_event_4yr = Y_batch[patient_idx, target_indices, washout_start_4yr:target_timepoint].sum().item() > 0
            if washout_start_9yr < target_timepoint:
                had_outcome_event_9yr = Y_batch[patient_idx, target_indices, washout_start_9yr:target_timepoint].sum().item() > 0
        
        # For each precursor, check if it was NEWLY developed in washout period
        for precursor_name, precursor_idx in precursor_indices.items():
            if precursor_idx >= Y_batch.shape[1]:
                continue  # Skip if precursor index is out of bounds
            
            # Check if precursor was present DURING washout period
            # This includes precursors that were present before OR developed during washout
            # Both are pre-clinical signals (as long as patient didn't have the outcome)
            had_precursor_during_4yr = False
            had_precursor_during_9yr = False
            if washout_start_4yr < target_timepoint and target_timepoint <= Y_batch.shape[2]:
                had_precursor_during_4yr = Y_batch[patient_idx, precursor_idx, washout_start_4yr:target_timepoint].sum().item() > 0
            if washout_start_9yr < target_timepoint and target_timepoint <= Y_batch.shape[2]:
                had_precursor_during_9yr = Y_batch[patient_idx, precursor_idx, washout_start_9yr:target_timepoint].sum().item() > 0
            
            # Track patients for debugging
            if had_precursor_during_4yr or had_precursor_during_9yr:
                precursor_check_debug[precursor_name] += 1
            
            # Analyze this patient-precursor combination
            patients_with_precursors_count += 1
            
            # Get lambda values for this patient across offsets
            # KEY COMPARISON: Compare lambda at the SAME future timepoint (t_enroll+9)
            # from models trained with different amounts of data:
            # - Offset 0: Model trained up to t_enroll, lambda at t_enroll+9
            # - Offset 9: Model trained up to t_enroll+9, lambda at t_enroll+9
            # This isolates model learning: if lambda changes, it's because the model
            # learned from additional data (ages t_enroll to t_enroll+9), not time progression.
            # 
            # If predictions change because patients developed real conditions (ages t_enroll to t_enroll+9),
            # that's conservative washout. If they change due to pre-clinical conditions detected
            # in the training data, that's accurate washout.
            
            # Target timepoint: t_enroll + 9 (future prediction)
            # Compare lambda at this SAME absolute timepoint from models trained at different offsets
            # This shows how predictions for the same future time change as models are trained
            # with more data (washout analysis)
            target_timepoint = t_enroll + 9
            
            # Check if target timepoint is valid (within lambda tensor bounds)
            if target_timepoint >= lambda_by_offset[0].shape[2] if lambda_by_offset[0] is not None else True:
                continue  # Skip if target timepoint is out of bounds
            
            lambda_values_by_offset = []
            for offset_idx, lambda_offset in enumerate(lambda_by_offset):
                if lambda_offset is None:
                    lambda_values_by_offset.append(None)
                    continue
                
                # Extract lambda at the SAME future timepoint (t_enroll+9) from all models
                # Model trained at offset k was trained up to t_enroll+k, but can predict
                # forward from there. We're comparing predictions at t_enroll+9 from all models.
                pred_time = target_timepoint
                
                if lambda_has_time:
                    # Lambda is [n_patients, n_signatures, n_timepoints]
                    if pred_time < lambda_offset.shape[2]:
                        lambda_patient = lambda_offset[patient_idx, :, pred_time]
                    else:
                        # Model can't predict this far into the future
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
            
            # Compare different washout periods:
            # 1. Full data (offset 9) vs 4-year washout (offset 5): predict at t_enroll+9
            # 2. Full data (offset 9) vs 9-year washout (offset 0): predict at t_enroll+9
            # This answers: "Predict for age 50 using data 0-49 vs 0-45"
            
            # Comparison 1: 4-year washout (offset 5 vs offset 9)
            # Predict at t_enroll+9: Model trained up to t_enroll+5 vs t_enroll+9
            if (lambda_values_by_offset[5] is not None and lambda_values_by_offset[9] is not None):
                lambda_change_4yr_washout = lambda_values_by_offset[9] - lambda_values_by_offset[5]
                
                for sig_idx in range(n_signatures):
                    key = f'Signature_{sig_idx}_4yr_washout'
                    signature_changes_by_precursor[precursor_name][key].append({
                        'change': lambda_change_4yr_washout[sig_idx],
                        'had_outcome_event': had_outcome_event_4yr,  # Real condition (ASCVD event)
                        'had_precursor': had_precursor_during_4yr  # Pre-clinical signal (precursor present during washout, whether before or developed during)
                    })
            
            # Comparison 2: 9-year washout (offset 0 vs offset 9)
            # Predict at t_enroll+9: Model trained up to t_enroll vs t_enroll+9
            if lambda_values_by_offset[0] is not None and lambda_values_by_offset[9] is not None:
                lambda_change_9yr_washout = lambda_values_by_offset[9] - lambda_values_by_offset[0]
                
                for sig_idx in range(n_signatures):
                    key = f'Signature_{sig_idx}_9yr_washout'
                    signature_changes_by_precursor[precursor_name][key].append({
                        'change': lambda_change_9yr_washout[sig_idx],
                        'had_outcome_event': had_outcome_event_9yr,  # Real condition (ASCVD event)
                        'had_precursor': had_precursor_during_9yr  # Pre-clinical signal (precursor present during washout, whether before or developed during)
                    })
    
    print(f"  Found {patients_with_precursors_count} patient-precursor combinations")
    print(f"  Patients with each precursor:")
    for name, count in precursor_check_debug.items():
        print(f"    {name}: {count} patients")
    
    # Summarize lambda changes by signature and washout period
    signature_summary = []
    for precursor_name, sig_changes in signature_changes_by_precursor.items():
        for sig_name, changes in sig_changes.items():
            if len(changes) > 0:
                # Check if changes are dictionaries (with event info) or just values
                if isinstance(changes[0], dict):
                    # Extract change values and event flags
                    change_values = [c['change'] for c in changes]
                    had_outcome_events = [c.get('had_outcome_event', False) for c in changes]
                    had_precursors = [c.get('had_precursor', False) for c in changes]
                    
                    # Separate patients by condition type:
                    # 1. WITH outcome events = real condition (conservative washout)
                    # 2. WITHOUT outcome events BUT WITH precursor = pre-clinical signal (accurate washout)
                    # 3. WITHOUT outcome events AND WITHOUT precursor = other reasons
                    changes_with_outcome = [c['change'] for c in changes if c.get('had_outcome_event', False)]
                    changes_with_precursor_only = [c['change'] for c in changes if not c.get('had_outcome_event', False) and c.get('had_precursor', False)]
                    changes_without_either = [c['change'] for c in changes if not c.get('had_outcome_event', False) and not c.get('had_precursor', False)]
                    
                    signature_summary.append({
                        'Precursor': precursor_name,
                        'Signature': sig_name,
                        'Washout_period': '4yr' if '4yr_washout' in sig_name else '9yr' if '9yr_washout' in sig_name else 'unknown',
                        'N_patients': len(changes),
                        'N_with_outcome': sum(had_outcome_events),  # Real conditions (conservative washout)
                        'N_with_precursor_only': sum([not h_out and h_prec for h_out, h_prec in zip(had_outcome_events, had_precursors)]),  # Pre-clinical signals (accurate washout)
                        'N_without_either': len(changes) - sum(had_outcome_events) - sum([not h_out and h_prec for h_out, h_prec in zip(had_outcome_events, had_precursors)]),
                        'Mean_change': np.mean(change_values),
                        'Mean_change_with_outcome': np.mean(changes_with_outcome) if len(changes_with_outcome) > 0 else np.nan,  # Conservative washout
                        'Mean_change_with_precursor_only': np.mean(changes_with_precursor_only) if len(changes_with_precursor_only) > 0 else np.nan,  # Accurate washout
                        'Mean_change_without_either': np.mean(changes_without_either) if len(changes_without_either) > 0 else np.nan,
                        'Median_change': np.median(change_values),
                        'Std_change': np.std(change_values),
                        'Abs_mean_change': np.mean(np.abs(change_values))
                    })
                else:
                    # Old format (just values)
                    signature_summary.append({
                        'Precursor': precursor_name,
                        'Signature': sig_name,
                        'Washout_period': 'unknown',
                        'N_patients': len(changes),
                        'N_with_events': np.nan,
                        'N_without_events': np.nan,
                        'Mean_change': np.mean(changes),
                        'Mean_change_with_events': np.nan,
                        'Mean_change_without_events': np.nan,
                        'Median_change': np.median(changes),
                        'Std_change': np.std(changes),
                        'Abs_mean_change': np.mean(np.abs(changes))
                    })
    
    print(f"  Generated {len(signature_summary)} signature change records")
    
    # Print interpretation
    if len(signature_summary) > 0:
        print("\n" + "="*80)
        print("WASHOUT INTERPRETATION:")
        print("="*80)
        print("Comparing predictions at t_enroll+9 from models with different washout periods:")
        print("  - 4yr washout: Model trained up to t_enroll+5 vs t_enroll+9")
        print("  - 9yr washout: Model trained up to t_enroll vs t_enroll+9")
        print("\nAnalyzing ALL patients (not just those with precursors before enrollment)")
        print("If predictions change:")
        print("  - WITH outcome events (ASCVD) in washout: Conservative washout (real conditions)")
        print("    → Model learned from patients who already had the outcome")
        print("  - WITHOUT outcome events BUT WITH precursor (present before OR developed during washout):")
        print("    → Accurate washout (pre-clinical signals detected)")
        print("    → Precursor was present during washout period (whether before or developed during)")
        print("  - WITHOUT outcome events AND WITHOUT precursor:")
        print("    → Other reasons (model refinement, etc.)")
        print("="*80)
    
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
                       default=['Hypercholesterolemia', 'Essential hypertension', 'Type 2 diabetes',
                               'Atrial fibrillation and flutter', 'Obesity', 
                               'Chronic Kidney Disease, Stage III', 'Rheumatoid arthritis',
                               'Sleep apnea', 'Peripheral vascular disease, unspecified'],
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
    pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_offset_local_vectorized_E_corrected/')
    pi_filename_pattern = 'pi_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun_pooledall.pt'
    
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
        args.start_idx, args.end_idx,
        cluster_assignments=cluster_assignments
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
                # Use prediction_change_4yr (4yr washout - 9yr washout, both predict at t_enroll+9)
                if 'prediction_change_4yr' in precursor_patients.columns:
                    mean_change = precursor_patients['prediction_change_4yr'].mean()
                    median_change = precursor_patients['prediction_change_4yr'].median()
                else:
                    # Fallback: calculate from offset predictions if available
                    if 'prediction_offset_0' in precursor_patients.columns and 'prediction_offset_5' in precursor_patients.columns:
                        changes = precursor_patients['prediction_offset_5'] - precursor_patients['prediction_offset_0']
                        mean_change = changes.mean()
                        median_change = changes.median()
                    else:
                        print(f"  ⚠️  Cannot calculate prediction change - missing columns")
                        continue
                
                event_rate = precursor_patients['had_event'].mean() * 100 if 'had_event' in precursor_patients.columns else np.nan
                
                print(f"\n{precursor}:")
                print(f"  N patients: {len(precursor_patients)}")
                print(f"  Mean prediction change (4yr washout - 9yr washout): {mean_change:.4f}")
                print(f"  Median prediction change: {median_change:.4f}")
                if not np.isnan(event_rate):
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

