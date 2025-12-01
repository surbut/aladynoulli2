#!/usr/bin/env python3
"""
Analyze MI (Myocardial Infarction) washout analysis with signature-based learning.

For each patient, tracks:
- Predictions at t9 from models m0, m5, m9 (all predicting at same timepoint)
- Signature 5 loadings at different timepoints
- MI status at t0, t5, t9 (cumulative)
- Precursor diseases at t0, t5, t9 (cumulative)

Categorizes washout based on what diseases developed when.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')

def load_essentials():
    """Load model essentials."""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def load_cluster_assignments():
    """Load cluster assignments for diseases."""
    path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt'
    
    if Path(path).exists():
        try:
            clusters = torch.load(path, weights_only=False)
            if isinstance(clusters, torch.Tensor):
                return clusters.numpy()
            elif isinstance(clusters, np.ndarray):
                return clusters
            elif isinstance(clusters, dict):
                if 'clusters' in clusters:
                    result = clusters['clusters'].numpy() if isinstance(clusters['clusters'], torch.Tensor) else clusters['clusters']
                    return result
                elif 'initial_clusters' in clusters:
                    result = clusters['initial_clusters'].numpy() if isinstance(clusters['initial_clusters'], torch.Tensor) else clusters['initial_clusters']
                    return result
        except Exception as e:
            print(f"⚠️  Error loading clusters: {e}")
    
    return None

def get_signature_5_diseases(disease_names, cluster_assignments):
    """Get all diseases in Signature 5 (cardiovascular cluster)."""
    if cluster_assignments is None:
        return []
    
    # Find Signature 5 (assuming it's cluster 5, but we should verify)
    # Actually, we need to find which signature MI belongs to, then get all diseases in that signature
    mi_idx = None
    for i, name in enumerate(disease_names):
        if 'Myocardial infarction' in name or 'Myocardial' in name:
            mi_idx = i
            break
    
    if mi_idx is None or mi_idx >= len(cluster_assignments):
        return []
    
    mi_signature = cluster_assignments[mi_idx]
    print(f"\n✓ MI belongs to Signature {mi_signature}")
    
    # Get all diseases in the same signature
    sig5_diseases = []
    sig5_indices = []
    for i, sig in enumerate(cluster_assignments):
        if sig == mi_signature and i < len(disease_names):
            sig5_diseases.append(disease_names[i])
            sig5_indices.append(i)
    
    print(f"✓ Found {len(sig5_diseases)} diseases in Signature {mi_signature}")
    print(f"  Examples: {sig5_diseases[:5]}")
    
    return sig5_indices, mi_signature

def load_model_checkpoints(pi_base_dir, start_idx, end_idx, max_offset=9):
    """Load model checkpoints and extract lambda (patient-specific parameters)."""
    model_filename_pattern = 'model_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun.pt'
    
    lambda_by_offset = []
    
    for k in range(max_offset + 1):
        model_filename = model_filename_pattern.format(k=k, start=start_idx, end=end_idx)
        model_path = pi_base_dir / model_filename
        
        if not model_path.exists():
            lambda_by_offset.append(None)
            continue
        
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Try different possible keys for lambda
            lambda_val = None
            if isinstance(checkpoint, dict):
                if 'lambda_' in checkpoint:
                    lambda_val = checkpoint['lambda_']
                elif 'model_state_dict' in checkpoint:
                    if 'lambda_' in checkpoint['model_state_dict']:
                        lambda_val = checkpoint['model_state_dict']['lambda_']
                    elif 'lambda' in checkpoint['model_state_dict']:
                        lambda_val = checkpoint['model_state_dict']['lambda']
                
                if lambda_val is not None:
                    if isinstance(lambda_val, torch.Tensor):
                        lambda_by_offset.append(lambda_val)
                    else:
                        lambda_by_offset.append(None)
                else:
                    lambda_by_offset.append(None)
            else:
                lambda_by_offset.append(None)
        except Exception as e:
            print(f"⚠️  Error loading model {k}: {e}")
            lambda_by_offset.append(None)
    
    return lambda_by_offset

def analyze_mi_washout(pi_batches, lambda_by_offset, Y_batch, E_batch, disease_names, 
                       mi_idx, sig5_indices, sig5_signature, cluster_assignments,
                       start_idx=0, end_idx=10000):
    """
    Analyze MI washout with signature-based learning.
    
    For each patient, tracks:
    - m0t9, m5t9, m9t9: Predictions at t9 from models m0, m5, m9
    - m0sig5t5, m0sig5t9, m9sig5t9: Signature 5 loadings
    - MI status at t0, t5, t9
    - Precursor diseases at t0, t5, t9
    """
    
    n_patients = pi_batches[0].shape[0]
    patient_analysis = []
    
    print(f"\nAnalyzing {n_patients} patients...")
    print(f"MI index: {mi_idx}")
    print(f"Signature 5 has {len(sig5_indices)} diseases")
    
    for patient_idx in range(n_patients):
        # Get enrollment time (E already contains age - 30)
        t_enroll = int(E_batch[patient_idx, 0].item()) if E_batch[patient_idx, 0] > 0 else 0
        
        if t_enroll < 0 or t_enroll + 9 >= pi_batches[0].shape[2]:
            continue
        
        t0 = t_enroll
        t5 = t_enroll + 5
        t9 = t_enroll + 9
        
        # Check if timepoints are valid
        if t9 >= pi_batches[0].shape[2] or t9 >= Y_batch.shape[2]:
            continue
        
        # ===== PREDICTIONS AT T9 FROM DIFFERENT MODELS =====
        # All models predict at the same timepoint (t9)
        m0t9 = np.nan
        m5t9 = np.nan
        m9t9 = np.nan
        
        # Model m0 (trained to t0) predicts at t9
        if len(pi_batches) > 0 and t9 < pi_batches[0].shape[2]:
            pi_mi_m0 = pi_batches[0][patient_idx, mi_idx, t9].item()
            m0t9 = pi_mi_m0
        
        # Model m5 (trained to t5) predicts at t9
        if len(pi_batches) > 5 and t9 < pi_batches[5].shape[2]:
            pi_mi_m5 = pi_batches[5][patient_idx, mi_idx, t9].item()
            m5t9 = pi_mi_m5
        
        # Model m9 (trained to t9) predicts at t9
        if len(pi_batches) > 9 and t9 < pi_batches[9].shape[2]:
            pi_mi_m9 = pi_batches[9][patient_idx, mi_idx, t9].item()
            m9t9 = pi_mi_m9
        
        # ===== SIGNATURE 5 LOADINGS =====
        m0sig5t5 = np.nan
        m0sig5t9 = np.nan
        m9sig5t9 = np.nan
        
        if lambda_by_offset[0] is not None and lambda_by_offset[0].shape[1] > sig5_signature:
            # Lambda shape: [n_patients, n_signatures, n_timepoints]
            if t5 < lambda_by_offset[0].shape[2]:
                m0sig5t5 = lambda_by_offset[0][patient_idx, sig5_signature, t5].item()
            if t9 < lambda_by_offset[0].shape[2]:
                m0sig5t9 = lambda_by_offset[0][patient_idx, sig5_signature, t9].item()
        
        if lambda_by_offset[9] is not None and lambda_by_offset[9].shape[1] > sig5_signature:
            if t9 < lambda_by_offset[9].shape[2]:
                m9sig5t9 = lambda_by_offset[9][patient_idx, sig5_signature, t9].item()
        
        # ===== DISEASE STATUS BY TIME PERIOD =====
        # Track diseases in 3 time periods: baseline (before t0), t0-t5, t5-t9
        
        # Get precursor indices (Signature 5 diseases excluding MI)
        precursor_indices = [idx for idx in sig5_indices if idx != mi_idx]
        
        # Baseline (before t0 / enrollment)
        MI_at_baseline = False
        precursors_at_baseline = []
        
        if t0 > 0:
            MI_at_baseline = Y_batch[patient_idx, mi_idx, :t0].sum().item() > 0
            
            for prec_idx in precursor_indices:
                if Y_batch[patient_idx, prec_idx, :t0].sum().item() > 0:
                    prec_name = disease_names[prec_idx] if prec_idx < len(disease_names) else f"Disease_{prec_idx}"
                    precursors_at_baseline.append(prec_name)
        
        # Interval 1: t0 to t5
        MI_between_t0_t5 = False
        precursors_between_t0_t5 = []
        
        if t5 > t0 and t5 <= Y_batch.shape[2]:
            MI_between_t0_t5 = Y_batch[patient_idx, mi_idx, t0:t5].sum().item() > 0
            
            for prec_idx in precursor_indices:
                if Y_batch[patient_idx, prec_idx, t0:t5].sum().item() > 0:
                    prec_name = disease_names[prec_idx] if prec_idx < len(disease_names) else f"Disease_{prec_idx}"
                    precursors_between_t0_t5.append(prec_name)
        
        # Interval 2: t5 to t9
        MI_between_t5_t9 = False
        precursors_between_t5_t9 = []
        
        if t9 > t5 and t9 <= Y_batch.shape[2]:
            MI_between_t5_t9 = Y_batch[patient_idx, mi_idx, t5:t9].sum().item() > 0
            
            for prec_idx in precursor_indices:
                if Y_batch[patient_idx, prec_idx, t5:t9].sum().item() > 0:
                    prec_name = disease_names[prec_idx] if prec_idx < len(disease_names) else f"Disease_{prec_idx}"
                    precursors_between_t5_t9.append(prec_name)
        
        # Cumulative status (for reference)
        MI_at_t0 = MI_at_baseline
        MI_at_t5 = MI_at_baseline or MI_between_t0_t5
        MI_at_t9 = MI_at_baseline or MI_between_t0_t5 or MI_between_t5_t9
        
        precursors_at_t0 = precursors_at_baseline
        precursors_at_t5 = list(set(precursors_at_baseline + precursors_between_t0_t5))
        precursors_at_t9 = list(set(precursors_at_baseline + precursors_between_t0_t5 + precursors_between_t5_t9))
        
        # Categorize washout based on what developed in intervals (not baseline)
        washout_category = 'neither'
        if MI_between_t0_t5 or MI_between_t5_t9:
            washout_category = 'conservative'  # Got MI in interval (real outcome)
        elif len(precursors_between_t0_t5) > 0 or len(precursors_between_t5_t9) > 0:
            washout_category = 'accurate'  # Got Signature 5 precursor in interval (pre-clinical signal)
        
        # Calculate enroll_age
        enroll_age = t_enroll + 30
        
        patient_analysis.append({
            'patient_idx': start_idx + patient_idx,
            'enroll_age': enroll_age,
            't_enroll': t_enroll,
            # Predictions at t9 from different models (3 models)
            'm0t9': m0t9,
            'm5t9': m5t9,
            'm9t9': m9t9,
            # Signature 5 loadings
            'm0sig5t5': m0sig5t5,
            'm0sig5t9': m0sig5t9,
            'm9sig5t9': m9sig5t9,
            # MI status by time period (3 periods)
            'MI_at_baseline': MI_at_baseline,  # Before t0
            'MI_between_t0_t5': MI_between_t0_t5,  # Interval 1
            'MI_between_t5_t9': MI_between_t5_t9,  # Interval 2
            # MI cumulative (for reference)
            'MI_at_t0': MI_at_t0,
            'MI_at_t5': MI_at_t5,
            'MI_at_t9': MI_at_t9,
            # Precursor counts by time period (3 periods)
            'n_precursors_at_baseline': len(precursors_at_baseline),
            'n_precursors_between_t0_t5': len(precursors_between_t0_t5),
            'n_precursors_between_t5_t9': len(precursors_between_t5_t9),
            # Precursor counts cumulative (for reference)
            'n_precursors_at_t0': len(precursors_at_t0),
            'n_precursors_at_t5': len(precursors_at_t5),
            'n_precursors_at_t9': len(precursors_at_t9),
            # Precursor lists by time period (comma-separated)
            'precursors_at_baseline': ','.join(precursors_at_baseline) if precursors_at_baseline else '',
            'precursors_between_t0_t5': ','.join(precursors_between_t0_t5) if precursors_between_t0_t5 else '',
            'precursors_between_t5_t9': ','.join(precursors_between_t5_t9) if precursors_between_t5_t9 else '',
            # Precursor lists cumulative (for reference)
            'precursors_at_t0': ','.join(precursors_at_t0) if precursors_at_t0 else '',
            'precursors_at_t5': ','.join(precursors_at_t5) if precursors_at_t5 else '',
            'precursors_at_t9': ','.join(precursors_at_t9) if precursors_at_t9 else '',
            # Washout category (based on intervals, not baseline)
            'washout_category': washout_category
        })
    
    return pd.DataFrame(patient_analysis)

def main():
    parser = argparse.ArgumentParser(description='Analyze MI washout with signature-based learning')
    parser.add_argument('--start_idx', type=int, default=0, help='Start patient index')
    parser.add_argument('--end_idx', type=int, default=10000, help='End patient index')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MI WASHOUT ANALYSIS WITH SIGNATURE-BASED LEARNING")
    print("="*80)
    print(f"Batch: {args.start_idx}-{args.end_idx}")
    
    # Load essentials
    print("\nLoading essentials...")
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Find MI index
    mi_idx = None
    for i, name in enumerate(disease_names):
        if 'Myocardial infarction' in name:
            mi_idx = i
            print(f"✓ Found MI at index {mi_idx}: {name}")
            break
    
    if mi_idx is None:
        raise ValueError("MI not found in disease names!")
    
    # Load cluster assignments
    print("\nLoading cluster assignments...")
    cluster_assignments = load_cluster_assignments()
    if cluster_assignments is None:
        raise ValueError("Could not load cluster assignments!")
    
    print(f"✓ Loaded clusters: {len(cluster_assignments)} diseases")
    
    # Get Signature 5 diseases
    sig5_indices, sig5_signature = get_signature_5_diseases(disease_names, cluster_assignments)
    
    # Load data batch
    print(f"\nLoading data batch {args.start_idx}-{args.end_idx}...")
    Y_batch = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_batch = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    
    # Subset to batch
    Y_batch = Y_batch[args.start_idx:args.end_idx]
    E_batch = E_batch[args.start_idx:args.end_idx]
    
    # Load pi batches
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
        
        pi_batch = torch.load(pi_path, weights_only=False)
        pi_batches.append(pi_batch)
    
    print(f"✓ Loaded {len(pi_batches)} pi batches")
    
    # Load lambda (signature loadings)
    print("\nLoading model checkpoints to extract lambda...")
    lambda_by_offset = load_model_checkpoints(pi_base_dir, args.start_idx, args.end_idx, max_offset=9)
    
    # Analyze
    print("\n" + "="*80)
    print("ANALYZING MI WASHOUT")
    print("="*80)
    
    patient_df = analyze_mi_washout(
        pi_batches, lambda_by_offset, Y_batch, E_batch, disease_names,
        mi_idx, sig5_indices, sig5_signature, cluster_assignments,
        args.start_idx, args.end_idx
    )
    
    # Save results
    output_dir = Path('results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'mi_washout_analysis_batch_{args.start_idx}_{args.end_idx}.csv'
    patient_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total patients analyzed: {len(patient_df)}")
    print(f"\nWashout categories:")
    print(patient_df['washout_category'].value_counts())
    print(f"\nMI status at t9: {patient_df['MI_at_t9'].sum()} patients ({patient_df['MI_at_t9'].mean()*100:.1f}%)")
    print(f"\nPatients with Signature 5 precursors at t9: {patient_df['n_precursors_at_t9'].gt(0).sum()} ({patient_df['n_precursors_at_t9'].gt(0).mean()*100:.1f}%)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

