#!/usr/bin/env python3
"""
Calculate 30-year risk for patients filtered by max_censor > 70.

This script:
1. Loads filtered patient indices (patients with max_censor > 70)
2. Pulls predictions from pi_fullmode_400k.pt starting at timepoint 0 (age 30)
3. Calculates 30-year risk (from age 30 to age 60, timepoint 0 to 30)
4. Evaluates AUC for 30-year predictions
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse

# Add path to import evaluation functions (same pattern as generate_time_horizon_predictions.py)
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi_variable_followup,
    evaluate_major_diseases_wsex_with_bootstrap_from_pi
)

def load_filtered_indices(indices_path):
    """Load filtered patient indices"""
    print(f"Loading filtered patient indices from: {indices_path}")
    
    if indices_path.suffix == '.npy':
        indices = np.load(indices_path)
    elif indices_path.suffix == '.csv':
        df = pd.read_csv(indices_path)
        if 'original_index' in df.columns:
            indices = df['original_index'].values
        else:
            indices = df.index.values
    else:
        raise ValueError(f"Unknown file format: {indices_path.suffix}")
    
    print(f"  Loaded {len(indices)} patient indices")
    print(f"  Index range: {indices.min()} to {indices.max()}")
    
    return indices

def load_pi_predictions_from_age70_filtered(age70_dir, indices):
    """
    Load and concatenate pi predictions from age70_filtered directory.
    
    The pi files are saved as:
    pi_fixedphi_age_40_offset_0_filtered_censor70.0_batch_{start}_{end}.pt
    """
    print(f"\nLoading pi predictions from age70_filtered directory: {age70_dir}")
    
    age70_path = Path(age70_dir)
    if not age70_path.exists():
        raise FileNotFoundError(f"age70_filtered directory not found: {age70_dir}")
    
    # Find all pi files
    pi_files = sorted(age70_path.glob('pi_fixedphi_age_40_offset_0_filtered_censor70.0_batch_*.pt'))
    
    if len(pi_files) == 0:
        raise FileNotFoundError(f"No pi prediction files found in {age70_dir}")
    
    print(f"  Found {len(pi_files)} pi prediction files")
    
    # Load and concatenate all batches
    pi_batches = []
    for pi_file in pi_files:
        print(f"    Loading {pi_file.name}...")
        pi_batch = torch.load(pi_file, map_location='cpu', weights_only=False)
        pi_batches.append(pi_batch)
    
    # Concatenate all batches
    pi_concatenated = torch.cat(pi_batches, dim=0)
    print(f"  Concatenated pi shape: {pi_concatenated.shape}")
    
    # Note: The pi predictions from age70_filtered models start at timepoint 10 (age 40)
    # But we need predictions from timepoint 0 (age 30) for 30-year risk calculation
    # So we'll need to either:
    # 1. Use pi_fullmode_400k.pt for timepoints 0-10, then age70_filtered for 10-40
    # 2. Or just use age70_filtered and note that we're calculating from age 40
    
    return pi_concatenated

def calculate_30year_cumulative_risk(pi, start_timepoint=0, end_timepoint=30):
    """
    Calculate cumulative 30-year risk from start_timepoint to end_timepoint.
    
    For each disease, calculates: 1 - product(1 - pi[t]) for t in [start, end]
    This gives the probability of developing the disease within the time window.
    """
    print(f"\nCalculating 30-year cumulative risk (timepoints {start_timepoint} to {end_timepoint})...")
    
    N, D, T = pi.shape
    
    if end_timepoint > T:
        print(f"  Warning: end_timepoint {end_timepoint} > T {T}, using T-1")
        end_timepoint = T - 1
    
    # Extract predictions for the time window
    pi_window = pi[:, :, start_timepoint:end_timepoint+1]  # Shape: (N, D, 31)
    
    # Calculate cumulative risk: 1 - product(1 - pi[t])
    # For each timepoint, probability of NOT getting disease = (1 - pi[t])
    # Cumulative probability of NOT getting disease = product(1 - pi[t])
    # Cumulative probability of getting disease = 1 - product(1 - pi[t])
    
    # Convert to numpy for easier computation
    pi_window_np = pi_window.numpy()
    
    # Calculate product of (1 - pi) across timepoints
    one_minus_pi = 1 - pi_window_np  # Shape: (N, D, 31)
    
    # Product across time dimension (axis=2)
    cumulative_no_event = np.prod(one_minus_pi, axis=2)  # Shape: (N, D)
    
    # Cumulative risk = 1 - cumulative_no_event
    cumulative_risk = 1 - cumulative_no_event  # Shape: (N, D)
    
    print(f"  Cumulative risk shape: {cumulative_risk.shape}")
    print(f"  Risk range: {cumulative_risk.min():.6f} to {cumulative_risk.max():.6f}")
    
    return cumulative_risk

def load_and_subset_data_for_filtered_patients(indices, data_dir):
    """
    Load Y, E, and pce_df, then subset to filtered patient indices.
    """
    print(f"\nLoading Y, E, and patient data...")
    
    # Load Y and E
    Y_full = torch.load(Path(data_dir) / 'Y_tensor.pt', map_location='cpu', weights_only=False)
    E_full = torch.load(Path(data_dir) / 'E_matrix_corrected.pt', map_location='cpu', weights_only=False)
    
    print(f"  Full Y shape: {Y_full.shape}")
    print(f"  Full E shape: {E_full.shape}")
    
    # Subset to filtered patients
    print(f"  Subsetting to {len(indices)} filtered patients...")
    Y_filtered = Y_full[indices]
    E_filtered = E_full[indices]
    
    print(f"  Filtered Y shape: {Y_filtered.shape}")
    print(f"  Filtered E shape: {E_filtered.shape}")
    
    # Load pce_df (patient characteristics)
    csv_path = Path(data_dir) / 'baselinagefamh_withpcs.csv'
    if not csv_path.exists():
        csv_path = Path(data_dir).parent / 'baselinagefamh_withpcs.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find baselinagefamh_withpcs.csv")
    
    pce_df_full = pd.read_csv(csv_path)
    print(f"  Loaded pce_df with {len(pce_df_full)} patients")
    
    # Subset to filtered patients
    pce_df_filtered = pce_df_full.iloc[indices].reset_index(drop=True)
    print(f"  Filtered pce_df shape: {pce_df_filtered.shape}")
    
    return Y_filtered, E_filtered, pce_df_filtered

def main():
    parser = argparse.ArgumentParser(description='Calculate 30-year risk for age 70 filtered patients')
    parser.add_argument('--indices_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/age70_filtered/filtered_patient_indices.npy',
                       help='Path to filtered patient indices (.npy or .csv)')
    parser.add_argument('--age70_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/age70_filtered/output/age70_filtered',
                       help='Directory containing age70_filtered pi predictions')
    parser.add_argument('--pi_full_path', type=str, default=None,
                       help='Optional: Path to full pi predictions (pi_fullmode_400k.pt) for timepoints 0-10')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running',
                       help='Directory containing Y_tensor.pt, E_matrix_corrected.pt, baselinagefamh_withpcs.csv')
    parser.add_argument('--disease_names_path', type=str, default=None,
                       help='Path to disease names CSV (default: data_dir/disease_names.csv)')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for AUC CI (default: 100)')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory for results (default: same as indices_path directory)')
    parser.add_argument('--start_timepoint', type=int, default=0,
                       help='Start timepoint (default: 0, age 30)')
    parser.add_argument('--end_timepoint', type=int, default=30,
                       help='End timepoint (default: 30, age 60)')
    parser.add_argument('--calculate_auc', action='store_true',
                       help='Calculate AUC for 30-year predictions')
    
    args = parser.parse_args()
    
    # Load filtered indices
    indices_path = Path(args.indices_path)
    if not indices_path.exists():
        # Try alternative paths
        alt_paths = [
            Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/filtered_patient_indices.npy'),
            Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/filtered_patient_indices.npy'),
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                indices_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find filtered patient indices. Tried: {args.indices_path} and alternatives")
    
    filtered_indices = load_filtered_indices(indices_path)
    
    # Load pi predictions from age70_filtered directory
    pi_filtered = load_pi_predictions_from_age70_filtered(args.age70_dir, filtered_indices)
    
    # If pi_full_path is provided, we can combine it with age70_filtered predictions
    # to get predictions from timepoint 0 (age 30)
    if args.pi_full_path and Path(args.pi_full_path).exists():
        print(f"\nLoading full pi predictions for timepoints 0-10 from: {args.pi_full_path}")
        pi_full = torch.load(args.pi_full_path, map_location='cpu', weights_only=False)
        pi_full_filtered = pi_full[filtered_indices]
        
        # Combine: use full pi for timepoints 0-10, age70_filtered for timepoints 10+
        print(f"  Combining: full pi shape {pi_full_filtered.shape}, age70 pi shape {pi_filtered.shape}")
        if pi_full_filtered.shape[2] >= 11 and pi_filtered.shape[2] >= 11:
            # Use full pi for timepoints 0-10, age70 for timepoints 10+
            pi_combined = torch.cat([
                pi_full_filtered[:, :, :11],  # Timepoints 0-10
                pi_filtered[:, :, 11:]  # Timepoints 11+ from age70_filtered
            ], dim=2)
            pi_filtered = pi_combined
            print(f"  Combined pi shape: {pi_filtered.shape}")
        else:
            print(f"  Warning: Cannot combine, using age70_filtered predictions only")
    
    # Calculate 30-year cumulative risk
    cumulative_risk = calculate_30year_cumulative_risk(
        pi_filtered, 
        start_timepoint=args.start_timepoint,
        end_timepoint=args.end_timepoint
    )
    
    # Save cumulative risk
    if args.output_dir is None:
        output_dir = indices_path.parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    risk_output_path = output_dir / f'cumulative_30year_risk_age70_filtered.pt'
    torch.save(torch.tensor(cumulative_risk), risk_output_path)
    print(f"\n✓ Saved cumulative 30-year risk to: {risk_output_path}")
    
    # If requested, calculate AUC using the same evaluation function as generate_time_horizon_predictions.py
    if args.calculate_auc:
        print("\n" + "="*80)
        print("CALCULATING 30-YEAR AUC FROM AGE 40")
        print("="*80)
        
        # Load Y, E, and pce_df, subset to filtered patients
        data_dir = Path(args.data_dir) if args.data_dir else Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running')
        Y_filtered, E_filtered, pce_df_filtered = load_and_subset_data_for_filtered_patients(
            filtered_indices, data_dir
        )
        
        # Load disease names
        disease_names_path = Path(args.disease_names_path) if args.disease_names_path else data_dir / 'disease_names.csv'
        disease_names_df = pd.read_csv(disease_names_path)
        if 'disease_name' in disease_names_df.columns:
            disease_names = disease_names_df['disease_name'].tolist()
        else:
            disease_names = disease_names_df.iloc[:, 1].tolist()
        
        print(f"\nEvaluating 30-year predictions from age 40...")
        print(f"  Using pi predictions from age70_filtered models (age 40, timepoint 10)")
        print(f"  Follow-up duration: 30 years (age 40 to 70, timepoint 10 to 40)")
        
        # Evaluate using the same function as generate_time_horizon_predictions.py
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi_filtered,
            Y_100k=Y_filtered,
            E_100k=E_filtered,
            disease_names=disease_names,
            pce_df=pce_df_filtered,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=30  # 30-year predictions from age 40
        )
        
        # Save results
        results_df = pd.DataFrame({
            disease: {
                'AUC': metrics['auc'],
                'CI_lower': metrics['ci_lower'],
                'CI_upper': metrics['ci_upper'],
                'n_events': metrics['n_events'],
                'n_total': metrics['n_total'],
                'event_rate': metrics['event_rate']
            }
            for disease, metrics in results.items()
        }).T
        
        results_df = results_df.sort_values('AUC', ascending=False)
        
        auc_output_path = output_dir / f'30year_auc_age70_filtered_from_age40.csv'
        results_df.to_csv(auc_output_path)
        print(f"\n✓ Saved AUC results to: {auc_output_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("30-YEAR AUC SUMMARY (FROM AGE 40)")
        print(f"{'='*80}")
        print(f"Total diseases evaluated: {len(results)}")
        print(f"Mean AUC: {results_df['AUC'].mean():.4f}")
        print(f"Median AUC: {results_df['AUC'].median():.4f}")
        print(f"\nTop 10 diseases by AUC:")
        print(results_df.head(10).to_string())
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Filtered patients: {len(filtered_indices)}")
    print(f"Time window: timepoint {args.start_timepoint} to {args.end_timepoint} (age {30+args.start_timepoint} to {30+args.end_timepoint})")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()

