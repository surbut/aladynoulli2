#!/usr/bin/env python
"""
Calculate 10-year AUC on POOLED leave-one-out predictions (Option B)

This is an alternative approach: pool all leave-one-out predictions together
and calculate one AUC on the combined dataset.

Usage:
    python calculate_leave_one_out_auc_pooled_correctedE.py
"""

import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add path for fig5utils
sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts'))
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi


def load_essentials(data_dir):
    """Load model essentials"""
    essentials_path = Path(data_dir) / 'model_essentials.pt'
    if not essentials_path.exists():
        raise FileNotFoundError(f"model_essentials not found: {essentials_path}")
    essentials = torch.load(str(essentials_path), weights_only=False)
    return essentials


def main():
    parser = argparse.ArgumentParser(description='Calculate 10-year AUC on pooled leave-one-out predictions')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Data directory')
    parser.add_argument('--predictions_base_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_correctedE/',
                       help='Base directory containing predictions')
    parser.add_argument('--pce_df_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv',
                       help='Path to covariates CSV file')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    parser.add_argument('--output_csv', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_auc_pooled_correctedE.csv',
                       help='Output CSV file for results')
    args = parser.parse_args()
    
    print("="*80)
    print("Calculate 10-Year AUC on Pooled Leave-One-Out Predictions (Corrected E)")
    print("="*80)
    print("This pools all leave-one-out predictions and calculates one AUC")
    print("="*80)
    
    # Load essentials
    print("\nLoading essentials...")
    essentials = load_essentials(args.data_dir)
    disease_names = essentials['disease_names']
    print(f"✓ Loaded {len(disease_names)} diseases")
    
    # Load full data tensors
    print("Loading full data tensors...")
    Y_full = torch.load(Path(args.data_dir) / 'Y_tensor.pt', weights_only=False)
    E_full = torch.load(Path(args.data_dir) / 'E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv(args.pce_df_path)
    print(f"✓ Loaded Y: {Y_full.shape}, E: {E_full.shape}, pce_df: {len(pce_df_full)} rows")
    
    # Load and pool all leave-one-out predictions
    print("\nLoading and pooling all leave-one-out predictions...")
    pi_batches = []
    valid_batches = []
    
    for batch_idx in range(args.total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        
        predictions_dir = Path(args.predictions_base_dir) / f'batch_{batch_idx}'
        pi_file = predictions_dir / f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        if not pi_file.exists():
            print(f"⚠️  Skipping batch {batch_idx} (file not found: {pi_file})")
            continue
        
        pi_batch = torch.load(str(pi_file), weights_only=False)
        pi_batches.append(pi_batch)
        valid_batches.append(batch_idx)
        print(f"✓ Loaded batch {batch_idx}: shape {pi_batch.shape}")
    
    if not pi_batches:
        print("✗ No prediction files found!")
        return
    
    # Concatenate all predictions
    print(f"\nConcatenating {len(pi_batches)} batches...")
    pi_pooled = torch.cat(pi_batches, dim=0)
    print(f"✓ Pooled pi shape: {pi_pooled.shape}")
    
    # Get corresponding Y, E, and pce_df for valid batches
    all_start_indices = [b * args.batch_size for b in valid_batches]
    all_end_indices = [(b + 1) * args.batch_size for b in valid_batches]
    
    Y_pooled = torch.cat([Y_full[start:end] for start, end in zip(all_start_indices, all_end_indices)], dim=0)
    E_pooled = torch.cat([E_full[start:end] for start, end in zip(all_start_indices, all_end_indices)], dim=0)
    
    pce_df_pooled = pd.concat([
        pce_df_full.iloc[start:end].copy() 
        for start, end in zip(all_start_indices, all_end_indices)
    ], ignore_index=True)
    
    print(f"✓ Pooled Y: {Y_pooled.shape}, E: {E_pooled.shape}, pce_df: {len(pce_df_pooled)} rows")
    
    # Calculate 10-year AUC on pooled data
    # Using EXACT same parameters as performance notebook
    print(f"\nCalculating 10-year AUC on pooled predictions...")
    print(f"  Using same evaluation function and parameters as performance notebook")
    print(f"  Total samples: {pi_pooled.shape[0]}")
    
    try:
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi_pooled,
            Y_100k=Y_pooled,
            E_100k=E_pooled,
            disease_names=disease_names,
            pce_df=pce_df_pooled,
            n_bootstraps=100,  # Same as performance notebook
            follow_up_duration_years=10,  # 10-year predictions
            patient_indices=None  # Evaluate all patients
        )
        
        # Store results
        all_results = []
        for disease_group, metrics in results.items():
            all_results.append({
                'disease_group': disease_group,
                'auc': metrics.get('auc', np.nan),
                'ci_lower': metrics.get('ci', (np.nan, np.nan))[0] if isinstance(metrics.get('ci'), tuple) else np.nan,
                'ci_upper': metrics.get('ci', (np.nan, np.nan))[1] if isinstance(metrics.get('ci'), tuple) else np.nan,
                'n_events': metrics.get('n_events', np.nan),
                'n_total': metrics.get('n_total', np.nan),
                'event_rate': metrics.get('event_rate', np.nan),
                'n_batches': len(valid_batches),
                'batches_included': ','.join(map(str, valid_batches))
            })
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('auc', ascending=False)
        results_df.to_csv(args.output_csv, index=False)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {args.output_csv}")
        print(f"Total disease groups: {len(results_df)}")
        print(f"Batches included: {len(valid_batches)}")
        print(f"{'='*80}")
        
        # Print summary
        print("\nTop 10 diseases by AUC:")
        for _, row in results_df.head(10).iterrows():
            print(f"  {row['disease_group']:25s} AUC: {row['auc']:.4f} ({row['ci_lower']:.4f}-{row['ci_upper']:.4f}) "
                  f"Events: {row['n_events']:.0f}/{row['n_total']:.0f} ({row['event_rate']*100:.2f}%)")
        
    except Exception as e:
        print(f"✗ Error calculating AUC: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

