#!/usr/bin/env python
"""
Calculate 10-year AUC for leave-one-out validation batches

This script calculates 10-year AUC for each excluded batch using the
predictions from leave-one-out validation.

Usage:
    python calculate_leave_one_out_auc_correctedE.py --batch 0
    python calculate_leave_one_out_auc_correctedE.py --all_batches
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
    parser = argparse.ArgumentParser(description='Calculate 10-year AUC for leave-one-out batches')
    parser.add_argument('--batch', type=int, default=None,
                       help='Single batch index to process (0-39)')
    parser.add_argument('--all_batches', action='store_true',
                       help='Process all batches (0-39)')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Data directory')
    parser.add_argument('--predictions_base_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_correctedE/',
                       help='Base directory containing predictions')
    parser.add_argument('--pce_df_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv',
                       help='Path to covariates CSV file (should match what was used in predictions)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    parser.add_argument('--output_csv', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_auc_results_correctedE.csv',
                       help='Output CSV file for results')
    args = parser.parse_args()
    
    print("="*80)
    print("Calculate 10-Year AUC for Leave-One-Out Validation (Corrected E)")
    print("="*80)
    
    # Determine which batches to process
    if args.all_batches:
        batches_to_process = list(range(args.total_batches))
    elif args.batch is not None:
        batches_to_process = [args.batch]
    else:
        print("Error: Must specify either --batch or --all_batches")
        return
    
    print(f"Batches to process: {batches_to_process}")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Load essentials
    print("Loading essentials...")
    essentials = load_essentials(args.data_dir)
    disease_names = essentials['disease_names']
    print(f"✓ Loaded {len(disease_names)} diseases")
    
    # Load full data tensors
    print("Loading full data tensors...")
    Y_full = torch.load(Path(args.data_dir) / 'Y_tensor.pt', weights_only=False)
    E_full = torch.load(Path(args.data_dir) / 'E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv(args.pce_df_path)
    print(f"✓ Loaded Y: {Y_full.shape}, E: {E_full.shape}, pce_df: {len(pce_df_full)} rows")
    
    # Store results
    all_results = []
    
    # Process each batch
    for batch_idx in batches_to_process:
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_idx} (samples {start_idx}-{end_idx})")
        print(f"{'='*80}")
        
        # Load predictions for this batch
        predictions_dir = Path(args.predictions_base_dir) / f'batch_{batch_idx}'
        pi_file = predictions_dir / f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        if not pi_file.exists():
            print(f"⚠️  Predictions file not found: {pi_file}")
            print(f"   Skipping batch {batch_idx}. Run predictions first.")
            continue
        
        print(f"Loading predictions from: {pi_file}")
        pi_batch = torch.load(str(pi_file), weights_only=False)
        print(f"✓ Loaded pi shape: {pi_batch.shape}")
        
        # Extract batch data
        Y_batch = Y_full[start_idx:end_idx]
        E_batch = E_full[start_idx:end_idx]
        pce_df_batch = pce_df_full.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        
        # Calculate 10-year AUC
        # Using EXACT same parameters as performance notebook:
        # - evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi
        # - n_bootstraps=100
        # - follow_up_duration_years=10
        # - patient_indices=None
        print(f"\nCalculating 10-year AUC for batch {batch_idx}...")
        print(f"  Using same evaluation function and parameters as performance notebook")
        try:
            results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
                pi=pi_batch,
                Y_100k=Y_batch,
                E_100k=E_batch,
                disease_names=disease_names,
                pce_df=pce_df_batch,
                n_bootstraps=100,  # Same as performance notebook
                follow_up_duration_years=10,  # 10-year predictions
                patient_indices=None  # Evaluate all patients in batch
            )
            
            # Store results with batch info
            for disease_group, metrics in results.items():
                all_results.append({
                    'batch_idx': batch_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'disease_group': disease_group,
                    'auc': metrics.get('auc', np.nan),
                    'ci_lower': metrics.get('ci', (np.nan, np.nan))[0] if isinstance(metrics.get('ci'), tuple) else np.nan,
                    'ci_upper': metrics.get('ci', (np.nan, np.nan))[1] if isinstance(metrics.get('ci'), tuple) else np.nan,
                    'n_events': metrics.get('n_events', np.nan),
                    'n_total': metrics.get('n_total', np.nan),
                    'event_rate': metrics.get('event_rate', np.nan),
                })
            
            print(f"✓ Calculated AUC for {len(results)} disease groups")
            
        except Exception as e:
            print(f"✗ Error calculating AUC for batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {args.output_csv}")
        print(f"Total results: {len(all_results)} disease-group combinations")
        print(f"{'='*80}")
        
        # Print summary
        print("\nSummary by batch:")
        for batch_idx in sorted(results_df['batch_idx'].unique()):
            batch_results = results_df[results_df['batch_idx'] == batch_idx]
            print(f"\nBatch {batch_idx}:")
            print(f"  Diseases evaluated: {len(batch_results)}")
            print(f"  Mean AUC: {batch_results['auc'].mean():.4f}")
            print(f"  AUC range: [{batch_results['auc'].min():.4f}, {batch_results['auc'].max():.4f}]")
    else:
        print("\n⚠️  No results to save!")


if __name__ == '__main__':
    main()

