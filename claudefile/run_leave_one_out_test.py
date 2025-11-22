#!/usr/bin/env python
"""
Quick leave-one-out validation test for a few batches

This script:
1. Creates leave-one-out checkpoints excluding test batches
2. Runs predictions on excluded batches
3. Compares performance to original pooled results

Usage:
    python run_leave_one_out_test.py --test_batches 39 38 37
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(description='Run leave-one-out validation for test batches')
    parser.add_argument('--test_batches', type=int, nargs='+', default=[39, 38, 37],
                       help='Batch indices to test (will exclude from pooling)')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Data directory')
    parser.add_argument('--skip_predictions', action='store_true',
                       help='Skip running predictions (just create checkpoints)')
    args = parser.parse_args()
    
    print("="*80)
    print("Leave-One-Out Validation Test")
    print("="*80)
    print(f"Test batches: {args.test_batches}")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Step 1: Create leave-one-out checkpoints
    print("\n" + "="*80)
    print("STEP 1: Creating leave-one-out checkpoints")
    print("="*80)
    
    for exclude_batch in args.test_batches:
        cmd = [
            sys.executable,
            'claudefile/create_leave_one_out_checkpoints.py',
            '--exclude_batch', str(exclude_batch),
            '--analysis_type', 'retrospective',
            '--data_dir', args.data_dir,
            '--retrospective_pattern', '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_*_*.pt',
            '--output_dir', args.data_dir,
            '--total_batches', '40'
        ]
        
        success = run_command(cmd, f"Creating checkpoint excluding batch {exclude_batch}")
        if not success:
            print(f"Failed to create checkpoint for batch {exclude_batch}")
            continue
    
    if args.skip_predictions:
        print("\n" + "="*80)
        print("Checkpoints created. Skipping predictions (--skip_predictions flag set).")
        print("="*80)
        return
    
    # Step 2: Run predictions on excluded batches
    print("\n" + "="*80)
    print("STEP 2: Running predictions on excluded batches")
    print("="*80)
    
    for exclude_batch in args.test_batches:
        start_idx = exclude_batch * 10000
        end_idx = (exclude_batch + 1) * 10000
        output_dir = f"{args.data_dir}leave_one_out_validation/batch_{exclude_batch}/"
        
        checkpoint_path = f"{args.data_dir}master_for_fitting_pooled_all_data_exclude_batch_{exclude_batch}.pt"
        
        cmd = [
            sys.executable,
            'claudefile/run_aladyn_predict_with_master.py',
            '--trained_model_path', checkpoint_path,
            '--data_dir', args.data_dir,
            '--output_dir', output_dir,
            '--start_batch', str(exclude_batch),
            '--max_batches', '1',
            '--covariates_path', '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
        ]
        
        success = run_command(cmd, f"Running predictions on batch {exclude_batch} (samples {start_idx}-{end_idx})")
        if not success:
            print(f"Failed to run predictions for batch {exclude_batch}")
            continue
    
    print("\n" + "="*80)
    print("Leave-one-out validation complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Compare AUCs from leave-one-out predictions vs. original pooled predictions")
    print("2. If AUCs are similar, pooling is robust (no overfitting)")
    print("3. If AUCs drop significantly, there may be some overfitting")


if __name__ == '__main__':
    main()

