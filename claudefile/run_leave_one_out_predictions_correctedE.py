#!/usr/bin/env python
"""
Run predictions for leave-one-out validation on corrected E batches

This script:
1. For each batch (0-39), runs predictions using the leave-one-out checkpoint
2. Saves predictions (pi) for each excluded batch

Usage:
    python run_leave_one_out_predictions_correctedE.py --batch 0
    python run_leave_one_out_predictions_correctedE.py --all_batches
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description, log_file=None):
    """Run a command and return success status"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    if log_file:
        print(f"Logging to: {log_file}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Write to log file if specified
        if log_file:
            with open(log_file, 'w') as f:
                f.write(f"{description}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*80}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            if log_file:
                print(f"Check log file for details: {log_file}")
            return False
        
        # Print a summary
        if log_file:
            print(f"✓ Output logged to: {log_file}")
            # Print last few lines to show progress
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    print("Last few lines of output:")
                    for line in lines[-5:]:
                        print(f"  {line}")
        else:
            print(result.stdout)
        return True
    except Exception as e:
        print(f"ERROR running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run leave-one-out predictions for corrected E batches')
    parser.add_argument('--batch', type=int, default=None,
                       help='Single batch index to process (0-39)')
    parser.add_argument('--all_batches', action='store_true',
                       help='Process all batches (0-39)')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Data directory')
    parser.add_argument('--output_base_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_correctedE/',
                       help='Base output directory for predictions')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for log files (default: output_base_dir/logs)')
    args = parser.parse_args()
    
    # Set up log directory
    if args.log_dir is None:
        args.log_dir = f"{args.output_base_dir}logs/"
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Leave-One-Out Predictions for Corrected E Batches")
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
    
    # Process each batch
    for batch_idx in batches_to_process:
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        output_dir = f"{args.output_base_dir}batch_{batch_idx}/"
        
        checkpoint_path = f"{args.data_dir}master_for_fitting_pooled_correctedE_exclude_batch_{batch_idx}.pt"
        
        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping batch {batch_idx}. Run create_leave_one_out_checkpoints_correctedE.py first.")
            continue
        
        cmd = [
            sys.executable,
            'claudefile/run_aladyn_predict_with_master_vector_cenosrE.py',
            '--trained_model_path', checkpoint_path,
            '--data_dir', args.data_dir,
            '--output_dir', output_dir,
            '--start_batch', str(batch_idx),
            '--max_batches', '1',
            '--covariates_path', '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
        ]
        
        log_file = f"{args.log_dir}batch_{batch_idx}.log"
        success = run_command(
            cmd, 
            f"Running predictions on batch {batch_idx} (samples {start_idx}-{end_idx})",
            log_file=log_file
        )
        if not success:
            print(f"Failed to run predictions for batch {batch_idx}")
            continue
    
    print("\n" + "="*80)
    print("Leave-one-out predictions complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Calculate 10-year AUC for each batch using calculate_leave_one_out_auc_correctedE.py")
    print("2. Compare to overall pooled AUC")


if __name__ == '__main__':
    main()

