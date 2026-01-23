#!/usr/bin/env python
"""
Create leave-one-batch-out master checkpoints for corrected E validation

This script creates master checkpoints by pooling phi from all batches EXCEPT
one test batch from censor_e_batchrun_vectorized, allowing proper validation.

Usage:
    python create_leave_one_out_checkpoints_correctedE.py --exclude_batch 0
    python create_leave_one_out_checkpoints_correctedE.py --exclude_batches 0 1 2
"""

import torch
import numpy as np
import glob
import argparse
from pathlib import Path
import sys

# Import functions from create_master_checkpoints
sys.path.insert(0, str(Path(__file__).parent))
from create_master_checkpoints import extract_healthy_state_psi, create_master_checkpoint


def pool_phi_from_batches_excluding(pattern, exclude_batch_indices, total_batches=40):
    """
    Load and pool phi from batch files, excluding specified batch indices.
    
    Args:
        pattern: Pattern like '/path/to/enrollment_model_W0.0001_batch_*_*.pt'
        exclude_batch_indices: List of batch indices to exclude (0-based, e.g., [0] to exclude batch 0)
        total_batches: Total number of batches expected
    
    Returns:
        Pooled phi (mean across included batches) as numpy array
        List of included batch indices
    """
    all_phis = []
    included_indices = []
    
    # Find all matching files
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if len(files) != total_batches:
        print(f"Warning: Expected {total_batches} batches, found {len(files)}")
    
    # Process each batch
    for batch_idx in range(min(len(files), total_batches)):
        if batch_idx in exclude_batch_indices:
            print(f"  Excluding batch {batch_idx}")
            continue
        
        file_path = files[batch_idx]
        try:
            checkpoint = torch.load(file_path, weights_only=False)
            
            # Extract phi
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                print(f"Warning: No phi found in {Path(file_path).name}")
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            all_phis.append(phi)
            included_indices.append(batch_idx)
            print(f"  ✓ Included batch {batch_idx}: {Path(file_path).name}, shape: {phi.shape}")
            
        except Exception as e:
            print(f"Error loading batch {batch_idx} ({file_path}): {e}")
            continue
    
    if len(all_phis) == 0:
        raise ValueError(f"No phi arrays loaded (all batches excluded?)")
    
    # Stack and compute mean
    phi_stack = np.stack(all_phis, axis=0)  # (n_batches, K, D, T)
    phi_pooled = np.mean(phi_stack, axis=0)  # (K, D, T)
    
    print(f"\nPooled phi from {len(all_phis)} batches (excluded: {exclude_batch_indices})")
    print(f"Included batch indices: {included_indices}")
    print(f"Pooled phi shape: {phi_pooled.shape}")
    print(f"Pooled phi stats: min={phi_pooled.min():.4f}, max={phi_pooled.max():.4f}, mean={phi_pooled.mean():.4f}")
    
    return phi_pooled, included_indices


def main():
    parser = argparse.ArgumentParser(description='Create leave-one-batch-out master checkpoints for corrected E')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing initial_psi file')
    parser.add_argument('--batch_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt',
                       help='Pattern for corrected E batch files (NOTE: censor_e_batchrun_vectorized uses enrollment_model_W0.0001, NOT enrollment_model_VECTORIZED_W0.0001)')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Output directory for master checkpoints')
    parser.add_argument('--exclude_batch', type=int, nargs='+', default=[0],
                       help='Batch index(ices) to exclude from pooling (0-based, e.g., 0 to exclude batch 0)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Leave-One-Batch-Out Master Checkpoints (Corrected E)")
    print("="*80)
    print(f"Excluding batches: {args.exclude_batch}")
    print(f"Total batches: {args.total_batches}")
    print(f"Batch pattern: {args.batch_pattern}")
    
    # Load initial_psi
    print("\n1. Loading initial_psi...")
    initial_psi_path = Path(args.data_dir) / 'initial_psi_400k.pt'
    if not initial_psi_path.exists():
        raise FileNotFoundError(f"initial_psi file not found: {initial_psi_path}")
    
    initial_psi = torch.load(str(initial_psi_path), weights_only=False)
    if torch.is_tensor(initial_psi):
        initial_psi = initial_psi.cpu().numpy()
    print(f"✓ Loaded initial_psi, shape: {initial_psi.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exclude_str = '_'.join(map(str, sorted(args.exclude_batch)))
    
    # Create master checkpoint
    print(f"\n2. Pooling phi from corrected E batches (excluding {args.exclude_batch})...")
    try:
        phi_pooled, included_indices = pool_phi_from_batches_excluding(
            args.batch_pattern, 
            args.exclude_batch,
            args.total_batches
        )
        
        # Extract healthy state psi
        print("\n  Attempting to extract healthy state psi...")
        healthy_psi_actual = extract_healthy_state_psi(args.batch_pattern)
        
        output_path = output_dir / f'master_for_fitting_pooled_correctedE_exclude_batch_{exclude_str}.pt'
        create_master_checkpoint(
            phi_pooled,
            initial_psi,
            str(output_path),
            description=f"Pooled phi from corrected E batches excluding {args.exclude_batch} (included: {included_indices})",
            healthy_psi_actual=healthy_psi_actual
        )
        print(f"\n✓ Created checkpoint: {output_path}")
        
        # Compare to overall pooled phi
        print(f"\n3. Comparing to overall pooled phi...")
        master_checkpoint_path = output_dir / 'master_for_fitting_pooled_correctedE.pt'
        if master_checkpoint_path.exists():
            try:
                master_checkpoint = torch.load(str(master_checkpoint_path), weights_only=False)
                
                # Extract phi from master checkpoint
                if 'model_state_dict' in master_checkpoint and 'phi' in master_checkpoint['model_state_dict']:
                    phi_master = master_checkpoint['model_state_dict']['phi']
                elif 'phi' in master_checkpoint:
                    phi_master = master_checkpoint['phi']
                else:
                    print("  ⚠️  Could not find phi in master checkpoint")
                    phi_master = None
                
                if phi_master is not None:
                    # Convert to numpy if tensor
                    if torch.is_tensor(phi_master):
                        phi_master = phi_master.detach().cpu().numpy()
                    
                    # Compare shapes
                    if phi_pooled.shape != phi_master.shape:
                        print(f"  ⚠️  Shape mismatch: leave-one-out {phi_pooled.shape} vs master {phi_master.shape}")
                    else:
                        # Calculate differences
                        diff = np.abs(phi_pooled - phi_master)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        median_diff = np.median(diff)
                        std_diff = np.std(diff)
                        
                        # Calculate relative differences
                        abs_phi_master = np.abs(phi_master)
                        rel_diff = diff / (abs_phi_master + 1e-10)  # Add small epsilon to avoid division by zero
                        max_rel_diff = np.max(rel_diff)
                        mean_rel_diff = np.mean(rel_diff)
                        
                        print(f"  Comparison to master_for_fitting_pooled_correctedE.pt:")
                        print(f"    Shape: {phi_pooled.shape}")
                        print(f"    Max absolute difference: {max_diff:.6f}")
                        print(f"    Mean absolute difference: {mean_diff:.6f}")
                        print(f"    Median absolute difference: {median_diff:.6f}")
                        print(f"    Std absolute difference: {std_diff:.6f}")
                        print(f"    Max relative difference: {max_rel_diff:.6f}")
                        print(f"    Mean relative difference: {mean_rel_diff:.6f}")
                        
                        # Check if differences are small
                        # For leave-one-out (excluding 1/40 = 2.5% of batches), we expect small differences
                        if mean_diff < 1e-4 and max_rel_diff < 0.01:  # Mean < 0.0001 and relative < 1%
                            print(f"  ✓ Excellent: Very close to master (mean diff < 1e-4, rel diff < 1%)")
                        elif mean_diff < 1e-3 and max_rel_diff < 0.02:  # Mean < 0.001 and relative < 2%
                            print(f"  ✓ Good: Close to master (mean diff < 1e-3, rel diff < 2%)")
                        elif mean_diff < 1e-2 and max_rel_diff < 0.05:  # Mean < 0.01 and relative < 5%
                            print(f"  ✓ Acceptable: Moderate difference (mean diff < 1e-2, rel diff < 5%)")
                        else:
                            print(f"  ⚠️  Larger difference (mean diff >= 1e-2 or rel diff >= 5%)")
                            
            except Exception as e:
                print(f"  ⚠️  Could not compare to master checkpoint: {e}")
        else:
            print(f"  ⚠️  Master checkpoint not found: {master_checkpoint_path}")
            print(f"     (This is expected if you haven't created the overall pooled checkpoint yet)")
    except Exception as e:
        print(f"✗ Error creating master checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Leave-one-out checkpoint creation complete!")
    print("="*80)
    print(f"\nCreated checkpoint excluding batches: {args.exclude_batch}")
    print(f"\nNext steps:")
    print(f"1. Run predictions on excluded batch(es) using this checkpoint")
    print(f"2. Calculate 10-year AUC for each excluded batch")
    print(f"3. Compare to overall pooled AUC")


if __name__ == '__main__':
    main()

