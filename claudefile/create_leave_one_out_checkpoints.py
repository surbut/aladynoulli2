#!/usr/bin/env python
"""
Create leave-one-batch-out master checkpoints for validation

This script creates master checkpoints by pooling phi from all batches EXCEPT
one test batch, allowing proper validation of the pooling approach.

Usage:
    python create_leave_one_out_checkpoints.py --exclude_batch 39
    python create_leave_one_out_checkpoints.py --exclude_batches 39 38 37
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
        exclude_batch_indices: List of batch indices to exclude (0-based, e.g., [39] to exclude batch 39)
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
    parser = argparse.ArgumentParser(description='Create leave-one-batch-out master checkpoints')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing initial_psi file')
    parser.add_argument('--retrospective_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_*_*.pt',
                       help='Pattern for retrospective batch files')
    parser.add_argument('--enrollment_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/enrollment_model_W0.0001_batch_*_*.pt',
                       help='Pattern for enrollment batch files')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Output directory for master checkpoints')
    parser.add_argument('--exclude_batch', type=int, nargs='+', default=[39],
                       help='Batch index(ices) to exclude from pooling (0-based, e.g., 39 to exclude batch 39)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    parser.add_argument('--analysis_type', type=str, choices=['retrospective', 'enrollment', 'both'], 
                       default='retrospective',
                       help='Which analysis type to create checkpoints for')
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Leave-One-Batch-Out Master Checkpoints")
    print("="*80)
    print(f"Excluding batches: {args.exclude_batch}")
    print(f"Total batches: {args.total_batches}")
    print(f"Analysis type: {args.analysis_type}")
    
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
    
    # Create master checkpoint for retrospective (if requested)
    if args.analysis_type in ['retrospective', 'both']:
        print(f"\n2. Pooling phi from retrospective batches (excluding {args.exclude_batch})...")
        try:
            phi_retrospective, included_indices = pool_phi_from_batches_excluding(
                args.retrospective_pattern, 
                args.exclude_batch,
                args.total_batches
            )
            
            # Extract healthy state psi
            print("\n  Attempting to extract healthy state psi...")
            healthy_psi_actual = extract_healthy_state_psi(args.retrospective_pattern)
            if healthy_psi_actual is None:
                healthy_psi_actual = extract_healthy_state_psi(args.enrollment_pattern)
            
            output_path_retro = output_dir / f'master_for_fitting_pooled_all_data_exclude_batch_{exclude_str}.pt'
            create_master_checkpoint(
                phi_retrospective,
                initial_psi,
                str(output_path_retro),
                description=f"Pooled phi from retrospective batches excluding {args.exclude_batch} (included: {included_indices})",
                healthy_psi_actual=healthy_psi_actual
            )
            print(f"\n✓ Created retrospective checkpoint: {output_path_retro}")
        except Exception as e:
            print(f"✗ Error creating retrospective master checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    # Create master checkpoint for enrollment (if requested)
    if args.analysis_type in ['enrollment', 'both']:
        print(f"\n3. Pooling phi from enrollment batches (excluding {args.exclude_batch})...")
        try:
            phi_enrollment, included_indices = pool_phi_from_batches_excluding(
                args.enrollment_pattern,
                args.exclude_batch,
                args.total_batches
            )
            
            # Extract healthy state psi
            print("\n  Extracting healthy state psi from enrollment batches...")
            healthy_psi_actual = extract_healthy_state_psi(args.enrollment_pattern)
            
            output_path_enroll = output_dir / f'master_for_fitting_pooled_enrollment_data_exclude_batch_{exclude_str}.pt'
            create_master_checkpoint(
                phi_enrollment,
                initial_psi,
                str(output_path_enroll),
                description=f"Pooled phi from enrollment batches excluding {args.exclude_batch} (included: {included_indices})",
                healthy_psi_actual=healthy_psi_actual
            )
            print(f"\n✓ Created enrollment checkpoint: {output_path_enroll}")
        except Exception as e:
            print(f"✗ Error creating enrollment master checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Leave-one-out checkpoint creation complete!")
    print("="*80)
    print(f"\nCreated checkpoints excluding batches: {args.exclude_batch}")
    print(f"\nNext steps:")
    print(f"1. Run predictions on excluded batch(es) using these checkpoints")
    print(f"2. Compare performance to the original pooled results")
    print(f"3. This validates that pooling doesn't overfit to the test batches")


if __name__ == '__main__':
    main()

