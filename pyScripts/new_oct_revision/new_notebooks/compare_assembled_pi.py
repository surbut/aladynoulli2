#!/usr/bin/env python3
"""
Compare assembled PI batches from AWS directory with pi_full_400k.pt

Assembles all pi_enroll_fixedphi_sex_*_*.pt files (40 batches of 10K each)
into a single 400K x 348 x 52 tensor and compares with pi_full_400k.pt
"""

import torch
import numpy as np
from pathlib import Path
import glob

def assemble_pi_batches(batch_dir):
    """
    Assemble all PI batch files into a single tensor.
    
    Parameters:
    -----------
    batch_dir : str or Path
        Directory containing pi_enroll_fixedphi_sex_*_*.pt files
    
    Returns:
    --------
    pi_assembled : torch.Tensor
        Assembled tensor [400000, 348, 52]
    """
    batch_dir = Path(batch_dir)
    
    # Find all PI batch files
    pattern = str(batch_dir / 'pi_enroll_fixedphi_sex_*_*.pt')
    batch_files = glob.glob(pattern)
    
    if len(batch_files) == 0:
        raise ValueError(f"No PI batch files found matching: {pattern}")
    
    # Sort by batch indices (extract start index from filename)
    def get_start_idx(filename):
        name = Path(filename).name
        # Extract start index from pi_enroll_fixedphi_sex_START_END.pt
        parts = name.split('_')
        for i, part in enumerate(parts):
            if part.isdigit() and i < len(parts) - 1:
                return int(part)
        return 0
    
    batch_files = sorted(batch_files, key=get_start_idx)
    
    print(f"Found {len(batch_files)} PI batch files")
    
    # Check for missing batches
    expected_batches = 40  # 0-10000, 10000-20000, ..., 390000-400000
    batch_size = 10000
    missing_batches = []
    for i in range(expected_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        expected_file = batch_dir / f'pi_enroll_fixedphi_sex_{start}_{end}.pt'
        if not expected_file.exists():
            missing_batches.append((start, end))
    
    if missing_batches:
        print(f"\n⚠️  WARNING: {len(missing_batches)} batches are missing:")
        for start, end in missing_batches:
            print(f"  - {start}_{end}")
        print(f"Will proceed with {len(batch_files)} available batches")
    
    # Load and stack batches
    batches = []
    total_patients = 0
    
    for i, batch_file in enumerate(batch_files):
        print(f"Loading batch {i+1}/{len(batch_files)}: {Path(batch_file).name}")
        pi_batch = torch.load(batch_file, map_location='cpu', weights_only=False)
        
        # Handle different possible formats
        if isinstance(pi_batch, dict):
            if 'pi' in pi_batch:
                pi_batch = pi_batch['pi']
            elif 'pi_tensor' in pi_batch:
                pi_batch = pi_batch['pi_tensor']
            else:
                # Try to find tensor-like values
                for key in pi_batch.keys():
                    val = pi_batch[key]
                    if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                        pi_batch = val
                        print(f"  Using key '{key}' from dict")
                        break
        
        if not isinstance(pi_batch, torch.Tensor):
            raise ValueError(f"Batch file {batch_file} does not contain a torch.Tensor")
        
        print(f"  Shape: {pi_batch.shape}")
        batches.append(pi_batch)
        total_patients += pi_batch.shape[0]
    
    # Stack all batches
    print(f"\nStacking {len(batches)} batches...")
    pi_assembled = torch.cat(batches, dim=0)
    
    print(f"Assembled shape: {pi_assembled.shape}")
    print(f"Total patients: {total_patients}")
    
    return pi_assembled


def compare_pi_tensors(pi1, pi2, tolerance=1e-6):
    """
    Compare two PI tensors for equality.
    
    Parameters:
    -----------
    pi1 : torch.Tensor
        First tensor
    pi2 : torch.Tensor
        Second tensor
    tolerance : float
        Numerical tolerance for comparison
    
    Returns:
    --------
    is_equal : bool
        Whether tensors are equal within tolerance
    max_diff : float
        Maximum absolute difference
    mean_diff : float
        Mean absolute difference
    """
    print("\n" + "="*80)
    print("COMPARING PI TENSORS")
    print("="*80)
    
    print(f"\nTensor 1 shape: {pi1.shape}")
    print(f"Tensor 2 shape: {pi2.shape}")
    
    if pi1.shape != pi2.shape:
        print(f"❌ Shapes don't match!")
        return False, None, None
    
    # Convert to numpy for comparison
    pi1_np = pi1.detach().cpu().numpy() if isinstance(pi1, torch.Tensor) else pi1
    pi2_np = pi2.detach().cpu().numpy() if isinstance(pi2, torch.Tensor) else pi2
    
    # Calculate differences
    diff = np.abs(pi1_np - pi2_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    # Check equality
    is_equal = max_diff < tolerance
    
    if is_equal:
        print(f"\n✅ Tensors are EQUAL (within tolerance {tolerance:.2e})")
    else:
        print(f"\n❌ Tensors are DIFFERENT (max diff {max_diff:.2e} > tolerance {tolerance:.2e})")
        
        # Show where differences occur
        diff_mask = diff > tolerance
        n_different = np.sum(diff_mask)
        total_elements = diff.size
        pct_different = 100 * n_different / total_elements
        
        print(f"\nElements with difference > {tolerance:.2e}: {n_different:,} / {total_elements:,} ({pct_different:.4f}%)")
        
        if n_different > 0:
            # Find locations of largest differences
            flat_idx = np.argmax(diff)
            idx_3d = np.unravel_index(flat_idx, diff.shape)
            print(f"\nLargest difference at index {idx_3d}:")
            print(f"  Tensor 1 value: {pi1_np[idx_3d]:.6e}")
            print(f"  Tensor 2 value: {pi2_np[idx_3d]:.6e}")
            print(f"  Difference: {diff[idx_3d]:.6e}")
    
    return is_equal, max_diff, mean_diff


def main():
    # Paths
    batch_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/models_fromAWS_enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/retrospective_pooled'
    pi_full_path = '/Users/sarahurbut/Downloads/pi_full_400k.pt'
    
    print("="*80)
    print("COMPARING ASSEMBLED PI BATCHES WITH pi_full_400k.pt")
    print("="*80)
    
    # Assemble batches
    print("\n1. Assembling PI batches from AWS directory...")
    try:
        pi_assembled = assemble_pi_batches(batch_dir)
    except Exception as e:
        print(f"\n❌ Error assembling batches: {e}")
        return
    
    # Load reference
    print("\n2. Loading reference pi_full_400k.pt...")
    try:
        pi_full = torch.load(pi_full_path, map_location='cpu', weights_only=False)
        
        # Handle dict format
        if isinstance(pi_full, dict):
            if 'pi' in pi_full:
                pi_full = pi_full['pi']
            elif 'pi_tensor' in pi_full:
                pi_full = pi_full['pi_tensor']
            else:
                # Try to find tensor-like values
                for key in pi_full.keys():
                    val = pi_full[key]
                    if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                        pi_full = val
                        print(f"  Using key '{key}' from dict")
                        break
        
        print(f"  Shape: {pi_full.shape}")
    except Exception as e:
        print(f"\n❌ Error loading reference: {e}")
        return
    
    # Compare
    print("\n3. Comparing tensors...")
    is_equal, max_diff, mean_diff = compare_pi_tensors(pi_assembled, pi_full)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if is_equal:
        print("✅ ASSEMBLED PI BATCHES MATCH pi_full_400k.pt")
        print("   The assembled tensor from AWS batches is identical to the reference.")
    else:
        print("❌ ASSEMBLED PI BATCHES DO NOT MATCH pi_full_400k.pt")
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")
        print("\n   Possible reasons:")
        print("   - Different model checkpoints used")
        print("   - Different random seeds")
        print("   - Different preprocessing")
        print("   - Files from different runs")


if __name__ == '__main__':
    main()

