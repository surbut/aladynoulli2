#!/usr/bin/env python3
"""
Verify that pi_full_400k.pt matches concatenation of batch files.

Checks:
1. Load pi_full_400k.pt from ~/Downloads/
2. Load all batch files from ~/Downloads/pi_batches/
3. Concatenate batches in order (0-10000, 10000-20000, ..., 390000-400000)
4. Compare concatenated tensor with pi_full_400k.pt
"""

import torch
import numpy as np
from pathlib import Path
import re

def extract_batch_indices(filename):
    """Extract start and end indices from filename like pi_enroll_fixedphi_sex_0_10000.pt"""
    match = re.search(r'(\d+)_(\d+)\.pt$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_and_assemble_batches(batch_dir, max_patients=400000, batch_size=10000):
    """Load all batch files and assemble in order."""
    batch_dir = Path(batch_dir).expanduser()
    
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    
    # Find all batch files
    batch_files = list(batch_dir.glob('pi_enroll_fixedphi_sex_*.pt'))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {batch_dir}")
    
    print(f"Found {len(batch_files)} batch files")
    
    # Sort by start index
    batch_info = []
    for f in batch_files:
        start, end = extract_batch_indices(f.name)
        if start is not None and end is not None:
            batch_info.append((start, end, f))
    
    batch_info.sort(key=lambda x: x[0])  # Sort by start index
    
    # Verify we have all batches
    expected_batches = max_patients // batch_size
    if len(batch_info) != expected_batches:
        print(f"⚠️  Warning: Expected {expected_batches} batches, found {len(batch_info)}")
    
    # Load and concatenate batches
    print("\nLoading and concatenating batches...")
    pi_batches = []
    
    for i, (start, end, filepath) in enumerate(batch_info):
        print(f"  Batch {i+1}/{len(batch_info)}: {filepath.name} (indices {start}-{end})")
        
        batch_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Handle different formats
        if isinstance(batch_data, dict):
            if 'pi' in batch_data:
                pi_batch = batch_data['pi']
            elif 'pi_tensor' in batch_data:
                pi_batch = batch_data['pi_tensor']
            else:
                # Try to find tensor-like values
                for key in batch_data.keys():
                    val = batch_data[key]
                    if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                        pi_batch = val
                        print(f"    Using key '{key}' from dict")
                        break
                else:
                    raise ValueError(f"Could not find pi tensor in {filepath}")
        elif isinstance(batch_data, torch.Tensor):
            pi_batch = batch_data
        else:
            raise ValueError(f"Unexpected data type in {filepath}: {type(batch_data)}")
        
        print(f"    Shape: {pi_batch.shape}")
        pi_batches.append(pi_batch)
    
    # Concatenate all batches
    print("\nConcatenating batches...")
    pi_assembled = torch.cat(pi_batches, dim=0)
    print(f"Assembled shape: {pi_assembled.shape}")
    
    return pi_assembled, batch_info

def load_pi_full(pi_full_path):
    """Load pi_full_400k.pt and handle different formats."""
    pi_full_path = Path(pi_full_path).expanduser()
    
    if not pi_full_path.exists():
        raise FileNotFoundError(f"pi_full_400k.pt not found: {pi_full_path}")
    
    print(f"\nLoading pi_full_400k.pt: {pi_full_path}")
    pi_full_data = torch.load(pi_full_path, map_location='cpu', weights_only=False)
    
    # Handle different formats
    if isinstance(pi_full_data, dict):
        if 'pi' in pi_full_data:
            pi_full = pi_full_data['pi']
        elif 'pi_tensor' in pi_full_data:
            pi_full = pi_full_data['pi_tensor']
        else:
            # Try to find tensor-like values
            for key in pi_full_data.keys():
                val = pi_full_data[key]
                if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                    pi_full = val
                    print(f"  Using key '{key}' from dict")
                    break
            else:
                raise ValueError(f"Could not find pi tensor in {pi_full_path}")
    elif isinstance(pi_full_data, torch.Tensor):
        pi_full = pi_full_data
    else:
        raise ValueError(f"Unexpected data type in {pi_full_path}: {type(pi_full_data)}")
    
    print(f"  Shape: {pi_full.shape}")
    return pi_full

def compare_tensors(pi1, pi2, tolerance=1e-6):
    """Compare two pi tensors."""
    print("\n" + "="*80)
    print("COMPARING TENSORS")
    print("="*80)
    
    # Check shapes
    if pi1.shape != pi2.shape:
        print(f"❌ Shape mismatch!")
        print(f"  Assembled: {pi1.shape}")
        print(f"  pi_full_400k: {pi2.shape}")
        return False, None, None
    
    print(f"✅ Shapes match: {pi1.shape}")
    
    # Compare values
    pi1_np = pi1.cpu().numpy()
    pi2_np = pi2.cpu().numpy()
    
    max_diff = np.max(np.abs(pi1_np - pi2_np))
    mean_diff = np.mean(np.abs(pi1_np - pi2_np))
    
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Tolerance: {tolerance:.2e}")
    
    is_equal = max_diff < tolerance
    
    if is_equal:
        print(f"\n✅ TENSORS MATCH (within tolerance {tolerance:.2e})")
    else:
        print(f"\n❌ TENSORS DO NOT MATCH (max diff {max_diff:.2e} > tolerance {tolerance:.2e})")
    
    return is_equal, max_diff, mean_diff

def main():
    pi_full_path = Path('~/Downloads/pi_full_400k.pt')
    batch_dir = Path('~/Downloads/pi_batches/')
    
    print("="*80)
    print("VERIFYING pi_full_400k.pt ASSEMBLY")
    print("="*80)
    print(f"pi_full_400k.pt: {pi_full_path.expanduser()}")
    print(f"Batch directory: {batch_dir.expanduser()}")
    print("="*80)
    
    # Load pi_full_400k.pt
    try:
        pi_full = load_pi_full(pi_full_path)
    except Exception as e:
        print(f"\n❌ Error loading pi_full_400k.pt: {e}")
        return
    
    # Load and assemble batches
    try:
        pi_assembled, batch_info = load_and_assemble_batches(batch_dir)
    except Exception as e:
        print(f"\n❌ Error assembling batches: {e}")
        return
    
    # Compare
    is_equal, max_diff, mean_diff = compare_tensors(pi_assembled, pi_full)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Batches processed: {len(batch_info)}")
    print(f"Assembled shape: {pi_assembled.shape}")
    print(f"pi_full_400k shape: {pi_full.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if is_equal:
        print("\n✅ VERIFICATION PASSED: pi_full_400k.pt matches concatenated batches!")
    else:
        print("\n❌ VERIFICATION FAILED: pi_full_400k.pt does NOT match concatenated batches!")
        print("\nPossible reasons:")
        print("  1. pi_full_400k.pt was created from different batch files")
        print("  2. Batch files have been modified since assembly")
        print("  3. Different assembly order or missing batches")

if __name__ == '__main__':
    main()

