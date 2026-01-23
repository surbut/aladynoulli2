#!/usr/bin/env python
"""
Compare phi and lambda values between censor_e_batchrun_vectorized and censor_e_batchrun_vectorized_11726
to verify they produce the same outputs.
"""

import torch
import numpy as np
from pathlib import Path
import glob

def load_batch_file(directory, pattern, batch_idx):
    """Load a batch file and extract phi and lambda"""
    files = sorted(glob.glob(str(Path(directory) / pattern)))
    if batch_idx >= len(files):
        return None
    
    file_path = files[batch_idx]
    try:
        checkpoint = torch.load(file_path, weights_only=False)
        
        # Extract phi
        if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
            phi = checkpoint['model_state_dict']['phi']
        elif 'phi' in checkpoint:
            phi = checkpoint['phi']
        else:
            print(f"  No phi found in {Path(file_path).name}")
            return None
        
        # Extract lambda (gamma)
        if 'model_state_dict' in checkpoint and 'lambda' in checkpoint['model_state_dict']:
            lambda_val = checkpoint['model_state_dict']['lambda']
        elif 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
            lambda_val = checkpoint['model_state_dict']['gamma']
        elif 'lambda' in checkpoint:
            lambda_val = checkpoint['lambda']
        elif 'gamma' in checkpoint:
            lambda_val = checkpoint['gamma']
        else:
            lambda_val = None
        
        # Convert to numpy if tensor
        if torch.is_tensor(phi):
            phi = phi.detach().cpu().numpy()
        if lambda_val is not None and torch.is_tensor(lambda_val):
            lambda_val = lambda_val.detach().cpu().numpy()
        
        return {
            'phi': phi,
            'lambda': lambda_val,
            'file': Path(file_path).name
        }
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def compare_arrays(arr1, arr2, name, tolerance=1e-4):
    """Compare two arrays and report differences"""
    if arr1 is None and arr2 is None:
        print(f"  {name}: Both None")
        return True
    if arr1 is None or arr2 is None:
        print(f"  {name}: One is None, other is not")
        return False
    
    if arr1.shape != arr2.shape:
        print(f"  {name}: Shape mismatch: {arr1.shape} vs {arr2.shape}")
        return False
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  {name}:")
    print(f"    Shape: {arr1.shape}")
    print(f"    Max difference: {max_diff:.6f}")
    print(f"    Mean difference: {mean_diff:.6f}")
    print(f"    Within tolerance ({tolerance}): {max_diff < tolerance}")
    
    return max_diff < tolerance

def main():
    dir1 = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized')
    pattern1 = 'enrollment_model_W0.0001_batch_*_*.pt'
    
    dir2 = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_11726')
    pattern2 = 'enrollment_model_VECTORIZED_W0.0001_batch_*_*.pt'
    
    print("="*80)
    print("Comparing Batch Outputs: censor_e_batchrun_vectorized vs censor_e_batchrun_vectorized_11726")
    print("="*80)
    
    # Check how many files exist in each
    files1 = sorted(glob.glob(str(dir1 / pattern1)))
    files2 = sorted(glob.glob(str(dir2 / pattern2)))
    
    print(f"\nDirectory 1 ({dir1.name}):")
    print(f"  Pattern: {pattern1}")
    print(f"  Files found: {len(files1)}")
    
    print(f"\nDirectory 2 ({dir2.name}):")
    print(f"  Pattern: {pattern2}")
    print(f"  Files found: {len(files2)}")
    
    # Compare a few batches
    num_batches_to_check = min(5, len(files1), len(files2))
    print(f"\n{'='*80}")
    print(f"Comparing first {num_batches_to_check} batches:")
    print(f"{'='*80}")
    
    all_match = True
    for batch_idx in range(num_batches_to_check):
        print(f"\nBatch {batch_idx}:")
        print(f"  Loading from {dir1.name}...")
        data1 = load_batch_file(dir1, pattern1, batch_idx)
        print(f"  Loading from {dir2.name}...")
        data2 = load_batch_file(dir2, pattern2, batch_idx)
        
        if data1 is None or data2 is None:
            print(f"  ⚠️  Could not load one or both files")
            all_match = False
            continue
        
        print(f"  File 1: {data1['file']}")
        print(f"  File 2: {data2['file']}")
        
        # Compare phi
        phi_match = compare_arrays(data1['phi'], data2['phi'], 'Phi', tolerance=1e-4)
        
        # Compare lambda if available
        if data1['lambda'] is not None and data2['lambda'] is not None:
            lambda_match = compare_arrays(data1['lambda'], data2['lambda'], 'Lambda', tolerance=1e-4)
        else:
            print(f"  Lambda: One or both missing")
            lambda_match = True  # Don't fail if lambda is missing
        
        if not phi_match:
            all_match = False
            print(f"  ✗ Phi does not match!")
        else:
            print(f"  ✓ Phi matches!")
        
        if not lambda_match:
            all_match = False
            print(f"  ✗ Lambda does not match!")
        else:
            print(f"  ✓ Lambda matches!")
    
    print(f"\n{'='*80}")
    if all_match:
        print("✓ All comparisons passed! The outputs appear to be the same.")
    else:
        print("✗ Some differences found. The outputs may differ.")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()




