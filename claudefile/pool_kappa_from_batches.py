#!/usr/bin/env python
"""
Pool kappa values from training batches to get population-level kappa.

This script:
- Loads kappa from all training batch files
- Computes mean kappa (pooled across batches)
- Shows statistics to verify consistency

Usage:
    python pool_kappa_from_batches.py
"""

import torch
import numpy as np
import glob
from pathlib import Path

def pool_kappa_from_batches(pattern, max_batches=None):
    """
    Load and pool kappa from all batch files matching the pattern.
    
    Args:
        pattern: Pattern like '/path/to/enrollment_model_W0.0001_batch_*_*.pt'
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled kappa (mean across batches) as float
    """
    all_kappas = []
    
    # Find all matching files
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, weights_only=False)
            
            # Extract kappa
            kappa = None
            if 'model_state_dict' in checkpoint and 'kappa' in checkpoint['model_state_dict']:
                kappa = checkpoint['model_state_dict']['kappa']
            elif 'kappa' in checkpoint:
                kappa = checkpoint['kappa']
            else:
                print(f"Warning: No kappa found in {Path(file_path).name}")
                continue
            
            # Convert to float if tensor
            if torch.is_tensor(kappa):
                kappa = kappa.item()
            elif isinstance(kappa, np.ndarray):
                kappa = float(kappa)
            else:
                kappa = float(kappa)
            
            all_kappas.append(kappa)
            print(f"  {Path(file_path).name}: kappa = {kappa:.6f}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(all_kappas) == 0:
        raise ValueError(f"No kappa values loaded from pattern: {pattern}")
    
    # Compute mean
    kappa_array = np.array(all_kappas)
    kappa_pooled = np.mean(kappa_array)
    
    print(f"\n{'='*80}")
    print(f"KAPPA POOLING RESULTS")
    print(f"{'='*80}")
    print(f"  Number of batches: {len(all_kappas)}")
    print(f"  Pooled kappa (mean): {kappa_pooled:.6f}")
    print(f"  Std kappa: {kappa_array.std():.6f}")
    print(f"  Min kappa: {kappa_array.min():.6f}")
    print(f"  Max kappa: {kappa_array.max():.6f}")
    print(f"  Range: {kappa_array.max() - kappa_array.min():.6f}")
    
    if kappa_array.std() > 0.01:
        print(f"  ⚠️  WARNING: Kappa varies significantly across batches (std = {kappa_array.std():.6f})")
    else:
        print(f"  ✓ Kappa is relatively consistent across batches")
    
    return kappa_pooled, kappa_array

def main():
    # Training batches pattern
    train_pattern = "/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*.pt"
    
    print("="*80)
    print("POOLING KAPPA FROM TRAINING BATCHES")
    print("="*80)
    
    try:
        kappa_pooled, kappa_array = pool_kappa_from_batches(train_pattern, max_batches=None)
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")
        print(f"  Use pooled kappa = {kappa_pooled:.6f} in master checkpoint")
        print(f"  This should be fixed during prediction (not learned per batch)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

