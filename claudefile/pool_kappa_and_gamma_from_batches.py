#!/usr/bin/env python
"""
Pool kappa and gamma values from training batches to get population-level parameters.

This script:
- Loads kappa and gamma from all training batch files in censor_e_batchrun_vectorized
- Computes mean kappa (pooled across batches)
- Computes mean gamma (pooled across batches)
- Saves the pooled values for use in fixed-kappa, fixed-gamma models

Usage:
    python pool_kappa_and_gamma_from_batches.py
"""

import torch
import numpy as np
import glob
from pathlib import Path
import argparse

def pool_kappa_and_gamma_from_batches(pattern, max_batches=None):
    """
    Load and pool both kappa and gamma from all batch files matching the pattern.
    
    Args:
        pattern: Pattern like '/path/to/enrollment_model_W0.0001_batch_*_*.pt'
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        kappa_pooled: Pooled kappa (mean across batches) as float
        gamma_pooled: Pooled gamma (mean across batches) as numpy array (P, K)
        kappa_array: Array of all kappa values
        n_batches: Number of batches processed
    """
    all_kappas = []
    all_gammas = []
    
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
            
            if kappa is not None:
                # Convert to float if tensor
                if torch.is_tensor(kappa):
                    kappa = kappa.item()
                elif isinstance(kappa, np.ndarray):
                    kappa = float(kappa)
                else:
                    kappa = float(kappa)
                all_kappas.append(kappa)
            
            # Extract gamma
            gamma = None
            if 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
            
            if gamma is not None:
                # Convert to numpy if tensor
                if torch.is_tensor(gamma):
                    gamma = gamma.detach().cpu().numpy()
                elif not isinstance(gamma, np.ndarray):
                    gamma = np.array(gamma)
                
                # Check if gamma is all zeros
                if not np.allclose(gamma, 0):
                    all_gammas.append(gamma)
                    if len(all_gammas) <= 3:  # Print first 3
                        print(f"  Loaded kappa={kappa:.6f}, gamma shape={gamma.shape} from {Path(file_path).name}")
                else:
                    print(f"  Warning: {Path(file_path).name} has gamma=0 (possibly untrained)")
            else:
                print(f"  Warning: No gamma found in {Path(file_path).name}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(all_kappas) == 0:
        raise ValueError(f"No kappa values loaded from pattern: {pattern}")
    
    if len(all_gammas) == 0:
        raise ValueError(f"No gamma arrays loaded from pattern: {pattern}")
    
    # Compute mean kappa
    kappa_array = np.array(all_kappas)
    kappa_pooled = np.mean(kappa_array)
    
    # Stack and compute mean gamma
    gamma_stack = np.stack(all_gammas, axis=0)  # (n_batches, P, K_total)
    gamma_pooled = np.mean(gamma_stack, axis=0)  # (P, K_total)
    
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
    
    print(f"\n{'='*80}")
    print(f"GAMMA POOLING RESULTS")
    print(f"{'='*80}")
    print(f"  Number of batches: {len(all_gammas)}")
    print(f"  Pooled gamma shape: {gamma_pooled.shape}")
    print(f"  Stats: min={gamma_pooled.min():.6f}, max={gamma_pooled.max():.6f}, mean={gamma_pooled.mean():.6f}")
    print(f"  Mean |γ|: {np.abs(gamma_pooled).mean():.6f}, Max |γ|: {np.abs(gamma_pooled).max():.6f}")
    
    return kappa_pooled, gamma_pooled, kappa_array, len(all_gammas)

def main():
    parser = argparse.ArgumentParser(description='Pool kappa and gamma from training batch files')
    parser.add_argument('--batch_pattern', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*.pt',
                        help='Pattern for training batch files to pool from')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to pool (None = all)')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                        help='Directory to save pooled kappa and gamma')
    args = parser.parse_args()

    print("="*80)
    print("POOLING KAPPA AND GAMMA FROM TRAINING BATCHES")
    print("="*80)
    
    try:
        kappa_pooled, gamma_pooled, kappa_array, n_batches = pool_kappa_and_gamma_from_batches(
            args.batch_pattern, args.max_batches
        )
        
        # Save pooled values
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save kappa
        kappa_path = output_dir / 'pooled_kappa.pt'
        torch.save({'kappa': kappa_pooled, 'kappa_array': kappa_array, 'n_batches': n_batches}, kappa_path)
        print(f"\n✓ Saved pooled kappa to: {kappa_path}")
        
        # Save gamma
        gamma_path = output_dir / 'pooled_gamma.pt'
        torch.save({'gamma': gamma_pooled, 'n_batches': n_batches}, gamma_path)
        print(f"✓ Saved pooled gamma to: {gamma_path}")
        
        # Save both together for convenience
        combined_path = output_dir / 'pooled_kappa_gamma.pt'
        torch.save({
            'kappa': kappa_pooled,
            'gamma': gamma_pooled,
            'kappa_array': kappa_array,
            'n_batches': n_batches
        }, combined_path)
        print(f"✓ Saved combined kappa+gamma to: {combined_path}")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")
        print(f"  Use pooled kappa = {kappa_pooled:.6f} in fixed-kappa models")
        print(f"  Use pooled gamma (shape {gamma_pooled.shape}) in fixed-gamma models")
        print(f"  Both should be fixed during prediction (not learned per batch)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

