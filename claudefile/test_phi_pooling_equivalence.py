#!/usr/bin/env python3
"""
Test script to verify that pre-pooled phi matches on-the-fly pooling from batch files.

This verifies that:
1. master_for_fitting_pooled_correctedE.pt contains the same phi as averaging all batch files
2. generate_fig2_signatures.py and plot_ukb_sigs.py are using equivalent data
"""

import torch
import numpy as np
import glob
from pathlib import Path

def test_phi_pooling_equivalence():
    """Test that pre-pooled phi matches on-the-fly pooling."""
    
    print("="*80)
    print("TESTING PHI POOLING EQUIVALENCE")
    print("="*80)
    
    # 1. Load pre-pooled phi from master checkpoint
    print("\n1. Loading pre-pooled phi from master checkpoint...")
    master_checkpoint_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt'
    master_checkpoint = torch.load(master_checkpoint_path, map_location='cpu', weights_only=False)
    phi_prepooled = master_checkpoint['model_state_dict']['phi']
    if torch.is_tensor(phi_prepooled):
        phi_prepooled = phi_prepooled.detach().cpu().numpy()
    
    print(f"   Pre-pooled phi shape: {phi_prepooled.shape}")
    print(f"   Pre-pooled phi range: [{phi_prepooled.min():.6f}, {phi_prepooled.max():.6f}]")
    print(f"   Pre-pooled phi mean: {phi_prepooled.mean():.6f}, std: {phi_prepooled.std():.6f}")
    
    # 2. Load all batch files and compute mean on-the-fly
    print("\n2. Loading batch files and computing mean on-the-fly...")
    batch_pattern = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt'
    batch_files = sorted(glob.glob(batch_pattern))
    print(f"   Found {len(batch_files)} batch files")
    
    all_phis = []
    for batch_file in batch_files:
        try:
            ckpt = torch.load(batch_file, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in ckpt and 'phi' in ckpt['model_state_dict']:
                phi = ckpt['model_state_dict']['phi']
            elif 'phi' in ckpt:
                phi = ckpt['phi']
            else:
                print(f"   Warning: No phi found in {Path(batch_file).name}")
                continue
            
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            all_phis.append(phi)
        except Exception as e:
            print(f"   Error loading {Path(batch_file).name}: {e}")
            continue
    
    if len(all_phis) == 0:
        raise ValueError("No phi arrays loaded from batch files!")
    
    print(f"   Successfully loaded {len(all_phis)} batch phi arrays")
    
    # Stack and compute mean
    phi_stack = np.stack(all_phis, axis=0)  # (n_batches, K, D, T)
    phi_on_the_fly = np.mean(phi_stack, axis=0)  # (K, D, T)
    
    print(f"   On-the-fly pooled phi shape: {phi_on_the_fly.shape}")
    print(f"   On-the-fly pooled phi range: [{phi_on_the_fly.min():.6f}, {phi_on_the_fly.max():.6f}]")
    print(f"   On-the-fly pooled phi mean: {phi_on_the_fly.mean():.6f}, std: {phi_on_the_fly.std():.6f}")
    
    # 3. Compare the two
    print("\n3. Comparing pre-pooled vs on-the-fly pooled phi...")
    
    # Check shapes match
    if phi_prepooled.shape != phi_on_the_fly.shape:
        print(f"   ❌ SHAPE MISMATCH!")
        print(f"      Pre-pooled: {phi_prepooled.shape}")
        print(f"      On-the-fly: {phi_on_the_fly.shape}")
        return False
    else:
        print(f"   ✓ Shapes match: {phi_prepooled.shape}")
    
    # Compute differences
    diff = phi_prepooled - phi_on_the_fly
    abs_diff = np.abs(diff)
    
    print(f"\n   Difference statistics:")
    print(f"      Mean absolute difference: {abs_diff.mean():.10f}")
    print(f"      Median absolute difference: {np.median(abs_diff):.10f}")
    print(f"      Max absolute difference: {abs_diff.max():.10f}")
    print(f"      Min absolute difference: {abs_diff.min():.10f}")
    print(f"      Std of differences: {diff.std():.10f}")
    
    # Check if they're essentially identical (within numerical precision)
    tolerance = 1e-6  # Very small tolerance for floating point comparison
    max_abs_diff = abs_diff.max()
    
    if max_abs_diff < tolerance:
        print(f"\n   ✓ SUCCESS: Pre-pooled and on-the-fly pooled phi are identical!")
        print(f"      Max difference ({max_abs_diff:.2e}) is within tolerance ({tolerance:.2e})")
        return True
    else:
        print(f"\n   ⚠ WARNING: Differences exceed tolerance!")
        print(f"      Max difference ({max_abs_diff:.2e}) > tolerance ({tolerance:.2e})")
        
        # Find where the largest differences are
        max_diff_indices = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"\n   Largest difference at:")
        print(f"      Signature: {max_diff_indices[0]}")
        print(f"      Disease: {max_diff_indices[1]}")
        print(f"      Timepoint: {max_diff_indices[2]}")
        print(f"      Pre-pooled value: {phi_prepooled[max_diff_indices]:.10f}")
        print(f"      On-the-fly value: {phi_on_the_fly[max_diff_indices]:.10f}")
        print(f"      Difference: {diff[max_diff_indices]:.10f}")
        
        # Check relative difference
        rel_diff = abs_diff / (np.abs(phi_prepooled) + 1e-10)  # Add small value to avoid division by zero
        max_rel_diff = rel_diff.max()
        print(f"\n   Max relative difference: {max_rel_diff:.2e}")
        
        if max_rel_diff < 1e-5:  # 0.001% relative difference
            print(f"   ✓ Relative differences are small - likely just numerical precision")
            return True
        else:
            print(f"   ❌ Relative differences are large - may indicate a real difference")
            return False
    
    # 4. Test sigmoid transformation equivalence
    print("\n4. Testing sigmoid transformation equivalence...")
    from scipy.special import expit as sigmoid
    
    prob_prepooled = sigmoid(phi_prepooled)
    prob_on_the_fly = sigmoid(phi_on_the_fly)
    
    prob_diff = np.abs(prob_prepooled - prob_on_the_fly)
    max_prob_diff = prob_diff.max()
    
    print(f"   Max absolute difference in probabilities: {max_prob_diff:.10f}")
    
    if max_prob_diff < tolerance:
        print(f"   ✓ SUCCESS: Sigmoid-transformed probabilities are identical!")
    else:
        print(f"   ⚠ WARNING: Differences in probabilities exceed tolerance!")
        print(f"      This is expected if phi values differ (sigmoid amplifies differences)")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return max_abs_diff < tolerance


if __name__ == "__main__":
    success = test_phi_pooling_equivalence()
    exit(0 if success else 1)

