#!/usr/bin/env python
"""
Detailed comparison of checkpoints to identify where differences come from
"""

import torch
import numpy as np

# Paths
old_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt'
new_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_batch_0_10000.pt'

print("="*80)
print("DETAILED CHECKPOINT COMPARISON")
print("="*80)

# Load checkpoints
old_ckpt = torch.load(old_ckpt_path, map_location='cpu', weights_only=False)
new_ckpt = torch.load(new_ckpt_path, map_location='cpu', weights_only=False)

# Compare args/settings
print("\n1. COMPARING ARGUMENTS/SETTINGS:")
print("-" * 80)
if 'args' in old_ckpt and 'args' in new_ckpt:
    old_args = old_ckpt['args']
    new_args = new_ckpt['args']
    
    for key in set(list(old_args.keys()) + list(new_args.keys())):
        old_val = old_args.get(key, 'NOT IN OLD')
        new_val = new_args.get(key, 'NOT IN NEW')
        if old_val != new_val:
            print(f"  ✗ {key}: OLD={old_val}, NEW={new_val}")
        else:
            print(f"  ✓ {key}: {old_val}")

# Compare phi in detail
print("\n2. DETAILED PHI COMPARISON:")
print("-" * 80)
old_phi = old_ckpt['phi'].cpu() if isinstance(old_ckpt['phi'], torch.Tensor) else torch.tensor(old_ckpt['phi'])
new_phi = new_ckpt['phi'].cpu() if isinstance(new_ckpt['phi'], torch.Tensor) else torch.tensor(new_ckpt['phi'])

diff = torch.abs(old_phi - new_phi)
print(f"Phi shape: {old_phi.shape}")
print(f"Max difference: {diff.max().item():.6e}")
print(f"Mean difference: {diff.mean().item():.6e}")
print(f"Std difference: {diff.std().item():.6e}")

# Find where max difference occurs
max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
print(f"\nMax difference location: signature={max_idx[0]}, disease={max_idx[1]}, time={max_idx[2]}")
print(f"  Old phi value: {old_phi[max_idx].item():.6f}")
print(f"  New phi value: {new_phi[max_idx].item():.6f}")
print(f"  Difference: {diff[max_idx].item():.6e}")

# Check if differences are systematic or random
print(f"\nDifferences by signature:")
for k in range(min(5, old_phi.shape[0])):  # Check first 5 signatures
    sig_diff = diff[k].max().item()
    print(f"  Signature {k}: max diff = {sig_diff:.6e}")

# Compare other parameters
print("\n3. COMPARING OTHER PARAMETERS:")
print("-" * 80)

# Lambda
if 'model_state_dict' in old_ckpt and 'model_state_dict' in new_ckpt:
    old_state = old_ckpt['model_state_dict']
    new_state = new_ckpt['model_state_dict']
    
    if 'lambda_' in old_state and 'lambda_' in new_state:
        old_lambda = old_state['lambda_'].cpu()
        new_lambda = new_state['lambda_'].cpu()
        lambda_diff = torch.abs(old_lambda - new_lambda)
        print(f"Lambda max diff: {lambda_diff.max().item():.6e}, mean: {lambda_diff.mean().item():.6e}")
    
    if 'gamma' in old_state and 'gamma' in new_state:
        old_gamma = old_state['gamma'].cpu()
        new_gamma = new_state['gamma'].cpu()
        gamma_diff = torch.abs(old_gamma - new_gamma)
        print(f"Gamma max diff: {gamma_diff.max().item():.6e}, mean: {gamma_diff.mean().item():.6e}")

# Check if it's just initialization differences
print("\n4. CHECKING INITIALIZATION:")
print("-" * 80)
print("If phi differs but other params are similar, it might be:")
print("  - Different random initialization")
print("  - Different convergence (local minima)")
print("  - Numerical differences in GP prior computation")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("A max difference of 1.1% could be due to:")
print("1. Different random initialization (even with same seed, if code paths differ)")
print("2. Floating point accumulation differences")
print("3. The old checkpoint might have been from a different training run")
print("\nIf the MEAN difference is small (6e-06), the models are very similar.")
print("Consider checking if the training loss curves match.")

