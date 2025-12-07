#!/usr/bin/env python
"""
Compare pi (disease probabilities) between old and new checkpoints
"""

import torch
import numpy as np

# Paths
old_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt'
new_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_batch_0_10000.pt'

print("="*80)
print("COMPARING PI VALUES")
print("="*80)

# Load checkpoints
old_ckpt = torch.load(old_ckpt_path, map_location='cpu', weights_only=False)
new_ckpt = torch.load(new_ckpt_path, map_location='cpu', weights_only=False)

# Check if pi is saved directly
if 'pi' in old_ckpt and 'pi' in new_ckpt:
    print("\nPi found directly in checkpoints!")
    old_pi = old_ckpt['pi'].cpu() if isinstance(old_ckpt['pi'], torch.Tensor) else torch.tensor(old_ckpt['pi'])
    new_pi = new_ckpt['pi'].cpu() if isinstance(new_ckpt['pi'], torch.Tensor) else torch.tensor(new_ckpt['pi'])
else:
    print("\nComputing pi from saved parameters...")
    
    # Get parameters
    old_state = old_ckpt.get('model_state_dict', old_ckpt)
    new_state = new_ckpt.get('model_state_dict', new_ckpt)
    
    # Extract phi, lambda, kappa
    old_phi = old_state['phi'].cpu() if 'phi' in old_state else old_ckpt['phi'].cpu()
    new_phi = new_state['phi'].cpu() if 'phi' in new_state else new_ckpt['phi'].cpu()
    
    old_lambda = old_state['lambda_'].cpu() if 'lambda_' in old_state else None
    new_lambda = new_state['lambda_'].cpu() if 'lambda_' in new_state else None
    
    old_kappa = old_state.get('kappa', torch.tensor([1.0])).cpu() if 'kappa' in old_state else torch.tensor([1.0])
    new_kappa = new_state.get('kappa', torch.tensor([1.0])).cpu() if 'kappa' in new_state else torch.tensor([1.0])
    
    # Compute pi = sigmoid(softmax(lambda) @ sigmoid(phi)) * kappa
    # pi = einsum('nkt,kdt->ndt', theta, phi_prob) * kappa
    # where theta = softmax(lambda) and phi_prob = sigmoid(phi)
    
    if old_lambda is not None and new_lambda is not None:
        # Compute theta
        old_theta = torch.softmax(old_lambda, dim=1)  # [N, K, T]
        new_theta = torch.softmax(new_lambda, dim=1)
        
        # Compute phi_prob
        old_phi_prob = torch.sigmoid(old_phi)  # [K, D, T]
        new_phi_prob = torch.sigmoid(new_phi)
        
        # Compute pi
        old_pi = torch.einsum('nkt,kdt->ndt', old_theta, old_phi_prob) * old_kappa
        new_pi = torch.einsum('nkt,kdt->ndt', new_theta, new_phi_prob) * new_kappa
        
        # Clamp to avoid numerical issues
        epsilon = 1e-8
        old_pi = torch.clamp(old_pi, epsilon, 1 - epsilon)
        new_pi = torch.clamp(new_pi, epsilon, 1 - epsilon)
    else:
        print("  ✗ Cannot compute pi - lambda not found in checkpoints")
        print("  Available keys in old_ckpt:", list(old_ckpt.keys()))
        print("  Available keys in new_ckpt:", list(new_ckpt.keys()))
        exit(1)

# Compare pi values
print(f"\nPi shape: {old_pi.shape}")
print(f"Expected: [N, D, T] = [patients, diseases, timepoints]")

diff = torch.abs(old_pi - new_pi)
max_diff = diff.max().item()
mean_diff = diff.mean().item()
std_diff = diff.std().item()

print(f"\nPi comparison:")
print(f"  Max difference: {max_diff:.6e}")
print(f"  Mean difference: {mean_diff:.6e}")
print(f"  Std difference: {std_diff:.6e}")

# Find where max difference occurs
max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
print(f"\nMax difference location: patient={max_idx[0]}, disease={max_idx[1]}, time={max_idx[2]}")
print(f"  Old pi value: {old_pi[max_idx].item():.6f}")
print(f"  New pi value: {new_pi[max_idx].item():.6f}")
print(f"  Difference: {diff[max_idx].item():.6e}")
print(f"  Relative difference: {(diff[max_idx] / (old_pi[max_idx] + 1e-8)).item() * 100:.4f}%")

# Compare by disease (average across patients and time)
print(f"\nMean pi difference by disease (top 10):")
disease_diffs = diff.mean(dim=(0, 2))  # Average over patients and time
top_diseases = torch.topk(disease_diffs, k=min(10, disease_diffs.shape[0]))
for i, (d_idx, diff_val) in enumerate(zip(top_diseases.indices, top_diseases.values)):
    print(f"  Disease {d_idx.item()}: mean diff = {diff_val.item():.6e}")

# Compare by timepoint (average across patients and diseases)
print(f"\nMean pi difference by timepoint:")
time_diffs = diff.mean(dim=(0, 1))  # Average over patients and diseases
for t in range(min(10, time_diffs.shape[0])):
    print(f"  Time {t}: mean diff = {time_diffs[t].item():.6e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if max_diff < 1e-4:
    print("✓ PI VALUES MATCH CLOSELY (max diff < 1e-4)")
elif max_diff < 1e-2:
    print("⚠ PI VALUES ARE SIMILAR (max diff < 1e-2)")
    print("  Small differences expected due to phi/lambda differences")
else:
    print("✗ PI VALUES DIFFER SIGNIFICANTLY (max diff >= 1e-2)")

print(f"\nMean difference: {mean_diff:.6e} (very small = good)")
print(f"Max difference: {max_diff:.6e}")

