#!/usr/bin/env python
"""
Compare new vectorized checkpoint with old checkpoint to verify they match
"""

import torch
import numpy as np
from pathlib import Path

# Paths
old_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt'
new_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_batch_0_10000.pt'

print("="*80)
print("COMPARING CHECKPOINTS")
print("="*80)

# Load checkpoints
print(f"\nLoading old checkpoint: {old_ckpt_path}")
old_ckpt = torch.load(old_ckpt_path, map_location='cpu', weights_only=False)

print(f"Loading new checkpoint: {new_ckpt_path}")
new_ckpt = torch.load(new_ckpt_path, map_location='cpu', weights_only=False)

# Check what keys are in each
print("\n" + "="*80)
print("CHECKPOINT KEYS")
print("="*80)
print(f"Old checkpoint keys: {list(old_ckpt.keys())}")
print(f"New checkpoint keys: {list(new_ckpt.keys())}")

# Compare phi (most important)
print("\n" + "="*80)
print("COMPARING PHI VALUES")
print("="*80)
if 'phi' in old_ckpt and 'phi' in new_ckpt:
    old_phi = old_ckpt['phi']
    new_phi = new_ckpt['phi']
    
    if isinstance(old_phi, torch.Tensor):
        old_phi = old_phi.cpu()
    if isinstance(new_phi, torch.Tensor):
        new_phi = new_phi.cpu()
    
    print(f"Old phi shape: {old_phi.shape}")
    print(f"New phi shape: {new_phi.shape}")
    
    if old_phi.shape == new_phi.shape:
        diff = torch.abs(old_phi - new_phi)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nPhi comparison:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        print(f"  Relative max diff: {max_diff / (torch.abs(old_phi).max().item() + 1e-8):.6e}")
        
        if max_diff < 1e-4:
            print("  ✓ PHI VALUES MATCH (within tolerance)")
        else:
            print("  ✗ PHI VALUES DIFFER")
            
        # Show some sample values
        print(f"\nSample phi values (first signature, first disease, first 5 timepoints):")
        print(f"  Old: {old_phi[0, 0, :5].numpy()}")
        print(f"  New: {new_phi[0, 0, :5].numpy()}")
    else:
        print("  ✗ PHI SHAPES DON'T MATCH!")
else:
    print("  ✗ PHI NOT FOUND IN ONE OR BOTH CHECKPOINTS")

# Compare psi if available
print("\n" + "="*80)
print("COMPARING PSI VALUES")
print("="*80)
if 'psi' in old_ckpt and 'psi' in new_ckpt:
    old_psi = old_ckpt['psi']
    new_psi = new_ckpt['psi']
    
    if isinstance(old_psi, torch.Tensor):
        old_psi = old_psi.cpu()
    if isinstance(new_psi, torch.Tensor):
        new_psi = new_psi.cpu()
    
    print(f"Old psi shape: {old_psi.shape}")
    print(f"New psi shape: {new_psi.shape}")
    
    if old_psi.shape == new_psi.shape:
        diff = torch.abs(old_psi - new_psi)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nPsi comparison:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        if max_diff < 1e-4:
            print("  ✓ PSI VALUES MATCH (within tolerance)")
        else:
            print("  ✗ PSI VALUES DIFFER")
    else:
        print("  ✗ PSI SHAPES DON'T MATCH!")
else:
    print("  (PSI not in checkpoints or not saved)")

# Compare clusters
print("\n" + "="*80)
print("COMPARING CLUSTERS")
print("="*80)
if 'clusters' in old_ckpt and 'clusters' in new_ckpt:
    old_clusters = old_ckpt['clusters']
    new_clusters = new_ckpt['clusters']
    
    if isinstance(old_clusters, torch.Tensor):
        old_clusters = old_clusters.cpu().numpy()
    if isinstance(new_clusters, torch.Tensor):
        new_clusters = new_clusters.cpu().numpy()
    
    if np.array_equal(old_clusters, new_clusters):
        print("  ✓ CLUSTERS MATCH EXACTLY")
    else:
        print("  ✗ CLUSTERS DIFFER")
        diff_count = np.sum(old_clusters != new_clusters)
        print(f"  Number of differences: {diff_count} / {len(old_clusters)}")
else:
    print("  (Clusters not in checkpoints)")

# Compare other parameters if available
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Checkpoint comparison complete!")
print("\nIf phi values match closely (< 1e-4), the vectorized version")
print("produces the same results as the old non-vectorized version.")
