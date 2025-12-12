import torch
import numpy as np
import glob
import os
from pathlib import Path

# Paths
model_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full'
initial_clusters_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt'

# Pattern to match
pattern = 'enrollment_model_W0.0001_batch_*_*.pt'

# Find matching files
model_files = glob.glob(os.path.join(model_dir, pattern))
model_files.sort()

print(f"Found {len(model_files)} matching model files:")
for f in model_files:
    print(f"  - {os.path.basename(f)}")

if len(model_files) == 0:
    print(f"\nNo files found matching pattern: {pattern}")
    print(f"Searching in: {model_dir}")
    exit(1)

# Load initial clusters
print(f"\nLoading initial clusters from: {initial_clusters_path}")
initial_clusters = torch.load(initial_clusters_path, map_location='cpu', weights_only=False)
if isinstance(initial_clusters, torch.Tensor):
    initial_clusters = initial_clusters.numpy()
print(f"Initial clusters shape: {initial_clusters.shape}")
print(f"Initial clusters range: [{initial_clusters.min()}, {initial_clusters.max()}]")

# Check each model file
print(f"\n{'='*80}")
print(f"CHECKING MAX SIGNATURE PER DISEASE vs INITIAL CLUSTERS")
print(f"{'='*80}")

all_match = True

for model_file in model_files:
    print(f"\n{'='*80}")
    print(f"Checking: {os.path.basename(model_file)}")
    print(f"{'='*80}")
    
    # Load model
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    
    # Extract psi
    if 'model_state_dict' in checkpoint:
        psi = checkpoint['model_state_dict']['psi']
    elif 'psi' in checkpoint:
        psi = checkpoint['psi']
    else:
        print(f"  ERROR: Could not find psi in checkpoint!")
        all_match = False
        continue
    
    if torch.is_tensor(psi):
        psi = psi.detach()
    
    print(f"  Psi shape: {psi.shape}")  # Should be [K_total x D]
    
    # Compute max signature per disease
    # Take argmax across signatures (psi is already [K_total x D], no time dimension)
    max_sig_per_disease = torch.argmax(psi, dim=0).numpy()  # [D]
    
    print(f"  Max sig per disease shape: {max_sig_per_disease.shape}")
    print(f"  Max sig per disease range: [{max_sig_per_disease.min()}, {max_sig_per_disease.max()}]")
    
    # Compare with initial clusters
    if max_sig_per_disease.shape[0] != len(initial_clusters):
        print(f"  ERROR: Shape mismatch!")
        print(f"    Max sig shape: {max_sig_per_disease.shape[0]}")
        print(f"    Initial clusters shape: {len(initial_clusters)}")
        all_match = False
        continue
    
    matches = (max_sig_per_disease == initial_clusters)
    n_matches = matches.sum()
    n_total = len(initial_clusters)
    match_rate = 100 * n_matches / n_total
    
    print(f"\n  Comparison Results:")
    print(f"    Matches: {n_matches} / {n_total} ({match_rate:.2f}%)")
    print(f"    Mismatches: {n_total - n_matches}")
    
    if n_matches == n_total:
        print(f"  ✓ PERFECT MATCH!")
    else:
        print(f"  ✗ MISMATCHES FOUND")
        all_match = False
        
        # Show mismatches
        mismatch_indices = np.where(~matches)[0]
        print(f"\n  First 10 mismatches:")
        for idx in mismatch_indices[:10]:
            print(f"    Disease {idx}: max_sig={max_sig_per_disease[idx]}, initial_cluster={initial_clusters[idx]}")
        
        if len(mismatch_indices) > 10:
            print(f"    ... and {len(mismatch_indices) - 10} more")

print(f"\n{'='*80}")
if all_match:
    print(f"✓ ALL MODELS MATCH INITIAL CLUSTERS PERFECTLY!")
else:
    print(f"✗ SOME MODELS HAVE MISMATCHES")
print(f"{'='*80}")

