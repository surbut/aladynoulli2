#!/usr/bin/env python3
"""
Compare phi values between aou_batches (with corrected E) and aou_batches_old (original)
to verify that fixing negative values in E didn't significantly change phi.
"""

import torch
import numpy as np
from pathlib import Path
import glob

# Configuration
AOU_BATCH_DIR_NEW = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/aou_batches')
AOU_BATCH_DIR_OLD = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/aou_batches_old')
N_BATCHES_TO_CHECK = 25  # Check all 25 batches
TOLERANCE = 1e-4  # Tolerance for "basically the same"

print("="*80)
print("COMPARING AOU PHI: NEW (corrected E) vs OLD")
print("="*80)
print(f"New batch directory: {AOU_BATCH_DIR_NEW}")
print(f"Old batch directory: {AOU_BATCH_DIR_OLD}")
print(f"Number of batches to check: {N_BATCHES_TO_CHECK}")
print(f"Tolerance for similarity: {TOLERANCE}")
print("="*80)

# Find batch files in both directories
new_pattern = str(AOU_BATCH_DIR_NEW / 'aou_model_batch_*_*_*.pt')
old_pattern = str(AOU_BATCH_DIR_OLD / 'aou_model_batch_*_*_*.pt')

new_batch_files = sorted(glob.glob(new_pattern))
old_batch_files = sorted(glob.glob(old_pattern))

print(f"\nFound {len(new_batch_files)} batch files in NEW directory")
print(f"Found {len(old_batch_files)} batch files in OLD directory")

if len(new_batch_files) == 0:
    raise FileNotFoundError(f"No batch files found in {AOU_BATCH_DIR_NEW}")
if len(old_batch_files) == 0:
    raise FileNotFoundError(f"No batch files found in {AOU_BATCH_DIR_OLD}")

# Extract batch numbers and sort by batch index
def extract_batch_idx(filename):
    """Extract batch index from filename like 'aou_model_batch_5_50000_60000.pt'"""
    return int(Path(filename).name.split('_')[3])

new_batch_files = sorted(new_batch_files, key=extract_batch_idx)
old_batch_files = sorted(old_batch_files, key=extract_batch_idx)

print(f"\nChecking first {N_BATCHES_TO_CHECK} batches from each directory...\n")

# Load and compare phi from corresponding batches
comparisons = []
all_close = True
max_diff_overall = 0
mean_diff_overall = 0

for i in range(min(N_BATCHES_TO_CHECK, len(new_batch_files), len(old_batch_files))):
    new_file = new_batch_files[i]
    old_file = old_batch_files[i]
    
    batch_idx = extract_batch_idx(new_file)
    print(f"Batch {batch_idx}: Comparing {Path(new_file).name} vs {Path(old_file).name}")
    
    try:
        # Load new checkpoint
        new_ckpt = torch.load(new_file, map_location='cpu', weights_only=False)
        new_phi = None
        if 'model_state_dict' in new_ckpt and 'phi' in new_ckpt['model_state_dict']:
            new_phi = new_ckpt['model_state_dict']['phi']
        elif 'phi' in new_ckpt:
            new_phi = new_ckpt['phi']
        else:
            print(f"  ⚠️  Warning: No phi found in NEW batch {batch_idx}")
            continue
        
        # Load old checkpoint
        old_ckpt = torch.load(old_file, map_location='cpu', weights_only=False)
        old_phi = None
        if 'model_state_dict' in old_ckpt and 'phi' in old_ckpt['model_state_dict']:
            old_phi = old_ckpt['model_state_dict']['phi']
        elif 'phi' in old_ckpt:
            old_phi = old_ckpt['phi']
        else:
            print(f"  ⚠️  Warning: No phi found in OLD batch {batch_idx}")
            continue
        
        # Convert to numpy
        if torch.is_tensor(new_phi):
            new_phi = new_phi.detach().cpu().numpy()
        else:
            new_phi = np.array(new_phi)
            
        if torch.is_tensor(old_phi):
            old_phi = old_phi.detach().cpu().numpy()
        else:
            old_phi = np.array(old_phi)
        
        # Check shapes
        if new_phi.shape != old_phi.shape:
            print(f"  ⚠️  Shapes differ! New: {new_phi.shape}, Old: {old_phi.shape}")
            all_close = False
            continue
        
        # Calculate differences
        diff = np.abs(new_phi - old_phi)
        max_diff = diff.max()
        mean_diff = diff.mean()
        relative_max_diff = max_diff / (np.abs(new_phi).max() + 1e-8)
        
        # Update overall stats
        max_diff_overall = max(max_diff_overall, max_diff)
        mean_diff_overall = max(mean_diff_overall, mean_diff)
        
        # Check if close
        is_close = np.allclose(new_phi, old_phi, atol=TOLERANCE)
        if not is_close:
            all_close = False
        
        status = "✓" if is_close else "✗"
        print(f"  {status} Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Relative: {relative_max_diff:.6e}")
        print(f"    Close (tol={TOLERANCE})? {is_close}")
        
        comparisons.append({
            'batch_idx': batch_idx,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'relative_max_diff': relative_max_diff,
            'is_close': is_close
        })
        
    except Exception as e:
        print(f"  ✗ Error comparing batch {batch_idx}: {e}")
        continue

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total batches compared: {len(comparisons)}")
print(f"Overall max difference: {max_diff_overall:.6e}")
print(f"Overall mean difference: {mean_diff_overall:.6e}")

if all_close and len(comparisons) > 0:
    print(f"\n✓ SUCCESS: All phi values are consistent between NEW and OLD!")
    print(f"  Fixing negative E values did NOT significantly change phi")
elif len(comparisons) > 0:
    # Count how many are close
    close_count = sum(1 for c in comparisons if c['is_close'])
    print(f"\n⚠️  WARNING: {len(comparisons) - close_count} out of {len(comparisons)} batches differ!")
    print(f"  {close_count} batches are close (within tolerance)")
    print(f"  {len(comparisons) - close_count} batches differ (may need investigation)")
    
    # Show worst differences
    if len(comparisons) > 0:
        comparisons_sorted = sorted(comparisons, key=lambda x: x['max_diff'], reverse=True)
        print(f"\nWorst 5 differences:")
        for comp in comparisons_sorted[:5]:
            status = "✓" if comp['is_close'] else "✗"
            print(f"  {status} Batch {comp['batch_idx']}: max_diff={comp['max_diff']:.6e}, mean_diff={comp['mean_diff']:.6e}")

# Show sample comparison from first batch
if len(comparisons) > 0:
    print(f"\n{'='*80}")
    print("SAMPLE COMPARISON (Batch 0)")
    print(f"{'='*80}")
    try:
        new_ckpt = torch.load(new_batch_files[0], map_location='cpu', weights_only=False)
        old_ckpt = torch.load(old_batch_files[0], map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in new_ckpt and 'phi' in new_ckpt['model_state_dict']:
            new_phi = new_ckpt['model_state_dict']['phi']
        else:
            new_phi = new_ckpt['phi']
            
        if 'model_state_dict' in old_ckpt and 'phi' in old_ckpt['model_state_dict']:
            old_phi = old_ckpt['model_state_dict']['phi']
        else:
            old_phi = old_ckpt['phi']
        
        if torch.is_tensor(new_phi):
            new_phi = new_phi.detach().cpu().numpy()
        if torch.is_tensor(old_phi):
            old_phi = old_phi.detach().cpu().numpy()
        
        print(f"Shape: {new_phi.shape}")
        print(f"First signature, first disease, first 5 timepoints:")
        print(f"  NEW: {new_phi[0, 0, :5]}")
        print(f"  OLD: {old_phi[0, 0, :5]}")
        print(f"  DIFF: {np.abs(new_phi[0, 0, :5] - old_phi[0, 0, :5])}")
    except Exception as e:
        print(f"Could not show sample: {e}")

print(f"\n{'='*80}")
print("COMPARISON COMPLETE")
print(f"{'='*80}")
