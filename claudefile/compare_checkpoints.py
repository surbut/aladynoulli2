#!/usr/bin/env python3
"""
Compare two model checkpoints to see if they're identical.
"""

import torch
import numpy as np

# Load checkpoints
rplap = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/model_enroll_fixedphi_sex_0_10000.pt', weights_only=False)
rp = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/retrospective_pooled/model_enroll_fixedphi_sex_0_10000.pt', weights_only=False)

print("="*80)
print("COMPARING CHECKPOINTS")
print("="*80)

# Check keys
print("\n1. CHECKING KEYS:")
print(f"   rplap keys: {sorted(rplap.keys())}")
print(f"   rp keys: {sorted(rp.keys())}")

if set(rplap.keys()) == set(rp.keys()):
    print("   ✓ Same keys")
else:
    print("   ✗ Different keys!")
    print(f"   Only in rplap: {set(rplap.keys()) - set(rp.keys())}")
    print(f"   Only in rp: {set(rp.keys()) - set(rplap.keys())}")

# Compare each key
print("\n2. COMPARING VALUES:")
all_match = True

for key in sorted(set(rplap.keys()) | set(rp.keys())):
    if key not in rplap:
        print(f"   ✗ {key}: Missing in rplap")
        all_match = False
        continue
    if key not in rp:
        print(f"   ✗ {key}: Missing in rp")
        all_match = False
        continue
    
    val1 = rplap[key]
    val2 = rp[key]
    
    # Handle different types
    # Check for tensors first (both val1 and val2)
    if isinstance(val1, torch.Tensor) or isinstance(val2, torch.Tensor):
        # Convert both to numpy
        if isinstance(val1, torch.Tensor):
            val1_np = val1.cpu().numpy()
        else:
            val1_np = np.array(val1)
        
        if isinstance(val2, torch.Tensor):
            val2_np = val2.cpu().numpy()
        else:
            val2_np = np.array(val2)
        
        if val1_np.shape != val2_np.shape:
            print(f"   ✗ {key}: Shape mismatch - {val1_np.shape} vs {val2_np.shape}")
            all_match = False
        elif np.allclose(val1_np, val2_np, rtol=1e-5, atol=1e-8):
            print(f"   ✓ {key}: Identical (shape: {val1_np.shape})")
        else:
            diff = np.abs(val1_np - val2_np)
            print(f"   ✗ {key}: Different!")
            print(f"      Shape: {val1_np.shape}")
            print(f"      Max diff: {diff.max():.6e}")
            print(f"      Mean diff: {diff.mean():.6e}")
            print(f"      Num different: {(diff > 1e-6).sum()} / {diff.size}")
            all_match = False
    
    elif isinstance(val1, dict):
        if isinstance(val2, dict):
            if val1.keys() == val2.keys():
                print(f"   ✓ {key}: Both dicts with same keys: {sorted(val1.keys())}")
                # Could recurse here if needed
            else:
                print(f"   ✗ {key}: Dict keys differ")
                print(f"      rplap keys: {sorted(val1.keys())}")
                print(f"      rp keys: {sorted(val2.keys())}")
                all_match = False
        else:
            print(f"   ✗ {key}: Type mismatch - dict vs {type(val2)}")
            all_match = False
    
    elif isinstance(val1, (int, float, str, bool)):
        if val1 == val2:
            print(f"   ✓ {key}: Identical ({val1})")
        else:
            print(f"   ✗ {key}: Different - {val1} vs {val2}")
            all_match = False
    
    elif isinstance(val1, np.ndarray):
        # Handle numpy arrays
        if isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                print(f"   ✗ {key}: Shape mismatch - {val1.shape} vs {val2.shape}")
                all_match = False
            elif np.allclose(val1, val2, rtol=1e-5, atol=1e-8):
                print(f"   ✓ {key}: Identical (shape: {val1.shape})")
            else:
                diff = np.abs(val1 - val2)
                print(f"   ✗ {key}: Different!")
                print(f"      Max diff: {diff.max():.6e}, Mean: {diff.mean():.6e}")
                all_match = False
        else:
            print(f"   ✗ {key}: Type mismatch - numpy array vs {type(val2)}")
            all_match = False
    
    elif isinstance(val1, (list, tuple)):
        if isinstance(val2, (list, tuple)) and len(val1) == len(val2):
            # Check if elements are tensors/arrays
            try:
                if all(v1 == v2 for v1, v2 in zip(val1, val2)):
                    print(f"   ✓ {key}: Identical (length: {len(val1)})")
                else:
                    print(f"   ✗ {key}: Different values")
                    all_match = False
            except (RuntimeError, ValueError):
                # Comparison failed (likely tensors), try element-wise
                print(f"   ? {key}: Complex comparison needed (length: {len(val1)})")
        else:
            print(f"   ✗ {key}: Type/length mismatch")
            all_match = False
    
    else:
        # For simple types, use == comparison
        try:
            if val1 == val2:
                print(f"   ✓ {key}: Identical")
            else:
                print(f"   ✗ {key}: Different - {val1} vs {val2}")
                all_match = False
        except (RuntimeError, ValueError) as e:
            # Comparison failed (likely tensors that weren't caught)
            print(f"   ? {key}: Cannot compare directly (type: {type(val1)}), error: {e}")
            # Try converting to numpy if possible
            try:
                if isinstance(val1, torch.Tensor):
                    val1_np = val1.cpu().numpy()
                else:
                    val1_np = np.array(val1)
                if isinstance(val2, torch.Tensor):
                    val2_np = val2.cpu().numpy()
                else:
                    val2_np = np.array(val2)
                
                if np.allclose(val1_np, val2_np, rtol=1e-5, atol=1e-8):
                    print(f"      ✓ Identical after conversion")
                else:
                    print(f"      ✗ Different after conversion")
                    all_match = False
            except:
                print(f"      ⚠️  Could not compare")
                all_match = False

print("\n" + "="*80)
if all_match:
    print("✓ CHECKPOINTS ARE IDENTICAL")
else:
    print("✗ CHECKPOINTS DIFFER")
print("="*80)

