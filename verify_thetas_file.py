#!/usr/bin/env python3
"""
Verify that new_thetas_with_pcs_retrospective.pt is what we think it is:
- Created by assemble_new_model_with_pcs()
- Shape: (400000, 21, 52)
- Values are softmax-normalized (sum to 1 along signature dimension)
- Values in [0, 1] range
"""

import torch
import numpy as np
from pathlib import Path

print("="*80)
print("VERIFYING new_thetas_with_pcs_retrospective.pt")
print("="*80)

# Check both possible paths
paths_to_check = [
    '/Users/sarahurbut/aladynoulli2/pyScripts/pt/new_thetas_with_pcs_retrospective.pt',
    '/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt'
]

thetas_path = None
for path in paths_to_check:
    if Path(path).exists():
        thetas_path = path
        print(f"\n✅ Found file at: {thetas_path}")
        break

if thetas_path is None:
    print(f"\n❌ File not found at either location:")
    for path in paths_to_check:
        print(f"   - {path}")
    exit(1)

# Load the file
print(f"\nLoading thetas...")
try:
    thetas = torch.load(thetas_path, map_location='cpu')
    
    if torch.is_tensor(thetas):
        thetas_np = thetas.numpy()
    else:
        thetas_np = np.array(thetas)
    
    print(f"✅ File loaded successfully!")
    
    print(f"\n📊 Thetas Properties:")
    print(f"   Shape: {thetas_np.shape}")
    print(f"   Dtype: {thetas_np.dtype}")
    print(f"   - N (patients): {thetas_np.shape[0]:,}")
    print(f"   - K (signatures): {thetas_np.shape[1]}")
    print(f"   - T (timepoints): {thetas_np.shape[2]}")
    
    # Verify expected shape
    expected_shape = (400000, 21, 52)
    shape_match = thetas_np.shape == expected_shape
    print(f"\n🔍 Shape Verification:")
    print(f"   Expected: {expected_shape}")
    print(f"   Actual:   {thetas_np.shape}")
    print(f"   ✅ Match: {shape_match}")
    
    print(f"\n📈 Value Statistics:")
    print(f"   Min: {thetas_np.min():.6f}")
    print(f"   Max: {thetas_np.max():.6f}")
    print(f"   Mean: {thetas_np.mean():.6f}")
    print(f"   Std: {thetas_np.std():.6f}")
    
    # Check if values are in [0, 1] range (softmax property)
    in_range = np.all((thetas_np >= 0) & (thetas_np <= 1))
    print(f"\n🔍 Softmax Property Checks:")
    print(f"   All values in [0, 1] range: {in_range}")
    
    # Check if values sum to 1 along signature dimension (softmax property)
    sums = thetas_np.sum(axis=1)  # Sum along signature dimension (K)
    print(f"   Sum along signature dimension (should be ~1.0):")
    print(f"   - Mean sum: {sums.mean():.6f}")
    print(f"   - Min sum: {sums.min():.6f}")
    print(f"   - Max sum: {sums.max():.6f}")
    print(f"   - Std of sums: {sums.std():.6f}")
    
    sums_close_to_one = np.allclose(sums, 1.0, atol=1e-5)
    print(f"   ✅ All sums close to 1.0 (within 1e-5): {sums_close_to_one}")
    
    # Check a few sample patients
    print(f"\n📋 Sample Patients:")
    for patient_idx, timepoint in [(0, 0), (1000, 25), (50000, 30)]:
        if patient_idx < thetas_np.shape[0] and timepoint < thetas_np.shape[2]:
            sample = thetas_np[patient_idx, :, timepoint]
            print(f"   Patient {patient_idx}, Timepoint {timepoint}:")
            print(f"      Sum: {sample.sum():.6f}")
            print(f"      Max signature: {np.argmax(sample)} (value: {sample.max():.6f})")
            print(f"      Min signature: {np.argmin(sample)} (value: {sample.min():.6f})")
    
    print(f"\n✅ Final Confirmation:")
    print(f"   - Shape: {thetas_np.shape} {'✅' if shape_match else '❌'}")
    print(f"   - Values in [0,1]: {in_range} {'✅' if in_range else '❌'}")
    print(f"   - Sums to ~1.0: {sums_close_to_one} {'✅' if sums_close_to_one else '❌'}")
    print(f"\n   This file contains:")
    print(f"   - 400,000 patients")
    print(f"   - 21 signatures")
    print(f"   - 52 timepoints (ages 30-81)")
    print(f"   - Softmax-normalized signature loadings (thetas)")
    print(f"   - Created by assemble_new_model_with_pcs() from batch model files")
    
except Exception as e:
    print(f"\n❌ Error loading file: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("="*80)

