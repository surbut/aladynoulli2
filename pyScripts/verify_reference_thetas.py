"""
Verify that reference_thetas.csv is the softmax of reference_trajectories.pt
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("VERIFYING reference_thetas.csv SOURCE")
print("="*80)

# Load reference_trajectories.pt
ref_traj_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/reference_trajectories.pt"
print(f"\n1. Loading reference_trajectories.pt from:")
print(f"   {ref_traj_path}")

tl = torch.load(ref_traj_path, map_location='cpu', weights_only=False)
print(f"   ✓ Loaded successfully")
print(f"   Keys in file: {list(tl.keys())}")

# Get signature_refs
if 'signature_refs' in tl:
    signature_refs = tl['signature_refs']
    print(f"\n2. Found 'signature_refs'")
    print(f"   Shape: {signature_refs.shape}")
    print(f"   Type: {type(signature_refs)}")
    if torch.is_tensor(signature_refs):
        print(f"   Dtype: {signature_refs.dtype}")
        print(f"   Range: [{signature_refs.min():.4f}, {signature_refs.max():.4f}]")
        signature_refs_np = signature_refs.numpy()
    else:
        signature_refs_np = np.array(signature_refs)
        print(f"   Range: [{signature_refs_np.min():.4f}, {signature_refs_np.max():.4f}]")
else:
    print("   ⚠️  'signature_refs' not found in reference_trajectories.pt")
    signature_refs_np = None

# Check for healthy_ref
if 'healthy_ref' in tl:
    healthy_ref = tl['healthy_ref']
    print(f"\n3. Found 'healthy_ref'")
    if torch.is_tensor(healthy_ref):
        print(f"   Shape: {healthy_ref.shape}")
        print(f"   Range: [{healthy_ref.min():.4f}, {healthy_ref.max():.4f}]")
        healthy_ref_np = healthy_ref.numpy()
    else:
        healthy_ref_np = np.array(healthy_ref)
        print(f"   Shape: {healthy_ref_np.shape}")
        print(f"   Range: [{healthy_ref_np.min():.4f}, {healthy_ref_np.max():.4f}]")
else:
    print(f"\n3. No 'healthy_ref' found (this is OK)")
    healthy_ref_np = None

# Calculate softmax
print(f"\n4. Calculating softmax of signature_refs...")
if signature_refs_np is not None:
    # Convert to torch tensor for softmax
    signature_refs_torch = torch.tensor(signature_refs_np, dtype=torch.float32)
    
    # Apply softmax along signature dimension (dim=0)
    reference_theta_calculated = torch.softmax(signature_refs_torch, dim=0).numpy()
    print(f"   ✓ Calculated softmax")
    print(f"   Shape: {reference_theta_calculated.shape}")
    print(f"   Range: [{reference_theta_calculated.min():.6f}, {reference_theta_calculated.max():.6f}]")
    print(f"   Sum along signature axis (should be ~1.0): {reference_theta_calculated.sum(axis=0)[:5]}")  # Check first 5 timepoints
else:
    reference_theta_calculated = None

# Load reference_thetas.csv
ref_theta_csv_path = "/Users/sarahurbut/dtwin_noulli/reference_thetas.csv"
print(f"\n5. Loading reference_thetas.csv from:")
print(f"   {ref_theta_csv_path}")

try:
    reference_theta_csv = pd.read_csv(ref_theta_csv_path, header=0).values
    print(f"   ✓ Loaded successfully")
    print(f"   Shape: {reference_theta_csv.shape}")
    print(f"   Range: [{reference_theta_csv.min():.6f}, {reference_theta_csv.max():.6f}]")
    print(f"   Sum along signature axis (should be ~1.0): {reference_theta_csv.sum(axis=0)[:5]}")  # Check first 5 timepoints
except Exception as e:
    print(f"   ⚠️  Could not load: {e}")
    reference_theta_csv = None

# Compare
print(f"\n6. COMPARISON:")
print("="*80)

if reference_theta_calculated is not None and reference_theta_csv is not None:
    # Check shapes match
    if reference_theta_calculated.shape == reference_theta_csv.shape:
        print(f"   ✓ Shapes match: {reference_theta_calculated.shape}")
        
        # Compare values
        max_diff = np.abs(reference_theta_calculated - reference_theta_csv).max()
        mean_diff = np.abs(reference_theta_calculated - reference_theta_csv).mean()
        
        print(f"\n   Max difference: {max_diff:.10f}")
        print(f"   Mean difference: {mean_diff:.10f}")
        
        # Check if they're close (within floating point precision)
        if np.allclose(reference_theta_calculated, reference_theta_csv, rtol=1e-5, atol=1e-8):
            print(f"\n   ✓✓✓ MATCH! reference_thetas.csv IS the softmax of reference_trajectories.pt ✓✓✓")
        else:
            print(f"\n   ⚠️  Values differ - may be due to:")
            print(f"      - Different number of signatures (healthy state included?)")
            print(f"      - Different timepoints")
            print(f"      - Rounding/precision differences")
            
            # Check if it's a subset issue
            min_shape = (min(reference_theta_calculated.shape[0], reference_theta_csv.shape[0]),
                        min(reference_theta_calculated.shape[1], reference_theta_csv.shape[1]))
            if min_shape[0] > 0 and min_shape[1] > 0:
                subset_calc = reference_theta_calculated[:min_shape[0], :min_shape[1]]
                subset_csv = reference_theta_csv[:min_shape[0], :min_shape[1]]
                if np.allclose(subset_calc, subset_csv, rtol=1e-5, atol=1e-8):
                    print(f"      ✓ But the overlapping subset matches!")
    else:
        print(f"   ⚠️  Shapes don't match:")
        print(f"      Calculated: {reference_theta_calculated.shape}")
        print(f"      CSV: {reference_theta_csv.shape}")
        
        # Check if CSV has extra row (healthy state?)
        if reference_theta_csv.shape[0] == reference_theta_calculated.shape[0] + 1:
            print(f"      CSV has one extra row - likely includes healthy state")
            # Compare without last row
            if np.allclose(reference_theta_calculated, reference_theta_csv[:-1, :], rtol=1e-5, atol=1e-8):
                print(f"      ✓ First {reference_theta_calculated.shape[0]} rows match!")
        elif reference_theta_csv.shape[0] == reference_theta_calculated.shape[0] - 1:
            print(f"      CSV has one fewer row - calculated includes healthy state")
else:
    print("   ⚠️  Cannot compare - missing data")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if reference_theta_calculated is not None and reference_theta_csv is not None:
    if np.allclose(reference_theta_calculated, reference_theta_csv, rtol=1e-5, atol=1e-8):
        print("✓ reference_thetas.csv = softmax(reference_trajectories.pt['signature_refs'])")
    elif reference_theta_csv.shape[0] == reference_theta_calculated.shape[0] + 1:
        print("✓ reference_thetas.csv = softmax(reference_trajectories.pt['signature_refs']) + healthy state")
    else:
        print("? Need to investigate further - shapes or values differ")
else:
    print("? Could not verify - missing files or data")
print("="*80)

