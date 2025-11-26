#!/usr/bin/env python3
"""
Quick check to see what E matrix was used in the retrospective models.
We can't directly check from the saved model, but we can check:
1. What E file exists and its properties
2. Compare with what run_aladyn_batch.py loads
"""

import torch
from pathlib import Path

data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')

print("="*60)
print("CHECKING E MATRICES")
print("="*60)

# Check what E files exist
e_files = {
    'E_matrix.pt': data_dir / 'E_matrix.pt',
    'E_enrollment_full.pt': data_dir / 'E_enrollment_full.pt',
}

for name, path in e_files.items():
    if path.exists():
        print(f"\n✓ {name} exists")
        try:
            E = torch.load(path, weights_only=False)
            if torch.is_tensor(E):
                print(f"  Shape: {E.shape}")
                print(f"  Dtype: {E.dtype}")
                print(f"  Sample values (first patient, first 5 timepoints): {E[0, :5] if len(E.shape) > 1 else 'N/A'}")
            else:
                print(f"  Type: {type(E)}")
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"\n✗ {name} does NOT exist")

print("\n" + "="*60)
print("WHAT run_aladyn_batch.py LOADS:")
print("="*60)
print("  E = torch.load(base_path + 'E_matrix.pt')")
print("  → Uses E_matrix.pt (full retrospective)")
print("\n" + "="*60)

