#!/usr/bin/env python
"""
Quick verification: Check if E_enrollment_full.pt matches E_matrix.pt censored at enrollment age
"""

import torch
import numpy as np
import pandas as pd
import os

# Paths
data_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
csv_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'

print("="*80)
print("VERIFICATION: E_enrollment_full.pt vs E_matrix.pt (offset 0 censoring)")
print("="*80)

# Load E matrices
print("\nLoading E matrices...")
E_enrollment = torch.load(os.path.join(data_dir, 'E_enrollment_full.pt'), weights_only=False)
E_matrix = torch.load(os.path.join(data_dir, 'E_matrix.pt'), weights_only=False)

print(f"E_enrollment_full.pt shape: {E_enrollment.shape}")
print(f"E_matrix.pt shape: {E_matrix.shape}")

# Load CSV to get enrollment ages
print("\nLoading enrollment ages...")
fh_processed = pd.read_csv(csv_path)
print(f"Loaded {len(fh_processed)} patients")

# Check first N patients (or all if small)
n_check = min(10000, len(fh_processed))
print(f"\nChecking first {n_check} patients...")

# Manually censor E_matrix at enrollment age (offset 0)
print("\nCreating manually censored E_matrix (offset 0)...")
E_matrix_censored = E_matrix.clone()

total_times_changed = 0
mismatches = 0

for patient_idx in range(n_check):
    if patient_idx >= len(fh_processed):
        break
    
    enrollment_age = fh_processed.iloc[patient_idx]['age']
    time_since_30 = max(0, enrollment_age - 30)
    
    # Cap event times at enrollment age
    original_times = E_matrix_censored[patient_idx, :].clone()
    E_matrix_censored[patient_idx, :] = torch.minimum(
        E_matrix_censored[patient_idx, :],
        torch.full_like(E_matrix_censored[patient_idx, :], time_since_30)
    )
    
    times_changed = torch.sum(E_matrix_censored[patient_idx, :] != original_times).item()
    total_times_changed += times_changed
    
    # Compare with E_enrollment
    if not torch.allclose(E_matrix_censored[patient_idx, :], E_enrollment[patient_idx, :], rtol=1e-5, atol=1e-6):
        mismatches += 1
        if mismatches <= 5:  # Show first 5 mismatches
            diff = (E_matrix_censored[patient_idx, :] - E_enrollment[patient_idx, :]).abs()
            max_diff_idx = diff.argmax().item()
            print(f"\n  Patient {patient_idx}: enrollment_age={enrollment_age:.1f}, time_since_30={time_since_30:.1f}")
            print(f"    Max diff: {diff.max().item():.6f} at disease {max_diff_idx}")
            print(f"    E_matrix_censored[{max_diff_idx}]: {E_matrix_censored[patient_idx, max_diff_idx].item():.2f}")
            print(f"    E_enrollment[{max_diff_idx}]: {E_enrollment[patient_idx, max_diff_idx].item():.2f}")

print(f"\n{'='*80}")
print("SUMMARY:")
print(f"{'='*80}")
print(f"Total patients checked: {n_check}")
print(f"Total event times changed by censoring: {total_times_changed}")
print(f"Patients with mismatches: {mismatches} / {n_check}")

# Overall comparison
if mismatches == 0:
    print("\n✓ PERFECT MATCH! E_enrollment_full.pt matches E_matrix.pt censored at enrollment age.")
else:
    # Check overall similarity
    max_diff = (E_matrix_censored[:n_check] - E_enrollment[:n_check]).abs().max().item()
    mean_diff = (E_matrix_censored[:n_check] - E_enrollment[:n_check]).abs().mean().item()
    
    print(f"\n⚠ MISMATCHES FOUND:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Mismatch rate: {mismatches/n_check*100:.2f}%")
    
    if max_diff < 0.01 and mismatches/n_check < 0.01:
        print("\n  → Differences are very small (<0.01), likely due to:")
        print("    - Different floating point precision")
        print("    - Different censoring implementation details")
        print("    - E_enrollment_full.pt may have been created with slightly different logic")
    else:
        print("\n  → Significant differences found - may indicate different censoring logic")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
if mismatches == 0:
    print("E_enrollment_full.pt is equivalent to E_matrix.pt censored at enrollment age (offset 0)")
else:
    print("E_enrollment_full.pt is similar but not identical to E_matrix.pt censored at enrollment age")
    print("Check the differences above to understand the discrepancy")

