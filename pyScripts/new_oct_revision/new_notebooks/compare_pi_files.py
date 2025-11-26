#!/usr/bin/env python3
"""
Compare the assembled pi file with the existing pi_full_400k.pt file
to verify they're the same.
"""

import torch
from pathlib import Path

print("="*80)
print("COMPARING PI FILES")
print("="*80)

# Paths
pi_assembled = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_FULL.pt')
pi_downloaded = Path('/Users/sarahurbut/Downloads/pi_full_400k.pt')

print(f"\n1. Assembled file: {pi_assembled}")
print(f"   Exists: {pi_assembled.exists()}")
if pi_assembled.exists():
    size_mb = pi_assembled.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")

print(f"\n2. Downloaded file: {pi_downloaded}")
print(f"   Exists: {pi_downloaded.exists()}")
if pi_downloaded.exists():
    size_mb = pi_downloaded.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")

if not pi_assembled.exists():
    print("\n⚠️  Assembled file does not exist - cannot compare")
    exit(1)

if not pi_downloaded.exists():
    print("\n⚠️  Downloaded file does not exist - cannot compare")
    exit(1)

# Load both files
print("\n" + "="*80)
print("LOADING FILES")
print("="*80)

print("\nLoading assembled file...")
pi_assembled_tensor = torch.load(pi_assembled, weights_only=False)
print(f"  Shape: {pi_assembled_tensor.shape}")
print(f"  Dtype: {pi_assembled_tensor.dtype}")

print("\nLoading downloaded file...")
pi_downloaded_tensor = torch.load(pi_downloaded, weights_only=False)
print(f"  Shape: {pi_downloaded_tensor.shape}")
print(f"  Dtype: {pi_downloaded_tensor.dtype}")

# Compare shapes
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

if pi_assembled_tensor.shape != pi_downloaded_tensor.shape:
    print(f"\n❌ SHAPES DIFFER:")
    print(f"   Assembled: {pi_assembled_tensor.shape}")
    print(f"   Downloaded: {pi_downloaded_tensor.shape}")
else:
    print(f"\n✓ Shapes match: {pi_assembled_tensor.shape}")

# Compare values
print("\nComparing values...")
if pi_assembled_tensor.shape == pi_downloaded_tensor.shape:
    # Check if they're exactly equal
    are_equal = torch.equal(pi_assembled_tensor, pi_downloaded_tensor)
    
    if are_equal:
        print("✓ Files are IDENTICAL (all values match exactly)")
    else:
        # Check how different they are
        diff = torch.abs(pi_assembled_tensor - pi_downloaded_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\n❌ Files are DIFFERENT:")
        print(f"   Max difference: {max_diff:.10f}")
        print(f"   Mean difference: {mean_diff:.10f}")
        
        # Check if differences are just floating point precision
        if max_diff < 1e-6:
            print(f"\n⚠️  Differences are very small (< 1e-6) - likely just floating point precision")
            print(f"   Files are effectively the same")
        else:
            print(f"\n⚠️  Differences are significant - files are different")
            
            # Sample some differences
            print(f"\nSampling differences (first 10 non-zero):")
            nonzero_diff = diff[diff > 1e-10]
            if len(nonzero_diff) > 0:
                print(f"   Found {len(nonzero_diff)} non-zero differences")
                print(f"   Sample values: {nonzero_diff[:10].tolist()}")
else:
    print("⚠️  Cannot compare values - shapes don't match")

# Compare first few values as sample
print("\n" + "="*80)
print("SAMPLE VALUES (first 5 patients, first disease, first 5 timepoints)")
print("="*80)

if pi_assembled_tensor.shape == pi_downloaded_tensor.shape:
    sample_assembled = pi_assembled_tensor[:5, 0, :5]
    sample_downloaded = pi_downloaded_tensor[:5, 0, :5]
    
    print("\nAssembled:")
    print(sample_assembled)
    print("\nDownloaded:")
    print(sample_downloaded)
    print("\nDifference:")
    print(sample_assembled - sample_downloaded)

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)

