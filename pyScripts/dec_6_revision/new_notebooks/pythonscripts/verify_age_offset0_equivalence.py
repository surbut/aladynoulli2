#!/usr/bin/env python3
"""
Verify that age_offset 0 predictions match the enrollment predictions.
Compare:
- age_offset_files/pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun.pt
- enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_0_10000.pt
"""

import torch
from pathlib import Path
import numpy as np

def compare_tensors(t1, t2, name1="Tensor 1", name2="Tensor 2", tolerance=1e-6):
    """Compare two tensors and report differences."""
    print(f"\n{'='*80}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*80}")
    
    print(f"\n{name1} shape: {t1.shape}")
    print(f"{name2} shape: {t2.shape}")
    
    if t1.shape != t2.shape:
        print(f"❌ SHAPE MISMATCH!")
        return False
    
    # Element-wise comparison
    diff = t1 - t2
    abs_diff = torch.abs(diff)
    
    print(f"\nElement-wise differences:")
    print(f"  Max absolute difference: {abs_diff.max().item():.2e}")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.2e}")
    print(f"  Median absolute difference: {abs_diff.median().item():.2e}")
    print(f"  Std of differences: {abs_diff.std().item():.2e}")
    
    # Count differences
    n_elements = t1.numel()
    n_different = (abs_diff > tolerance).sum().item()
    pct_different = 100 * n_different / n_elements
    
    print(f"\nDifferences > {tolerance}:")
    print(f"  Number: {n_different:,} / {n_elements:,}")
    print(f"  Percentage: {pct_different:.6f}%")
    
    # Check if they're equal within tolerance
    are_equal = torch.allclose(t1, t2, atol=tolerance, rtol=0)
    
    if are_equal:
        print(f"\n✅ TENSORS ARE EQUAL (within tolerance {tolerance})")
    else:
        print(f"\n❌ TENSORS DIFFER (exceed tolerance {tolerance})")
        
        # Show some example differences
        if n_different > 0:
            print(f"\nExample differences (first 10):")
            flat_diff = abs_diff.flatten()
            flat_t1 = t1.flatten()
            flat_t2 = t2.flatten()
            
            # Find indices with largest differences
            top_diff_indices = torch.topk(flat_diff, min(10, n_different)).indices
            
            print(f"  Index    | {name1:20s} | {name2:20s} | Difference")
            print(f"  {'-'*70}")
            for idx in top_diff_indices[:10]:
                val1 = flat_t1[idx].item()
                val2 = flat_t2[idx].item()
                diff_val = flat_diff[idx].item()
                print(f"  {idx:8d} | {val1:20.10f} | {val2:20.10f} | {diff_val:12.2e}")
    
    return are_equal

def main():
    base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox')
    base_dir_two = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal')
    
    # Paths
    age_offset_path = base_dir_two / 'age_offset_local_vectorized_E_corrected' / 'pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt'
    enrollment_path = base_dir / 'enrollment_predictions_fixedphi_correctedE_vectorized' / 'pi_enroll_fixedphi_sex_0_10000.pt'
    
    print("="*80)
    print("VERIFYING AGE OFFSET 0 EQUIVALENCE")
    print("="*80)
    print(f"\nAge offset file:")
    print(f"  {age_offset_path}")
    print(f"\nEnrollment file:")
    print(f"  {enrollment_path}")
    
    # Load files
    if not age_offset_path.exists():
        print(f"\n❌ Age offset file not found: {age_offset_path}")
        return
    
    if not enrollment_path.exists():
        print(f"\n❌ Enrollment file not found: {enrollment_path}")
        return
    
    print("\nLoading files...")
    pi_age_offset = torch.load(str(age_offset_path), weights_only=False)
    pi_enrollment = torch.load(str(enrollment_path), weights_only=False)
    
    print(f"✓ Loaded age_offset file: {pi_age_offset.shape}")
    print(f"✓ Loaded enrollment file: {pi_enrollment.shape}")
    
    # Compare with different tolerances
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Strict comparison (exact match)
    print("\n1. STRICT COMPARISON (exact match):")
    exact_match = compare_tensors(
        pi_age_offset, 
        pi_enrollment,
        name1="Age Offset 0",
        name2="Enrollment",
        tolerance=0.0
    )
    
    # Relaxed comparison (numerical precision)
    print("\n2. RELAXED COMPARISON (numerical precision, tol=1e-6):")
    close_match = compare_tensors(
        pi_age_offset,
        pi_enrollment,
        name1="Age Offset 0",
        name2="Enrollment",
        tolerance=1e-6
    )
    
    # Very relaxed comparison (practical equivalence)
    print("\n3. VERY RELAXED COMPARISON (practical equivalence, tol=1e-4):")
    practical_match = compare_tensors(
        pi_age_offset,
        pi_enrollment,
        name1="Age Offset 0",
        name2="Enrollment",
        tolerance=1e-4
    )
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nAge Offset 0:")
    print(f"  Min: {pi_age_offset.min().item():.6f}")
    print(f"  Max: {pi_age_offset.max().item():.6f}")
    print(f"  Mean: {pi_age_offset.mean().item():.6f}")
    print(f"  Std: {pi_age_offset.std().item():.6f}")
    
    print(f"\nEnrollment:")
    print(f"  Min: {pi_enrollment.min().item():.6f}")
    print(f"  Max: {pi_enrollment.max().item():.6f}")
    print(f"  Mean: {pi_enrollment.mean().item():.6f}")
    print(f"  Std: {pi_enrollment.std().item():.6f}")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    if exact_match:
        print("✅ EXACT MATCH: Files are identical")
    elif close_match:
        print("✅ CLOSE MATCH: Files differ only by numerical precision (< 1e-6)")
    elif practical_match:
        print("⚠️  PRACTICAL MATCH: Files differ slightly but may be practically equivalent (< 1e-4)")
    else:
        print("❌ MISMATCH: Files differ significantly")
        print("\nThis suggests the age_offset 0 predictions may not match enrollment predictions.")
        print("You may need to investigate why they differ.")

if __name__ == '__main__':
    main()

