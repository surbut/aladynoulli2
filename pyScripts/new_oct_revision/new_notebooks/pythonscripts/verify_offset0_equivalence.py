#!/usr/bin/env python3
"""
Verify that forAWS_offsetmasterfix.py (offset 0) and run_aladyn_predict_with_master.py
should give the same results for batch 0-10K.

This checks:
1. Does manually capping E_matrix.pt at enrollment age match E_enrollment_full.pt?
2. If so, the two scripts should produce identical pi tensors for batch 0-10K at offset 0.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

def verify_E_equivalence():
    """Check if manually capping E_matrix.pt matches E_enrollment_full.pt"""
    print("="*80)
    print("VERIFYING E MATRIX EQUIVALENCE")
    print("="*80)
    
    data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    csv_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    
    # Load E matrices
    print("\nLoading E matrices...")
    E_matrix = torch.load(data_dir / 'E_matrix.pt', weights_only=False)
    E_enrollment = torch.load(data_dir / 'E_enrollment_full.pt', weights_only=False)
    
    print(f"E_matrix.pt shape: {E_matrix.shape}")
    print(f"E_enrollment_full.pt shape: {E_enrollment.shape}")
    
    # Load enrollment ages
    print("\nLoading enrollment ages...")
    fh_processed = pd.read_csv(csv_path)
    
    # Check batch 0-10K
    start_idx, end_idx = 0, 10000
    print(f"\nChecking batch {start_idx}-{end_idx}...")
    
    # Manually cap E_matrix at enrollment age (offset 0 logic)
    print("\nManually capping E_matrix at enrollment age (offset 0)...")
    E_matrix_capped = E_matrix[start_idx:end_idx].clone()
    
    total_changed = 0
    mismatches = 0
    
    for patient_idx in range(end_idx - start_idx):
        if patient_idx >= len(fh_processed):
            break
        
        enrollment_age = fh_processed.iloc[start_idx + patient_idx]['age']
        time_since_30 = max(0, enrollment_age - 30)
        
        # Cap event times (offset 0 logic from forAWS_offsetmasterfix.py)
        original_times = E_matrix_capped[patient_idx, :].clone()
        E_matrix_capped[patient_idx, :] = torch.minimum(
            E_matrix_capped[patient_idx, :],
            torch.full_like(E_matrix_capped[patient_idx, :], time_since_30)
        )
        
        times_changed = torch.sum(E_matrix_capped[patient_idx, :] != original_times).item()
        total_changed += times_changed
        
        # Compare with E_enrollment
        if not torch.allclose(E_matrix_capped[patient_idx, :], E_enrollment[start_idx + patient_idx, :], rtol=1e-5, atol=1e-6):
            mismatches += 1
            if mismatches <= 5:
                diff = (E_matrix_capped[patient_idx, :] - E_enrollment[start_idx + patient_idx, :]).abs()
                max_diff_idx = diff.argmax().item()
                print(f"\n  Patient {patient_idx}: enrollment_age={enrollment_age:.1f}, time_since_30={time_since_30:.1f}")
                print(f"    Max diff: {diff.max().item():.6f} at disease {max_diff_idx}")
                print(f"    E_matrix_capped[{max_diff_idx}]: {E_matrix_capped[patient_idx, max_diff_idx].item():.2f}")
                print(f"    E_enrollment[{max_diff_idx}]: {E_enrollment[start_idx + patient_idx, max_diff_idx].item():.2f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Total patients checked: {end_idx - start_idx}")
    print(f"Total event times changed by capping: {total_changed}")
    print(f"Patients with mismatches: {mismatches} / {end_idx - start_idx}")
    
    if mismatches == 0:
        print("\n✅ PERFECT MATCH!")
        print("   Manually capping E_matrix.pt matches E_enrollment_full.pt")
        print("   → The two scripts SHOULD produce identical results for batch 0-10K at offset 0")
    else:
        max_diff = (E_matrix_capped - E_enrollment[start_idx:end_idx]).abs().max().item()
        mean_diff = (E_matrix_capped - E_enrollment[start_idx:end_idx]).abs().mean().item()
        
        print(f"\n⚠ MISMATCHES FOUND:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Mismatch rate: {mismatches/(end_idx-start_idx)*100:.2f}%")
        
        if max_diff < 0.01:
            print("\n  → Differences are very small (<0.01)")
            print("    Likely due to:")
            print("    - Different capping implementation")
            print("    - Floating point precision")
            print("    - Different enrollment ages used")
        else:
            print("\n  → Significant differences found")
            print("    May indicate different capping logic")
    
    return mismatches == 0


def compare_pi_tensors():
    """Compare pi tensors from the two scripts for batch 0-10K at offset 0"""
    print("\n" + "="*80)
    print("COMPARING PI TENSORS")
    print("="*80)
    
    # Age offset pi (from forAWS_offsetmasterfix.py, offset 0, batch 0-10K)
    pi_age_offset_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/age_offset_files') / \
                         'pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun.pt'
    
    # Washout pi (from run_aladyn_predict_with_master.py, batch 0-10K slice)
    pi_washout_path = Path('/Users/sarahurbut/Downloads/pi_full_400k.pt')
    
    if not pi_age_offset_path.exists():
        print(f"❌ Age offset pi not found: {pi_age_offset_path}")
        return False
    
    if not pi_washout_path.exists():
        print(f"❌ Washout pi not found: {pi_washout_path}")
        return False
    
    print(f"\nLoading age_offset pi: {pi_age_offset_path}")
    pi_age_offset = torch.load(pi_age_offset_path, weights_only=False)
    
    print(f"Loading washout pi: {pi_washout_path}")
    pi_washout_full = torch.load(pi_washout_path, weights_only=False)
    pi_washout_batch = pi_washout_full[:10000]
    
    print(f"\nAge offset pi shape: {pi_age_offset.shape}")
    print(f"Washout pi (batch) shape: {pi_washout_batch.shape}")
    
    if pi_age_offset.shape != pi_washout_batch.shape:
        print(f"❌ Shape mismatch!")
        return False
    
    # Compare tensors
    if torch.allclose(pi_age_offset, pi_washout_batch, rtol=1e-4, atol=1e-5):
        print("✅ Pi tensors match!")
        return True
    else:
        print("❌ Pi tensors differ!")
        max_diff = (pi_age_offset - pi_washout_batch).abs().max().item()
        mean_diff = (pi_age_offset - pi_washout_batch).abs().mean().item()
        print(f"   Max difference: {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")
        return False


if __name__ == "__main__":
    print("="*80)
    print("VERIFYING OFFSET 0 EQUIVALENCE")
    print("="*80)
    print("\nQuestion: Should forAWS_offsetmasterfix.py (offset 0) and")
    print("          run_aladyn_predict_with_master.py give same results?")
    print("="*80)
    
    # Check E matrix equivalence
    E_match = verify_E_equivalence()
    
    # Compare pi tensors
    pi_match = compare_pi_tensors()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    if E_match and pi_match:
        print("✅ YES - They should and DO produce identical results!")
        print("   - E matrices match (manual capping = pre-capped)")
        print("   - Pi tensors match")
    elif E_match and not pi_match:
        print("⚠️  E matrices match, but pi tensors differ")
        print("   - This suggests differences in:")
        print("     * Model initialization")
        print("     * Training dynamics")
        print("     * Random seed effects")
        print("     * Numerical precision")
    elif not E_match:
        print("⚠️  E matrices differ")
        print("   - Manual capping doesn't exactly match E_enrollment_full.pt")
        print("   - This explains why pi tensors differ")
        print("   - Check how E_enrollment_full.pt was created")

