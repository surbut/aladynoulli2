#!/usr/bin/env python3
"""
Compare the saved thetas file VALUES with what assemble_new_model_with_pcs() would create
by loading sample batches and checking if the values match exactly
"""

import torch
import numpy as np
from scipy.special import softmax
from pathlib import Path

print("="*80)
print("COMPARING SAVED THETAS VALUES vs ASSEMBLED FROM BATCHES")
print("="*80)

# Load the saved thetas file
saved_path = '/Users/sarahurbut/aladynoulli2/pyScripts/pt/new_thetas_with_pcs_retrospective.pt'
print(f"\n1. Loading saved thetas from: {saved_path}")
saved_thetas = torch.load(saved_path, map_location='cpu', weights_only=False)
if torch.is_tensor(saved_thetas):
    saved_thetas = saved_thetas.numpy()
print(f"   Shape: {saved_thetas.shape}")

# Load a sample batch and compute thetas from it
base_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full"
batch_start = 0
batch_end = 10000
filename = f"enrollment_model_W0.0001_batch_{batch_start}_{batch_end}.pt"
filepath = Path(base_path) / filename

print(f"\n2. Loading sample batch: {filename}")
if filepath.exists():
    model = torch.load(filepath, map_location='cpu', weights_only=False)
    lambda_batch = model['model_state_dict']['lambda_'].numpy()
    print(f"   Lambda shape: {lambda_batch.shape}")
    
    # Apply softmax to get thetas (same as assemble_new_model_with_pcs does)
    batch_thetas = softmax(lambda_batch, axis=1)
    print(f"   Thetas shape (from batch): {batch_thetas.shape}")
    
    # Compare with saved thetas for the same patients
    print(f"\n3. Comparing VALUES for first {batch_end - batch_start} patients:")
    saved_subset = saved_thetas[batch_start:batch_end, :, :]
    print(f"   Saved thetas subset shape: {saved_subset.shape}")
    print(f"   Batch thetas shape: {batch_thetas.shape}")
    
    # Check if they match
    if saved_subset.shape == batch_thetas.shape:
        # Compare values
        max_diff = np.abs(saved_subset - batch_thetas).max()
        mean_diff = np.abs(saved_subset - batch_thetas).mean()
        all_close = np.allclose(saved_subset, batch_thetas, atol=1e-6, rtol=1e-6)
        
        print(f"\n   Value Comparison Results:")
        print(f"   - Max absolute difference: {max_diff:.12f}")
        print(f"   - Mean absolute difference: {mean_diff:.12f}")
        print(f"   - All values close (within 1e-6): {all_close}")
        
        if all_close:
            print(f"\n   ✅ CONFIRMED: Saved thetas VALUES match assembled thetas!")
            print(f"   The file contains the exact same values as assemble_new_model_with_pcs() would create")
        else:
            print(f"\n   ⚠️  Values differ - checking with looser tolerance...")
            all_close_loose = np.allclose(saved_subset, batch_thetas, atol=1e-5, rtol=1e-5)
            print(f"   - All values close (within 1e-5): {all_close_loose}")
            
        # Show detailed sample comparisons
        print(f"\n   Detailed Sample Comparisons:")
        for i, (p, s, t) in enumerate([(0, 0, 0), (100, 5, 10), (1000, 10, 25), (5000, 15, 30)]):
            if p < saved_subset.shape[0] and s < saved_subset.shape[1] and t < saved_subset.shape[2]:
                saved_val = saved_subset[p, s, t]
                batch_val = batch_thetas[p, s, t]
                diff = abs(saved_val - batch_val)
                print(f"   Patient {p}, Signature {s}, Timepoint {t}:")
                print(f"      Saved: {saved_val:.10f}")
                print(f"      Batch: {batch_val:.10f}")
                print(f"      Diff:  {diff:.12f}")
        
        # Check ALL 40 batches to be thorough
        print(f"\n4. Checking ALL 40 batches...")
        total_batches = 40
        batch_size = 10000
        all_match = True
        failed_batches = []
        
        for batch_num in range(total_batches):
            batch_start_test = batch_num * batch_size
            batch_end_test = min((batch_num + 1) * batch_size, 400000)
            filename_test = f"enrollment_model_W0.0001_batch_{batch_start_test}_{batch_end_test}.pt"
            filepath_test = Path(base_path) / filename_test
            
            if filepath_test.exists():
                try:
                    model_test = torch.load(filepath_test, map_location='cpu', weights_only=False)
                    lambda_test = model_test['model_state_dict']['lambda_'].numpy()
                    thetas_test = softmax(lambda_test, axis=1)
                    saved_test = saved_thetas[batch_start_test:batch_end_test, :, :]
                    
                    if saved_test.shape == thetas_test.shape:
                        max_diff_test = np.abs(saved_test - thetas_test).max()
                        all_close_test = np.allclose(saved_test, thetas_test, atol=1e-6, rtol=1e-6)
                        
                        if all_close_test:
                            status = "✅"
                        else:
                            status = "❌"
                            all_match = False
                            failed_batches.append(batch_num)
                        
                        # Show progress every 5 batches
                        if batch_num % 5 == 0 or not all_close_test:
                            print(f"   Batch {batch_num:2d} ({batch_start_test:6d}-{batch_end_test:6d}): max_diff={max_diff_test:.12f} {status}")
                    else:
                        print(f"   Batch {batch_num:2d}: ❌ shape mismatch (saved: {saved_test.shape}, batch: {thetas_test.shape})")
                        all_match = False
                        failed_batches.append(batch_num)
                except Exception as e:
                    print(f"   Batch {batch_num:2d}: ❌ Error loading: {e}")
                    all_match = False
                    failed_batches.append(batch_num)
            else:
                print(f"   Batch {batch_num:2d}: ❌ file not found: {filename_test}")
                all_match = False
                failed_batches.append(batch_num)
        
        print(f"\n   Summary:")
        print(f"   - Total batches checked: {total_batches}")
        if all_match:
            print(f"   - ✅ ALL batches match perfectly!")
        else:
            print(f"   - ❌ {len(failed_batches)} batch(es) failed: {failed_batches}")
        
    else:
        print(f"   ⚠️  Shape mismatch!")
        print(f"   Saved: {saved_subset.shape}, Batch: {batch_thetas.shape}")
else:
    print(f"   ❌ Batch file not found: {filepath}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If all comparisons show values match (within 1e-6), then the saved file")
print("contains the exact same thetas as assemble_new_model_with_pcs() would create.")
print("="*80)

