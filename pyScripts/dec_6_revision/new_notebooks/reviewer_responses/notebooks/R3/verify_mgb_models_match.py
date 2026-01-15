"""
Verify that mgb_model_initialized.pt and mgb_model_trained_correctedE.pt
both use the same corrected E and prevalence.
"""

import torch
import numpy as np
from pathlib import Path

print("="*80)
print("VERIFYING MGB MODELS USE SAME CORRECTED E AND PREVALENCE")
print("="*80)

# Paths
mgb_initialized_path = '/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt'
mgb_trained_path = '/Users/sarahurbut/aladynoulli2/mgb_model_trained_correctedE.pt'
mgb_E_corrected_path = '/Users/sarahurbut/aladynoulli2/mgb_E_corrected.pt'
mgb_prevalence_corrected_path = '/Users/sarahurbut/aladynoulli2/mgb_prevalence_corrected_E.pt'

# Load corrected files (ground truth)
print("\n1. Loading corrected E and prevalence (ground truth)...")
try:
    mgb_E_corrected = torch.load(mgb_E_corrected_path, map_location='cpu')
    mgb_prevalence_corrected = torch.load(mgb_prevalence_corrected_path, map_location='cpu')
    print(f"  ✓ mgb_E_corrected.pt shape: {mgb_E_corrected.shape}")
    print(f"  ✓ mgb_prevalence_corrected_E.pt shape: {mgb_prevalence_corrected.shape}")
except Exception as e:
    print(f"  ✗ Error loading corrected files: {e}")
    exit(1)

# Check mgb_model_initialized.pt
print("\n2. Checking mgb_model_initialized.pt...")
if Path(mgb_initialized_path).exists():
    try:
        mgb_initialized = torch.load(mgb_initialized_path, map_location='cpu')
        print(f"  ✓ File exists")
        print(f"  Keys: {list(mgb_initialized.keys())}")
        
        # Check prevalence
        if 'prevalence_t' in mgb_initialized:
            prev_initialized = mgb_initialized['prevalence_t']
            if torch.is_tensor(prev_initialized):
                prev_initialized = prev_initialized.detach()
            
            print(f"  Prevalence shape: {prev_initialized.shape}")
            
            if prev_initialized.shape == mgb_prevalence_corrected.shape:
                if torch.is_tensor(prev_initialized):
                    diff = torch.abs(prev_initialized - mgb_prevalence_corrected)
                else:
                    diff = np.abs(prev_initialized - mgb_prevalence_corrected)
                
                max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
                mean_diff = diff.mean().item() if torch.is_tensor(diff) else diff.mean()
                
                print(f"  Prevalence comparison with corrected:")
                print(f"    Max difference: {max_diff:.10f}")
                print(f"    Mean difference: {mean_diff:.10f}")
                
                if max_diff < 1e-6:
                    print(f"    ✓ PERFECT MATCH!")
                    initialized_prevalence_match = True
                elif max_diff < 1e-3:
                    print(f"    ✓ Very close match (likely numerical precision)")
                    initialized_prevalence_match = True
                else:
                    print(f"    ✗ MISMATCH DETECTED")
                    initialized_prevalence_match = False
            else:
                print(f"    ✗ Shape mismatch: {prev_initialized.shape} vs {mgb_prevalence_corrected.shape}")
                initialized_prevalence_match = False
        else:
            print(f"  ⚠ No 'prevalence_t' in checkpoint")
            initialized_prevalence_match = None
            
        # Note: E is not stored in model checkpoints, it's data
        # But we can note that the model should have been initialized with corrected E
        print(f"  Note: E is data (not stored in checkpoint), but model should use mgb_E_corrected.pt")
        
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        initialized_prevalence_match = None
else:
    print(f"  ✗ File does not exist: {mgb_initialized_path}")
    initialized_prevalence_match = None

# Check mgb_model_trained_correctedE.pt
print("\n3. Checking mgb_model_trained_correctedE.pt...")
if Path(mgb_trained_path).exists():
    try:
        mgb_trained = torch.load(mgb_trained_path, map_location='cpu')
        print(f"  ✓ File exists")
        print(f"  Keys: {list(mgb_trained.keys())}")
        
        # Check prevalence
        if 'prevalence_t' in mgb_trained:
            prev_trained = mgb_trained['prevalence_t']
            if torch.is_tensor(prev_trained):
                prev_trained = prev_trained.detach()
            
            print(f"  Prevalence shape: {prev_trained.shape}")
            
            if prev_trained.shape == mgb_prevalence_corrected.shape:
                if torch.is_tensor(prev_trained):
                    diff = torch.abs(prev_trained - mgb_prevalence_corrected)
                else:
                    diff = np.abs(prev_trained - mgb_prevalence_corrected)
                
                max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
                mean_diff = diff.mean().item() if torch.is_tensor(diff) else diff.mean()
                
                print(f"  Prevalence comparison with corrected:")
                print(f"    Max difference: {max_diff:.10f}")
                print(f"    Mean difference: {mean_diff:.10f}")
                
                if max_diff < 1e-6:
                    print(f"    ✓ PERFECT MATCH!")
                    trained_prevalence_match = True
                elif max_diff < 1e-3:
                    print(f"    ✓ Very close match (likely numerical precision)")
                    trained_prevalence_match = True
                else:
                    print(f"    ✗ MISMATCH DETECTED")
                    trained_prevalence_match = False
            else:
                print(f"    ✗ Shape mismatch: {prev_trained.shape} vs {mgb_prevalence_corrected.shape}")
                trained_prevalence_match = False
        else:
            print(f"  ⚠ No 'prevalence_t' in checkpoint")
            trained_prevalence_match = None
            
        print(f"  Note: E is data (not stored in checkpoint), but model should use mgb_E_corrected.pt")
        
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        trained_prevalence_match = None
else:
    print(f"  ⚠ File does not exist: {mgb_trained_path}")
    print(f"    (This is OK - mgb_model_initialized.pt might be the trained model)")
    trained_prevalence_match = None

# Compare both models if both exist
if initialized_prevalence_match is not None and trained_prevalence_match is not None:
    print("\n4. Comparing prevalence between both models...")
    try:
        prev_initialized = mgb_initialized['prevalence_t']
        prev_trained = mgb_trained['prevalence_t']
        
        if torch.is_tensor(prev_initialized):
            prev_initialized = prev_initialized.detach()
        if torch.is_tensor(prev_trained):
            prev_trained = prev_trained.detach()
        
        if prev_initialized.shape == prev_trained.shape:
            if torch.is_tensor(prev_initialized):
                diff = torch.abs(prev_initialized - prev_trained)
            else:
                diff = np.abs(prev_initialized - prev_trained)
            
            max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
            mean_diff = diff.mean().item() if torch.is_tensor(diff) else diff.mean()
            
            print(f"  Max difference: {max_diff:.10f}")
            print(f"  Mean difference: {mean_diff:.10f}")
            
            if max_diff < 1e-6:
                print(f"  ✓ PERFECT MATCH between models!")
            elif max_diff < 1e-3:
                print(f"  ✓ Very close match (likely numerical precision)")
            else:
                print(f"  ✗ MISMATCH between models")
    except Exception as e:
        print(f"  ⚠ Could not compare: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nmgb_model_initialized.pt:")
print(f"  Exists: {Path(mgb_initialized_path).exists()}")
if initialized_prevalence_match is True:
    print(f"  ✓ Uses corrected prevalence")
elif initialized_prevalence_match is False:
    print(f"  ✗ Does NOT match corrected prevalence")
else:
    print(f"  ? Cannot verify (prevalence_t not in checkpoint or file not found)")

if Path(mgb_trained_path).exists():
    print(f"\nmgb_model_trained_correctedE.pt:")
    print(f"  Exists: True")
    if trained_prevalence_match is True:
        print(f"  ✓ Uses corrected prevalence")
    elif trained_prevalence_match is False:
        print(f"  ✗ Does NOT match corrected prevalence")
    else:
        print(f"  ? Cannot verify (prevalence_t not in checkpoint)")
else:
    print(f"\nmgb_model_trained_correctedE.pt:")
    print(f"  Exists: False (not found)")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

