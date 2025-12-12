import torch
import numpy as np

print("="*80)
print("VERIFYING CORRECTED E MATRICES AND PREVALENCE MATCH MODEL CHECKPOINTS")
print("="*80)

# ============================================================================
# AOU Verification
# ============================================================================
print("\n" + "="*80)
print("AOU VERIFICATION")
print("="*80)

# Load standalone files
aou_E_corrected = torch.load('/Users/sarahurbut/aladynoulli2/aou_E_corrected.pt', map_location='cpu')
aou_prevalence_corrected = torch.load('/Users/sarahurbut/aladynoulli2/aou_prevalence_corrected_E.pt', map_location='cpu')

print(f"\nStandalone files:")
print(f"  aou_E_corrected.pt shape: {aou_E_corrected.shape}")
print(f"  aou_prevalence_corrected_E.pt shape: {aou_prevalence_corrected.shape}")

# Load model checkpoint (if it exists and contains these values)
try:
    aou_model = torch.load('/Users/sarahurbut/aladynoulli2/aou_model_initialized.pt', map_location='cpu')
    print(f"\nModel checkpoint keys: {list(aou_model.keys())}")
    
    # Check if prevalence is stored in model
    if 'prevalence_t' in aou_model:
        model_prevalence = aou_model['prevalence_t']
        if torch.is_tensor(model_prevalence):
            model_prevalence = model_prevalence.detach()
        
        print(f"\nModel checkpoint prevalence_t shape: {model_prevalence.shape}")
        
        # Compare prevalences
        if model_prevalence.shape == aou_prevalence_corrected.shape:
            if torch.is_tensor(model_prevalence):
                diff = torch.abs(model_prevalence - aou_prevalence_corrected)
            else:
                diff = np.abs(model_prevalence - aou_prevalence_corrected)
            
            max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
            mean_diff = diff.mean().item() if torch.is_tensor(diff) else diff.mean()
            
            print(f"\nPrevalence comparison:")
            print(f"  Max difference: {max_diff:.10f}")
            print(f"  Mean difference: {mean_diff:.10f}")
            
            if max_diff < 1e-6:
                print(f"  ✓ PERFECT MATCH!")
            elif max_diff < 1e-3:
                print(f"  ✓ Very close match (likely numerical precision)")
            else:
                print(f"  ✗ MISMATCH DETECTED")
        else:
            print(f"  ✗ Shape mismatch!")
    else:
        print(f"  ⚠ Model checkpoint does not contain 'prevalence_t'")
        
except FileNotFoundError:
    print(f"\n⚠ Model checkpoint not found: aou_model_initialized.pt")
except Exception as e:
    print(f"\n⚠ Error loading model: {e}")

# ============================================================================
# MGB Verification
# ============================================================================
print("\n" + "="*80)
print("MGB VERIFICATION")
print("="*80)

# Load standalone files
mgb_E_corrected = torch.load('/Users/sarahurbut/aladynoulli2/mgb_E_corrected.pt', map_location='cpu')
mgb_prevalence_corrected = torch.load('/Users/sarahurbut/aladynoulli2/mgb_prevalence_corrected_E.pt', map_location='cpu')

print(f"\nStandalone files:")
print(f"  mgb_E_corrected.pt shape: {mgb_E_corrected.shape}")
print(f"  mgb_prevalence_corrected_E.pt shape: {mgb_prevalence_corrected.shape}")

# Load model checkpoint (if it exists and contains these values)
try:
    mgb_model = torch.load('/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt', map_location='cpu')
    print(f"\nModel checkpoint keys: {list(mgb_model.keys())}")
    
    # Check if prevalence is stored in model
    if 'prevalence_t' in mgb_model:
        model_prevalence = mgb_model['prevalence_t']
        if torch.is_tensor(model_prevalence):
            model_prevalence = model_prevalence.detach()
        
        print(f"\nModel checkpoint prevalence_t shape: {model_prevalence.shape}")
        
        # Compare prevalences
        if model_prevalence.shape == mgb_prevalence_corrected.shape:
            if torch.is_tensor(model_prevalence):
                diff = torch.abs(model_prevalence - mgb_prevalence_corrected)
            else:
                diff = np.abs(model_prevalence - mgb_prevalence_corrected)
            
            max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
            mean_diff = diff.mean().item() if torch.is_tensor(diff) else diff.mean()
            
            print(f"\nPrevalence comparison:")
            print(f"  Max difference: {max_diff:.10f}")
            print(f"  Mean difference: {mean_diff:.10f}")
            
            if max_diff < 1e-6:
                print(f"  ✓ PERFECT MATCH!")
            elif max_diff < 1e-3:
                print(f"  ✓ Very close match (likely numerical precision)")
            else:
                print(f"  ✗ MISMATCH DETECTED")
        else:
            print(f"  ✗ Shape mismatch!")
    else:
        print(f"  ⚠ Model checkpoint does not contain 'prevalence_t'")
        
except FileNotFoundError:
    print(f"\n⚠ Model checkpoint not found: mgb_model_initialized.pt")
except Exception as e:
    print(f"\n⚠ Error loading model: {e}")

# ============================================================================
# Check if trained models use the corrected files
# ============================================================================
print("\n" + "="*80)
print("CHECKING TRAINED MODELS USE CORRECTED FILES")
print("="*80)

# Check AOU trained models (if any exist)
import glob
import os

aou_trained_pattern = '/Users/sarahurbut/aladynoulli2/aou_model_*.pt'
aou_trained_files = [f for f in glob.glob(aou_trained_pattern) if 'initialized' not in f]

if aou_trained_files:
    print(f"\nFound {len(aou_trained_files)} AOU trained model file(s)")
    for f in aou_trained_files[:3]:  # Check first 3
        try:
            checkpoint = torch.load(f, map_location='cpu')
            if 'prevalence_t' in checkpoint:
                prev = checkpoint['prevalence_t']
                if torch.is_tensor(prev):
                    prev = prev.detach()
                
                if prev.shape == aou_prevalence_corrected.shape:
                    diff = torch.abs(prev - aou_prevalence_corrected) if torch.is_tensor(prev) else np.abs(prev - aou_prevalence_corrected)
                    max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
                    print(f"  {os.path.basename(f)}: max_diff={max_diff:.10f} {'✓' if max_diff < 1e-6 else '✗'}")
        except Exception as e:
            print(f"  {os.path.basename(f)}: Error - {e}")

# Check MGB trained models
mgb_trained_pattern = '/Users/sarahurbut/aladynoulli2/mgb_model_*.pt'
mgb_trained_files = [f for f in glob.glob(mgb_trained_pattern) if 'initialized' not in f]

if mgb_trained_files:
    print(f"\nFound {len(mgb_trained_files)} MGB trained model file(s)")
    for f in mgb_trained_files[:3]:  # Check first 3
        try:
            checkpoint = torch.load(f, map_location='cpu')
            if 'prevalence_t' in checkpoint:
                prev = checkpoint['prevalence_t']
                if torch.is_tensor(prev):
                    prev = prev.detach()
                
                if prev.shape == mgb_prevalence_corrected.shape:
                    diff = torch.abs(prev - mgb_prevalence_corrected) if torch.is_tensor(prev) else np.abs(prev - mgb_prevalence_corrected)
                    max_diff = diff.max().item() if torch.is_tensor(diff) else diff.max()
                    print(f"  {os.path.basename(f)}: max_diff={max_diff:.10f} {'✓' if max_diff < 1e-6 else '✗'}")
        except Exception as e:
            print(f"  {os.path.basename(f)}: Error - {e}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

