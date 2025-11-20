# Master Checkpoint Creation - Verification

## Data Source Verification

### ✅ Retrospective Master Checkpoint (`master_for_fitting_pooled_all_data.pt`)

**Phi Source:**
- ✅ Loaded from: `enrollment_retrospective_full/enrollment_model_W0.0001_batch_*_*.pt`
- ✅ Pooled by: Computing mean across all retrospective batches
- ✅ **NO enrollment data used for phi**

**Psi Source:**
- ✅ Initial psi: `initial_psi_400k.pt` (shared, population-level)
- ✅ Healthy state psi: Extracted from retrospective batches ONLY
- ✅ **NO enrollment data used for healthy state psi**

**Healthy State Phi:**
- ✅ Extracted from retrospective batches ONLY
- ✅ **NO enrollment data used for healthy state phi**

---

### ✅ Enrollment Master Checkpoint (`master_for_fitting_pooled_enrollment_data.pt`)

**Phi Source:**
- ✅ Loaded from: `enrollment_prediction_jointphi_sex_pcs/enrollment_model_W0.0001_batch_*_*.pt`
- ✅ Pooled by: Computing mean across all enrollment batches
- ✅ **NO retrospective data used for phi**

**Psi Source:**
- ✅ Initial psi: `initial_psi_400k.pt` (shared, population-level)
- ✅ Healthy state psi: Extracted from enrollment batches ONLY
- ✅ **NO retrospective data used for healthy state psi**

**Healthy State Phi:**
- ✅ Extracted from enrollment batches ONLY
- ✅ **NO retrospective data used for healthy state phi**

---

## Key Points

1. **No Data Mixing**: Each master checkpoint uses ONLY its respective data source:
   - Retrospective checkpoint = retrospective batches only
   - Enrollment checkpoint = enrollment batches only

2. **Shared Components**:
   - `initial_psi_400k.pt` is shared (this is correct - it's population-level initialization)
   - Healthy state values are extracted from the same source as phi (no cross-contamination)

3. **Initialization Fix**:
   - The script now uses `psi_total` (from master checkpoint) for gamma initialization
   - This ensures consistent initialization across batches
   - Previously used `initial_psi` which was less accurate

4. **Healthy State Padding**:
   - Both phi and psi are padded with healthy state if needed (shape 20 → 21)
   - Healthy state values are extracted from the SAME data source as the phi
   - No cross-contamination between retrospective and enrollment

---

## Verification Commands

To verify the checkpoints were created correctly:

```python
import torch
import numpy as np

# Check retrospective checkpoint
retro = torch.load('master_for_fitting_pooled_all_data.pt', weights_only=False)
print("Retrospective checkpoint:")
print(f"  Description: {retro.get('description', 'N/A')}")
print(f"  Phi shape: {retro['model_state_dict']['phi'].shape}")
print(f"  Psi shape: {retro['model_state_dict']['psi'].shape}")

# Check enrollment checkpoint
enroll = torch.load('master_for_fitting_pooled_enrollment_data.pt', weights_only=False)
print("\nEnrollment checkpoint:")
print(f"  Description: {enroll.get('description', 'N/A')}")
print(f"  Phi shape: {enroll['model_state_dict']['phi'].shape}")
print(f"  Psi shape: {enroll['model_state_dict']['psi'].shape}")

# Verify shapes are correct (should be 21, D, T for phi and 21, D for psi)
assert retro['model_state_dict']['phi'].shape[0] == 21, "Phi should have 21 signatures (20 + healthy)"
assert retro['model_state_dict']['psi'].shape[0] == 21, "Psi should have 21 signatures (20 + healthy)"
assert enroll['model_state_dict']['phi'].shape[0] == 21, "Phi should have 21 signatures (20 + healthy)"
assert enroll['model_state_dict']['psi'].shape[0] == 21, "Psi should have 21 signatures (20 + healthy)"

print("\n✅ All checks passed!")
```

---

## Summary

**✅ NO CHEATING**: Each master checkpoint is created using ONLY its designated data source:
- Retrospective checkpoint uses retrospective batches exclusively
- Enrollment checkpoint uses enrollment batches exclusively
- No cross-contamination between data sources
- Healthy state values extracted from the same source as phi

The initialization fix ensures consistent gamma initialization using `psi_total` from the master checkpoint, which is more accurate than using the old `initial_psi` file.

