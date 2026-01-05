#!/usr/bin/env python3
"""
Compare gamma values between test run (lambda_reg=0.1) and no-penalty version.
"""

import torch

# Load test checkpoint (with penalty lambda_reg=0.1)
test_checkpoint = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_withLR_test/model_enroll_fixedphi_sex_0_10000.pt'
test_model = torch.load(test_checkpoint, map_location='cpu', weights_only=False)

# Load no-penalty checkpoint for comparison  
no_penalty_checkpoint = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/model_enroll_fixedphi_sex_0_10000.pt'
no_penalty_model = torch.load(no_penalty_checkpoint, map_location='cpu', weights_only=False)

# Extract gamma
gamma_test = test_model['model_state_dict']['gamma']
gamma_no_penalty = no_penalty_model['model_state_dict']['gamma']

print("="*80)
print("GAMMA COMPARISON: lambda_reg=0.1 vs No Penalty")
print("="*80)

print("\nTest (lambda_reg=0.1) gamma stats:")
print(f"  Mean abs: {gamma_test.abs().mean():.6f}")
print(f"  Max abs: {gamma_test.abs().max():.6f}")
print(f"  Std: {gamma_test.std():.6f}")
print(f"  Shape: {gamma_test.shape}")

print("\nNo-penalty gamma stats:")
print(f"  Mean abs: {gamma_no_penalty.abs().mean():.6f}")
print(f"  Max abs: {gamma_no_penalty.abs().max():.6f}")
print(f"  Std: {gamma_no_penalty.std():.6f}")
print(f"  Shape: {gamma_no_penalty.shape}")

print("\n" + "="*80)
print("RATIOS (Test / No-Penalty)")
print("="*80)
ratio_mean = gamma_test.abs().mean() / gamma_no_penalty.abs().mean()
ratio_max = gamma_test.abs().max() / gamma_no_penalty.abs().max()
ratio_std = gamma_test.std() / gamma_no_penalty.std()

print(f"Mean abs ratio: {ratio_mean:.4f}x")
print(f"Max abs ratio:  {ratio_max:.4f}x")
print(f"Std ratio:      {ratio_std:.4f}x")

print(f"\nAre they identical? {torch.allclose(gamma_test, gamma_no_penalty, atol=1e-6)}")
max_diff = (gamma_test - gamma_no_penalty).abs().max()
print(f"Max difference: {max_diff:.10f}")

# Check per-signature
print("\n" + "="*80)
print("PER-SIGNATURE COMPARISON (Mean Absolute Gamma)")
print("="*80)
for k in range(gamma_test.shape[1]):
    test_mean = gamma_test[:, k].abs().mean()
    no_penalty_mean = gamma_no_penalty[:, k].abs().mean()
    ratio = test_mean / no_penalty_mean if no_penalty_mean > 0 else float('inf')
    print(f"Signature {k}: Test={test_mean:.6f}, No-Penalty={no_penalty_mean:.6f}, Ratio={ratio:.4f}x")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if ratio_mean < 0.5:
    print(f"✓ Penalty is working! Gamma shrunk by {1/ratio_mean:.2f}x")
elif ratio_mean > 0.9:
    print(f"✗ Penalty may not be working - gamma barely changed (ratio={ratio_mean:.4f})")
else:
    print(f"⚠ Penalty has moderate effect - gamma shrunk by {1/ratio_mean:.2f}x")








