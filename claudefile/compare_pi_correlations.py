#!/usr/bin/env python
"""
Compare correlations between old and new pi values
This checks if the patterns/relationships are preserved
"""

import torch
import numpy as np
from scipy.stats import pearsonr

# Paths
old_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt'
new_ckpt_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_batch_0_10000.pt'

print("="*80)
print("COMPARING PI CORRELATIONS")
print("="*80)

# Load checkpoints
old_ckpt = torch.load(old_ckpt_path, map_location='cpu', weights_only=False)
new_ckpt = torch.load(new_ckpt_path, map_location='cpu', weights_only=False)

# Get parameters and compute pi
old_state = old_ckpt.get('model_state_dict', old_ckpt)
new_state = new_ckpt.get('model_state_dict', new_ckpt)

old_phi = old_state['phi'].cpu() if 'phi' in old_state else old_ckpt['phi'].cpu()
new_phi = new_state['phi'].cpu() if 'phi' in new_state else new_ckpt['phi'].cpu()

old_lambda = old_state['lambda_'].cpu() if 'lambda_' in old_state else None
new_lambda = new_state['lambda_'].cpu() if 'lambda_' in new_state else None

old_kappa = old_state.get('kappa', torch.tensor([1.0])).cpu() if 'kappa' in old_state else torch.tensor([1.0])
new_kappa = new_state.get('kappa', torch.tensor([1.0])).cpu() if 'kappa' in new_state else torch.tensor([1.0])

# Compute pi
old_theta = torch.softmax(old_lambda, dim=1)  # [N, K, T]
new_theta = torch.softmax(new_lambda, dim=1)

old_phi_prob = torch.sigmoid(old_phi)  # [K, D, T]
new_phi_prob = torch.sigmoid(new_phi)

old_pi = torch.einsum('nkt,kdt->ndt', old_theta, old_phi_prob) * old_kappa
new_pi = torch.einsum('nkt,kdt->ndt', new_theta, new_phi_prob) * new_kappa

epsilon = 1e-8
old_pi = torch.clamp(old_pi, epsilon, 1 - epsilon)
new_pi = torch.clamp(new_pi, epsilon, 1 - epsilon)

print(f"\nPi shape: {old_pi.shape}")

# Flatten for overall correlation
old_flat = old_pi.flatten().numpy()
new_flat = new_pi.flatten().numpy()

corr_overall, p_val_overall = pearsonr(old_flat, new_flat)
print(f"\n1. OVERALL CORRELATION (all pi values flattened):")
print(f"   Correlation: {corr_overall:.8f}")
print(f"   P-value: {p_val_overall:.2e}")
print(f"   R²: {corr_overall**2:.8f}")

# Correlation by disease (across all patients and timepoints)
print(f"\n2. CORRELATION BY DISEASE (top 10 best, bottom 10 worst):")
disease_corrs = []
for d in range(old_pi.shape[1]):
    old_d = old_pi[:, d, :].flatten().numpy()
    new_d = new_pi[:, d, :].flatten().numpy()
    corr, _ = pearsonr(old_d, new_d)
    disease_corrs.append((d, corr))

disease_corrs_sorted = sorted(disease_corrs, key=lambda x: x[1])
print(f"\n   Top 10 best correlations:")
for d, corr in disease_corrs_sorted[-10:]:
    print(f"     Disease {d}: r = {corr:.8f}")

print(f"\n   Bottom 10 worst correlations:")
for d, corr in disease_corrs_sorted[:10]:
    print(f"     Disease {d}: r = {corr:.8f}")

# Correlation by timepoint (across all patients and diseases)
print(f"\n3. CORRELATION BY TIMEPOINT:")
time_corrs = []
for t in range(old_pi.shape[2]):
    old_t = old_pi[:, :, t].flatten().numpy()
    new_t = new_pi[:, :, t].flatten().numpy()
    corr, _ = pearsonr(old_t, new_t)
    time_corrs.append((t, corr))

for t, corr in time_corrs[:10]:  # Show first 10
    print(f"   Time {t}: r = {corr:.8f}")

print(f"\n   Min correlation: {min(time_corrs, key=lambda x: x[1])[1]:.8f} at time {min(time_corrs, key=lambda x: x[1])[0]}")
print(f"   Max correlation: {max(time_corrs, key=lambda x: x[1])[1]:.8f} at time {max(time_corrs, key=lambda x: x[1])[0]}")

# Correlation by patient (across all diseases and timepoints)
print(f"\n4. CORRELATION BY PATIENT (sample of 10):")
patient_corrs = []
for n in range(min(100, old_pi.shape[0])):  # Sample first 100 patients
    old_n = old_pi[n, :, :].flatten().numpy()
    new_n = new_pi[n, :, :].flatten().numpy()
    corr, _ = pearsonr(old_n, new_n)
    patient_corrs.append((n, corr))

patient_corrs_sorted = sorted(patient_corrs, key=lambda x: x[1])
print(f"\n   Sample correlations (showing range):")
for n, corr in patient_corrs_sorted[:5]:  # Worst 5
    print(f"     Patient {n}: r = {corr:.8f}")
print(f"     ...")
for n, corr in patient_corrs_sorted[-5:]:  # Best 5
    print(f"     Patient {n}: r = {corr:.8f}")

# Summary statistics
all_disease_corrs = [c for _, c in disease_corrs]
all_time_corrs = [c for _, c in time_corrs]
all_patient_corrs = [c for _, c in patient_corrs]

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Overall correlation: {corr_overall:.8f}")
print(f"\nBy disease:")
print(f"  Mean correlation: {np.mean(all_disease_corrs):.8f}")
print(f"  Min correlation: {np.min(all_disease_corrs):.8f}")
print(f"  Max correlation: {np.max(all_disease_corrs):.8f}")
print(f"  Std correlation: {np.std(all_disease_corrs):.8f}")

print(f"\nBy timepoint:")
print(f"  Mean correlation: {np.mean(all_time_corrs):.8f}")
print(f"  Min correlation: {np.min(all_time_corrs):.8f}")
print(f"  Max correlation: {np.max(all_time_corrs):.8f}")

print(f"\nBy patient (sample of {len(all_patient_corrs)}):")
print(f"  Mean correlation: {np.mean(all_patient_corrs):.8f}")
print(f"  Min correlation: {np.min(all_patient_corrs):.8f}")
print(f"  Max correlation: {np.max(all_patient_corrs):.8f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
if corr_overall > 0.99:
    print("✓ EXCELLENT: Correlations > 0.99 indicate the patterns are nearly identical")
elif corr_overall > 0.95:
    print("✓ VERY GOOD: Correlations > 0.95 indicate strong agreement in patterns")
elif corr_overall > 0.90:
    print("⚠ GOOD: Correlations > 0.90 indicate good agreement, but some differences exist")
else:
    print("✗ WARNING: Correlations < 0.90 suggest significant pattern differences")

print(f"\nHigh correlations mean the relative patterns/relationships are preserved,")
print(f"even if absolute values differ slightly (which is expected given phi/lambda differences).")

