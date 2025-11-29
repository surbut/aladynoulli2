"""
Compare two pi prediction files to check for differences.
Specifically comparing:
1. ~/Downloads/pi_full_400k.pt (from AWS)
2. Library/CloudStorage/enrollment_predictions_fixedphi/RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_FULL.pt (local)
"""

import torch
from pathlib import Path
import numpy as np
import pandas as pd
import gc

print("="*80)
print("COMPARING PI PREDICTION FILES")
print("="*80)

# Paths
pi_aws_path = Path.home() / "Downloads" / "pi_full_400k.pt"
pi_local_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_FULL.pt")

# Load both files - SAMPLE ONLY to avoid memory issues
SAMPLE_SIZE = 10000  # Compare first 10K patients only

print(f"\n1. Loading AWS pi file (sampling first {SAMPLE_SIZE} patients)...")
if pi_aws_path.exists():
    # Load only first SAMPLE_SIZE patients
    pi_aws_full = torch.load(str(pi_aws_path), weights_only=False)
    print(f"   ✓ Loaded full: shape {pi_aws_full.shape}")
    sample_n = min(SAMPLE_SIZE, pi_aws_full.shape[0])
    pi_aws = pi_aws_full[:sample_n]
    del pi_aws_full  # Free memory
    gc.collect()
    print(f"   ✓ Using sample: shape {pi_aws.shape}")
else:
    print(f"   ⚠️  File not found!")
    pi_aws = None

print(f"\n2. Loading local pi file (sampling first {SAMPLE_SIZE} patients)...")
if pi_local_path.exists():
    # Load only first SAMPLE_SIZE patients
    pi_local_full = torch.load(str(pi_local_path), weights_only=False)
    print(f"   ✓ Loaded full: shape {pi_local_full.shape}")
    sample_n = min(SAMPLE_SIZE, pi_local_full.shape[0])
    pi_local = pi_local_full[:sample_n]
    del pi_local_full  # Free memory
    gc.collect()
    print(f"   ✓ Using sample: shape {pi_local.shape}")
else:
    print(f"   ⚠️  File not found!")
    pi_local = None

if pi_aws is None or pi_local is None:
    print("\n⚠️  Cannot compare - one or both files missing")
    exit(1)

# Check shapes
print(f"\n3. Shape comparison (sample):")
print(f"   AWS:   {pi_aws.shape}")
print(f"   Local: {pi_local.shape}")

if pi_aws.shape != pi_local.shape:
    print(f"   ⚠️  Shapes differ! Using minimum dimensions for comparison")
    min_n = min(pi_aws.shape[0], pi_local.shape[0])
    min_d = min(pi_aws.shape[1], pi_local.shape[1])
    min_t = min(pi_aws.shape[2], pi_local.shape[2])
    pi_aws = pi_aws[:min_n, :min_d, :min_t]
    pi_local = pi_local[:min_n, :min_d, :min_t]
    print(f"   Using subset: ({min_n}, {min_d}, {min_t})")

# Overall statistics
print(f"\n4. Overall statistics:")
print(f"   AWS - Mean: {pi_aws.mean().item():.6f}, Std: {pi_aws.std().item():.6f}")
print(f"   Local - Mean: {pi_local.mean().item():.6f}, Std: {pi_local.std().item():.6f}")

# Calculate differences on sample
print(f"\n5. Difference statistics (on {pi_aws.shape[0]} patient sample):")

diff = pi_aws - pi_local
abs_diff = torch.abs(diff)

print(f"   Mean absolute difference: {abs_diff.mean().item():.6f}")
print(f"   Max absolute difference: {abs_diff.max().item():.6f}")
print(f"   Median absolute difference: {abs_diff.median().item():.6f}")
print(f"   Std of differences: {diff.std().item():.6f}")

# Check if they're identical
if torch.allclose(pi_aws, pi_local, rtol=1e-5, atol=1e-6):
    print(f"\n   ✓ Files are very similar (within tolerance)")
else:
    print(f"\n   ⚠️  Files differ (but may still be very close)")

# Global comparison by dimension
print(f"\n6. Global comparison by dimension (on {pi_aws.shape[0]} patient sample):")

# Per-patient differences (average across diseases and time)
per_patient_diff = torch.abs(pi_aws - pi_local).mean(dim=(1, 2))  # Shape: (N,)
print(f"\n   Per-patient mean absolute differences:")
print(f"   Mean: {per_patient_diff.mean().item():.6f}")
print(f"   Std: {per_patient_diff.std().item():.6f}")
print(f"   Min: {per_patient_diff.min().item():.6f}")
print(f"   Max: {per_patient_diff.max().item():.6f}")
print(f"   Median: {per_patient_diff.median().item():.6f}")

# Per-disease differences (average across patients and time)
per_disease_diff = torch.abs(pi_aws - pi_local).mean(dim=(0, 2))  # Shape: (D,)
print(f"\n   Per-disease mean absolute differences:")
print(f"   Mean: {per_disease_diff.mean().item():.6f}")
print(f"   Std: {per_disease_diff.std().item():.6f}")
print(f"   Min: {per_disease_diff.min().item():.6f}")
print(f"   Max: {per_disease_diff.max().item():.6f}")
print(f"   Median: {per_disease_diff.median().item():.6f}")

# Per-timepoint differences (average across patients and diseases)
per_time_diff = torch.abs(pi_aws - pi_local).mean(dim=(0, 1))  # Shape: (T,)
print(f"\n   Per-timepoint mean absolute differences:")
print(f"   Mean: {per_time_diff.mean().item():.6f}")
print(f"   Std: {per_time_diff.std().item():.6f}")
print(f"   Min: {per_time_diff.min().item():.6f}")
print(f"   Max: {per_time_diff.max().item():.6f}")
print(f"   Median: {per_time_diff.median().item():.6f}")

# Find patients with largest differences
top_10_patients = torch.argsort(per_patient_diff, descending=True)[:10]
print(f"\n   Top 10 patients with largest differences:")
for i, p_idx in enumerate(top_10_patients):
    p_idx_int = p_idx.item()
    diff_val = per_patient_diff[p_idx_int].item()
    print(f"   {i+1}. Patient {p_idx_int}: mean diff = {diff_val:.6f}")

# Load disease names for disease-level analysis
disease_names_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv")
disease_names = None
if disease_names_path.exists():
    disease_names_df = pd.read_csv(disease_names_path)
    disease_names = disease_names_df.iloc[:, 1].tolist()
    if len(disease_names) > 0 and str(disease_names[0]).lower() == 'x':
        disease_names = disease_names[1:]
    disease_names = [str(name) if pd.notna(name) else f"Disease_{i}" for i, name in enumerate(disease_names)]

# Find diseases with largest differences
top_10_diseases = torch.argsort(per_disease_diff, descending=True)[:10]
print(f"\n   Top 10 diseases with largest differences:")
for i, d_idx in enumerate(top_10_diseases):
    d_idx_int = d_idx.item()
    diff_val = per_disease_diff[d_idx_int].item()
    if disease_names and d_idx_int < len(disease_names):
        print(f"   {i+1}. Disease {d_idx_int} ({disease_names[d_idx_int][:50]}): mean diff = {diff_val:.6f}")
    else:
        print(f"   {i+1}. Disease {d_idx_int}: mean diff = {diff_val:.6f}")

# Patient 5565 specific (if exists)
patient_idx = 5565
if patient_idx < pi_aws.shape[0]:
    print(f"\n7. Patient {patient_idx} specific comparison:")
    
    pi_aws_patient = pi_aws[patient_idx]  # Shape: (D, T)
    pi_local_patient = pi_local[patient_idx]  # Shape: (D, T)
    
    patient_diff = torch.abs(pi_aws_patient - pi_local_patient)
    print(f"   Mean absolute difference: {patient_diff.mean().item():.6f}")
    print(f"   Max absolute difference: {patient_diff.max().item():.6f}")
    
    if disease_names:
        # Find prostate cancer index
        prostate_idx = None
        for i, name in enumerate(disease_names):
            if 'prostate' in str(name).lower() and ('cancer' in str(name).lower() or 'carcinoma' in str(name).lower() or 'malignant' in str(name).lower()):
                prostate_idx = i
                break
        
        if prostate_idx is not None and prostate_idx < pi_aws_patient.shape[0]:
            print(f"\n   Prostate cancer (index {prostate_idx}):")
            # Average across time
            aws_prostate_mean = pi_aws_patient[prostate_idx].mean().item()
            local_prostate_mean = pi_local_patient[prostate_idx].mean().item()
            print(f"   AWS mean: {aws_prostate_mean:.6f}")
            print(f"   Local mean: {local_prostate_mean:.6f}")
            print(f"   Difference: {abs(aws_prostate_mean - local_prostate_mean):.6f}")
            
            # Max across time
            aws_prostate_max = pi_aws_patient[prostate_idx].max().item()
            local_prostate_max = pi_local_patient[prostate_idx].max().item()
            print(f"   AWS max: {aws_prostate_max:.6f}")
            print(f"   Local max: {local_prostate_max:.6f}")
            print(f"   Difference: {abs(aws_prostate_max - local_prostate_max):.6f}")
            
            # Calculate risk ratios (pi over baseline)
            print(f"\n   Risk Ratio (pi / population baseline):")
            
            # Calculate population baseline (mean across all patients in sample)
            aws_prostate_baseline = pi_aws[:, prostate_idx, :].mean().item()
            local_prostate_baseline = pi_local[:, prostate_idx, :].mean().item()
            
            print(f"   Population baseline:")
            print(f"   AWS:   {aws_prostate_baseline:.6f}")
            print(f"   Local: {local_prostate_baseline:.6f}")
            print(f"   Difference: {abs(aws_prostate_baseline - local_prostate_baseline):.6f}")
            
            # Calculate risk ratios using mean pi
            aws_rr_mean = aws_prostate_mean / aws_prostate_baseline if aws_prostate_baseline > 0 else 0
            local_rr_mean = local_prostate_mean / local_prostate_baseline if local_prostate_baseline > 0 else 0
            
            print(f"\n   Risk Ratio (mean pi / baseline):")
            print(f"   AWS:   {aws_rr_mean:.4f}x")
            print(f"   Local: {local_rr_mean:.4f}x")
            print(f"   Difference: {abs(aws_rr_mean - local_rr_mean):.4f}x")
            
            # Calculate risk ratios using max pi
            aws_rr_max = aws_prostate_max / aws_prostate_baseline if aws_prostate_baseline > 0 else 0
            local_rr_max = local_prostate_max / local_prostate_baseline if local_prostate_baseline > 0 else 0
            
            print(f"\n   Risk Ratio (max pi / baseline):")
            print(f"   AWS:   {aws_rr_max:.4f}x")
            print(f"   Local: {local_rr_max:.4f}x")
            print(f"   Difference: {abs(aws_rr_max - local_rr_max):.4f}x")
            
            # Also check at specific time point if we know when MI occurred
            # (This would require knowing the time point, but we can show all time points)
            print(f"\n   Risk Ratio at each time point:")
            print(f"   Time  AWS_pi    Local_pi  AWS_baseline  Local_baseline  AWS_RR   Local_RR  Diff")
            print(f"   " + "-"*80)
            for t in range(min(10, pi_aws_patient.shape[1])):  # Show first 10 time points
                aws_pi_t = pi_aws_patient[prostate_idx, t].item()
                local_pi_t = pi_local_patient[prostate_idx, t].item()
                aws_baseline_t = pi_aws[:, prostate_idx, t].mean().item()
                local_baseline_t = pi_local[:, prostate_idx, t].mean().item()
                aws_rr_t = aws_pi_t / aws_baseline_t if aws_baseline_t > 0 else 0
                local_rr_t = local_pi_t / local_baseline_t if local_baseline_t > 0 else 0
                print(f"   {t:2d}  {aws_pi_t:.6f}  {local_pi_t:.6f}  {aws_baseline_t:.6f}  {local_baseline_t:.6f}  {aws_rr_t:.4f}x  {local_rr_t:.4f}x  {abs(aws_rr_t - local_rr_t):.4f}x")

# Sample comparison across patients
print(f"\n8. Sample comparison (first 10 patients, first disease, first time point):")
print(f"   AWS:   {pi_aws[:10, 0, 0].tolist()}")
print(f"   Local: {pi_local[:10, 0, 0].tolist()}")
print(f"   Diff:  {(pi_aws[:10, 0, 0] - pi_local[:10, 0, 0]).tolist()}")

# Correlation
print(f"\n9. Correlation analysis:")
# Flatten and calculate correlation
pi_aws_flat = pi_aws.flatten()
pi_local_flat = pi_local.flatten()
correlation = torch.corrcoef(torch.stack([pi_aws_flat, pi_local_flat]))[0, 1]
print(f"   Correlation: {correlation.item():.6f}")

# Percentage of values that are very close
close_threshold = 1e-5
very_close = (abs_diff < close_threshold).sum().item()
total = abs_diff.numel()
pct_close = (very_close / total) * 100
print(f"   Values within {close_threshold}: {very_close}/{total} ({pct_close:.2f}%)")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
