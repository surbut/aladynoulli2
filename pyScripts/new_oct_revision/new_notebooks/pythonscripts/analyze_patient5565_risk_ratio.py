"""
Analyze why patient 5565's prostate cancer risk ratio differs between AWS and local pi files.
"""

import torch
from pathlib import Path
import pandas as pd
import numpy as np

print("="*80)
print("ANALYZING PATIENT 5565 RISK RATIO DIFFERENCE")
print("="*80)

# Load both pi files - only load what we need
pi_aws_path = Path.home() / "Downloads" / "pi_full_400k.pt"
pi_local_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_FULL.pt")

patient_idx = 5565
SAMPLE_SIZE = max(10000, patient_idx + 1000)  # Load enough for patient 5565 + baseline calculation

print(f"\n1. Loading pi files (first {SAMPLE_SIZE} patients for patient {patient_idx} analysis)...")
print("   (This may take a moment to load the files...)")

# Load in chunks to avoid memory issues
pi_aws_full = torch.load(str(pi_aws_path), weights_only=False, map_location='cpu')
pi_local_full = torch.load(str(pi_local_path), weights_only=False, map_location='cpu')

sample_n = min(SAMPLE_SIZE, pi_aws_full.shape[0])
pi_aws = pi_aws_full[:sample_n]
pi_local = pi_local_full[:sample_n]

# Free memory
del pi_aws_full, pi_local_full
import gc
gc.collect()

print(f"   ✓ Loaded AWS: {pi_aws.shape}")
print(f"   ✓ Loaded Local: {pi_local.shape}")

# Load disease names
disease_names_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv")
disease_names_df = pd.read_csv(disease_names_path)
disease_names = disease_names_df.iloc[:, 1].tolist()
if len(disease_names) > 0 and str(disease_names[0]).lower() == 'x':
    disease_names = disease_names[1:]
disease_names = [str(name) if pd.notna(name) else f"Disease_{i}" for i, name in enumerate(disease_names)]

# Find prostate cancer index
prostate_idx = None
for i, name in enumerate(disease_names):
    if 'prostate' in str(name).lower() and ('cancer' in str(name).lower() or 'carcinoma' in str(name).lower() or 'malignant' in str(name).lower()):
        prostate_idx = i
        print(f"\n2. Found prostate cancer: index {i} - {disease_names[i]}")
        break

if prostate_idx is None:
    print("⚠️  Could not find prostate cancer!")
    exit(1)

patient_idx = 5565
if patient_idx >= pi_aws.shape[0]:
    print(f"\n⚠️  Patient {patient_idx} not in sample (only have {pi_aws.shape[0]} patients)")
    print("   Using patient 0 instead for demonstration...")
    patient_idx = 0

print(f"\n3. Patient {patient_idx} prostate cancer analysis:")

# Get patient's pi values
pi_aws_patient = pi_aws[patient_idx, prostate_idx, :]  # Shape: (T,)
pi_local_patient = pi_local[patient_idx, prostate_idx, :]  # Shape: (T,)

# Calculate population baselines at each time point
pi_aws_baseline = pi_aws[:, prostate_idx, :].mean(dim=0)  # Shape: (T,)
pi_local_baseline = pi_local[:, prostate_idx, :].mean(dim=0)  # Shape: (T,)

# Calculate risk ratios at each time point
aws_rr = pi_aws_patient / pi_aws_baseline  # Shape: (T,)
local_rr = pi_local_patient / pi_local_baseline  # Shape: (T,)

print(f"\n   Time Point  AWS_pi      Local_pi    AWS_baseline  Local_baseline  AWS_RR    Local_RR   RR_Diff")
print(f"   " + "-"*95)

for t in range(min(52, pi_aws_patient.shape[0])):
    aws_pi_t = pi_aws_patient[t].item()
    local_pi_t = pi_local_patient[t].item()
    aws_baseline_t = pi_aws_baseline[t].item()
    local_baseline_t = pi_local_baseline[t].item()
    aws_rr_t = aws_rr[t].item()
    local_rr_t = local_rr[t].item()
    rr_diff = abs(aws_rr_t - local_rr_t)
    
    # Highlight large differences
    marker = " ⚠️" if rr_diff > 0.5 else ""
    print(f"   {t:2d}         {aws_pi_t:.6f}  {local_pi_t:.6f}  {aws_baseline_t:.6f}  {local_baseline_t:.6f}  {aws_rr_t:.4f}x  {local_rr_t:.4f}x  {rr_diff:.4f}x{marker}")

# Find time point with maximum risk ratio difference
max_diff_idx = torch.argmax(torch.abs(aws_rr - local_rr)).item()
max_diff_val = torch.abs(aws_rr - local_rr)[max_diff_idx].item()

print(f"\n4. Maximum risk ratio difference:")
print(f"   Time point: {max_diff_idx}")
print(f"   AWS RR: {aws_rr[max_diff_idx].item():.4f}x")
print(f"   Local RR: {local_rr[max_diff_idx].item():.4f}x")
print(f"   Difference: {max_diff_val:.4f}x")

# Show the components
print(f"\n   Components at time {max_diff_idx}:")
print(f"   AWS pi: {pi_aws_patient[max_diff_idx].item():.6f}")
print(f"   Local pi: {pi_local_patient[max_diff_idx].item():.6f}")
print(f"   Pi difference: {abs(pi_aws_patient[max_diff_idx].item() - pi_local_patient[max_diff_idx].item()):.6f}")
print(f"   AWS baseline: {pi_aws_baseline[max_diff_idx].item():.6f}")
print(f"   Local baseline: {pi_local_baseline[max_diff_idx].item():.6f}")
print(f"   Baseline difference: {abs(pi_aws_baseline[max_diff_idx].item() - pi_local_baseline[max_diff_idx].item()):.6f}")

# Calculate average risk ratios
aws_rr_mean = aws_rr.mean().item()
local_rr_mean = local_rr.mean().item()
print(f"\n5. Average risk ratios (across all time points):")
print(f"   AWS: {aws_rr_mean:.4f}x")
print(f"   Local: {local_rr_mean:.4f}x")
print(f"   Difference: {abs(aws_rr_mean - local_rr_mean):.4f}x")

# Show max risk ratios
aws_rr_max = aws_rr.max().item()
local_rr_max = local_rr.max().item()
print(f"\n6. Maximum risk ratios (across all time points):")
print(f"   AWS: {aws_rr_max:.4f}x")
print(f"   Local: {local_rr_max:.4f}x")
print(f"   Difference: {abs(aws_rr_max - local_rr_max):.4f}x")

# Explain why small pi differences lead to large RR differences
print(f"\n7. Why small pi differences lead to large RR differences:")
print(f"   When baseline is small, even tiny differences in pi get amplified.")
print(f"   Example at time {max_diff_idx}:")
print(f"   - Pi difference: {abs(pi_aws_patient[max_diff_idx].item() - pi_local_patient[max_diff_idx].item()):.6f}")
print(f"   - Baseline: ~{pi_aws_baseline[max_diff_idx].item():.6f}")
print(f"   - Small pi difference / small baseline = large relative difference")
print(f"   - This is why RR differs more than pi values themselves")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

