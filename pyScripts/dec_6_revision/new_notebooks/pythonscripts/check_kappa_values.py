"""
Extract and compare kappa values from IPW correction batches.
Checks both ipwbatchrun113 directory and main results/batch_* directories.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("CHECKING KAPPA VALUES FROM IPW CORRECTION BATCHES")
print("="*80)

# Check both locations
ipw_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/ipwbatchrun113')
results_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')

def extract_kappa_from_batches(batch_dir, label=""):
    """Extract kappa values from batches in a directory."""
    kappa_full_list = []
    kappa_biased_list = []
    kappa_biased_ipw_list = []
    
    print(f"\n{label}")
    print("-" * 80)
    print(f"Extracting kappa values from: {batch_dir}")
    
    for batch_idx in range(1, 6):
        batch_path = batch_dir / f'batch_{batch_idx}'
        
        if not batch_path.exists():
            print(f"  Batch {batch_idx}: Directory not found")
            continue
        
        # Load model files
        model_full_path = batch_path / 'model_full.pt'
        model_biased_path = batch_path / 'model_biased.pt'
        model_biased_ipw_path = batch_path / 'model_biased_ipw.pt'
        
        if model_full_path.exists():
            full_data = torch.load(model_full_path, weights_only=False)
            kappa_full = full_data['kappa']
            if torch.is_tensor(kappa_full):
                kappa_full = kappa_full.item() if kappa_full.numel() == 1 else kappa_full.mean().item()
            kappa_full_list.append(kappa_full)
            print(f"  Batch {batch_idx}: Full kappa = {kappa_full:.6f}")
        else:
            print(f"  Batch {batch_idx}: model_full.pt not found")
        
        if model_biased_path.exists():
            biased_data = torch.load(model_biased_path, weights_only=False)
            kappa_biased = biased_data['kappa']
            if torch.is_tensor(kappa_biased):
                kappa_biased = kappa_biased.item() if kappa_biased.numel() == 1 else kappa_biased.mean().item()
            kappa_biased_list.append(kappa_biased)
            print(f"  Batch {batch_idx}: Biased (no IPW) kappa = {kappa_biased:.6f}")
        else:
            print(f"  Batch {batch_idx}: model_biased.pt not found")
        
        if model_biased_ipw_path.exists():
            biased_ipw_data = torch.load(model_biased_ipw_path, weights_only=False)
            kappa_biased_ipw = biased_ipw_data['kappa']
            if torch.is_tensor(kappa_biased_ipw):
                kappa_biased_ipw = kappa_biased_ipw.item() if kappa_biased_ipw.numel() == 1 else kappa_biased_ipw.mean().item()
            kappa_biased_ipw_list.append(kappa_biased_ipw)
            print(f"  Batch {batch_idx}: Biased (with IPW) kappa = {kappa_biased_ipw:.6f}")
        else:
            print(f"  Batch {batch_idx}: model_biased_ipw.pt not found")
    
    return kappa_full_list, kappa_biased_list, kappa_biased_ipw_list

# Extract from both locations
results_ipw = extract_kappa_from_batches(ipw_dir, "LOCATION 1: ipwbatchrun113")
results_main = extract_kappa_from_batches(results_dir, "LOCATION 2: results/batch_*")

kappa_full_ipw, kappa_biased_ipw, kappa_biased_ipw_ipw = results_ipw
kappa_full_main, kappa_biased_main, kappa_biased_ipw_main = results_main

def print_kappa_summary(kappa_full_list, kappa_biased_list, kappa_biased_ipw_list, label=""):
    """Print summary statistics for kappa values."""
    print(f"\n{'='*80}")
    print(f"KAPPA SUMMARY {label}")
    print("="*80)
    
    if len(kappa_full_list) > 0:
        kappa_full_arr = np.array(kappa_full_list)
        print(f"\nFull Population:")
        print(f"  Mean: {kappa_full_arr.mean():.6f}")
        print(f"  Std:  {kappa_full_arr.std():.6f}")
        print(f"  Min:  {kappa_full_arr.min():.6f}")
        print(f"  Max:  {kappa_full_arr.max():.6f}")
        print(f"  Values: {kappa_full_arr}")
    
    if len(kappa_biased_list) > 0:
        kappa_biased_arr = np.array(kappa_biased_list)
        print(f"\nBiased (no IPW):")
        print(f"  Mean: {kappa_biased_arr.mean():.6f}")
        print(f"  Std:  {kappa_biased_arr.std():.6f}")
        print(f"  Min:  {kappa_biased_arr.min():.6f}")
        print(f"  Max:  {kappa_biased_arr.max():.6f}")
        print(f"  Values: {kappa_biased_arr}")
    
    if len(kappa_biased_ipw_list) > 0:
        kappa_biased_ipw_arr = np.array(kappa_biased_ipw_list)
        print(f"\nBiased (with IPW):")
        print(f"  Mean: {kappa_biased_ipw_arr.mean():.6f}")
        print(f"  Std:  {kappa_biased_ipw_arr.std():.6f}")
        print(f"  Min:  {kappa_biased_ipw_arr.min():.6f}")
        print(f"  Max:  {kappa_biased_ipw_arr.max():.6f}")
        print(f"  Values: {kappa_biased_ipw_arr}")
    
    if len(kappa_full_list) > 0 and len(kappa_biased_list) > 0:
        print(f"\nDifference (Full - Biased no IPW):")
        diff = np.array(kappa_full_list) - np.array(kappa_biased_list)
        print(f"  Mean: {diff.mean():.6f}")
        print(f"  Std:  {diff.std():.6f}")
        print(f"  Values: {diff}")
    
    if len(kappa_full_list) > 0 and len(kappa_biased_ipw_list) > 0:
        print(f"\nDifference (Full - Biased with IPW):")
        diff_ipw = np.array(kappa_full_list) - np.array(kappa_biased_ipw_list)
        print(f"  Mean: {diff_ipw.mean():.6f}")
        print(f"  Std:  {diff_ipw.std():.6f}")
        print(f"  Values: {diff_ipw}")
    
    if len(kappa_biased_list) > 0 and len(kappa_biased_ipw_list) > 0:
        print(f"\nDifference (Biased no IPW - Biased with IPW):")
        diff_between = np.array(kappa_biased_list) - np.array(kappa_biased_ipw_list)
        print(f"  Mean: {diff_between.mean():.6f}")
        print(f"  Std:  {diff_between.std():.6f}")
        print(f"  Values: {diff_between}")

# Print summaries for both locations
print_kappa_summary(kappa_full_ipw, kappa_biased_ipw, kappa_biased_ipw_ipw, "(ipwbatchrun113)")
print_kappa_summary(kappa_full_main, kappa_biased_main, kappa_biased_ipw_main, "(results/batch_*)")

# Compare the two locations
if len(kappa_full_ipw) > 0 and len(kappa_full_main) > 0:
    print(f"\n{'='*80}")
    print("COMPARISON BETWEEN LOCATIONS")
    print("="*80)
    
    print(f"\nFull Population - Difference (ipwbatchrun113 - results/batch_*):")
    if len(kappa_full_ipw) == len(kappa_full_main):
        diff_full = np.array(kappa_full_ipw) - np.array(kappa_full_main)
        print(f"  Mean: {diff_full.mean():.6f}")
        print(f"  Std:  {diff_full.std():.6f}")
        print(f"  Values: {diff_full}")
    else:
        print(f"  Cannot compare: different number of batches ({len(kappa_full_ipw)} vs {len(kappa_full_main)})")
    
    print(f"\nBiased (no IPW) - Difference (ipwbatchrun113 - results/batch_*):")
    if len(kappa_biased_ipw) == len(kappa_biased_main):
        diff_biased = np.array(kappa_biased_ipw) - np.array(kappa_biased_main)
        print(f"  Mean: {diff_biased.mean():.6f}")
        print(f"  Std:  {diff_biased.std():.6f}")
        print(f"  Values: {diff_biased}")
    else:
        print(f"  Cannot compare: different number of batches ({len(kappa_biased_ipw)} vs {len(kappa_biased_main)})")
    
    print(f"\nBiased (with IPW) - Difference (ipwbatchrun113 - results/batch_*):")
    if len(kappa_biased_ipw_ipw) == len(kappa_biased_ipw_main):
        diff_ipw = np.array(kappa_biased_ipw_ipw) - np.array(kappa_biased_ipw_main)
        print(f"  Mean: {diff_ipw.mean():.6f}")
        print(f"  Std:  {diff_ipw.std():.6f}")
        print(f"  Values: {diff_ipw}")
    else:
        print(f"  Cannot compare: different number of batches ({len(kappa_biased_ipw_ipw)} vs {len(kappa_biased_ipw_main)})")

print("\n" + "="*80)
print("DONE")
print("="*80)

