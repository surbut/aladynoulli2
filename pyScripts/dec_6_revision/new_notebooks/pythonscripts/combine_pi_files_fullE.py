#!/usr/bin/env python3
"""
Combine pi prediction files from enrollment_predictions_fixedphi_RETROSPECTIVE_pooled_withfullE
into a single pi_enroll_fixedphi_sex_FULL.pt file.
"""

import torch
from pathlib import Path
import re

def extract_range(filename):
    """Extract start and stop indices from filename like pi_enroll_fixedphi_sex_0_10000.pt"""
    match = re.search(r'(\d+)_(\d+)\.pt$', filename.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def combine_pi_files(directory):
    """Combine all pi files in directory into a single FULL file."""
    directory = Path(directory)
    
    # Find all pi files
    pi_files = list(directory.glob("pi_enroll_fixedphi_sex_*.pt"))
    
    # Filter out the FULL file if it exists
    pi_files = [f for f in pi_files if "FULL" not in f.name]
    
    if not pi_files:
        print(f"No pi files found in {directory}")
        return
    
    # Sort by start index
    pi_files_with_ranges = []
    for f in pi_files:
        start, stop = extract_range(f)
        if start is not None:
            pi_files_with_ranges.append((start, stop, f))
    
    pi_files_with_ranges.sort(key=lambda x: x[0])
    
    print(f"Found {len(pi_files_with_ranges)} pi files to combine")
    print(f"Range: {pi_files_with_ranges[0][0]} to {pi_files_with_ranges[-1][1]}")
    
    # Load and concatenate
    pi_batches = []
    expected_start = pi_files_with_ranges[0][0]
    
    for start, stop, filepath in pi_files_with_ranges:
        if start != expected_start:
            print(f"⚠️  Warning: Gap detected! Expected start={expected_start}, got start={start}")
        
        try:
            pi_batch = torch.load(str(filepath), weights_only=False)
            print(f"✓ Loaded {filepath.name}, shape: {pi_batch.shape}, range: {start}-{stop}")
            pi_batches.append(pi_batch)
            expected_start = stop
        except Exception as e:
            print(f"✗ Error loading {filepath.name}: {e}")
            continue
    
    if not pi_batches:
        print("✗ No files loaded successfully!")
        return
    
    # Concatenate along patient dimension (dim=0)
    print(f"\nConcatenating {len(pi_batches)} batches...")
    pi_full = torch.cat(pi_batches, dim=0)
    print(f"Final shape: {pi_full.shape}")
    
    # Save combined file
    full_filename = directory / "pi_enroll_fixedphi_sex_FULL.pt"
    torch.save(pi_full, full_filename)
    print(f"✓ Saved combined predictions to {full_filename}")
    
    # Verify the file
    pi_loaded = torch.load(str(full_filename), weights_only=False)
    print(f"✓ Verified saved file: {pi_loaded.shape}")
    
    return pi_full

if __name__ == '__main__':
    base_dir = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox")
    directory = base_dir / "enrollment_predictions_fixedphi_RETROSPECTIVE_pooled_withfullE"
    
    print("="*80)
    print("COMBINING PI FILES")
    print("="*80)
    print(f"Directory: {directory}")
    
    combine_pi_files(directory)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

