#!/usr/bin/env python3
"""
Verify that pi_full_400k.pt was assembled correctly from batch files.

Checks each batch file against the corresponding slice of pi_full_400k.pt
"""

import torch
import numpy as np
from pathlib import Path
import glob

def verify_pi_assembly():
    """
    Verify that pi_full_400k.pt matches assembled batches.
    """
    batch_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/models_fromAWS_enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/retrospective_pooled')
    pi_full_path = '/Users/sarahurbut/Downloads/pi_full_400k.pt'
    
    print("="*80)
    print("VERIFYING PI ASSEMBLY")
    print("="*80)
    
    # Load full PI tensor
    print("\n1. Loading pi_full_400k.pt...")
    pi_full = torch.load(pi_full_path, map_location='cpu', weights_only=False)
    
    # Handle dict format
    if isinstance(pi_full, dict):
        if 'pi' in pi_full:
            pi_full = pi_full['pi']
        elif 'pi_tensor' in pi_full:
            pi_full = pi_full['pi_tensor']
        else:
            for key in pi_full.keys():
                val = pi_full[key]
                if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                    pi_full = val
                    print(f"  Using key '{key}' from dict")
                    break
    
    print(f"  Shape: {pi_full.shape}")
    
    # Find all batch files
    pattern = str(batch_dir / 'pi_enroll_fixedphi_sex_*_*.pt')
    batch_files = glob.glob(pattern)
    
    # Sort by start index
    def get_start_idx(filename):
        name = Path(filename).name
        parts = name.split('_')
        for i, part in enumerate(parts):
            if part.isdigit() and i < len(parts) - 1:
                return int(part)
        return 0
    
    batch_files = sorted(batch_files, key=get_start_idx)
    
    print(f"\n2. Found {len(batch_files)} batch files")
    print(f"   Expected: 40 batches (0-10000, 10000-20000, ..., 390000-400000)")
    
    # Check each batch
    print("\n3. Verifying each batch...")
    print("-"*80)
    
    all_match = True
    missing_batches = []
    
    for batch_file in batch_files:
        # Extract start and end indices from filename
        name = Path(batch_file).name
        # Format: pi_enroll_fixedphi_sex_START_END.pt
        parts = name.replace('.pt', '').split('_')
        start_idx = None
        end_idx = None
        
        for i, part in enumerate(parts):
            if part.isdigit():
                if start_idx is None:
                    start_idx = int(part)
                elif end_idx is None:
                    end_idx = int(part)
                    break
        
        if start_idx is None or end_idx is None:
            print(f"⚠️  Could not parse indices from: {name}")
            continue
        
        # Load batch file
        pi_batch = torch.load(batch_file, map_location='cpu', weights_only=False)
        
        # Handle dict format
        if isinstance(pi_batch, dict):
            if 'pi' in pi_batch:
                pi_batch = pi_batch['pi']
            elif 'pi_tensor' in pi_batch:
                pi_batch = pi_batch['pi_tensor']
            else:
                for key in pi_batch.keys():
                    val = pi_batch[key]
                    if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                        pi_batch = val
                        break
        
        # Extract corresponding slice from full tensor
        pi_slice = pi_full[start_idx:end_idx]
        
        # Compare
        if pi_batch.shape != pi_slice.shape:
            print(f"❌ {name}: Shape mismatch!")
            print(f"   Batch shape: {pi_batch.shape}")
            print(f"   Slice shape: {pi_slice.shape}")
            all_match = False
            continue
        
        # Check if values match
        max_diff = torch.max(torch.abs(pi_batch - pi_slice)).item()
        mean_diff = torch.mean(torch.abs(pi_batch - pi_slice)).item()
        
        if max_diff < 1e-6:
            print(f"✅ {name}: Match (max_diff={max_diff:.2e})")
        else:
            print(f"❌ {name}: MISMATCH!")
            print(f"   Max diff: {max_diff:.2e}")
            print(f"   Mean diff: {mean_diff:.2e}")
            all_match = False
    
    # Check for missing batches
    print("\n4. Checking for missing batches...")
    print("-"*80)
    
    batch_size = 10000
    expected_batches = 40
    
    for i in range(expected_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        expected_file = batch_dir / f'pi_enroll_fixedphi_sex_{start}_{end}.pt'
        if not expected_file.exists():
            missing_batches.append((start, end))
            print(f"⚠️  Missing batch: {start}_{end}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_match and len(missing_batches) == 0:
        print("✅ ALL BATCHES MATCH AND NONE ARE MISSING")
        print("   pi_full_400k.pt was assembled correctly!")
    elif all_match:
        print(f"✅ ALL AVAILABLE BATCHES MATCH")
        print(f"   {len(missing_batches)} batches are missing (expected {expected_batches} total)")
        print("   pi_full_400k.pt matches the available batches")
    else:
        print("❌ SOME BATCHES DO NOT MATCH")
        print("   pi_full_400k.pt may have been assembled incorrectly")
    
    if missing_batches:
        print(f"\nMissing batches: {len(missing_batches)}")
        for start, end in missing_batches:
            print(f"  - {start}_{end}")


if __name__ == '__main__':
    verify_pi_assembly()

