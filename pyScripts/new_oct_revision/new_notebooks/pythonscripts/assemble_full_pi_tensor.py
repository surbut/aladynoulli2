#!/usr/bin/env python3
"""
Assemble batch pi tensors into a single full pi tensor for 0-400K patients.

This script concatenates all batch pi tensors (0-10000, 10000-20000, ..., 390000-400000)
into a single tensor covering 0-400K patients.

Usage:
    python assemble_full_pi_tensor.py --approach pooled_retrospective
    python assemble_full_pi_tensor.py --approach pooled_enrollment
"""

import argparse
import torch
from pathlib import Path
import sys

def assemble_pi_tensors(base_dir, approach_name, max_patients=400000, batch_size=10000):
    """
    Assemble batch pi tensors into a single full tensor.
    
    Parameters:
    -----------
    base_dir : Path
        Base directory containing batch pi tensors
    approach_name : str
        Name of the approach (for output filename)
    max_patients : int
        Maximum number of patients to include (default 400000)
    batch_size : int
        Size of each batch (default 10000)
    """
    print("="*80)
    print(f"ASSEMBLING FULL PI TENSOR: {approach_name.upper()}")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Max patients: {max_patients}")
    print(f"Batch size: {batch_size}")
    print("="*80)
    
    # Calculate number of batches needed
    n_batches = max_patients // batch_size
    print(f"\nWill assemble {n_batches} batches (0-{max_patients})")
    
    # Load and concatenate batches
    pi_batches = []
    total_loaded = 0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        
        pi_file = base_dir / f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        if not pi_file.exists():
            print(f"WARNING: Batch file not found: {pi_file}")
            print(f"Skipping batch {i} ({start_idx}-{end_idx})")
            continue
        
        print(f"Loading batch {i+1}/{n_batches}: {start_idx}-{end_idx}...", end=' ')
        try:
            pi_batch = torch.load(pi_file, weights_only=False)
            print(f"✓ Shape: {pi_batch.shape}")
            pi_batches.append(pi_batch)
            total_loaded += pi_batch.shape[0]
        except Exception as e:
            print(f"✗ Error loading: {e}")
            continue
    
    if not pi_batches:
        raise ValueError("No batch pi tensors were successfully loaded!")
    
    print(f"\nLoaded {len(pi_batches)} batches, total patients: {total_loaded}")
    
    # Concatenate all batches
    print("\nConcatenating batches...")
    pi_full = torch.cat(pi_batches, dim=0)
    print(f"Full pi tensor shape: {pi_full.shape}")
    
    # Verify we have the right number of patients
    if pi_full.shape[0] != max_patients:
        print(f"\nWARNING: Expected {max_patients} patients, got {pi_full.shape[0]}")
        print(f"Subsetting to first {max_patients} patients...")
        pi_full = pi_full[:max_patients]
        print(f"Final shape: {pi_full.shape}")
    
    # Save full tensor
    output_file = base_dir / f'pi_enroll_fixedphi_sex_FULL.pt'
    print(f"\nSaving full pi tensor to: {output_file}")
    torch.save(pi_full, output_file)
    print(f"✓ Saved successfully!")
    
    # Print file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("ASSEMBLY COMPLETE")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Tensor shape: {pi_full.shape}")
    print(f"Total patients: {pi_full.shape[0]}")
    
    return pi_full

def main():
    parser = argparse.ArgumentParser(description='Assemble batch pi tensors into full tensor')
    parser.add_argument('--approach', type=str, required=True,
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to assemble')
    parser.add_argument('--max_patients', type=int, default=400000,
                       help='Maximum number of patients (default 400000)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (default 10000)')
    
    args = parser.parse_args()
    
    # Set up paths based on approach
    if args.approach == 'pooled_enrollment':
        base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled')
        approach_name = 'pooled_enrollment'
    elif args.approach == 'pooled_retrospective':
        base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled')
        approach_name = 'pooled_retrospective'
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Assemble the full tensor
    pi_full = assemble_pi_tensors(
        base_dir=base_dir,
        approach_name=approach_name,
        max_patients=args.max_patients,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()

