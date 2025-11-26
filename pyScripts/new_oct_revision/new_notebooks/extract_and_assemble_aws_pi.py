#!/usr/bin/env python3
"""
Extract pi batch files from AWS tar archive and assemble into full tensor.

This script:
1. Extracts only pi batch files from the tar (memory efficient)
2. Assembles them incrementally (loads one batch at a time)
3. Saves the final assembled pi tensor

Memory-efficient: Processes one batch at a time instead of loading all at once.

Usage:
    python extract_and_assemble_aws_pi.py --tar_path /path/to/results.tar.gz --output_path /path/to/output.pt
"""

import argparse
import sys
import os
import tarfile
import torch
import tempfile
from pathlib import Path
from tqdm import tqdm

def extract_pi_batches_from_tar(tar_path, extract_dir, batch_size=10000, max_patients=400000):
    """
    Extract only pi batch files from tar archive.
    
    Args:
        tar_path: Path to tar.gz file
        extract_dir: Directory to extract files to
        batch_size: Size of each batch (default 10000)
        max_patients: Maximum number of patients to process (default 400000)
    
    Returns:
        List of extracted pi file paths, sorted by batch number
    """
    print("="*80)
    print("EXTRACTING PI BATCH FILES FROM TAR ARCHIVE")
    print("="*80)
    print(f"Tar file: {tar_path}")
    print(f"Extract directory: {extract_dir}")
    print(f"Max patients: {max_patients}")
    print("="*80)
    
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate number of batches
    n_batches = (max_patients + batch_size - 1) // batch_size
    
    print(f"\nLooking for {n_batches} batch files (0-{max_patients})...")
    
    # Pattern to match pi batch files
    # Example: pi_enroll_fixedphi_sex_0_10000.pt
    pi_patterns = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, max_patients)
        # Try different possible naming patterns
        patterns = [
            f"pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt",
            f"pi_enroll_fixedphi_sex_{start_idx}_{end_idx}_try2.pt",
            f"pi_enroll_fixedphi_sex_{start_idx}_{end_idx}_try2_withpcs.pt",
            f"pi_enroll_fixedphi_sex_{start_idx}_{end_idx}_try2_withpcs_newrun.pt",
            f"**/pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt",
            f"**/pi_enroll_fixedphi_sex_{start_idx}_{end_idx}_try2.pt",
        ]
        pi_patterns.extend(patterns)
    
    extracted_files = []
    
    print("\nOpening tar archive...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get list of all members
        all_members = tar.getmembers()
        print(f"Total files in archive: {len(all_members)}")
        
        # Find pi batch files
        pi_members = []
        for member in all_members:
            name = member.name
            # Check if it matches any pattern
            for pattern in pi_patterns:
                if pattern.replace('**/', '') in name or name.endswith(pattern.replace('**/', '')):
                    pi_members.append(member)
                    break
        
        print(f"\nFound {len(pi_members)} pi batch files")
        
        if len(pi_members) == 0:
            print("\n⚠️  WARNING: No pi batch files found!")
            print("Available files (first 20):")
            for member in all_members[:20]:
                print(f"  {member.name}")
            return []
        
        # Extract pi files
        print("\nExtracting pi batch files...")
        for member in tqdm(pi_members, desc="Extracting"):
            # Extract to temp location first
            tar.extract(member, path=extract_dir)
            extracted_path = extract_dir / member.name
            
            # If it's in a subdirectory, move to root
            if extracted_path.parent != extract_dir:
                final_path = extract_dir / extracted_path.name
                if extracted_path.exists():
                    extracted_path.rename(final_path)
                    # Remove empty subdirectories
                    try:
                        extracted_path.parent.rmdir()
                    except:
                        pass
                    extracted_path = final_path
            
            if extracted_path.exists():
                extracted_files.append(extracted_path)
    
    # Sort by batch number
    def get_batch_num(path):
        name = path.name
        # Extract start index from filename
        try:
            parts = name.split('_')
            for i, part in enumerate(parts):
                if part.isdigit() and i < len(parts) - 1:
                    return int(part)
        except:
            return 0
        return 0
    
    extracted_files.sort(key=get_batch_num)
    
    print(f"\n✓ Extracted {len(extracted_files)} pi batch files")
    return extracted_files


def assemble_pi_tensor(pi_files, output_path, max_patients=400000):
    """
    Assemble pi batch files into full tensor (memory-efficient).
    
    Args:
        pi_files: List of pi batch file paths, sorted by batch number
        output_path: Path to save assembled tensor
        max_patients: Maximum number of patients (default 400000)
    """
    print("\n" + "="*80)
    print("ASSEMBLING PI TENSOR")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Max patients: {max_patients}")
    print("="*80)
    
    if not pi_files:
        raise ValueError("No pi files provided!")
    
    print(f"\nLoading {len(pi_files)} batch files...")
    
    # Load first batch to get shape
    print(f"\nLoading first batch: {pi_files[0].name}")
    first_batch = torch.load(pi_files[0], weights_only=False)
    print(f"  Shape: {first_batch.shape}")
    print(f"  Dtype: {first_batch.dtype}")
    
    # Calculate total size needed
    batch_size = first_batch.shape[0]
    n_diseases = first_batch.shape[1]
    n_timepoints = first_batch.shape[2]
    
    total_batches = len(pi_files)
    total_patients = min(total_batches * batch_size, max_patients)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Total patients: {total_patients}")
    print(f"Tensor shape: [{total_patients}, {n_diseases}, {n_timepoints}]")
    
    # Pre-allocate full tensor
    print("\nAllocating full tensor...")
    pi_full = torch.zeros((total_patients, n_diseases, n_timepoints), 
                          dtype=first_batch.dtype)
    
    # Load and concatenate batches incrementally
    current_idx = 0
    for i, pi_file in enumerate(tqdm(pi_files, desc="Assembling")):
        if current_idx >= max_patients:
            break
        
        # Load batch
        batch = torch.load(pi_file, weights_only=False)
        batch_size_actual = batch.shape[0]
        
        # Calculate how many patients to take from this batch
        remaining = max_patients - current_idx
        take = min(batch_size_actual, remaining)
        
        # Copy to full tensor
        pi_full[current_idx:current_idx + take] = batch[:take]
        current_idx += take
        
        # Free memory
        del batch
    
    # Trim to exact size if needed
    if current_idx < max_patients:
        pi_full = pi_full[:current_idx]
    
    print(f"\n✓ Assembled tensor shape: {pi_full.shape}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pi_full, output_path)
    
    # Verify
    print(f"✓ Saved {output_path}")
    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  File size: {file_size_gb:.2f} GB")
    
    return pi_full


def main():
    parser = argparse.ArgumentParser(description='Extract and assemble AWS pi tensors')
    parser.add_argument('--tar_path', type=str, required=True,
                       help='Path to AWS tar.gz archive')
    parser.add_argument('--extract_dir', type=str,
                       default=None,
                       help='Directory to extract files to (default: temp directory)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save assembled pi tensor')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Size of each batch (default: 10000)')
    parser.add_argument('--max_patients', type=int, default=400000,
                       help='Maximum number of patients (default: 400000)')
    parser.add_argument('--keep_extracted', action='store_true',
                       help='Keep extracted batch files (default: delete after assembly)')
    
    args = parser.parse_args()
    
    # Set up extract directory
    if args.extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix='aws_pi_extract_')
        print(f"Using temporary directory: {extract_dir}")
    else:
        extract_dir = args.extract_dir
    
    try:
        # Extract pi batch files
        pi_files = extract_pi_batches_from_tar(
            args.tar_path, 
            extract_dir,
            batch_size=args.batch_size,
            max_patients=args.max_patients
        )
        
        if not pi_files:
            print("\n❌ ERROR: No pi files extracted!")
            return 1
        
        # Assemble tensor
        assemble_pi_tensor(
            pi_files,
            args.output_path,
            max_patients=args.max_patients
        )
        
        # Clean up extracted files if requested
        if not args.keep_extracted:
            print(f"\nCleaning up extracted files in {extract_dir}...")
            import shutil
            shutil.rmtree(extract_dir)
            print("✓ Cleaned up")
        
        print("\n" + "="*80)
        print("ASSEMBLY COMPLETE")
        print("="*80)
        print(f"Output: {args.output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())


