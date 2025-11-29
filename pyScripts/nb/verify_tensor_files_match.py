#!/usr/bin/env python3
"""
Verify that tensor files match between different locations.
This script compares Y_tensor.pt, E_matrix.pt, and G_matrix.pt files.
"""

import torch
from pathlib import Path
import sys

def compare_tensors(file1: Path, file2: Path, name: str) -> bool:
    """Compare two tensor files and return True if they match exactly."""
    print(f"\n{'='*80}")
    print(f"Comparing {name}")
    print(f"{'='*80}")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    if not file1.exists():
        print(f"  ❌ File 1 does not exist!")
        return False
    
    if not file2.exists():
        print(f"  ❌ File 2 does not exist!")
        return False
    
    try:
        tensor1 = torch.load(str(file1), weights_only=False, map_location='cpu')
        tensor2 = torch.load(str(file2), weights_only=False, map_location='cpu')
        
        print(f"  Shape 1: {tensor1.shape}")
        print(f"  Shape 2: {tensor2.shape}")
        
        if tensor1.shape != tensor2.shape:
            print(f"  ❌ Shapes don't match!")
            return False
        
        # Check if tensors are exactly equal
        if torch.equal(tensor1, tensor2):
            print(f"  ✅ Tensors match exactly!")
            return True
        else:
            # Check how many elements differ
            diff = (tensor1 != tensor2).sum().item()
            total = tensor1.numel()
            pct_diff = (diff / total) * 100
            print(f"  ❌ Tensors differ: {diff:,} / {total:,} elements ({pct_diff:.6f}%)")
            
            # Show some statistics
            if diff > 0:
                diff_mask = (tensor1 != tensor2)
                print(f"  First 10 differing values:")
                diff_indices = torch.nonzero(diff_mask, as_tuple=False)[:10]
                for idx in diff_indices:
                    idx_tuple = tuple(idx.tolist())
                    print(f"    Index {idx_tuple}: {tensor1[idx_tuple].item()} vs {tensor2[idx_tuple].item()}")
            
            return False
            
    except Exception as e:
        print(f"  ❌ Error loading/comparing: {e}")
        return False


def main():
    """Main comparison function."""
    if len(sys.argv) < 3:
        print("Usage: python verify_tensor_files_match.py <dir1> <dir2>")
        print("\nExample:")
        print("  python verify_tensor_files_match.py /path/to/backup /path/to/aws_download")
        sys.exit(1)
    
    dir1 = Path(sys.argv[1])
    dir2 = Path(sys.argv[2])
    
    if not dir1.exists():
        print(f"❌ Directory 1 does not exist: {dir1}")
        sys.exit(1)
    
    if not dir2.exists():
        print(f"❌ Directory 2 does not exist: {dir2}")
        sys.exit(1)
    
    print("="*80)
    print("TENSOR FILE VERIFICATION")
    print("="*80)
    print(f"\nComparing files between:")
    print(f"  Directory 1: {dir1}")
    print(f"  Directory 2: {dir2}")
    
    files_to_check = [
        ("Y_tensor.pt", "Y tensor"),
        ("E_matrix.pt", "E matrix"),
        ("G_matrix.pt", "G matrix")
    ]
    
    all_match = True
    for filename, name in files_to_check:
        file1 = dir1 / filename
        file2 = dir2 / filename
        
        if compare_tensors(file1, file2, name):
            continue
        else:
            all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("✅ ALL FILES MATCH!")
        print("="*80)
        return 0
    else:
        print("❌ FILES DO NOT MATCH!")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

