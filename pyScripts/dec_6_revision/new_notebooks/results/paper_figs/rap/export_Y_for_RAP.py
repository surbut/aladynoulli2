#!/usr/bin/env python3
"""
Export Y array to RDS format for R analysis on RAP.

This script loads the Y tensor and exports it in R-readable format.
"""

import torch
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from pathlib import Path
import sys

# Activate R interfaces
numpy2ri.activate()
pandas2ri.activate()

def export_Y_to_RDS(y_path, output_path=None, condense_to_binary=True):
    """
    Export Y tensor to RDS format.
    
    Args:
        y_path: Path to Y_tensor.pt file
        output_path: Output RDS file path (default: same directory as input)
        condense_to_binary: If True, also create binary (ever/never) version
    """
    print("="*60)
    print("EXPORTING Y ARRAY TO RDS FORMAT")
    print("="*60)
    
    y_path = Path(y_path)
    if not y_path.exists():
        raise FileNotFoundError(f"Y tensor not found: {y_path}")
    
    print(f"\nLoading Y tensor from: {y_path}")
    Y = torch.load(y_path, weights_only=False, map_location='cpu')
    
    if isinstance(Y, torch.Tensor):
        Y_np = Y.numpy()
    else:
        Y_np = np.array(Y)
    
    print(f"  Y shape: {Y_np.shape} (N={Y_np.shape[0]}, D={Y_np.shape[1]}, T={Y_np.shape[2]})")
    
    # Convert to R array
    Y_r = numpy2ri.numpy2rpy(Y_np)
    
    # Set output path
    if output_path is None:
        output_path = y_path.parent / "Y_tensor.rds"
    else:
        output_path = Path(output_path)
    
    # Save as RDS
    print(f"\nSaving to RDS: {output_path}")
    robjects.r['saveRDS'](Y_r, str(output_path))
    print(f"✓ Saved full Y array: {output_path}")
    
    # Also create binary (condensed) version
    if condense_to_binary:
        print("\nCreating binary (ever/never) version...")
        Y_binary = (Y_np.sum(axis=2) > 0).astype(int)  # Any occurrence across time
        Y_binary_r = numpy2ri.numpy2rpy(Y_binary)
        
        binary_output = output_path.parent / "Y_binary.rds"
        robjects.r['saveRDS'](Y_binary_r, str(binary_output))
        print(f"✓ Saved binary Y array: {binary_output}")
        print(f"  Shape: {Y_binary.shape} (N x D)")
    
    return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Y tensor to RDS format")
    parser.add_argument('--y_path', type=str, required=True,
                       help='Path to Y_tensor.pt file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output RDS file path (default: Y_tensor.rds in same directory)')
    parser.add_argument('--no-binary', action='store_true',
                       help='Skip creating binary version')
    
    args = parser.parse_args()
    
    export_Y_to_RDS(
        args.y_path,
        output_path=args.output,
        condense_to_binary=not args.no_binary
    )
    
    print("\n✅ Export complete!")
    print("   Upload the .rds file to DNAnexus:")
    print(f"   dx upload {args.output or Path(args.y_path).parent / 'Y_tensor.rds'} --path {args.project_id}:/project/to_Sarah/Signature/")


if __name__ == '__main__':
    main()

