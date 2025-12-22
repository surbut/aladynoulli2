#!/usr/bin/env python3
"""
Condense Y tensor (N x D x T) to binary matrix (N x D) for RAP analysis.

Converts the full temporal Y tensor to a simple binary matrix indicating
whether each patient ever had each disease (1 = yes, 0 = no).

This dramatically reduces file size while preserving the information needed
for rare variant-disease correlation analysis.
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

def condense_Y_tensor(y_path, output_path=None, save_csv=False):
    """
    Condense Y tensor from (N x D x T) to binary (N x D) matrix.
    
    Args:
        y_path: Path to Y_tensor.pt file
        output_path: Output RDS file path (default: Y_binary.rds in same directory)
        save_csv: Also save as CSV for easy inspection
    """
    print("="*60)
    print("CONDENSING Y TENSOR TO BINARY MATRIX")
    print("="*60)
    
    y_path = Path(y_path).expanduser()
    if not y_path.exists():
        raise FileNotFoundError(f"Y tensor not found: {y_path}")
    
    print(f"\nLoading Y tensor from: {y_path}")
    print("  (This may take a moment for large files...)")
    
    Y = torch.load(y_path, weights_only=False, map_location='cpu')
    
    if isinstance(Y, torch.Tensor):
        Y_np = Y.numpy()
    else:
        Y_np = np.array(Y)
    
    print(f"  Original Y shape: {Y_np.shape}")
    print(f"    N (patients): {Y_np.shape[0]:,}")
    print(f"    D (diseases): {Y_np.shape[1]:,}")
    print(f"    T (timepoints): {Y_np.shape[2]:,}")
    
    # Condense: Any occurrence across time = 1, else 0
    print("\nCondensing to binary (ever/never) matrix...")
    Y_binary = (Y_np.sum(axis=2) > 0).astype(int)
    
    print(f"  Condensed shape: {Y_binary.shape}")
    print(f"    N (patients): {Y_binary.shape[0]:,}")
    print(f"    D (diseases): {Y_binary.shape[1]:,}")
    
    # Calculate statistics
    total_disease_occurrences = Y_binary.sum()
    patients_with_disease = (Y_binary.sum(axis=1) > 0).sum()
    diseases_with_cases = (Y_binary.sum(axis=0) > 0).sum()
    
    print(f"\n  Statistics:")
    print(f"    Total disease occurrences: {total_disease_occurrences:,}")
    print(f"    Patients with at least one disease: {patients_with_disease:,} ({100*patients_with_disease/Y_binary.shape[0]:.1f}%)")
    print(f"    Diseases with at least one case: {diseases_with_cases:,} ({100*diseases_with_cases/Y_binary.shape[1]:.1f}%)")
    
    # Set output path
    if output_path is None:
        output_path = y_path.parent / "Y_binary.rds"
    else:
        output_path = Path(output_path).expanduser()
    
    # Convert to R array
    print(f"\nConverting to R format...")
    Y_binary_r = numpy2ri.numpy2rpy(Y_binary)
    
    # Save as RDS
    print(f"Saving to RDS: {output_path}")
    robjects.r['saveRDS'](Y_binary_r, str(output_path))
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved binary Y matrix: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Also save as CSV if requested (for inspection)
    if save_csv:
        csv_path = output_path.with_suffix('.csv')
        print(f"\nSaving CSV version (for inspection): {csv_path}")
        # Note: CSV will be huge, so we'll save a summary instead
        # Save disease names as column headers if available
        pd.DataFrame(Y_binary).to_csv(csv_path, index=False)
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"  CSV file size: {csv_size_mb:.1f} MB")
    
    # Save metadata (disease names, patient IDs if available)
    metadata_path = output_path.parent / "Y_binary_metadata.rds"
    metadata = list()
    
    # Try to find disease names
    disease_names_path = y_path.parent / "disease_names.csv"
    if disease_names_path.exists():
        print(f"\nFound disease names file: {disease_names_path}")
        disease_names_df = pd.read_csv(disease_names_path)
        # Usually first column is index, second is name
        if disease_names_df.shape[1] >= 2:
            disease_names = disease_names_df.iloc[:, 1].tolist()
            # Remove header if it's "X" or similar
            if len(disease_names) > 0 and str(disease_names[0]).lower() in ['x', 'name', 'disease']:
                disease_names = disease_names[1:]
            # Pad or trim to match D
            if len(disease_names) < Y_binary.shape[1]:
                disease_names.extend([f"Disease_{i}" for i in range(len(disease_names), Y_binary.shape[1])])
            disease_names = disease_names[:Y_binary.shape[1]]
        else:
            disease_names = [f"Disease_{i}" for i in range(Y_binary.shape[1])]
    else:
        disease_names = [f"Disease_{i}" for i in range(Y_binary.shape[1])]
    
    metadata_dict = {
        'disease_names': disease_names,
        'n_patients': Y_binary.shape[0],
        'n_diseases': Y_binary.shape[1],
        'shape': Y_binary.shape
    }
    
    # Convert to R list
    metadata_r = robjects.ListVector(metadata_dict)
    robjects.r['saveRDS'](metadata_r, str(metadata_path))
    print(f"✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*60)
    print("✅ CONDENSATION COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Upload to DNAnexus:")
    print(f"   dx upload {output_path} --path project-GJQXvjjJ1JqJ43PZ6y1XPqG9:/project/to_Sarah/Signature/")
    print(f"2. Upload metadata (optional):")
    print(f"   dx upload {metadata_path} --path project-GJQXvjjJ1JqJ43PZ6y1XPqG9:/project/to_Sarah/Signature/")
    print(f"\nThe R script will automatically find and load this file!")
    
    return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Condense Y tensor to binary matrix for RAP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python condense_Y_for_RAP.py --y_path ~/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt
  
  # With custom output path
  python condense_Y_for_RAP.py --y_path Y_tensor.pt --output Y_binary.rds
  
  # Also save CSV (for inspection, but will be large)
  python condense_Y_for_RAP.py --y_path Y_tensor.pt --save_csv
        """
    )
    parser.add_argument('--y_path', type=str, 
                       default='~/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt',
                       help='Path to Y_tensor.pt file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output RDS file path (default: Y_binary.rds in same directory)')
    parser.add_argument('--save_csv', action='store_true',
                       help='Also save as CSV (for inspection, but will be large)')
    
    args = parser.parse_args()
    
    try:
        output_file = condense_Y_tensor(
            args.y_path,
            output_path=args.output,
            save_csv=args.save_csv
        )
        print(f"\n✅ Success! Output saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

