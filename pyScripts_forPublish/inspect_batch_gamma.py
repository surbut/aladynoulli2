#!/usr/bin/env python3
"""
Inspect gamma values from a single batch checkpoint to debug averaging issues.

Usage:
    python inspect_batch_gamma.py <checkpoint_path>
"""

import sys
import torch
import numpy as np

def inspect_batch_gamma(checkpoint_path):
    """Inspect gamma values from a batch checkpoint."""
    print(f"Inspecting gamma from: {checkpoint_path}")
    print("="*80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Print all keys in checkpoint
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # Try to extract gamma
        gamma = None
        gamma_location = None
        
        if 'model_state_dict' in checkpoint:
            print("\nmodel_state_dict keys:")
            for key in checkpoint['model_state_dict'].keys():
                print(f"  - {key}")
                if 'gamma' in key.lower():
                    print(f"    ^ Found gamma-related key!")
            
            if 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
                gamma_location = "model_state_dict['gamma']"
        
        if gamma is None and 'gamma' in checkpoint:
            gamma = checkpoint['gamma']
            gamma_location = "checkpoint['gamma']"
        
        if gamma is None:
            print("\n❌ ERROR: Could not find gamma in checkpoint!")
            return
        
        print(f"\n✓ Found gamma at: {gamma_location}")
        
        # Convert to numpy if tensor
        if torch.is_tensor(gamma):
            print(f"  Type: torch.Tensor")
            print(f"  Requires grad: {gamma.requires_grad if hasattr(gamma, 'requires_grad') else 'N/A'}")
            gamma_np = gamma.detach().cpu().numpy()
        else:
            print(f"  Type: {type(gamma)}")
            gamma_np = np.array(gamma)
        
        print(f"  Shape: {gamma_np.shape}")
        print(f"  Dtype: {gamma_np.dtype}")
        
        # Statistics
        print(f"\nGamma Statistics:")
        print(f"  Min: {np.min(gamma_np):.6f}")
        print(f"  Max: {np.max(gamma_np):.6f}")
        print(f"  Mean: {np.mean(gamma_np):.6f}")
        print(f"  Std: {np.std(gamma_np):.6f}")
        print(f"  Median: {np.median(gamma_np):.6f}")
        
        # Check for zeros
        total_elements = gamma_np.size
        zero_count = np.count_nonzero(gamma_np == 0)
        near_zero_count = np.count_nonzero(np.abs(gamma_np) < 1e-6)
        non_zero_count = np.count_nonzero(gamma_np)
        
        print(f"\nZero Analysis:")
        print(f"  Total elements: {total_elements}")
        print(f"  Exactly zero: {zero_count} ({100*zero_count/total_elements:.2f}%)")
        print(f"  Near zero (<1e-6): {near_zero_count} ({100*near_zero_count/total_elements:.2f}%)")
        print(f"  Non-zero: {non_zero_count} ({100*non_zero_count/total_elements:.2f}%)")
        
        # Show some sample values
        print(f"\nSample Values (first 5x5):")
        if len(gamma_np.shape) == 2:
            print(gamma_np[:5, :5])
        elif len(gamma_np.shape) == 1:
            print(gamma_np[:10])
        else:
            print(f"  Shape {gamma_np.shape} - showing flattened first 10:")
            print(gamma_np.flatten()[:10])
        
        # Show largest absolute values
        print(f"\nTop 10 Largest Absolute Values:")
        abs_gamma = np.abs(gamma_np)
        flat_indices = np.argsort(abs_gamma.flatten())[-10:][::-1]
        if len(gamma_np.shape) == 2:
            P, K = gamma_np.shape
            for idx in flat_indices:
                p_idx = idx // K
                k_idx = idx % K
                val = gamma_np[p_idx, k_idx]
                print(f"  PRS {p_idx}, Signature {k_idx}: {val:.6f}")
        else:
            for idx in flat_indices:
                val = gamma_np.flatten()[idx]
                print(f"  Index {idx}: {val:.6f}")
        
        # Check if all zeros
        if np.allclose(gamma_np, 0):
            print(f"\n⚠️  WARNING: Gamma is all zeros! This batch may not have been trained.")
        else:
            print(f"\n✓ Gamma has non-zero values")
        
        return gamma_np
        
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_batch_gamma.py <checkpoint_path>")
        print("\nExample:")
        print("  python inspect_batch_gamma.py '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_batch_gamma(checkpoint_path)

