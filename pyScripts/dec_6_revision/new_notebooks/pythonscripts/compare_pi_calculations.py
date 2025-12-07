"""
Compare pi calculations:
1. Load pre-computed pi from enrollment predictions
2. Compute pi from model checkpoints using compute_pi_from_fullmode_models.py approach
3. Compare the two to ensure they match
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add path to import utils
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred

def compare_pi_calculations():
    """Compare pre-computed pi vs computed from checkpoints"""
    
    # Paths
    precomputed_pi_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/pi_fullmode_400k.pt')
    model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized')
    
    print("="*80)
    print("COMPARING PI CALCULATIONS")
    print("="*80)
    
    # Load pre-computed pi
    if precomputed_pi_path.exists():
        print(f"\n1. Loading pre-computed pi from: {precomputed_pi_path}")
        pi_precomputed = torch.load(str(precomputed_pi_path), map_location='cpu', weights_only=False)
        print(f"   Shape: {pi_precomputed.shape}")
        print(f"   Range: [{pi_precomputed.min().item():.6f}, {pi_precomputed.max().item():.6f}]")
    else:
        print(f"\n⚠️  Pre-computed pi not found: {precomputed_pi_path}")
        return
    
    # Find model files
    import re
    def extract_start_idx(filename):
        match = re.search(r'(\d+)_(\d+)\.pt$', filename.name)
        return int(match.group(1)) if match else 0
    
    model_files = list(model_dir.glob('enrollment_model_W0.0001_batch_*_*.pt'))
    model_files = [f for f in model_files if 'FULL' not in f.name]
    model_files = sorted(model_files, key=extract_start_idx)
    
    if not model_files:
        print(f"\n⚠️  No model files found in {model_dir}")
        return
    
    print(f"\n2. Found {len(model_files)} model files")
    print(f"   Computing pi from checkpoints using calculate_pi_pred...")
    
    # Compute pi from first batch for comparison
    first_model = model_files[0]
    print(f"\n   Processing first batch: {first_model.name}")
    
    checkpoint = torch.load(first_model, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Extract parameters
    lambda_ = state_dict['lambda_'].cpu()  # [N, K, T]
    phi = state_dict['phi'].cpu()  # [K, D, T]
    kappa = state_dict.get('kappa', torch.tensor(1.0))
    if torch.is_tensor(kappa):
        kappa = kappa.cpu()
        if kappa.numel() == 1:
            kappa = kappa.item()
        else:
            kappa = kappa.mean().item()
    
    print(f"   Lambda shape: {lambda_.shape}")
    print(f"   Phi shape: {phi.shape}")
    print(f"   Kappa: {kappa}")
    
    # Compute pi using utils function
    pi_computed = calculate_pi_pred(lambda_, phi, kappa)
    
    print(f"\n   Computed pi shape: {pi_computed.shape}")
    print(f"   Computed pi range: [{pi_computed.min().item():.6f}, {pi_computed.max().item():.6f}]")
    
    # Compare with pre-computed (first batch)
    pi_precomputed_batch = pi_precomputed[:len(pi_computed)]
    
    print(f"\n3. Comparing pre-computed vs computed (first batch):")
    print(f"   Pre-computed shape: {pi_precomputed_batch.shape}")
    print(f"   Computed shape: {pi_computed.shape}")
    
    # Check if shapes match
    if pi_precomputed_batch.shape == pi_computed.shape:
        # Compute differences
        diff = torch.abs(pi_precomputed_batch - pi_computed)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\n   Max absolute difference: {max_diff:.8f}")
        print(f"   Mean absolute difference: {mean_diff:.8f}")
        
        # Check if they're essentially the same (within numerical precision)
        if max_diff < 1e-5:
            print(f"\n   ✓ MATCH! Differences are within numerical precision (< 1e-5)")
        elif max_diff < 1e-3:
            print(f"\n   ⚠️  Close match (differences < 1e-3), may be due to different batches or numerical precision")
        else:
            print(f"\n   ⚠️  Significant differences detected! May indicate different computation methods")
            
        # Show sample values
        print(f"\n   Sample comparison (first patient, first disease, first timepoint):")
        print(f"     Pre-computed: {pi_precomputed_batch[0, 0, 0].item():.8f}")
        print(f"     Computed:     {pi_computed[0, 0, 0].item():.8f}")
        print(f"     Difference:   {diff[0, 0, 0].item():.8f}")
    else:
        print(f"\n   ⚠️  Shape mismatch!")
        print(f"   Pre-computed: {pi_precomputed_batch.shape}")
        print(f"   Computed:     {pi_computed.shape}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    compare_pi_calculations()

