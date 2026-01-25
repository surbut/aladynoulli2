#!/usr/bin/env python
"""
Check kappa values across prediction batches and training batches
to see if kappa is being learned per batch (which would be wrong).
"""

import torch
import numpy as np
from pathlib import Path
import glob

def check_kappa_in_checkpoints(checkpoint_dir, pattern, description):
    """Check kappa values in checkpoints matching pattern"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    
    checkpoint_files = sorted(glob.glob(str(Path(checkpoint_dir) / pattern)))
    
    if not checkpoint_files:
        print(f"  No files found matching: {pattern}")
        return None
    
    print(f"  Found {len(checkpoint_files)} checkpoint files")
    
    kappa_values = []
    for ckpt_file in checkpoint_files[:10]:  # Check first 10
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            
            # Try different ways kappa might be stored
            kappa = None
            if 'model_state_dict' in ckpt:
                if 'kappa' in ckpt['model_state_dict']:
                    kappa = ckpt['model_state_dict']['kappa'].item()
                elif 'kappa' in ckpt['model_state_dict']:
                    kappa = ckpt['model_state_dict']['kappa']
                    if torch.is_tensor(kappa):
                        kappa = kappa.item()
            elif 'kappa' in ckpt:
                kappa = ckpt['kappa']
                if torch.is_tensor(kappa):
                    kappa = kappa.item()
            
            if kappa is not None:
                kappa_values.append(kappa)
                print(f"  {Path(ckpt_file).name}: kappa = {kappa:.6f}")
            else:
                print(f"  {Path(ckpt_file).name}: kappa not found")
                
        except Exception as e:
            print(f"  {Path(ckpt_file).name}: Error loading - {e}")
    
    if kappa_values:
        kappa_array = np.array(kappa_values)
        print(f"\n  Summary:")
        print(f"    Mean kappa: {kappa_array.mean():.6f}")
        print(f"    Std kappa: {kappa_array.std():.6f}")
        print(f"    Min kappa: {kappa_array.min():.6f}")
        print(f"    Max kappa: {kappa_array.max():.6f}")
        print(f"    Range: {kappa_array.max() - kappa_array.min():.6f}")
        
        if kappa_array.std() > 0.001:
            print(f"    ⚠️  WARNING: Kappa varies across batches! (std = {kappa_array.std():.6f})")
        else:
            print(f"    ✓ Kappa is consistent across batches")
        
        return kappa_array
    else:
        print(f"  No kappa values found")
        return None

def main():
    # Check prediction batches
    pred_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/"
    pred_pattern = "model_enroll_*.pt"
    pred_kappas = check_kappa_in_checkpoints(pred_dir, pred_pattern, 
                                             "PREDICTION BATCHES (enrollment_predictions_fixedphi_correctedE_vectorized)")
    
    # Check training batches
    train_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/"
    train_pattern = "enrollment_model_W0.0001_batch_*.pt"
    train_kappas = check_kappa_in_checkpoints(train_dir, train_pattern,
                                             "TRAINING BATCHES (censor_e_batchrun_vectorized)")
    
    # Compare if both exist
    if pred_kappas is not None and train_kappas is not None:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"  Prediction batches mean kappa: {pred_kappas.mean():.6f}")
        print(f"  Training batches mean kappa: {train_kappas.mean():.6f}")
        print(f"  Difference: {abs(pred_kappas.mean() - train_kappas.mean()):.6f}")
        
        if abs(pred_kappas.mean() - train_kappas.mean()) > 0.01:
            print(f"  ⚠️  WARNING: Significant difference between prediction and training kappas!")
        else:
            print(f"  ✓ Kappas are similar between prediction and training")

if __name__ == '__main__':
    main()

