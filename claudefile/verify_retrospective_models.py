#!/usr/bin/env python3
"""
Verify that enrollment_retrospective_full models were created with joint phi estimation.

Checks:
1. Whether phi is saved in checkpoint (joint estimation saves phi)
2. Whether phi differs between batches (joint = different, fixed = same)
3. Model class used (from args if available)
"""

import torch
import numpy as np
from pathlib import Path
import glob

def verify_retrospective_models(model_dir):
    """Verify models in enrollment_retrospective_full directory."""
    
    model_dir = Path(model_dir)
    pattern = str(model_dir / 'enrollment_model_W0.0001_batch_*_*.pt')
    model_files = sorted(glob.glob(pattern))
    
    if not model_files:
        print(f"❌ No model files found matching: {pattern}")
        return
    
    print(f"Found {len(model_files)} model files")
    print(f"Checking first 3 batches...\n")
    
    phis = []
    has_phi = []
    model_classes = []
    
    for i, model_file in enumerate(model_files[:3]):
        print(f"{'='*60}")
        print(f"Batch {i+1}: {Path(model_file).name}")
        print(f"{'='*60}")
        
        try:
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            # Check 1: Does it have phi saved?
            if 'phi' in checkpoint:
                phi = checkpoint['phi']
                if torch.is_tensor(phi):
                    phi = phi.cpu().numpy()
                phis.append(phi)
                has_phi.append(True)
                print(f"✓ Has 'phi' key: YES")
                print(f"  Phi shape: {phi.shape}")
                print(f"  Phi mean: {phi.mean():.6f}, std: {phi.std():.6f}")
            else:
                has_phi.append(False)
                print(f"✗ Has 'phi' key: NO")
            
            # Check 2: Check model_state_dict for phi
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'phi' in state_dict:
                    phi_from_state = state_dict['phi']
                    if torch.is_tensor(phi_from_state):
                        phi_from_state = phi_from_state.cpu().numpy()
                    print(f"✓ Has 'phi' in model_state_dict: YES")
                    print(f"  Phi shape: {phi_from_state.shape}")
                    
                    # Compare with top-level phi if both exist
                    if 'phi' in checkpoint:
                        if np.allclose(phi, phi_from_state):
                            print(f"  ✓ Top-level phi matches model_state_dict phi")
                        else:
                            print(f"  ⚠️  Top-level phi differs from model_state_dict phi")
                else:
                    print(f"✗ Has 'phi' in model_state_dict: NO")
            
            # Check 3: Check args for script info
            if 'args' in checkpoint:
                args = checkpoint['args']
                print(f"\nModel arguments:")
                if 'start_index' in args:
                    print(f"  start_index: {args['start_index']}")
                if 'end_index' in args:
                    print(f"  end_index: {args['end_index']}")
                if 'num_epochs' in args:
                    print(f"  num_epochs: {args['num_epochs']}")
            
            # Check 4: Check for clusters (joint estimation uses clusters)
            if 'clusters' in checkpoint:
                clusters = checkpoint['clusters']
                if torch.is_tensor(clusters):
                    clusters = clusters.cpu().numpy()
                print(f"\n✓ Has 'clusters': YES")
                print(f"  Clusters shape: {clusters.shape}")
            else:
                print(f"\n✗ Has 'clusters': NO")
            
            print()
            
        except Exception as e:
            print(f"✗ ERROR loading {model_file}: {e}\n")
            continue
    
    # Check 5: Compare phi between batches
    print(f"{'='*60}")
    print("COMPARING PHI BETWEEN BATCHES")
    print(f"{'='*60}")
    
    if len(phis) >= 2:
        phi1, phi2 = phis[0], phis[1]
        
        if phi1.shape == phi2.shape:
            diff = np.abs(phi1 - phi2)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"\nComparing Batch 0 vs Batch 1:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Are they identical? {np.allclose(phi1, phi2)}")
            
            if np.allclose(phi1, phi2):
                print(f"\n⚠️  WARNING: Phi values are IDENTICAL between batches!")
                print(f"   This suggests FIXED PHI (not joint estimation)")
            else:
                print(f"\n✓ Phi values DIFFER between batches")
                print(f"   This confirms JOINT ESTIMATION (phi learned per batch)")
        else:
            print(f"⚠️  Cannot compare: phi shapes differ ({phi1.shape} vs {phi2.shape})")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if all(has_phi):
        print("✓ All batches have 'phi' saved")
    else:
        print("✗ Some batches missing 'phi'")
    
    if len(phis) >= 2 and not np.allclose(phis[0], phis[1]):
        print("✓ Phi differs between batches → JOINT ESTIMATION CONFIRMED")
    elif len(phis) >= 2:
        print("✗ Phi is identical between batches → FIXED PHI (not joint)")
    
    print(f"\n{'='*60}")

if __name__ == '__main__':
    model_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full'
    verify_retrospective_models(model_dir)

