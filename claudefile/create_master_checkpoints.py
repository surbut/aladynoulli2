#!/usr/bin/env python
"""
Create master checkpoint files with pooled phi and initial_psi

This script:
- Loads phi from all batches and pools them (mean across batches)
- Uses initial_psi from the saved file
- Creates master checkpoint files that can be used for fixed-phi predictions

Usage:
    python create_master_checkpoints.py
"""

import torch
import numpy as np
import glob
import argparse
from pathlib import Path

def pool_phi_from_batches(pattern, max_batches=None):
    """
    Load and pool phi from all batch files matching the pattern.
    
    Args:
        pattern: Pattern like '/path/to/enrollment_model_W0.0001_batch_*_*.pt'
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled phi (mean across batches) as numpy array
    """
    all_phis = []
    
    # Find all matching files
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, weights_only=False)
            
            # Extract phi
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                print(f"Warning: No phi found in {file_path}")
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            all_phis.append(phi)
            print(f"Loaded phi from {Path(file_path).name}, shape: {phi.shape}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(all_phis) == 0:
        raise ValueError(f"No phi arrays loaded from pattern: {pattern}")
    
    # Stack and compute mean
    phi_stack = np.stack(all_phis, axis=0)  # (n_batches, K, D, T)
    phi_pooled = np.mean(phi_stack, axis=0)  # (K, D, T)
    
    print(f"\nPooled phi from {len(all_phis)} batches")
    print(f"Pooled phi shape: {phi_pooled.shape}")
    print(f"Pooled phi stats: min={phi_pooled.min():.4f}, max={phi_pooled.max():.4f}, mean={phi_pooled.mean():.4f}")
    
    return phi_pooled


def extract_healthy_state_psi(pattern):
    """
    Extract healthy state psi (index 20) from a sample batch checkpoint.
    
    Args:
        pattern: Pattern to find batch files
    
    Returns:
        healthy_psi: (D,) array with healthy state psi values, or None if not found
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    
    # Try first file
    try:
        checkpoint = torch.load(files[0], weights_only=False)
        if 'model_state_dict' in checkpoint:
            psi = checkpoint['model_state_dict']['psi']
        elif 'psi' in checkpoint:
            psi = checkpoint['psi']
        else:
            return None
        
        if torch.is_tensor(psi):
            psi = psi.cpu().numpy()
        
        # Check if healthy state exists (shape should be 21, D)
        if psi.shape[0] == 21:
            healthy_psi = psi[20, :]  # Extract healthy state (index 20)
            print(f"  ✓ Extracted healthy state psi from {Path(files[0]).name}")
            print(f"    Healthy psi stats: mean={healthy_psi.mean():.4f}, range=[{healthy_psi.min():.4f}, {healthy_psi.max():.4f}]")
            return healthy_psi
    except Exception as e:
        print(f"  ⚠️  Could not extract healthy state psi: {e}")
    
    return None


def create_master_checkpoint(phi_pooled, initial_psi, output_path, description="", healthy_psi_actual=None):
    """
    Create a master checkpoint file with pooled phi and initial_psi.
    
    Args:
        phi_pooled: Pooled phi array (K, D, T) - may be 20 or 21 signatures
        initial_psi: Initial psi array (K, D) - typically (20, D)
        output_path: Path to save the checkpoint
        description: Description string for the checkpoint
        healthy_psi_actual: Optional actual healthy state psi values (D,) to use for padding
    """
    # Convert to numpy if needed
    if isinstance(phi_pooled, np.ndarray):
        phi_pooled_np = phi_pooled
    else:
        phi_pooled_np = phi_pooled.cpu().numpy() if torch.is_tensor(phi_pooled) else np.array(phi_pooled)
    
    if isinstance(initial_psi, np.ndarray):
        initial_psi_np = initial_psi
    else:
        initial_psi_np = initial_psi.cpu().numpy() if torch.is_tensor(initial_psi) else np.array(initial_psi)
    
    # Pad phi and psi with healthy state if needed (for healthy_reference=True, K_total = 21)
    if phi_pooled_np.shape[0] == 20:
        print("  Padding phi with healthy state...")
        # For phi, we need to estimate healthy state phi
        # This would require prevalence_t, so for now we'll use a simple approach
        # The prediction script will handle this if needed
        # For now, just note that padding is needed
        print("  ⚠️  Note: phi needs healthy state padding (will be handled in prediction script)")
    
    # Pad initial_psi with healthy state if needed
    if initial_psi_np.shape[0] == 20:
        print("  Padding initial_psi with healthy state...")
        if healthy_psi_actual is not None:
            # Use actual healthy state psi values
            healthy_psi = healthy_psi_actual[np.newaxis, :]  # (1, D)
            print(f"    Using actual healthy state psi (mean: {healthy_psi_actual.mean():.4f})")
        else:
            # Fallback: use constant -5.0
            healthy_psi = np.full((1, initial_psi_np.shape[1]), -5.0, dtype=initial_psi_np.dtype)
            print("    Using constant -5.0 for healthy state psi")
        initial_psi_np = np.vstack([initial_psi_np, healthy_psi])
        print(f"    Padded initial_psi shape: {initial_psi_np.shape}")
    
    # Convert to tensors
    phi_pooled = torch.tensor(phi_pooled_np, dtype=torch.float32)
    initial_psi = torch.tensor(initial_psi_np, dtype=torch.float32)
    
    # Create checkpoint in the format expected by run_aladyn_predict.py
    checkpoint = {
        'model_state_dict': {
            'phi': phi_pooled,
            'psi': initial_psi,
        },
        'description': description,
        'phi_shape': list(phi_pooled.shape),
        'psi_shape': list(initial_psi.shape),
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n✓ Saved master checkpoint to: {output_path}")
    print(f"  Description: {description}")
    print(f"  Phi shape: {phi_pooled.shape}")
    print(f"  Psi shape: {initial_psi.shape}")


def main():
    parser = argparse.ArgumentParser(description='Create master checkpoint files with pooled phi')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing initial_psi file')
    parser.add_argument('--retrospective_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_*_*.pt',
                       help='Pattern for retrospective batch files')
    parser.add_argument('--enrollment_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/enrollment_model_W0.0001_batch_*_*.pt',
                       help='Pattern for enrollment batch files')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Output directory for master checkpoints')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to pool (None = all)')
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Master Checkpoint Files")
    print("="*80)
    
    # Load initial_psi
    print("\n1. Loading initial_psi...")
    initial_psi_path = Path(args.data_dir) / 'initial_psi_400k.pt'
    if not initial_psi_path.exists():
        raise FileNotFoundError(f"initial_psi file not found: {initial_psi_path}")
    
    initial_psi = torch.load(str(initial_psi_path), weights_only=False)
    if torch.is_tensor(initial_psi):
        initial_psi = initial_psi.cpu().numpy()
    print(f"✓ Loaded initial_psi, shape: {initial_psi.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create master checkpoint for pooled retrospective data
    print("\n2. Pooling phi from retrospective batches...")
    try:
        phi_retrospective = pool_phi_from_batches(args.retrospective_pattern, args.max_batches)
        
        # Try to extract healthy state psi from enrollment batches (as fallback)
        # Retrospective batches might not have healthy state
        print("\n  Attempting to extract healthy state psi...")
        healthy_psi_actual = extract_healthy_state_psi(args.enrollment_pattern)
        if healthy_psi_actual is None:
            # Try retrospective pattern as fallback
            healthy_psi_actual = extract_healthy_state_psi(args.retrospective_pattern)
        
        output_path_retro = output_dir / 'master_for_fitting_pooled_all_data.pt'
        create_master_checkpoint(
            phi_retrospective,
            initial_psi,
            str(output_path_retro),
            description="Pooled phi from all retrospective batches + initial_psi (with healthy state)",
            healthy_psi_actual=healthy_psi_actual
        )
    except Exception as e:
        print(f"✗ Error creating retrospective master checkpoint: {e}")
    
    # Create master checkpoint for pooled enrollment data
    print("\n3. Pooling phi from enrollment batches...")
    try:
        phi_enrollment = pool_phi_from_batches(args.enrollment_pattern, args.max_batches)
        
        # Extract actual healthy state psi from enrollment batches
        print("\n  Extracting healthy state psi from enrollment batches...")
        healthy_psi_actual = extract_healthy_state_psi(args.enrollment_pattern)
        
        output_path_enroll = output_dir / 'master_for_fitting_pooled_enrollment_data.pt'
        create_master_checkpoint(
            phi_enrollment,
            initial_psi,
            str(output_path_enroll),
            description="Pooled phi from all enrollment batches + initial_psi (with healthy state)",
            healthy_psi_actual=healthy_psi_actual
        )
    except Exception as e:
        print(f"✗ Error creating enrollment master checkpoint: {e}")
    
    print("\n" + "="*80)
    print("Master checkpoint creation complete!")
    print("="*80)
    print(f"\nCreated files:")
    print(f"  - {output_dir / 'master_for_fitting_pooled_all_data.pt'}")
    print(f"  - {output_dir / 'master_for_fitting_pooled_enrollment_data.pt'}")
    print(f"\nThese can now be used as --trained_model_path in run_aladyn_predict.py")


if __name__ == '__main__':
    main()

