#!/usr/bin/env python
"""
Create master checkpoint file with pooled phi and gamma from _nolr batches

This script:
- Loads phi and gamma from all _nolr batch files and pools them (mean across batches)
- Uses initial_psi from the saved file
- Creates master checkpoint file: master_for_fitting_pooled_correctedE_nolr.pt

Usage:
    python create_master_checkpoint_nolr.py
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
        pattern: Pattern like '/path/to/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt'
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


def pool_gamma_from_batches(pattern, max_batches=None):
    """
    Load and pool gamma from all batch files matching the pattern.
    
    Args:
        pattern: Pattern like '/path/to/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt'
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled gamma (mean across batches) as numpy array
    """
    all_gammas = []
    
    # Find all matching files
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, weights_only=False)
            
            # Extract gamma
            if 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
            else:
                print(f"Warning: No gamma found in {file_path}")
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(gamma):
                gamma = gamma.detach().cpu().numpy()
            elif not isinstance(gamma, np.ndarray):
                gamma = np.array(gamma)
            
            # Check if gamma is all zeros (might indicate untrained model)
            if np.allclose(gamma, 0):
                print(f"  Warning: {Path(file_path).name} has gamma=0 (possibly untrained)")
            else:
                all_gammas.append(gamma)
                print(f"Loaded gamma from {Path(file_path).name}, shape: {gamma.shape}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(all_gammas) == 0:
        raise ValueError(f"No gamma arrays loaded from pattern: {pattern}")
    
    # Stack and compute mean
    gamma_stack = np.stack(all_gammas, axis=0)  # (n_batches, P, K_total)
    gamma_pooled = np.mean(gamma_stack, axis=0)  # (P, K_total)
    
    print(f"\nPooled gamma from {len(all_gammas)} batches")
    print(f"Pooled gamma shape: {gamma_pooled.shape}")
    print(f"Pooled gamma stats: min={gamma_pooled.min():.6f}, max={gamma_pooled.max():.6f}, mean={gamma_pooled.mean():.6f}")
    print(f"Non-zero gamma values: {np.count_nonzero(gamma_pooled)}/{gamma_pooled.size}")
    
    return gamma_pooled


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


def create_master_checkpoint(phi_pooled, gamma_pooled, initial_psi, output_path, description="", healthy_psi_actual=None):
    """
    Create a master checkpoint file with pooled phi, gamma, and initial_psi.
    
    Args:
        phi_pooled: Pooled phi array (K, D, T) - may be 20 or 21 signatures
        gamma_pooled: Pooled gamma array (P, K_total)
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
    
    if isinstance(gamma_pooled, np.ndarray):
        gamma_pooled_np = gamma_pooled
    else:
        gamma_pooled_np = gamma_pooled.cpu().numpy() if torch.is_tensor(gamma_pooled) else np.array(gamma_pooled)
    
    if isinstance(initial_psi, np.ndarray):
        initial_psi_np = initial_psi
    else:
        initial_psi_np = initial_psi.cpu().numpy() if torch.is_tensor(initial_psi) else np.array(initial_psi)
    
    # Pad phi and psi with healthy state if needed (for healthy_reference=True, K_total = 21)
    if phi_pooled_np.shape[0] == 20:
        print("  Padding phi with healthy state...")
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
    gamma_pooled = torch.tensor(gamma_pooled_np, dtype=torch.float32)
    initial_psi = torch.tensor(initial_psi_np, dtype=torch.float32)
    
    # Create checkpoint in the format expected by the app
    checkpoint = {
        'model_state_dict': {
            'phi': phi_pooled,
            'psi': initial_psi,
            'gamma': gamma_pooled,  # Include pooled gamma
        },
        'description': description,
        'phi_shape': list(phi_pooled.shape),
        'psi_shape': list(initial_psi.shape),
        'gamma_shape': list(gamma_pooled.shape),
        'n_batches': len(glob.glob(pattern)) if 'pattern' in locals() else None,
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n✓ Saved master checkpoint to: {output_path}")
    print(f"  Description: {description}")
    print(f"  Phi shape: {phi_pooled.shape}")
    print(f"  Psi shape: {initial_psi.shape}")
    print(f"  Gamma shape: {gamma_pooled.shape}")


def main():
    parser = argparse.ArgumentParser(description='Create master checkpoint file with pooled phi and gamma from _nolr batches')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing initial_psi file')
    parser.add_argument('--batch_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt',
                       help='Pattern for _nolr batch files')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Output directory for master checkpoint')
    parser.add_argument('--output_name', type=str,
                       default='master_for_fitting_pooled_correctedE_nolr.pt',
                       help='Output filename for master checkpoint')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to pool (None = all)')
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Master Checkpoint from _nolr Batches (phi + gamma)")
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
    
    # Pool phi from batches
    print("\n2. Pooling phi from _nolr batches...")
    phi_pooled = pool_phi_from_batches(args.batch_pattern, args.max_batches)
    
    # Pool gamma from batches
    print("\n3. Pooling gamma from _nolr batches...")
    gamma_pooled = pool_gamma_from_batches(args.batch_pattern, args.max_batches)
    
    # Extract healthy state psi
    print("\n4. Extracting healthy state psi...")
    healthy_psi_actual = extract_healthy_state_psi(args.batch_pattern)
    
    # Create master checkpoint
    print("\n5. Creating master checkpoint...")
    output_path = output_dir / args.output_name
    create_master_checkpoint(
        phi_pooled,
        gamma_pooled,
        initial_psi,
        str(output_path),
        description="Pooled phi and gamma from _nolr batches (no lambda_reg) + initial_psi (with healthy state)",
        healthy_psi_actual=healthy_psi_actual
    )
    
    print("\n" + "="*80)
    print("Master checkpoint creation complete!")
    print("="*80)
    print(f"\nCreated file: {output_path}")
    print(f"\nThis can now be used in the realtime patient refit app!")


if __name__ == '__main__':
    main()