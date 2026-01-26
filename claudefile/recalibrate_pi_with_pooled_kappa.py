#!/usr/bin/env python3
"""
Post-hoc recalibrate pi predictions using pooled kappa.

Workflow:
1. Load original predictions (pi) where kappa was learned per batch (free kappa)
2. Load model checkpoints to extract learned kappa per batch
3. Load pooled kappa (from training)
4. Recalibrate: pi_calibrated = pi_original * (kappa_pooled / kappa_learned_per_batch)
5. Save recalibrated predictions

This gives:
- Best AUC (from free kappa during prediction)
- Good calibration (from pooled kappa post-hoc adjustment)

Usage:
    python recalibrate_pi_with_pooled_kappa.py \
        --pi_dir /path/to/original/predictions \
        --model_pattern "model_enroll_fixedphi_sex_*.pt" \
        --pooled_kappa_path /path/to/pooled_kappa.pt \
        --output_dir /path/to/output
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np
import glob
from tqdm import tqdm

def extract_kappa_from_checkpoint(checkpoint_path):
    """Extract kappa value from a model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try different ways kappa might be stored
        kappa = None
        if 'model_state_dict' in checkpoint:
            if 'kappa' in checkpoint['model_state_dict']:
                kappa = checkpoint['model_state_dict']['kappa']
                if torch.is_tensor(kappa):
                    kappa = kappa.item()
        elif 'kappa' in checkpoint:
            kappa = checkpoint['kappa']
            if torch.is_tensor(kappa):
                kappa = kappa.item()
        
        return kappa
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def get_batch_indices_from_filename(filename):
    """Extract start and stop indices from filename like 'pi_enroll_fixedphi_sex_0_10000.pt'"""
    basename = Path(filename).stem
    parts = basename.split('_')
    # Find the numeric parts (start and stop)
    start = None
    stop = None
    for i, part in enumerate(parts):
        if part.isdigit():
            if start is None:
                start = int(part)
            else:
                stop = int(part)
                break
    
    if start is None or stop is None:
        return None, None
    
    return start, stop

def main():
    parser = argparse.ArgumentParser(description='Post-hoc recalibrate pi with pooled kappa')
    parser.add_argument('--pi_dir', type=str, required=True,
                       help='Directory containing original pi predictions')
    parser.add_argument('--model_pattern', type=str, required=True,
                       help='Pattern for model checkpoint files (e.g., "model_enroll_fixedphi_sex_*.pt")')
    parser.add_argument('--pooled_kappa_path', type=str, required=True,
                       help='Path to pooled_kappa.pt file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for recalibrated predictions')
    parser.add_argument('--pi_pattern', type=str, default='pi_enroll_fixedphi_sex_*.pt',
                       help='Pattern for pi files (default: pi_enroll_fixedphi_sex_*.pt)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("POST-HOC RECALIBRATION WITH POOLED KAPPA")
    print("="*80)
    
    # Load pooled kappa
    print(f"\nLoading pooled kappa from {args.pooled_kappa_path}...")
    pooled_data = torch.load(args.pooled_kappa_path, map_location='cpu', weights_only=False)
    if 'kappa' in pooled_data:
        kappa_pooled = pooled_data['kappa']
        if torch.is_tensor(kappa_pooled):
            kappa_pooled = kappa_pooled.item()
    else:
        raise ValueError(f"No 'kappa' found in {args.pooled_kappa_path}")
    
    print(f"  Pooled kappa: {kappa_pooled:.6f}")
    
    # Find all model checkpoints
    pi_dir = Path(args.pi_dir)
    model_files = sorted(glob.glob(str(pi_dir / args.model_pattern)))
    pi_files = sorted(glob.glob(str(pi_dir / args.pi_pattern)))
    
    print(f"\nFound {len(model_files)} model checkpoints")
    print(f"Found {len(pi_files)} pi prediction files")
    
    if len(model_files) == 0:
        raise ValueError(f"No model files found matching: {args.model_pattern}")
    if len(pi_files) == 0:
        raise ValueError(f"No pi files found matching: {args.pi_pattern}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each batch
    print(f"\nProcessing batches...")
    recalibrated_pis = []
    batch_info = []
    
    for model_file in tqdm(model_files, desc="Recalibrating batches"):
        # Extract kappa from model checkpoint
        kappa_learned = extract_kappa_from_checkpoint(model_file)
        if kappa_learned is None:
            print(f"  ⚠️  Skipping {Path(model_file).name}: could not extract kappa")
            continue
        
        # Find corresponding pi file
        model_basename = Path(model_file).stem
        # Extract indices from model filename
        start, stop = get_batch_indices_from_filename(model_file)
        if start is None or stop is None:
            print(f"  ⚠️  Skipping {Path(model_file).name}: could not extract indices")
            continue
        
        # Find matching pi file
        pi_file = None
        for pf in pi_files:
            pf_start, pf_stop = get_batch_indices_from_filename(pf)
            if pf_start == start and pf_stop == stop:
                pi_file = pf
                break
        
        if pi_file is None:
            print(f"  ⚠️  Skipping {Path(model_file).name}: no matching pi file found")
            continue
        
        # Load original pi
        pi_original = torch.load(pi_file, map_location='cpu', weights_only=False)
        
        # Calculate recalibration factor
        recal_factor = kappa_pooled / kappa_learned
        
        # Recalibrate
        pi_calibrated = pi_original * recal_factor
        
        # Save recalibrated pi
        output_filename = output_dir / Path(pi_file).name.replace('.pt', '_recalibrated.pt')
        torch.save(pi_calibrated, output_filename)
        
        recalibrated_pis.append(pi_calibrated)
        batch_info.append({
            'start': start,
            'stop': stop,
            'kappa_learned': kappa_learned,
            'kappa_pooled': kappa_pooled,
            'recal_factor': recal_factor,
            'pi_shape': list(pi_calibrated.shape)
        })
        
        print(f"  ✓ Batch {start}-{stop}: kappa_learned={kappa_learned:.6f}, "
              f"recal_factor={recal_factor:.6f}, shape={pi_calibrated.shape}")
    
    # Concatenate all recalibrated predictions
    if recalibrated_pis:
        print(f"\nConcatenating {len(recalibrated_pis)} batches...")
        pi_full_recalibrated = torch.cat(recalibrated_pis, dim=0)
        print(f"  Full shape: {pi_full_recalibrated.shape}")
        
        # Save full recalibrated predictions
        full_filename = output_dir / "pi_enroll_fixedphi_sex_FULL_recalibrated.pt"
        torch.save(pi_full_recalibrated, full_filename)
        print(f"  ✓ Saved full recalibrated predictions to {full_filename}")
        
        # Save batch info
        info_filename = output_dir / "recalibration_info.pt"
        torch.save({
            'batch_info': batch_info,
            'kappa_pooled': kappa_pooled,
            'total_patients': pi_full_recalibrated.shape[0],
            'n_diseases': pi_full_recalibrated.shape[1],
            'n_timepoints': pi_full_recalibrated.shape[2],
        }, info_filename)
        print(f"  ✓ Saved recalibration info to {info_filename}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("RECALIBRATION SUMMARY")
        print(f"{'='*80}")
        kappa_learned_array = np.array([b['kappa_learned'] for b in batch_info])
        recal_factor_array = np.array([b['recal_factor'] for b in batch_info])
        print(f"  Number of batches: {len(batch_info)}")
        print(f"  Pooled kappa: {kappa_pooled:.6f}")
        print(f"  Learned kappa: mean={kappa_learned_array.mean():.6f}, "
              f"std={kappa_learned_array.std():.6f}, "
              f"range=[{kappa_learned_array.min():.6f}, {kappa_learned_array.max():.6f}]")
        print(f"  Recalibration factor: mean={recal_factor_array.mean():.6f}, "
              f"std={recal_factor_array.std():.6f}, "
              f"range=[{recal_factor_array.min():.6f}, {recal_factor_array.max():.6f}]")
        print(f"  Total patients: {pi_full_recalibrated.shape[0]}")
        print("="*80)
        print("COMPLETED")
        print("="*80)
    else:
        print("✗ No batches processed successfully!")


if __name__ == '__main__':
    main()
