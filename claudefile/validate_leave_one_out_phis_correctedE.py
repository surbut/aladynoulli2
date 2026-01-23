#!/usr/bin/env python3
"""
Validate phi values in leave-one-out checkpoints.

Checks for:
- NaN or inf values
- Extreme values
- Shape consistency
- Comparison to overall pooled phi
- Statistics (mean, std, min, max)
"""

import torch
import numpy as np
from pathlib import Path
import glob
import argparse


def validate_phi(phi, name="phi"):
    """Validate a phi array for common issues."""
    issues = []
    warnings = []
    
    # Convert to numpy if tensor
    if torch.is_tensor(phi):
        phi_np = phi.detach().cpu().numpy()
    else:
        phi_np = np.array(phi)
    
    # Check 1: NaN values
    nan_count = np.isnan(phi_np).sum()
    if nan_count > 0:
        issues.append(f"❌ {nan_count} NaN values found!")
    else:
        warnings.append("✓ No NaN values")
    
    # Check 2: Inf values
    inf_count = np.isinf(phi_np).sum()
    if inf_count > 0:
        issues.append(f"❌ {inf_count} Inf values found!")
    else:
        warnings.append("✓ No Inf values")
    
    # Check 3: Extreme values
    finite_phi = phi_np[np.isfinite(phi_np)]
    if len(finite_phi) > 0:
        abs_max = np.abs(finite_phi).max()
        abs_mean = np.abs(finite_phi).mean()
        abs_std = np.abs(finite_phi).std()
        
        if abs_max > 100:
            issues.append(f"⚠️  Very large values: max(abs) = {abs_max:.2f}")
        elif abs_max > 10:
            warnings.append(f"⚠️  Large values: max(abs) = {abs_max:.2f}")
        else:
            warnings.append(f"✓ Reasonable range: max(abs) = {abs_max:.2f}")
        
        stats = {
            'mean': np.mean(finite_phi),
            'std': np.std(finite_phi),
            'min': np.min(finite_phi),
            'max': np.max(finite_phi),
            'median': np.median(finite_phi),
            'abs_max': abs_max,
            'abs_mean': abs_mean,
        }
    else:
        stats = None
        issues.append("❌ No finite values!")
    
    return {
        'phi_np': phi_np,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'shape': phi_np.shape
    }


def main():
    parser = argparse.ArgumentParser(description='Validate leave-one-out checkpoint phi values')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Data directory containing checkpoints')
    parser.add_argument('--check_all', action='store_true',
                       help='Check all leave-one-out checkpoints (0-39)')
    parser.add_argument('--check_batch', type=int, nargs='+',
                       help='Check specific batch indices (e.g., --check_batch 0 1 2)')
    parser.add_argument('--compare_to_master', action='store_true',
                       help='Compare to overall pooled phi')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("="*80)
    print("VALIDATING LEAVE-ONE-OUT CHECKPOINT PHI VALUES")
    print("="*80)
    
    # Determine which batches to check
    if args.check_all:
        batch_indices = list(range(40))
    elif args.check_batch:
        batch_indices = args.check_batch
    else:
        # Default: check first 5 batches
        batch_indices = list(range(5))
        print(f"\n⚠️  No batches specified. Checking first 5 batches by default.")
        print(f"   Use --check_all to check all 40, or --check_batch N to check specific batches.")
    
    print(f"\nChecking batches: {batch_indices}")
    print(f"Data directory: {data_dir}\n")
    
    # Load master checkpoint for comparison if requested
    master_phi = None
    if args.compare_to_master:
        master_path = data_dir / 'master_for_fitting_pooled_correctedE.pt'
        if master_path.exists():
            print(f"Loading master checkpoint for comparison: {master_path.name}")
            master_ckpt = torch.load(str(master_path), map_location='cpu', weights_only=False)
            if 'model_state_dict' in master_ckpt and 'phi' in master_ckpt['model_state_dict']:
                master_phi = master_ckpt['model_state_dict']['phi']
            elif 'phi' in master_ckpt:
                master_phi = master_ckpt['phi']
            
            if master_phi is not None:
                if torch.is_tensor(master_phi):
                    master_phi = master_phi.detach().cpu().numpy()
                print(f"✓ Master phi shape: {master_phi.shape}\n")
            else:
                print("⚠️  Could not extract phi from master checkpoint\n")
        else:
            print(f"⚠️  Master checkpoint not found: {master_path}\n")
    
    # Validate each checkpoint
    all_results = {}
    all_issues = []
    
    for batch_idx in batch_indices:
        checkpoint_path = data_dir / f'master_for_fitting_pooled_correctedE_exclude_batch_{batch_idx}.pt'
        
        print(f"{'='*80}")
        print(f"BATCH {batch_idx}: {checkpoint_path.name}")
        print(f"{'='*80}")
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found!")
            all_issues.append(f"Batch {batch_idx}: Checkpoint not found")
            continue
        
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            # Extract phi
            phi = None
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                print("❌ No phi found in checkpoint!")
                all_issues.append(f"Batch {batch_idx}: No phi found")
                continue
            
            # Validate phi
            result = validate_phi(phi, f"Batch {batch_idx}")
            all_results[batch_idx] = result
            
            print(f"Shape: {result['shape']}")
            print()
            
            # Print warnings
            for warning in result['warnings']:
                print(f"  {warning}")
            
            # Print issues
            for issue in result['issues']:
                print(f"  {issue}")
                all_issues.append(f"Batch {batch_idx}: {issue}")
            
            # Print statistics
            if result['stats']:
                stats = result['stats']
                print(f"\n  Statistics:")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Std:  {stats['std']:.6f}")
                print(f"    Min:  {stats['min']:.6f}")
                print(f"    Max:  {stats['max']:.6f}")
                print(f"    Median: {stats['median']:.6f}")
                print(f"    Max(abs): {stats['abs_max']:.6f}")
            
            # Compare to master if available
            if master_phi is not None and result['phi_np'] is not None:
                if result['phi_np'].shape == master_phi.shape:
                    diff = np.abs(result['phi_np'] - master_phi)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    
                    # Relative difference
                    abs_master = np.abs(master_phi)
                    rel_diff = diff / (abs_master + 1e-10)
                    max_rel_diff = np.max(rel_diff)
                    mean_rel_diff = np.mean(rel_diff)
                    
                    print(f"\n  Comparison to master:")
                    print(f"    Max absolute diff: {max_diff:.6f}")
                    print(f"    Mean absolute diff: {mean_diff:.6f}")
                    print(f"    Max relative diff: {max_rel_diff:.6f}")
                    print(f"    Mean relative diff: {mean_rel_diff:.6f}")
                    
                    if mean_diff < 1e-3 and max_rel_diff < 0.02:
                        print(f"    ✓ Good agreement with master")
                    elif mean_diff < 1e-2 and max_rel_diff < 0.05:
                        print(f"    ⚠️  Moderate difference from master")
                    else:
                        print(f"    ❌ Large difference from master")
                        all_issues.append(f"Batch {batch_idx}: Large difference from master (mean={mean_diff:.6f}, rel={max_rel_diff:.6f})")
                else:
                    print(f"\n  ⚠️  Shape mismatch with master: {result['phi_np'].shape} vs {master_phi.shape}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            all_issues.append(f"Batch {batch_idx}: Error - {e}")
            print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Checked {len(all_results)} checkpoints")
    
    if len(all_issues) == 0:
        print("✓ All checkpoints passed validation!")
    else:
        print(f"⚠️  Found {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"  - {issue}")
    
    # Check shape consistency
    if len(all_results) > 1:
        shapes = [r['shape'] for r in all_results.values()]
        if len(set(shapes)) == 1:
            print(f"\n✓ All checkpoints have consistent shape: {shapes[0]}")
        else:
            print(f"\n❌ Shape inconsistency detected!")
            for batch_idx, shape in zip(all_results.keys(), shapes):
                print(f"  Batch {batch_idx}: {shape}")
    
    # Check statistics consistency
    if len(all_results) > 1 and all(r['stats'] is not None for r in all_results.values()):
        means = [r['stats']['mean'] for r in all_results.values()]
        stds = [r['stats']['std'] for r in all_results.values()]
        print(f"\nStatistics across checkpoints:")
        print(f"  Mean range: [{np.min(means):.6f}, {np.max(means):.6f}]")
        print(f"  Std range:  [{np.min(stds):.6f}, {np.max(stds):.6f}]")
        
        if np.max(means) - np.min(means) < 0.01:
            print(f"  ✓ Means are consistent")
        else:
            print(f"  ⚠️  Means vary significantly")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()




