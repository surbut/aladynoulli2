#!/usr/bin/env python
"""
Check gamma values from prediction fits to compare with enrollment batch gammas
Verifies that prediction gammas are larger but z-scores are similar (like October non-fixed)
"""
import torch
import numpy as np
import glob
import os
from pathlib import Path
from scipy import stats

def load_gamma_from_checkpoint(checkpoint_path):
    """Load gamma from a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        if 'gamma' in checkpoint['model_state_dict']:
            gamma = checkpoint['model_state_dict']['gamma']
            if torch.is_tensor(gamma):
                return gamma.detach().cpu().numpy()
            return np.array(gamma)
    
    if 'gamma' in checkpoint:
        gamma = checkpoint['gamma']
        if torch.is_tensor(gamma):
            return gamma.detach().cpu().numpy()
        return np.array(gamma)
    
    return None

def analyze_gamma_stats(gamma):
    """Compute statistics for gamma"""
    gamma_abs = np.abs(gamma)
    return {
        'mean_abs': np.mean(gamma_abs),
        'std_abs': np.std(gamma_abs),
        'max_abs': np.max(gamma_abs),
        'min_abs': np.min(gamma_abs),
        'median_abs': np.median(gamma_abs),
        'p95_abs': np.percentile(gamma_abs, 95),
        'p99_abs': np.percentile(gamma_abs, 99),
        'shape': gamma.shape,
        'non_zero_count': np.count_nonzero(gamma_abs > 1e-6),
        'total_count': gamma.size
    }

def main():
    # Path to prediction output directory
    prediction_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/"
    
    # Find all model checkpoint files
    pattern = os.path.join(prediction_dir, "model_enroll_fixedphi_sex_*.pt")
    checkpoint_files = sorted(glob.glob(pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print("="*80)
    
    all_gammas = []
    batch_stats = []
    
    for i, checkpoint_path in enumerate(checkpoint_files):
        gamma = load_gamma_from_checkpoint(checkpoint_path)
        if gamma is None:
            print(f"Warning: Could not load gamma from {os.path.basename(checkpoint_path)}")
            continue
        
        stats = analyze_gamma_stats(gamma)
        all_gammas.append(gamma)
        batch_stats.append({
            'file': os.path.basename(checkpoint_path),
            'stats': stats
        })
        
        print(f"\nBatch {i+1}: {os.path.basename(checkpoint_path)}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean |γ|: {stats['mean_abs']:.6f}")
        print(f"  Std |γ|: {stats['std_abs']:.6f}")
        print(f"  Max |γ|: {stats['max_abs']:.6f}")
        print(f"  Median |γ|: {stats['median_abs']:.6f}")
        print(f"  P95 |γ|: {stats['p95_abs']:.6f}")
        print(f"  P99 |γ|: {stats['p99_abs']:.6f}")
        print(f"  Non-zero: {stats['non_zero_count']}/{stats['total_count']}")
    
    if not all_gammas:
        print("No gamma values found!")
        return
    
    # Stack and compute overall statistics
    gamma_stack = np.stack(all_gammas)
    gamma_mean = np.mean(gamma_stack, axis=0)
    gamma_std = np.std(gamma_stack, axis=0)
    
    overall_stats = analyze_gamma_stats(gamma_mean)
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS (averaged across batches)")
    print("="*80)
    print(f"Shape: {overall_stats['shape']}")
    print(f"Mean |γ|: {overall_stats['mean_abs']:.6f}")
    print(f"Std |γ|: {overall_stats['std_abs']:.6f}")
    print(f"Max |γ|: {overall_stats['max_abs']:.6f}")
    print(f"Min |γ|: {overall_stats['min_abs']:.6f}")
    print(f"Median |γ|: {overall_stats['median_abs']:.6f}")
    print(f"P95 |γ|: {overall_stats['p95_abs']:.6f}")
    print(f"P99 |γ|: {overall_stats['p99_abs']:.6f}")
    print(f"Non-zero: {overall_stats['non_zero_count']}/{overall_stats['total_count']}")
    
    # Compute SEM and z-scores for prediction gammas
    gamma_sem_pred = gamma_std / np.sqrt(len(all_gammas))
    z_scores_pred = gamma_mean / (gamma_sem_pred + 1e-10)  # Add small epsilon to avoid division by zero
    
    print("\n" + "="*80)
    print("PREDICTION GAMMA Z-SCORES")
    print("="*80)
    valid_z_pred = z_scores_pred[~np.isnan(z_scores_pred) & ~np.isinf(z_scores_pred)]
    if len(valid_z_pred) > 0:
        print(f"  Z-score range: [{np.min(valid_z_pred):.2f}, {np.max(valid_z_pred):.2f}]")
        print(f"  Mean |z-score|: {np.mean(np.abs(valid_z_pred)):.2f}")
        print(f"  Median |z-score|: {np.median(np.abs(valid_z_pred)):.2f}")
        print(f"  |z-score| > 2: {(np.abs(valid_z_pred) > 2).sum()} / {len(valid_z_pred)} ({(np.abs(valid_z_pred) > 2).sum()/len(valid_z_pred)*100:.1f}%)")
        print(f"  |z-score| > 3: {(np.abs(valid_z_pred) > 3).sum()} / {len(valid_z_pred)} ({(np.abs(valid_z_pred) > 3).sum()/len(valid_z_pred)*100:.1f}%)")
    
    # Compare with enrollment batch gammas (with penalty)
    print("\n" + "="*80)
    print("COMPARISON WITH ENROLLMENT BATCH GAMMAS (with penalty)")
    print("="*80)
    
    # Try to load enrollment batch gammas
    enrollment_batch_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/"
    enrollment_pattern = "enrollment_model_W0.0001_batch_*_*.pt"
    enrollment_checkpoints = sorted(glob.glob(os.path.join(enrollment_batch_dir, enrollment_pattern)))
    
    if enrollment_checkpoints:
        print(f"\nLoading enrollment batch gammas from {len(enrollment_checkpoints)} batches...")
        enrollment_gammas = []
        for checkpoint_path in enrollment_checkpoints:
            gamma = load_gamma_from_checkpoint(checkpoint_path)
            if gamma is not None and not np.allclose(gamma, 0):
                enrollment_gammas.append(gamma)
        
        if enrollment_gammas:
            enrollment_stack = np.stack(enrollment_gammas)
            enrollment_mean = np.mean(enrollment_stack, axis=0)
            enrollment_std = np.std(enrollment_stack, axis=0)
            enrollment_sem = enrollment_std / np.sqrt(len(enrollment_gammas))
            enrollment_z_scores = enrollment_mean / (enrollment_sem + 1e-10)
            
            enrollment_stats = analyze_gamma_stats(enrollment_mean)
            
            print(f"\nEnrollment batch gamma stats (WITH penalty):")
            print(f"  Mean |γ|: {enrollment_stats['mean_abs']:.6f}")
            print(f"  Max |γ|: {enrollment_stats['max_abs']:.6f}")
            print(f"  P95 |γ|: {enrollment_stats['p95_abs']:.6f}")
            
            print(f"\nPrediction gamma stats (NO penalty):")
            print(f"  Mean |γ|: {overall_stats['mean_abs']:.6f}")
            print(f"  Max |γ|: {overall_stats['max_abs']:.6f}")
            print(f"  P95 |γ|: {overall_stats['p95_abs']:.6f}")
            
            print(f"\nMagnitude Comparison (Prediction / Enrollment):")
            ratio_mean = overall_stats['mean_abs'] / enrollment_stats['mean_abs'] if enrollment_stats['mean_abs'] > 0 else np.inf
            ratio_max = overall_stats['max_abs'] / enrollment_stats['max_abs'] if enrollment_stats['max_abs'] > 0 else np.inf
            ratio_p95 = overall_stats['p95_abs'] / enrollment_stats['p95_abs'] if enrollment_stats['p95_abs'] > 0 else np.inf
            
            print(f"  Mean ratio: {ratio_mean:.2f}x")
            print(f"  Max ratio: {ratio_max:.2f}x")
            print(f"  P95 ratio: {ratio_p95:.2f}x")
            
            # Compare z-scores
            print(f"\nZ-Score Comparison:")
            valid_z_enroll = enrollment_z_scores[~np.isnan(enrollment_z_scores) & ~np.isinf(enrollment_z_scores)]
            valid_z_pred_comp = z_scores_pred[~np.isnan(z_scores_pred) & ~np.isinf(z_scores_pred)]
            
            if len(valid_z_enroll) > 0 and len(valid_z_pred_comp) > 0:
                # Compare distributions
                print(f"  Enrollment mean |z-score|: {np.mean(np.abs(valid_z_enroll)):.2f}")
                print(f"  Prediction mean |z-score|: {np.mean(np.abs(valid_z_pred_comp)):.2f}")
                print(f"  Ratio: {np.mean(np.abs(valid_z_pred_comp)) / np.mean(np.abs(valid_z_enroll)):.2f}x")
                
                # Compare specific associations (if shapes match)
                if gamma_mean.shape == enrollment_mean.shape:
                    z_ratio = np.abs(z_scores_pred) / (np.abs(enrollment_z_scores) + 1e-10)
                    valid_ratio = z_ratio[~np.isnan(z_ratio) & ~np.isinf(z_ratio) & (np.abs(enrollment_z_scores) > 0.01)]
                    if len(valid_ratio) > 0:
                        print(f"  Mean z-score ratio (pred/enroll): {np.mean(valid_ratio):.2f}x")
                        print(f"  Median z-score ratio: {np.median(valid_ratio):.2f}x")
                        print(f"  Z-scores within 10%: {(np.abs(valid_ratio - 1.0) < 0.1).sum()} / {len(valid_ratio)} ({(np.abs(valid_ratio - 1.0) < 0.1).sum()/len(valid_ratio)*100:.1f}%)")
            
            if ratio_mean > 1.5:
                print(f"\n✓ Prediction gammas are {ratio_mean:.1f}x larger (expected - no weight_decay penalty)")
                if len(valid_z_enroll) > 0 and len(valid_z_pred_comp) > 0:
                    z_ratio_mean = np.mean(np.abs(valid_z_pred_comp)) / np.mean(np.abs(valid_z_enroll))
                    if 0.8 < z_ratio_mean < 1.2:
                        print(f"✓ Z-scores are similar ({z_ratio_mean:.2f}x) - consistent with October non-fixed pattern!")
                    else:
                        print(f"⚠️  Z-scores differ ({z_ratio_mean:.2f}x) - may need investigation")
    else:
        print(f"\n⚠️  Could not find enrollment batch checkpoints matching pattern: {enrollment_pattern}")
        print(f"  Searched in: {enrollment_batch_dir}")

if __name__ == "__main__":
    main()

