"""
Compare predictions from Fixed Pooled Phi vs Jointly Estimated Phi
Both using complete E matrix (full data)

This addresses: Are differences between fixed phi and joint phi concerning,
or just due to small absolute risks?
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compare_pi_predictions(pi_fixed_path, pi_joint_path, disease_names_path=None, 
                          sample_patients=1000):
    """
    Compare pi predictions from fixed pooled phi vs jointly estimated phi.
    
    Args:
        pi_fixed_path: Path to pi from fixed pooled phi (complete E)
        pi_joint_path: Path to pi from jointly estimated phi (complete E)
        disease_names_path: Path to disease names file
        sample_patients: Number of patients to sample for detailed comparison
    """
    print("="*80)
    print("COMPARING FIXED POOLED PHI vs JOINTLY ESTIMATED PHI")
    print("="*80)
    
    # Load pi predictions
    print("\n1. Loading pi predictions...")
    pi_fixed = torch.load(pi_fixed_path, weights_only=False)
    pi_joint = torch.load(pi_joint_path, weights_only=False)
    
    print(f"   Fixed phi pi shape: {pi_fixed.shape}")
    print(f"   Joint phi pi shape: {pi_joint.shape}")
    
    if pi_fixed.shape != pi_joint.shape:
        print(f"\n⚠️  WARNING: Shapes don't match!")
        print(f"   This may indicate different patient sets or timepoints")
        return None
    
    N, D, T = pi_fixed.shape
    print(f"   Patients: {N}, Diseases: {D}, Timepoints: {T}")
    
    # Load disease names if available
    if disease_names_path and Path(disease_names_path).exists():
        disease_names = torch.load(disease_names_path, weights_only=False)
        if isinstance(disease_names, dict):
            disease_names = disease_names.get('disease_names', disease_names)
    else:
        disease_names = [f"Disease_{i}" for i in range(D)]
    
    # Calculate differences
    print("\n2. Calculating differences...")
    pi_diff = torch.abs(pi_fixed - pi_joint)
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL DIFFERENCES")
    print("="*80)
    print(f"Mean absolute difference: {pi_diff.mean().item():.8f}")
    print(f"Median absolute difference: {pi_diff.median().item():.8f}")
    print(f"Max absolute difference: {pi_diff.max().item():.8f}")
    print(f"95th percentile difference: {torch.quantile(pi_diff, 0.95).item():.8f}")
    print(f"99th percentile difference: {torch.quantile(pi_diff, 0.99).item():.8f}")
    
    # Calculate relative differences (avoid division by zero)
    print("\n" + "="*80)
    print("RELATIVE DIFFERENCES (|fixed - joint| / max(fixed, joint))")
    print("="*80)
    pi_max = torch.maximum(pi_fixed, pi_joint)
    pi_max = torch.clamp(pi_max, min=1e-8)  # Avoid division by zero
    rel_diff = pi_diff / pi_max
    
    print(f"Mean relative difference: {rel_diff.mean().item():.4f} ({rel_diff.mean().item()*100:.2f}%)")
    print(f"Median relative difference: {rel_diff.median().item():.4f} ({rel_diff.median().item()*100:.2f}%)")
    print(f"Max relative difference: {rel_diff.max().item():.4f} ({rel_diff.max().item()*100:.2f}%)")
    
    # Disease-specific analysis
    print("\n" + "="*80)
    print("DISEASE-SPECIFIC ANALYSIS")
    print("="*80)
    
    disease_stats = []
    for d in range(D):
        pi_fixed_d = pi_fixed[:, d, :]
        pi_joint_d = pi_joint[:, d, :]
        
        # Mean absolute difference for this disease
        mean_abs_diff = pi_diff[:, d, :].mean().item()
        
        # Mean relative difference
        pi_max_d = torch.maximum(pi_fixed_d, pi_joint_d)
        pi_max_d = torch.clamp(pi_max_d, min=1e-8)
        rel_diff_d = (pi_diff[:, d, :] / pi_max_d).mean().item()
        
        # Mean absolute risk (to check if differences correlate with risk level)
        mean_risk_fixed = pi_fixed_d.mean().item()
        mean_risk_joint = pi_joint_d.mean().item()
        mean_risk = (mean_risk_fixed + mean_risk_joint) / 2
        
        disease_stats.append({
            'disease_idx': d,
            'disease_name': disease_names[d] if d < len(disease_names) else f"Disease_{d}",
            'mean_abs_diff': mean_abs_diff,
            'mean_rel_diff': rel_diff_d,
            'mean_risk': mean_risk,
            'mean_risk_fixed': mean_risk_fixed,
            'mean_risk_joint': mean_risk_joint,
        })
    
    df_stats = pd.DataFrame(disease_stats)
    df_stats = df_stats.sort_values('mean_abs_diff', ascending=False)
    
    print("\nTop 20 diseases by absolute difference:")
    print(df_stats.head(20)[['disease_name', 'mean_abs_diff', 'mean_rel_diff', 'mean_risk']].to_string(index=False))
    
    print("\nTop 20 diseases by relative difference:")
    df_stats_rel = df_stats.sort_values('mean_rel_diff', ascending=False)
    print(df_stats_rel.head(20)[['disease_name', 'mean_abs_diff', 'mean_rel_diff', 'mean_risk']].to_string(index=False))
    
    # Check correlation between risk level and differences
    print("\n" + "="*80)
    print("RISK LEVEL vs DIFFERENCES")
    print("="*80)
    
    # Group diseases by risk level
    low_risk = df_stats[df_stats['mean_risk'] < 0.001]
    med_risk = df_stats[(df_stats['mean_risk'] >= 0.001) & (df_stats['mean_risk'] < 0.01)]
    high_risk = df_stats[df_stats['mean_risk'] >= 0.01]
    
    print(f"\nLow risk diseases (mean risk < 0.001): {len(low_risk)}")
    if len(low_risk) > 0:
        print(f"  Mean absolute diff: {low_risk['mean_abs_diff'].mean():.8f}")
        print(f"  Mean relative diff: {low_risk['mean_rel_diff'].mean():.4f} ({low_risk['mean_rel_diff'].mean()*100:.2f}%)")
    
    print(f"\nMedium risk diseases (0.001 <= mean risk < 0.01): {len(med_risk)}")
    if len(med_risk) > 0:
        print(f"  Mean absolute diff: {med_risk['mean_abs_diff'].mean():.8f}")
        print(f"  Mean relative diff: {med_risk['mean_rel_diff'].mean():.4f} ({med_risk['mean_rel_diff'].mean()*100:.2f}%)")
    
    print(f"\nHigh risk diseases (mean risk >= 0.01): {len(high_risk)}")
    if len(high_risk) > 0:
        print(f"  Mean absolute diff: {high_risk['mean_abs_diff'].mean():.8f}")
        print(f"  Mean relative diff: {high_risk['mean_rel_diff'].mean():.4f} ({high_risk['mean_rel_diff'].mean()*100:.2f}%)")
    
    # Patient-specific analysis (sample)
    print("\n" + "="*80)
    print(f"PATIENT-SPECIFIC ANALYSIS (sampling {min(sample_patients, N)} patients)")
    print("="*80)
    
    patient_indices = np.random.choice(N, size=min(sample_patients, N), replace=False)
    
    patient_stats = []
    for p_idx in patient_indices:
        pi_fixed_p = pi_fixed[p_idx, :, :]
        pi_joint_p = pi_joint[p_idx, :, :]
        
        mean_abs_diff_p = pi_diff[p_idx, :, :].mean().item()
        max_abs_diff_p = pi_diff[p_idx, :, :].max().item()
        
        patient_stats.append({
            'patient_idx': p_idx,
            'mean_abs_diff': mean_abs_diff_p,
            'max_abs_diff': max_abs_diff_p,
        })
    
    df_patient = pd.DataFrame(patient_stats)
    print(f"\nPatient-level statistics:")
    print(f"  Mean absolute difference: {df_patient['mean_abs_diff'].mean():.8f}")
    print(f"  Max absolute difference: {df_patient['max_abs_diff'].mean():.8f}")
    print(f"  95th percentile: {df_patient['mean_abs_diff'].quantile(0.95):.8f}")
    
    # Visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Absolute difference vs Mean risk
    ax1 = axes[0, 0]
    ax1.scatter(df_stats['mean_risk'], df_stats['mean_abs_diff'], alpha=0.5, s=20)
    ax1.set_xlabel('Mean Risk (average of fixed and joint)')
    ax1.set_ylabel('Mean Absolute Difference')
    ax1.set_title('Absolute Difference vs Risk Level')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative difference vs Mean risk
    ax2 = axes[0, 1]
    ax2.scatter(df_stats['mean_risk'], df_stats['mean_rel_diff'], alpha=0.5, s=20)
    ax2.set_xlabel('Mean Risk (average of fixed and joint)')
    ax2.set_ylabel('Mean Relative Difference')
    ax2.set_title('Relative Difference vs Risk Level')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of absolute differences
    ax3 = axes[1, 0]
    ax3.hist(pi_diff.numpy().flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Absolute Difference')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Absolute Differences')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of relative differences
    ax4 = axes[1, 1]
    rel_diff_flat = rel_diff.numpy().flatten()
    rel_diff_flat = rel_diff_flat[rel_diff_flat < 1.0]  # Remove outliers
    ax4.hist(rel_diff_flat, bins=100, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Relative Difference')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Relative Differences')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fixed_vs_joint_phi_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to: fixed_vs_joint_phi_comparison.png")
    
    # Summary and interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nKey Questions:")
    print("1. Are differences concerning?")
    print(f"   → Mean absolute difference: {pi_diff.mean().item():.8f}")
    print(f"   → Mean relative difference: {rel_diff.mean().item()*100:.2f}%")
    
    if pi_diff.mean().item() < 0.0001:
        print("   ✓ Differences are very small (< 0.0001)")
    elif pi_diff.mean().item() < 0.001:
        print("   ⚠️  Differences are small but measurable (< 0.001)")
    else:
        print("   ⚠️  Differences are substantial (> 0.001)")
    
    print("\n2. Are differences larger for rare diseases (small absolute risk)?")
    if len(low_risk) > 0 and len(high_risk) > 0:
        low_rel = low_risk['mean_rel_diff'].mean()
        high_rel = high_risk['mean_rel_diff'].mean()
        if low_rel > high_rel * 1.5:
            print(f"   ✓ Yes - Low risk diseases have {low_rel/high_rel:.1f}x higher relative differences")
            print("   → This suggests differences are amplified for rare diseases")
        else:
            print(f"   → Relative differences are similar across risk levels")
    
    print("\n3. Clinical significance:")
    print("   → In prediction settings, you MUST use fixed phi (can't jointly estimate)")
    print("   → These differences show what happens when phi is fixed vs jointly estimated")
    print("   → If differences are small, fixed phi is a good approximation")
    print("   → If differences are large, may need to refine fixed phi estimation")
    
    print("\n" + "="*80)
    
    return {
        'pi_fixed': pi_fixed,
        'pi_joint': pi_joint,
        'pi_diff': pi_diff,
        'rel_diff': rel_diff,
        'df_stats': df_stats,
        'df_patient': df_patient,
    }

if __name__ == "__main__":
    # Example usage - update paths as needed
    pi_fixed_path = "/path/to/pi_fixed_pooled_completeE.pt"
    pi_joint_path = "/path/to/pi_joint_completeE.pt"
    disease_names_path = "/path/to/disease_names.pt"
    
    results = compare_pi_predictions(
        pi_fixed_path=pi_fixed_path,
        pi_joint_path=pi_joint_path,
        disease_names_path=disease_names_path,
        sample_patients=1000
    )

