#!/usr/bin/env python3
"""
Create Supplementary Figure S8: Sex-specific ROC curves vs PREVENT

This script generates S8 showing sex-specific ROC curves comparing Aladynoulli
10-year static predictions (from enrollment) vs PREVENT for ASCVD.
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')

def calculate_risks_from_enrollment(pi, Y, E, pce_df, ascvd_indices=[111, 112, 113, 114, 115, 116]):
    """
    Calculate ASCVD risks from enrollment time predictions for ALL patients.
    Uses 1-year risk (not converted to 10-year) to match evaluate_major_diseases_wsex_with_bootstrap_from_pi.
    
    Args:
        pi: Prediction tensor (N, D, T)
        Y: Outcome tensor (N, D, T)
        E: Event/censoring matrix (N, D)
        pce_df: DataFrame with enrollment ages and PREVENT scores
        ascvd_indices: List of disease indices for ASCVD
        
    Returns:
        our_risks: Array of 1-year risks from our model (for all patients)
        actual_10yr: Array of actual 10-year outcomes
        prevent_risks: Array of PREVENT risks (for all patients, prevent_impute should be available)
        patient_indices: Array of patient indices used
    """
    print("\n" + "="*80)
    print("CALCULATING RISKS FROM ENROLLMENT (ALL PATIENTS)")
    print("="*80)
    
    N, D, T = pi.shape
    
    # Skip calibration for ROC curves (AUC measures discrimination, not calibration)
    print("\nUsing uncalibrated predictions (calibration not needed for AUC comparison)")
    
    # Calculate risks for ALL patients
    our_risks = []
    actual_10yr = []
    prevent_risks = []
    patient_indices = []
    
    print(f"\nProcessing {N} patients...")
    valid_count = 0
    
    for patient_idx in range(N):
        if patient_idx >= len(pce_df):
            continue
        
        row = pce_df.iloc[patient_idx]
        
        # Get enrollment age (use 'age' column like compare_with_external_scores.py)
        if 'age' in pce_df.columns:
            enroll_age = row['age']
        elif 'age_enrolled' in pce_df.columns:
            enroll_age = row['age_enrolled']
        elif 'age_at_enroll' in pce_df.columns:
            enroll_age = row['age_at_enroll']
        else:
            continue
        
        if pd.isna(enroll_age):
            continue
        
        enroll_time = int(enroll_age - 30)
        
        # Check if we have enough follow-up
        if enroll_time < 0 or enroll_time + 10 >= T:
            continue
        
        # Get predictions at enrollment time
        pi_ascvd = pi[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk (matching evaluate_major_diseases_wsex_with_bootstrap_from_pi)
        # Use 1-year risk, not converted to 10-year (AUC is rank-based, so this matches static_10yr_results.csv)
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        our_risks.append(yearly_risk)
        
        # Get actual events over 10 years (match compare_with_external_scores.py logic)
        end_time_10yr = min(enroll_time + 10, Y.shape[2])
        event_10yr = False
        for d_idx in ascvd_indices:
            if d_idx >= Y.shape[1]:
                continue
            if np.any(Y[patient_idx, d_idx, enroll_time:end_time_10yr] > 0):
                event_10yr = True
                break
        actual_10yr.append(1 if event_10yr else 0)
        
        # Get PREVENT risk (prevent_impute should be available for everyone)
        prevent_risk = row.get('prevent_impute', np.nan)
        prevent_risks.append(prevent_risk)
        patient_indices.append(patient_idx)
        valid_count += 1
        
        if valid_count % 50000 == 0:
            print(f"  Processed {valid_count} patients...")
    
    print(f"\n✓ Processed {valid_count} valid patients")
    
    return (
        np.array(our_risks),
        np.array(actual_10yr),
        np.array(prevent_risks),
        np.array(patient_indices)
    )


def plot_sex_specific_roc(our_risks, prevent_risks, actual, sex_labels, output_path, figsize=(12, 10)):
    """
    Plot sex-specific ROC curves comparing our model vs PREVENT.
    
    Args:
        our_risks: Our model's 10-year risks
        prevent_risks: PREVENT 10-year risks
        actual: Actual 10-year outcomes
        sex_labels: Array of sex labels ('Male' or 'Female')
        output_path: Path to save figure
        figsize: Figure size
    """
    print("\n" + "="*80)
    print("PLOTTING SEX-SPECIFIC ROC CURVES")
    print("="*80)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Overall
    ax = axes[0]
    our_auc = roc_auc_score(actual, our_risks)
    prevent_auc = roc_auc_score(actual, prevent_risks)
    
    fpr_our, tpr_our, _ = roc_curve(actual, our_risks)
    fpr_prevent, tpr_prevent, _ = roc_curve(actual, prevent_risks)
    
    ax.plot(fpr_our, tpr_our, label=f'Aladynoulli (AUC={our_auc:.3f})', linewidth=2.5)
    ax.plot(fpr_prevent, tpr_prevent, label=f'PREVENT (AUC={prevent_auc:.3f})', linewidth=2.5, linestyle='--')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('All Patients', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Males
    ax = axes[1]
    male_mask = sex_labels == 'Male'
    if np.sum(male_mask) > 0:
        our_auc_m = roc_auc_score(actual[male_mask], our_risks[male_mask])
        prevent_auc_m = roc_auc_score(actual[male_mask], prevent_risks[male_mask])
        
        fpr_our_m, tpr_our_m, _ = roc_curve(actual[male_mask], our_risks[male_mask])
        fpr_prevent_m, tpr_prevent_m, _ = roc_curve(actual[male_mask], prevent_risks[male_mask])
        
        ax.plot(fpr_our_m, tpr_our_m, label=f'Aladynoulli (AUC={our_auc_m:.3f})', linewidth=2.5)
        ax.plot(fpr_prevent_m, tpr_prevent_m, label=f'PREVENT (AUC={prevent_auc_m:.3f})', linewidth=2.5, linestyle='--')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'Males (N={np.sum(male_mask):,})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No male patients', ha='center', va='center', transform=ax.transAxes)
    
    # Females
    ax = axes[2]
    female_mask = sex_labels == 'Female'
    if np.sum(female_mask) > 0:
        our_auc_f = roc_auc_score(actual[female_mask], our_risks[female_mask])
        prevent_auc_f = roc_auc_score(actual[female_mask], prevent_risks[female_mask])
        
        fpr_our_f, tpr_our_f, _ = roc_curve(actual[female_mask], our_risks[female_mask])
        fpr_prevent_f, tpr_prevent_f, _ = roc_curve(actual[female_mask], prevent_risks[female_mask])
        
        ax.plot(fpr_our_f, tpr_our_f, label=f'Aladynoulli (AUC={our_auc_f:.3f})', linewidth=2.5)
        ax.plot(fpr_prevent_f, tpr_prevent_f, label=f'PREVENT (AUC={prevent_auc_f:.3f})', linewidth=2.5, linestyle='--')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'Females (N={np.sum(female_mask):,})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No female patients', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('10-Year ASCVD Risk Prediction: Aladynoulli vs PREVENT', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved S8 figure to: {output_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Overall:")
    print(f"  Aladynoulli AUC: {our_auc:.3f}")
    print(f"  PREVENT AUC: {prevent_auc:.3f}")
    print(f"  N patients: {len(actual):,}")
    print(f"  N events: {np.sum(actual):,} ({100*np.mean(actual):.2f}%)")
    
    if np.sum(male_mask) > 0:
        print(f"\nMales:")
        print(f"  Aladynoulli AUC: {our_auc_m:.3f}")
        print(f"  PREVENT AUC: {prevent_auc_m:.3f}")
        print(f"  N patients: {np.sum(male_mask):,}")
        print(f"  N events: {np.sum(actual[male_mask]):,} ({100*np.mean(actual[male_mask]):.2f}%)")
    
    if np.sum(female_mask) > 0:
        print(f"\nFemales:")
        print(f"  Aladynoulli AUC: {our_auc_f:.3f}")
        print(f"  PREVENT AUC: {prevent_auc_f:.3f}")
        print(f"  N patients: {np.sum(female_mask):,}")
        print(f"  N events: {np.sum(actual[female_mask]):,} ({100*np.mean(actual[female_mask]):.2f}%)")


def main():
    """Main function to generate S8."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate S8: Sex-specific ROC curves vs PREVENT')
    parser.add_argument('--pi_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt',
                       help='Path to pooled pi tensor (to match static_10yr_results.csv)')
    parser.add_argument('--pce_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/ukb_pce_prevent_gail_qrisk3_combined.csv',
                       help='Path to PCE/PREVENT CSV file (use same as compare_with_external_scores.py)')
    parser.add_argument('--output_path', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s8/S8.pdf',
                       help='Output path for S8 figure')
    parser.add_argument('--ascvd_indices', type=int, nargs='+', default=[111, 112, 113, 114, 115, 116],
                       help='Disease indices for ASCVD')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING S8: SEX-SPECIFIC ROC CURVES VS PREVENT")
    print("="*80)
    print(f"PI path: {args.pi_path}")
    print(f"PCE/PREVENT file: {args.pce_path}")
    print(f"Output path: {args.output_path}")
    print(f"ASCVD indices: {args.ascvd_indices}")
    print("="*80)
    
    # Load pi predictions (use pooled file to match static_10yr_results.csv)
    print("\n" + "="*80)
    print("LOADING PREDICTIONS")
    print("="*80)
    pi_full = torch.load(args.pi_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(pi_full):
        pi_full = pi_full.numpy()
    N, D, T = pi_full.shape
    print(f"✓ Loaded pi predictions: shape {pi_full.shape}")
    
    # Cap at 400K to match generate_time_horizon_predictions.py
    MAX_PATIENTS = 400000
    if N > MAX_PATIENTS:
        print(f"Subsetting to first {MAX_PATIENTS} patients...")
        pi_full = pi_full[:MAX_PATIENTS]
        N = MAX_PATIENTS
    
    # Load Y and E
    print("\n" + "="*80)
    print("LOADING OUTCOME DATA")
    print("="*80)
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', 
                       map_location='cpu', weights_only=False)
    if torch.is_tensor(Y_full):
        Y_full = Y_full.numpy()
    
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt',
                       map_location='cpu', weights_only=False)
    if torch.is_tensor(E_full):
        E_full = E_full.numpy()
    
    print(f"✓ Loaded Y: {Y_full.shape}, E: {E_full.shape}")
    
    # Subset to match pi size
    Y_subset = Y_full[:N]
    E_subset = E_full[:N]
    print(f"✓ Subset to match pi: Y: {Y_subset.shape}, E: {E_subset.shape}")
    
    # Load PCE/PREVENT data
    print("\n" + "="*80)
    print("LOADING PCE/PREVENT DATA")
    print("="*80)
    pce_df = pd.read_csv(args.pce_path)
    print(f"✓ Loaded PCE/PREVENT data: {len(pce_df)} rows")
    print(f"  Columns: {list(pce_df.columns)[:10]}...")
    
    # Subset to match pi size
    pce_df_subset = pce_df.iloc[:N].reset_index(drop=True)
    print(f"✓ Subset to {len(pce_df_subset)} rows to match pi")
    
    # Ensure prevent_impute column exists (like compare_with_external_scores.py)
    if 'prevent_impute' not in pce_df_subset.columns:
        if 'prevent_base_ascvd_risk' in pce_df_subset.columns:
            pce_df_subset['prevent_impute'] = pce_df_subset['prevent_base_ascvd_risk']
            print("Created prevent_impute from prevent_base_ascvd_risk")
        else:
            # Impute missing values with mean
            pce_df_subset['prevent_impute'] = pce_df_subset.get('prevent_base_ascvd_risk', np.nan)
            mean_prevent = pce_df_subset['prevent_impute'].mean()
            pce_df_subset['prevent_impute'] = pce_df_subset['prevent_impute'].fillna(mean_prevent)
            print(f"Created prevent_impute column, imputed {pce_df_subset['prevent_impute'].isna().sum()} missing values")
    
    # Ensure sex column is correct
    if 'Sex' in pce_df_subset.columns:
        sex_col = 'Sex'
    elif 'sex' in pce_df_subset.columns:
        sex_col = 'sex'
    else:
        raise ValueError("Need 'Sex' or 'sex' column in PCE dataframe")
    
    # Calculate risks for ALL patients first (matching static_10yr_results.csv)
    our_risks_all, actual_all, prevent_risks_all, patient_indices_all = calculate_risks_from_enrollment(
        pi=pi_full,
        Y=Y_subset,
        E=E_subset,
        pce_df=pce_df_subset,
        ascvd_indices=args.ascvd_indices
    )
    
    # Use all patients (prevent_impute is imputed for everyone)
    valid_mask = np.isfinite(our_risks_all) & np.isfinite(prevent_risks_all) & np.isfinite(actual_all)
    our_risks = our_risks_all[valid_mask]
    prevent_risks = prevent_risks_all[valid_mask]
    actual = actual_all[valid_mask]
    sex_labels = pce_df_subset.iloc[patient_indices_all[valid_mask]][sex_col].values
    
    # Calculate Aladynoulli AUC (should match static_10yr_results.csv)
    our_auc_all = roc_auc_score(actual, our_risks)
    print(f"\n" + "="*80)
    print("ALADYNOULLI AUC (ALL PATIENTS - matching static_10yr_results.csv)")
    print("="*80)
    print(f"  Aladynoulli AUC: {our_auc_all:.4f}")
    print(f"  N patients: {len(our_risks):,}")
    print(f"  N events: {np.sum(actual):,} ({100*np.mean(actual):.2f}%)")
    print(f"\n✓ Using all patients (prevent_impute is imputed for everyone)")
    
    # Plot sex-specific ROC curves
    plot_sex_specific_roc(
        our_risks=our_risks,
        prevent_risks=prevent_risks,
        actual=actual,
        sex_labels=sex_labels,
        output_path=args.output_path,
        figsize=(15, 5)
    )
    
    print("\n" + "="*80)
    print("S8 GENERATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

