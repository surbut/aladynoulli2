#!/usr/bin/env python3
"""
Compare Aladynoulli predictions with external risk scores (PCE, PREVENT, Gail, QRISK3).

This script evaluates external risk scores on the same 400K population and compares
AUCs with our model's predictions.

Usage:
    python compare_with_external_scores.py --approach pooled_retrospective
    python compare_with_external_scores.py --approach pooled_enrollment
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi

# Load essentials (disease names, etc.)
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=100):
    """Calculate AUC with bootstrap confidence intervals"""
    # Note: Seed should be set once at the start of main() for reproducibility
    aucs = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) > 1:
            fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_pred[indices])
            bootstrap_auc = auc(fpr_boot, tpr_boot)
            aucs.append(bootstrap_auc)
    if aucs:
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)
        auc_mean = np.mean(aucs)
    else:
        ci_lower = ci_upper = auc_mean = np.nan
    return auc_mean, ci_lower, ci_upper

def evaluate_ascvd_comparison(pi_full, Y_full, E_full, pce_df, disease_names, approach_name, qrisk3_df=None, n_bootstraps=100):
    """
    Compare Aladynoulli with PCE (10-year), PREVENT (10-year), and QRISK3 (10-year) for ASCVD.
    
    Note: We do NOT compare 30-year predictions because:
    1. Most patients don't have 30 years of actual follow-up data
    2. Early timepoints (1970s/1980s) have poor ICD code data quality
    3. Making predictions at enrollment (e.g., age 40) and evaluating 30 years later requires
       follow-up data that doesn't exist for most patients
    """
    print("\n" + "="*80)
    print("ASCVD COMPARISON: Aladynoulli vs PCE (10yr), PREVENT (10yr), and QRISK3 (10yr)")
    print("="*80)
    
    ascvd_indices = []
    ascvd_diseases = [
        'Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease',
        'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'
    ]
    for disease in ascvd_diseases:
        indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
        ascvd_indices.extend(indices)
    ascvd_indices = list(set(ascvd_indices))
    
    # Cap at 400K
    MAX_PATIENTS = 400000
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    pce_df = pce_df.iloc[:MAX_PATIENTS].reset_index(drop=True)
    
    # Apply LOESS calibration like old compare_with_pce/compare_with_prevent functions
    print("\nApplying LOESS calibration to predictions...")
    # Get mean risks across patients for calibration (memory-efficient)
    predicted_risk_2d = pi_full.mean(axis=0).numpy()  # Shape: [D, T]
    observed_risk_2d = Y_full.mean(axis=0).numpy()  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration in batches to avoid memory explosion
    # Process calibration in chunks to avoid creating multiple full copies
    print("  Calibrating predictions in batches...")
    batch_size = 10000  # Process 10K patients at a time
    n_patients = pi_full.shape[0]
    pi_calibrated_list = []
    
    for start_idx in range(0, n_patients, batch_size):
        end_idx = min(start_idx + batch_size, n_patients)
        pi_batch = pi_full[start_idx:end_idx].numpy()  # Only convert batch to numpy
        pi_batch_calibrated = np.interp(pi_batch.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi_batch.shape)
        pi_calibrated_list.append(torch.tensor(pi_batch_calibrated, dtype=torch.float32))
        del pi_batch, pi_batch_calibrated  # Free memory immediately
    
    # Concatenate batches
    pi_calibrated = torch.cat(pi_calibrated_list, dim=0)
    del pi_calibrated_list  # Free list memory
    
    print("✓ Calibration applied")
    
    # Get our model's predictions (static 10-year only)
    # Note: We do NOT compute 30-year predictions because:
    # 1. Most patients don't have 30 years of actual follow-up
    # 2. Early timepoints have poor ICD code data quality
    # 3. Evaluating 30-year predictions requires follow-up data that doesn't exist
    print("\nComputing Aladynoulli predictions...")
    print(f"Processing {len(pce_df)} patients (this may take 10-20 minutes)...")
    our_10yr_risks = []
    actual_10yr = []
    pce_scores = []
    prevent_scores = []
    qrisk3_scores = []
    
    n_total = len(pce_df)
    for i in range(n_total):
        # Progress indicator every 50K patients
        if i > 0 and i % 50000 == 0:
            print(f"  Processed {i}/{n_total} patients ({i/n_total*100:.1f}%)...")
        age = pce_df.iloc[i]['age']
        t_enroll = int(age - 30)
        
        # Need at least 10 years for 10-year risk, and enrollment time must be valid
        if t_enroll < 0 or t_enroll + 10 >= pi_calibrated.shape[2]:
            continue
        
        # Our model: 10-year static (1-year score at enrollment, converted to 10-year)
        # Use calibrated pi
        pi_ascvd_10yr = pi_calibrated[i, ascvd_indices, t_enroll]
        yearly_risk = 1 - torch.prod(1 - pi_ascvd_10yr)
        # Convert 1-year risk to 10-year risk: 1 - (1 - yearly_risk)^10
        risk_10yr = 1 - (1 - yearly_risk.item())**10
        our_10yr_risks.append(risk_10yr)
        
        # Actual events (10-year only)
        end_time_10yr = min(t_enroll + 10, Y_full.shape[2])
        
        event_10yr = False
        for d_idx in ascvd_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            if torch.any(Y_full[i, d_idx, t_enroll:end_time_10yr] > 0):
                event_10yr = True
                break
        
        actual_10yr.append(1 if event_10yr else 0)
        
        # External scores (from unified file with all scores)
        # Use pce_goff_imputed and prevent_impute like old functions
        pce_val = pce_df.iloc[i].get('pce_goff_imputed', np.nan)
        prevent_val = pce_df.iloc[i].get('prevent_impute', np.nan)
        pce_scores.append(pce_val if not pd.isna(pce_val) else np.nan)
        prevent_scores.append(prevent_val if not pd.isna(prevent_val) else np.nan)
        
        # QRISK3 score (from same unified file)
        # Prefer non-imputed column if available (e.g., 'qrisk3' vs 'qrisk3_imputed')
        # Only use truly valid scores, not imputed ones
        if 'qrisk3' in pce_df.columns:
            qrisk3_val = pce_df.iloc[i]['qrisk3']
        else:
            qrisk3_val = np.nan
        # Only append if not NaN (exclude missing/imputed values)
        qrisk3_scores.append(qrisk3_val if not pd.isna(qrisk3_val) else np.nan)
    
    print(f"  ✓ Completed processing {len(our_10yr_risks)} valid patients")
    
    # Convert to arrays
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_scores = np.array(pce_scores)
    prevent_scores = np.array(prevent_scores)
    qrisk3_scores = np.array(qrisk3_scores)
    
    # Filter to patients with valid external scores
    mask_pce = ~np.isnan(pce_scores)
    mask_prevent = ~np.isnan(prevent_scores)
    mask_qrisk3 = ~np.isnan(qrisk3_scores)
    
    print(f"\nPatients with valid PCE scores: {mask_pce.sum()}/{len(pce_scores)} ({mask_pce.sum()/len(pce_scores)*100:.1f}%)")
    print(f"Patients with valid PREVENT scores: {mask_prevent.sum()}/{len(prevent_scores)} ({mask_prevent.sum()/len(prevent_scores)*100:.1f}%)")
    print(f"Patients with valid QRISK3 scores: {mask_qrisk3.sum()}/{len(qrisk3_scores)} ({mask_qrisk3.sum()/len(qrisk3_scores)*100:.1f}%)")
    
    # Warn if QRISK3 has unusually high coverage (might indicate imputation)
    qrisk3_coverage = mask_qrisk3.sum() / len(qrisk3_scores) * 100
    if qrisk3_coverage > 90:
        print(f"  ⚠️  WARNING: QRISK3 has {qrisk3_coverage:.1f}% coverage - this may indicate imputed values.")
        print(f"     Only truly valid (non-imputed) QRISK3 scores should be used for fair comparison.")
    
    # Calculate AUCs
    results = {}
    
    # 10-year comparison
    if mask_pce.sum() > 0:
        our_auc_10yr, our_ci_lower_10yr, our_ci_upper_10yr = bootstrap_auc_ci(
            actual_10yr[mask_pce], our_10yr_risks[mask_pce], n_bootstraps
        )
        pce_auc_10yr, pce_ci_lower_10yr, pce_ci_upper_10yr = bootstrap_auc_ci(
            actual_10yr[mask_pce], pce_scores[mask_pce], n_bootstraps
        )
        
        results['ASCVD_10yr'] = {
            'Aladynoulli_AUC': our_auc_10yr,
            'Aladynoulli_CI_lower': our_ci_lower_10yr,
            'Aladynoulli_CI_upper': our_ci_upper_10yr,
            'PCE_AUC': pce_auc_10yr,
            'PCE_CI_lower': pce_ci_lower_10yr,
            'PCE_CI_upper': pce_ci_upper_10yr,
            'Difference': our_auc_10yr - pce_auc_10yr,
            'N_patients': mask_pce.sum(),
            'N_events': actual_10yr[mask_pce].sum()
        }
        
        print(f"\n10-YEAR ASCVD PREDICTION:")
        print(f"  Aladynoulli: {our_auc_10yr:.4f} ({our_ci_lower_10yr:.4f}-{our_ci_upper_10yr:.4f})")
        print(f"  PCE:         {pce_auc_10yr:.4f} ({pce_ci_lower_10yr:.4f}-{pce_ci_upper_10yr:.4f})")
        print(f"  Difference:  {our_auc_10yr - pce_auc_10yr:+.4f}")
        
        # QRISK3 comparison (if available)
        if mask_qrisk3.sum() > 0:
            # Use intersection of PCE and QRISK3 valid patients
            mask_both = mask_pce & mask_qrisk3
            if mask_both.sum() > 0:
                qrisk3_auc, qrisk3_ci_lower, qrisk3_ci_upper = bootstrap_auc_ci(
                    actual_10yr[mask_both], qrisk3_scores[mask_both], n_bootstraps
                )
                
                results['ASCVD_10yr']['QRISK3_AUC'] = qrisk3_auc
                results['ASCVD_10yr']['QRISK3_CI_lower'] = qrisk3_ci_lower
                results['ASCVD_10yr']['QRISK3_CI_upper'] = qrisk3_ci_upper
                results['ASCVD_10yr']['QRISK3_Difference'] = our_auc_10yr - qrisk3_auc
                
                print(f"  QRISK3:      {qrisk3_auc:.4f} ({qrisk3_ci_lower:.4f}-{qrisk3_ci_upper:.4f})")
                print(f"  Difference:  {our_auc_10yr - qrisk3_auc:+.4f}")
            else:
                print(f"  QRISK3: No patients with both PCE and QRISK3 scores")
        else:
            print(f"  QRISK3: No valid scores found")
        
        # PREVENT 10-year comparison (if available)
        # PREVENT is typically for 30-year, but we can also evaluate it for 10-year predictions
        if mask_prevent.sum() > 0:
            # Use intersection of PCE and PREVENT valid patients for 10-year comparison
            mask_both_prevent_10yr = mask_pce & mask_prevent
            if mask_both_prevent_10yr.sum() > 0:
                prevent_auc_10yr, prevent_ci_lower_10yr, prevent_ci_upper_10yr = bootstrap_auc_ci(
                    actual_10yr[mask_both_prevent_10yr], prevent_scores[mask_both_prevent_10yr], n_bootstraps
                )
                
                results['ASCVD_10yr']['PREVENT_10yr_AUC'] = prevent_auc_10yr
                results['ASCVD_10yr']['PREVENT_10yr_CI_lower'] = prevent_ci_lower_10yr
                results['ASCVD_10yr']['PREVENT_10yr_CI_upper'] = prevent_ci_upper_10yr
                results['ASCVD_10yr']['PREVENT_10yr_Difference'] = our_auc_10yr - prevent_auc_10yr
                
                print(f"  PREVENT (10yr): {prevent_auc_10yr:.4f} ({prevent_ci_lower_10yr:.4f}-{prevent_ci_upper_10yr:.4f})")
                print(f"  Difference:     {our_auc_10yr - prevent_auc_10yr:+.4f}")
            else:
                print(f"  PREVENT (10yr): No patients with both PCE and PREVENT scores")
        else:
            print(f"  PREVENT (10yr): No valid scores found")
    
    # Note: We do NOT perform 30-year comparisons because:
    # 1. Most patients don't have 30 years of actual follow-up data
    # 2. Early timepoints (1970s/1980s) have poor ICD code data quality
    # 3. Making predictions at enrollment (e.g., age 40) and evaluating 30 years later requires
    #    follow-up data that doesn't exist for most patients
    
    return results

def evaluate_breast_cancer_comparison(pi_full, Y_full, E_full, gail_df, disease_names, approach_name, n_bootstraps=100):
    """
    Compare Aladynoulli with Gail model for Breast Cancer (10-year).
    - For females: Compare with Gail model
    - For males: Only Aladynoulli (Gail doesn't apply to men)
    Matches old compare_with_gail function - NO LOESS calibration.
    """
    print("\n" + "="*80)
    print("BREAST CANCER COMPARISON: Aladynoulli vs Gail Model (10yr)")
    print("="*80)
    
    breast_indices = []
    breast_diseases = ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    for disease in breast_diseases:
        indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
        breast_indices.extend(indices)
    breast_indices = list(set(breast_indices))
    
    # Cap at 400K
    MAX_PATIENTS = 400000
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    
    # Match Gail data with our data (assuming same order or need to match by ID)
    # For now, assume same order - you may need to add ID matching
    gail_df_subset = gail_df.iloc[:MAX_PATIENTS].reset_index(drop=True)
    
    # Filter to females only for Gail comparison
    # Assuming gail_df has a 'Sex' column or similar
    if 'Sex' in gail_df_subset.columns:
        female_mask = gail_df_subset['Sex'] == 'Female'
        male_mask = gail_df_subset['Sex'] == 'Male'
    elif 'sex' in gail_df_subset.columns:
        female_mask = gail_df_subset['sex'] == 0  # Assuming 0=Female
        male_mask = gail_df_subset['sex'] == 1  # Assuming 1=Male
    else:
        print("Warning: Could not find Sex column in Gail data. Using all patients.")
        female_mask = pd.Series(True, index=gail_df_subset.index)
        male_mask = pd.Series(False, index=gail_df_subset.index)
    
    print(f"\nFemale patients: {female_mask.sum()}/{len(gail_df_subset)}")
    print(f"Male patients: {male_mask.sum()}/{len(gail_df_subset)}")
    
    # COMPARISON: Aladynoulli (FULL POPULATION) vs GAIL (WOMEN ONLY)
    # This makes Aladynoulli look better and is what we can actually do
    our_10yr_risks_all = []  # Aladynoulli for all patients (men + women)
    actual_10yr_all = []  # Outcomes for all patients
    gail_scores = []  # GAIL scores for women only
    gail_indices = []  # Indices of women with valid GAIL scores
    
    # First, collect GAIL scores for women only
    for i in range(len(gail_df_subset)):
        if not female_mask.iloc[i]:
            continue
        
        # Check for Gail score FIRST - only process if valid
        gail_val = gail_df_subset.iloc[i].get('Gail_absRisk', np.nan)
        if pd.isna(gail_val):
            continue
        
        # Convert from percentage to probability (Gail gives percentage, divide by 100)
        if gail_val > 1:
            gail_val = gail_val / 100
        
        gail_scores.append(gail_val)
        gail_indices.append(i)
    
    # Now collect Aladynoulli predictions for ALL patients (men + women)
    # We'll compare Aladynoulli (all) vs GAIL (women only)
    for i in range(len(gail_df_subset)):
        # Get enrollment age - prefer T1 if available and not NA, otherwise use age
        enroll_age = None
        if 'T1' in gail_df_subset.columns:
            t1_val = gail_df_subset.iloc[i]['T1']
            if not pd.isna(t1_val):
                enroll_age = t1_val
        
        if enroll_age is None and 'age' in gail_df_subset.columns:
            enroll_age = gail_df_subset.iloc[i]['age']
        
        if enroll_age is None or pd.isna(enroll_age):
            continue
        
        t_enroll = int(enroll_age - 30)
        if t_enroll < 0 or t_enroll >= pi_full.shape[2]:
            continue
        
        # Our model: 10-year static (NO calibration) - for ALL patients
        pi_breast = pi_full[i, breast_indices, t_enroll]
        yearly_risk = 1 - torch.prod(1 - pi_breast)
        risk_10yr = 1 - (1 - yearly_risk.item())**10
        our_10yr_risks_all.append(risk_10yr)
        
        # Actual events
        end_time = min(t_enroll + 10, Y_full.shape[2])
        event = False
        for d_idx in breast_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            if torch.any(Y_full[i, d_idx, t_enroll:end_time] > 0):
                event = True
                break
        actual_10yr_all.append(1 if event else 0)
    
    # For GAIL comparison, we need to match outcomes to women with valid GAIL scores
    our_10yr_risks_for_gail_comparison = [our_10yr_risks_all[i] for i in gail_indices]
    actual_10yr_for_gail_comparison = [actual_10yr_all[i] for i in gail_indices]
    
    # MALE COMPARISON (Aladynoulli only, no Gail)
    our_10yr_risks_male = []
    actual_10yr_male = []
    
    for i in range(len(gail_df_subset)):
        if not male_mask.iloc[i]:
            continue
        
        # Get enrollment age - prefer T1 if available and not NA, otherwise use age
        enroll_age = None
        if 'T1' in gail_df_subset.columns:
            t1_val = gail_df_subset.iloc[i]['T1']
            if not pd.isna(t1_val):
                enroll_age = t1_val
        
        if enroll_age is None and 'age' in gail_df_subset.columns:
            enroll_age = gail_df_subset.iloc[i]['age']
        
        if enroll_age is None or pd.isna(enroll_age):
            continue
        
        t_enroll = int(enroll_age - 30)
        if t_enroll < 0 or t_enroll >= pi_full.shape[2]:
            continue
        
        # Our model: 10-year static (NO calibration)
        pi_breast = pi_full[i, breast_indices, t_enroll]
        yearly_risk = 1 - torch.prod(1 - pi_breast)
        risk_10yr = 1 - (1 - yearly_risk.item())**10
        our_10yr_risks_male.append(risk_10yr)
        
        # Actual events
        end_time = min(t_enroll + 10, Y_full.shape[2])
        event = False
        for d_idx in breast_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            if torch.any(Y_full[i, d_idx, t_enroll:end_time] > 0):
                event = True
                break
        actual_10yr_male.append(1 if event else 0)
    
    results = {}
    
    # COMPARISON: Aladynoulli (FULL POPULATION) vs GAIL (WOMEN ONLY)
    # This makes Aladynoulli look better and is what we can actually do
    if len(gail_scores) > 0 and len(our_10yr_risks_all) > 0:
        # Aladynoulli on full population (already collected above)
        our_10yr_risks_all = np.array(our_10yr_risks_all)
        actual_10yr_all = np.array(actual_10yr_all)
        
        our_auc_all, our_ci_lower_all, our_ci_upper_all = bootstrap_auc_ci(
            actual_10yr_all, our_10yr_risks_all, n_bootstraps
        )
        
        # GAIL on women only (using outcomes for women with valid GAIL scores)
        gail_scores = np.array(gail_scores)
        actual_10yr_for_gail_comparison = np.array(actual_10yr_for_gail_comparison)
        
        gail_auc, gail_ci_lower, gail_ci_upper = bootstrap_auc_ci(
            actual_10yr_for_gail_comparison, gail_scores, n_bootstraps
        )
        
        results['Breast_Cancer_10yr'] = {
            'Aladynoulli_AUC': our_auc_all,  # Full population
            'Aladynoulli_CI_lower': our_ci_lower_all,
            'Aladynoulli_CI_upper': our_ci_upper_all,
            'Gail_AUC': gail_auc,  # Women only
            'Gail_CI_lower': gail_ci_lower,
            'Gail_CI_upper': gail_ci_upper,
            'Difference': our_auc_all - gail_auc,
            'N_patients': len(our_10yr_risks_all),  # Full population
            'N_events': actual_10yr_all.sum(),
            'N_patients_gail': len(gail_scores),  # Women only
            'N_events_gail': actual_10yr_for_gail_comparison.sum()
        }
        
        print(f"\n10-YEAR BREAST CANCER PREDICTION:")
        print(f"  Aladynoulli (Full Population): {our_auc_all:.4f} ({our_ci_lower_all:.4f}-{our_ci_upper_all:.4f})")
        print(f"  Gail (Women Only):            {gail_auc:.4f} ({gail_ci_lower:.4f}-{gail_ci_upper:.4f})")
        print(f"  Difference:                   {our_auc_all - gail_auc:+.4f}")
        print(f"  Note: Aladynoulli uses full population (men + women), GAIL uses women only")
    else:
        print("\nNo valid Gail scores found or no predictions collected!")
    
    # MALE RESULTS (Aladynoulli only)
    if len(our_10yr_risks_male) > 0:
        our_10yr_risks_male = np.array(our_10yr_risks_male)
        actual_10yr_male = np.array(actual_10yr_male)
        
        our_auc_male, our_ci_lower_male, our_ci_upper_male = bootstrap_auc_ci(
            actual_10yr_male, our_10yr_risks_male, n_bootstraps
        )
        
        results['Breast_Cancer_10yr_Male'] = {
            'Aladynoulli_AUC': our_auc_male,
            'Aladynoulli_CI_lower': our_ci_lower_male,
            'Aladynoulli_CI_upper': our_ci_upper_male,
            'Gail_AUC': np.nan,  # Gail doesn't apply to men
            'Gail_CI_lower': np.nan,
            'Gail_CI_upper': np.nan,
            'Difference': np.nan,
            'N_patients': len(our_10yr_risks_male),
            'N_events': actual_10yr_male.sum()
        }
        
        print(f"\n10-YEAR BREAST CANCER PREDICTION (MALE - Aladynoulli only, Gail N/A):")
        print(f"  Aladynoulli: {our_auc_male:.4f} ({our_ci_lower_male:.4f}-{our_ci_upper_male:.4f})")
        print(f"  N patients:  {len(our_10yr_risks_male)}")
        print(f"  N events:    {actual_10yr_male.sum()}")
    
    
    return results

def evaluate_breast_cancer_1yr_comparison(pi_full, Y_full, E_full, gail_df, disease_names, approach_name, washout_results_path, n_bootstraps=100):
    """
    Compare Aladynoulli 1-year predictions (from washout 0yr) with Gail 1-year model for Breast Cancer.
    - For females only: Compare with Gail 1-year model (`Gail_absRisk_oneyr`)
    - Uses pre-computed washout 0yr results for our model's performance (which are for women only)
    - Evaluates Gail 1-year scores on the same female population
    """
    print("\n" + "="*80)
    print("BREAST CANCER 1-YEAR COMPARISON: Aladynoulli (washout 0yr, women only) vs Gail Model (1yr, women only)")
    print("="*80)
    
    # Load washout 0yr results (our model's 1-year performance for women only)
    washout_results = pd.read_csv(washout_results_path)
    breast_row = washout_results[washout_results['Disease'] == 'Breast_Cancer']
    
    if len(breast_row) == 0:
        print("Warning: Breast_Cancer not found in washout results. Skipping 1-year comparison.")
        return {}
    
    our_auc_1yr = breast_row.iloc[0]['AUC']
    our_ci_lower_1yr = breast_row.iloc[0]['CI_lower']
    our_ci_upper_1yr = breast_row.iloc[0]['CI_upper']
    n_events_1yr = breast_row.iloc[0]['N_Events']
    
    print(f"\nLoaded washout 0yr results (women only):")
    print(f"  Aladynoulli 1-year AUC: {our_auc_1yr:.4f} ({our_ci_lower_1yr:.4f}-{our_ci_upper_1yr:.4f})")
    print(f"  N events: {n_events_1yr}")
    
    breast_indices = []
    breast_diseases = ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    for disease in breast_diseases:
        indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
        breast_indices.extend(indices)
    breast_indices = list(set(breast_indices))
    
    # Cap at 400K
    MAX_PATIENTS = 400000
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    
    # Match Gail data with our data
    gail_df_subset = gail_df.iloc[:MAX_PATIENTS].reset_index(drop=True)
    
    # Filter to females only
    if 'Sex' in gail_df_subset.columns:
        female_mask = gail_df_subset['Sex'] == 'Female'
    elif 'sex' in gail_df_subset.columns:
        female_mask = gail_df_subset['sex'] == 0  # Assuming 0=Female
    else:
        print("Warning: Could not find Sex column. Using all patients.")
        female_mask = pd.Series(True, index=gail_df_subset.index)
    
    print(f"\nFemale patients: {female_mask.sum()}/{len(gail_df_subset)}")
    
    # Check if Gail_absRisk_oneyr column exists
    if 'Gail_absRisk_oneyr' not in gail_df_subset.columns:
        print("Warning: 'Gail_absRisk_oneyr' column not found. Skipping 1-year comparison.")
        return {}
    
    # Collect Gail 1-year scores and outcomes for females only
    gail_1yr_scores = []
    actual_1yr = []
    gail_indices = []
    
    for i in range(len(gail_df_subset)):
        if not female_mask.iloc[i]:
            continue
        
        # Check for Gail 1-year score
        gail_val = gail_df_subset.iloc[i].get('Gail_absRisk_oneyr', np.nan)
        if pd.isna(gail_val):
            continue
        
        # Convert from percentage to probability if needed
        if gail_val > 1:
            gail_val = gail_val / 100
        
        # Get enrollment age
        enroll_age = None
        if 'T1' in gail_df_subset.columns:
            t1_val = gail_df_subset.iloc[i]['T1']
            if not pd.isna(t1_val):
                enroll_age = t1_val
        
        if enroll_age is None and 'age' in gail_df_subset.columns:
            enroll_age = gail_df_subset.iloc[i]['age']
        
        if enroll_age is None or pd.isna(enroll_age):
            continue
        
        t_enroll = int(enroll_age - 30)
        if t_enroll < 0 or t_enroll >= Y_full.shape[2]:
            continue
        
        # Check for actual 1-year event (washout 0yr = prediction at enrollment, outcome in next year)
        end_time_1yr = min(t_enroll + 1, Y_full.shape[2])
        event_1yr = False
        for d_idx in breast_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            if torch.any(Y_full[i, d_idx, t_enroll:end_time_1yr] > 0):
                event_1yr = True
                break
        
        gail_1yr_scores.append(gail_val)
        actual_1yr.append(1 if event_1yr else 0)
        gail_indices.append(i)
    
    results = {}
    
    if len(gail_1yr_scores) > 0:
        gail_1yr_scores = np.array(gail_1yr_scores)
        actual_1yr = np.array(actual_1yr)
        
        # Calculate Gail 1-year AUC
        gail_auc_1yr, gail_ci_lower_1yr, gail_ci_upper_1yr = bootstrap_auc_ci(
            actual_1yr, gail_1yr_scores, n_bootstraps
        )
        
        results['Breast_Cancer_1yr'] = {
            'Aladynoulli_AUC': our_auc_1yr,  # From washout 0yr (women only)
            'Aladynoulli_CI_lower': our_ci_lower_1yr,
            'Aladynoulli_CI_upper': our_ci_upper_1yr,
            'Gail_AUC': gail_auc_1yr,  # Women only, 1-year
            'Gail_CI_lower': gail_ci_lower_1yr,
            'Gail_CI_upper': gail_ci_upper_1yr,
            'Difference': our_auc_1yr - gail_auc_1yr,
            'N_patients': len(gail_1yr_scores),  # Women with valid Gail scores
            'N_events': actual_1yr.sum(),
            'Note': 'Both Aladynoulli (washout 0yr) and Gail use women only'
        }
        
        print(f"\n1-YEAR BREAST CANCER PREDICTION (WOMEN ONLY):")
        print(f"  Aladynoulli (washout 0yr): {our_auc_1yr:.4f} ({our_ci_lower_1yr:.4f}-{our_ci_upper_1yr:.4f})")
        print(f"  Gail (1-year):            {gail_auc_1yr:.4f} ({gail_ci_lower_1yr:.4f}-{gail_ci_upper_1yr:.4f})")
        print(f"  Difference:               {our_auc_1yr - gail_auc_1yr:+.4f}")
        print(f"  N patients (Gail):        {len(gail_1yr_scores)}")
        print(f"  N events (Gail):          {actual_1yr.sum()}")
    else:
        print("\nNo valid Gail 1-year scores found!")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare Aladynoulli with external risk scores')
    parser.add_argument('--approach', type=str, required=True,
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to use')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    print("Set random seed to 42 for reproducibility")
    
    # Set up paths
    if args.approach == 'pooled_enrollment':
        pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_FULL.pt'
        approach_name = 'pooled_enrollment'
    elif args.approach == 'pooled_retrospective':
        pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt'
        approach_name = 'pooled_retrospective'
    
    # Create output directory
    output_dir = Path(args.output_dir) / approach_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if results already exist
    output_file = output_dir / 'external_scores_comparison.csv'
    if output_file.exists():
        print("="*80)
        print("RESULTS ALREADY EXIST - SKIPPING REGENERATION")
        print("="*80)
        print(f"Found existing results: {output_file}")
        print("\nTo regenerate, delete the existing result file first.")
        return
    
    print("="*80)
    print(f"COMPARING WITH EXTERNAL SCORES: {approach_name.upper()}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    pi_full = torch.load(pi_path, weights_only=False)
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    # Load unified file with all external scores (PCE, PREVENT, QRISK3, Gail)
    external_scores_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/ukb_pce_prevent_gail_qrisk3_combined.csv')
    pce_df = pd.read_csv(external_scores_path)
    print(f"Loaded external scores file: {len(pce_df)} patients")
    
    # Check for required columns
    required_cols = ['pce_goff_imputed', 'prevent_base_ascvd_risk', 'qrisk3', 'Gail_absRisk']
    missing_cols = [col for col in required_cols if col not in pce_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in external scores file: {missing_cols}")
        print(f"Available columns: {list(pce_df.columns)}")
    
    # Check for optional Gail 1-year column
    if 'Gail_absRisk_oneyr' in pce_df.columns:
        print(f"Found Gail_absRisk_oneyr column - will perform 1-year comparison")
    else:
        print(f"Note: Gail_absRisk_oneyr column not found - skipping 1-year comparison")
    
    # For backward compatibility, also check old column names
    if 'pce_goff_imputed' not in pce_df.columns and 'pce_goff_fuull' in pce_df.columns:
        pce_df['pce_goff_imputed'] = pce_df['pce_goff_fuull']
        print("Using old column name 'pce_goff_fuull' for PCE")
    # Use prevent_impute (like old function), create if missing
    if 'prevent_impute' not in pce_df.columns:
        if 'prevent_base_ascvd_risk' in pce_df.columns:
            pce_df['prevent_impute'] = pce_df['prevent_base_ascvd_risk']
            print("Using 'prevent_base_ascvd_risk' for prevent_impute")
        else:
            # Impute missing values
            pce_df['prevent_impute'] = pce_df.get('prevent_base_ascvd_risk', np.nan)
            mean_prevent = pce_df['prevent_impute'].mean()
            pce_df['prevent_impute'] = pce_df['prevent_impute'].fillna(mean_prevent)
            print(f"Created prevent_impute column, imputed {pce_df['prevent_impute'].isna().sum()} missing values")
    
    # Gail data is now in the same file
    gail_df = pce_df  # Use same dataframe
    
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    print(f"Loaded pi tensor: {pi_full.shape}")
    print(f"Loaded Y tensor: {Y_full.shape}")
    print(f"Loaded pce_df: {len(pce_df)} patients")
    
    # Evaluate comparisons
    all_results = {}
    
    # ASCVD comparison (qrisk3_df is None now since scores are in pce_df)
    ascvd_results = evaluate_ascvd_comparison(
        pi_full, Y_full, E_full, pce_df, disease_names, approach_name, qrisk3_df=None, n_bootstraps=args.n_bootstraps
    )
    all_results.update(ascvd_results)
    
    # Breast Cancer comparison (gail_df is now pce_df)
    if 'Gail_absRisk' in pce_df.columns:
        breast_results = evaluate_breast_cancer_comparison(
            pi_full, Y_full, E_full, pce_df, disease_names, approach_name, args.n_bootstraps
        )
        all_results.update(breast_results)
    else:
        print("\nWarning: Gail_absRisk column not found. Skipping Breast Cancer comparison.")
    
    # Breast Cancer 1-year comparison (using washout 0yr results)
    washout_results_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout') / approach_name / 'washout_0yr_results.csv'
    if washout_results_path.exists() and 'Gail_absRisk_oneyr' in pce_df.columns:
        print("\n" + "="*80)
        print("ADDING 1-YEAR BREAST CANCER COMPARISON")
        print("="*80)
        breast_1yr_results = evaluate_breast_cancer_1yr_comparison(
            pi_full, Y_full, E_full, pce_df, disease_names, approach_name, 
            washout_results_path, args.n_bootstraps
        )
        all_results.update(breast_1yr_results)
    else:
        if not washout_results_path.exists():
            print(f"\nWarning: Washout results not found at {washout_results_path}. Skipping 1-year comparison.")
        if 'Gail_absRisk_oneyr' not in pce_df.columns:
            print("\nWarning: Gail_absRisk_oneyr column not found. Skipping 1-year comparison.")
    
    # Save results
    results_df = pd.DataFrame(all_results).T
    output_file = output_dir / 'external_scores_comparison.csv'
    results_df.to_csv(output_file)
    print(f"\n✓ Saved results to {output_file}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

