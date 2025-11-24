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
    Compare Aladynoulli with PCE (10-year) and PREVENT (30-year) for ASCVD.
    """
    print("\n" + "="*80)
    print("ASCVD COMPARISON: Aladynoulli vs PCE (10yr) and PREVENT (30yr)")
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
    
    # Get our model's predictions (static 10-year and dynamic 30-year)
    print("\nComputing Aladynoulli predictions...")
    our_10yr_risks = []
    our_30yr_risks = []
    actual_10yr = []
    actual_30yr = []
    pce_scores = []
    prevent_scores = []
    qrisk3_scores = []
    
    for i in range(len(pce_df)):
        age = pce_df.iloc[i]['age']
        t_enroll = int(age - 30)
        
        if t_enroll < 0 or t_enroll >= pi_full.shape[2]:
            continue
        
        # Our model: 10-year static (1-year score at enrollment)
        pi_ascvd_10yr = pi_full[i, ascvd_indices, t_enroll]
        yearly_risk_10yr = 1 - torch.prod(1 - pi_ascvd_10yr)
        our_10yr_risks.append(yearly_risk_10yr.item())
        
        # Our model: 30-year dynamic (cumulative risk)
        yearly_risks_30yr = []
        for t in range(1, 31):
            if t_enroll + t >= pi_full.shape[2]:
                break
            pi_ascvd_t = pi_full[i, ascvd_indices, t_enroll + t]
            yearly_risk_t = 1 - torch.prod(1 - pi_ascvd_t)
            yearly_risks_30yr.append(yearly_risk_t.item())
        if yearly_risks_30yr:
            survival_prob_30yr = np.prod([1 - r for r in yearly_risks_30yr])
            our_30yr_risks.append(1 - survival_prob_30yr)
        else:
            our_30yr_risks.append(0.0)
        
        # Actual events
        end_time_10yr = min(t_enroll + 10, Y_full.shape[2])
        end_time_30yr = min(t_enroll + 30, Y_full.shape[2])
        
        event_10yr = False
        event_30yr = False
        for d_idx in ascvd_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            if not event_10yr and torch.any(Y_full[i, d_idx, t_enroll:end_time_10yr] > 0):
                event_10yr = True
            if not event_30yr and torch.any(Y_full[i, d_idx, t_enroll:end_time_30yr] > 0):
                event_30yr = True
            if event_10yr and event_30yr:
                break
        
        actual_10yr.append(1 if event_10yr else 0)
        actual_30yr.append(1 if event_30yr else 0)
        
        # External scores (from unified file with all scores)
        pce_val = pce_df.iloc[i].get('pce_goff_imputed', np.nan)
        prevent_val = pce_df.iloc[i].get('prevent_base_ascvd_risk', np.nan)
        pce_scores.append(pce_val if not pd.isna(pce_val) else np.nan)
        prevent_scores.append(prevent_val if not pd.isna(prevent_val) else np.nan)
        
        # QRISK3 score (from same unified file)
        qrisk3_val = pce_df.iloc[i].get('qrisk3', np.nan)
        qrisk3_scores.append(qrisk3_val if not pd.isna(qrisk3_val) else np.nan)
    
    # Convert to arrays
    our_10yr_risks = np.array(our_10yr_risks)
    our_30yr_risks = np.array(our_30yr_risks)
    actual_10yr = np.array(actual_10yr)
    actual_30yr = np.array(actual_30yr)
    pce_scores = np.array(pce_scores)
    prevent_scores = np.array(prevent_scores)
    qrisk3_scores = np.array(qrisk3_scores)
    
    # Filter to patients with valid external scores
    mask_pce = ~np.isnan(pce_scores)
    mask_prevent = ~np.isnan(prevent_scores)
    mask_qrisk3 = ~np.isnan(qrisk3_scores)
    
    print(f"\nPatients with valid PCE scores: {mask_pce.sum()}/{len(pce_scores)} ({mask_pce.sum()/len(pce_scores)*100:.1f}%)")
    print(f"Patients with valid PREVENT scores: {mask_prevent.sum()}/{len(prevent_scores)} ({mask_prevent.sum()/len(prevent_scores)*100:.1f}%)")
    
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
    
    # 30-year comparison
    if mask_prevent.sum() > 0:
        our_auc_30yr, our_ci_lower_30yr, our_ci_upper_30yr = bootstrap_auc_ci(
            actual_30yr[mask_prevent], our_30yr_risks[mask_prevent], n_bootstraps
        )
        prevent_auc_30yr, prevent_ci_lower_30yr, prevent_ci_upper_30yr = bootstrap_auc_ci(
            actual_30yr[mask_prevent], prevent_scores[mask_prevent], n_bootstraps
        )
        
        results['ASCVD_30yr'] = {
            'Aladynoulli_AUC': our_auc_30yr,
            'Aladynoulli_CI_lower': our_ci_lower_30yr,
            'Aladynoulli_CI_upper': our_ci_upper_30yr,
            'PREVENT_AUC': prevent_auc_30yr,
            'PREVENT_CI_lower': prevent_ci_lower_30yr,
            'PREVENT_CI_upper': prevent_ci_upper_30yr,
            'Difference': our_auc_30yr - prevent_auc_30yr,
            'N_patients': mask_prevent.sum(),
            'N_events': actual_30yr[mask_prevent].sum()
        }
        
        print(f"\n30-YEAR ASCVD PREDICTION:")
        print(f"  Aladynoulli: {our_auc_30yr:.4f} ({our_ci_lower_30yr:.4f}-{our_ci_upper_30yr:.4f})")
        print(f"  PREVENT:     {prevent_auc_30yr:.4f} ({prevent_ci_lower_30yr:.4f}-{prevent_ci_upper_30yr:.4f})")
        print(f"  Difference:  {our_auc_30yr - prevent_auc_30yr:+.4f}")
    
    return results

def evaluate_breast_cancer_comparison(pi_full, Y_full, E_full, gail_df, disease_names, approach_name, n_bootstraps=100):
    """
    Compare Aladynoulli with Gail model for Breast Cancer (10-year).
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
    
    # Filter to females only
    # Assuming gail_df has a 'Sex' column or similar
    if 'Sex' in gail_df_subset.columns:
        female_mask = gail_df_subset['Sex'] == 'Female'
    elif 'sex' in gail_df_subset.columns:
        female_mask = gail_df_subset['sex'] == 0  # Assuming 0=Female
    else:
        print("Warning: Could not find Sex column in Gail data. Using all patients.")
        female_mask = pd.Series(True, index=gail_df_subset.index)
    
    print(f"\nFemale patients: {female_mask.sum()}/{len(gail_df_subset)}")
    
    our_10yr_risks = []
    actual_10yr = []
    gail_scores = []
    
    for i in range(len(gail_df_subset)):
        if not female_mask.iloc[i]:
            continue
        
        # Get enrollment age from Gail data (assuming column name)
        if 'T1' in gail_df_subset.columns:
            enroll_age = gail_df_subset.iloc[i]['T1']
        elif 'age' in gail_df_subset.columns:
            enroll_age = gail_df_subset.iloc[i]['age']
        else:
            continue
        
        t_enroll = int(enroll_age - 30)
        if t_enroll < 0 or t_enroll >= pi_full.shape[2]:
            continue
        
        # Our model: 10-year static
        pi_breast = pi_full[i, breast_indices, t_enroll]
        yearly_risk = 1 - torch.prod(1 - pi_breast)
        risk_10yr = 1 - (1 - yearly_risk.item())**10
        our_10yr_risks.append(risk_10yr)
        
        # Actual events
        end_time = min(t_enroll + 10, Y_full.shape[2])
        event = False
        for d_idx in breast_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            if torch.any(Y_full[i, d_idx, t_enroll:end_time] > 0):
                event = True
                break
        actual_10yr.append(1 if event else 0)
        
        # Gail score
        gail_val = gail_df_subset.iloc[i].get('Gail_absRisk', np.nan)
        if pd.isna(gail_val):
            continue
        # Convert from percentage to probability if needed
        if gail_val > 1:
            gail_val = gail_val / 100
        gail_scores.append(gail_val)
    
    # Convert to arrays
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    gail_scores = np.array(gail_scores)
    
    print(f"\nPatients with valid Gail scores: {len(gail_scores)}")
    
    if len(gail_scores) > 0:
        our_auc, our_ci_lower, our_ci_upper = bootstrap_auc_ci(
            actual_10yr, our_10yr_risks, n_bootstraps
        )
        gail_auc, gail_ci_lower, gail_ci_upper = bootstrap_auc_ci(
            actual_10yr, gail_scores, n_bootstraps
        )
        
        results = {
            'Breast_Cancer_10yr': {
                'Aladynoulli_AUC': our_auc,
                'Aladynoulli_CI_lower': our_ci_lower,
                'Aladynoulli_CI_upper': our_ci_upper,
                'Gail_AUC': gail_auc,
                'Gail_CI_lower': gail_ci_lower,
                'Gail_CI_upper': gail_ci_upper,
                'Difference': our_auc - gail_auc,
                'N_patients': len(gail_scores),
                'N_events': actual_10yr.sum()
            }
        }
        
        print(f"\n10-YEAR BREAST CANCER PREDICTION:")
        print(f"  Aladynoulli: {our_auc:.4f} ({our_ci_lower:.4f}-{our_ci_upper:.4f})")
        print(f"  Gail:        {gail_auc:.4f} ({gail_ci_lower:.4f}-{gail_ci_upper:.4f})")
        print(f"  Difference:  {our_auc - gail_auc:+.4f}")
        
        return results
    else:
        print("\nNo valid Gail scores found!")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Compare Aladynoulli with external risk scores')
    parser.add_argument('--approach', type=str, required=True,
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to use')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set up paths
    if args.approach == 'pooled_enrollment':
        pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_FULL.pt'
        approach_name = 'pooled_enrollment'
    elif args.approach == 'pooled_retrospective':
        pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/pi_enroll_fixedphi_sex_FULL.pt'
        approach_name = 'pooled_retrospective'
    
    # Create output directory
    output_dir = Path(args.output_dir) / approach_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # For backward compatibility, also check old column names
    if 'pce_goff_imputed' not in pce_df.columns and 'pce_goff_fuull' in pce_df.columns:
        pce_df['pce_goff_imputed'] = pce_df['pce_goff_fuull']
        print("Using old column name 'pce_goff_fuull' for PCE")
    if 'prevent_base_ascvd_risk' not in pce_df.columns and 'prevent_impute' in pce_df.columns:
        pce_df['prevent_base_ascvd_risk'] = pce_df['prevent_impute']
        print("Using old column name 'prevent_impute' for PREVENT")
    
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

