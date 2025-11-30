#!/usr/bin/env python3
"""
Generate age-stratified ROC curves for ASCVD.

This script extracts predictions and outcomes for each age group and creates
ROC curves showing how discrimination varies across age groups.
"""

import argparse
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year
)

def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def extract_ascvd_predictions_for_roc(pi, Y_100k, E_100k, pce_df, disease_names, 
                                      age_group_indices, horizon_name, horizon_years, horizon_type):
    """
    Extract ASCVD predictions and outcomes for ROC curve plotting.
    
    Returns:
        risks: array of predicted risks
        outcomes: array of actual outcomes (0/1)
        auc_score: AUC value
    """
    # Find ASCVD disease indices
    ascvd_diseases = ['Myocardial infarction', 'Coronary atherosclerosis', 
                      'Other acute and subacute forms of ischemic heart disease',
                      'Unstable angina', 'Angina pectoris', 
                      'Other chronic ischemic heart disease']
    disease_indices = []
    for disease in ascvd_diseases:
        indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
        disease_indices.extend(indices)
    disease_indices = list(set(disease_indices))
    
    # Subset data to age group
    pi_subset = pi[age_group_indices]
    Y_subset = Y_100k[age_group_indices]
    E_subset = E_100k[age_group_indices]
    pce_df_subset = pce_df.iloc[age_group_indices].reset_index(drop=True)
    
    risks = []
    outcomes = []
    
    for i in range(len(age_group_indices)):
        age = pce_df_subset.iloc[i]['age']
        t_enroll = int(age - 30)
        
        if t_enroll < 0:
            continue
        
        # Exclude prevalent cases (ASCVD before enrollment)
        if torch.any(Y_subset[i, disease_indices, :t_enroll] > 0):
            continue
            
        if horizon_type == 'static':
            # Static 10-year: 1-year score for 10-year outcome
            if t_enroll + 1 >= pi_subset.shape[2]:
                continue
            # Get 1-year risk
            pi_ascvd = pi_subset[i, disease_indices, t_enroll + 1]
            yearly_risk = 1 - torch.prod(1 - pi_ascvd)
            risks.append(yearly_risk.item())
            # Check for events in 10 years
            end_time = min(t_enroll + horizon_years, Y_subset.shape[2])
            event = torch.any(Y_subset[i, disease_indices, t_enroll:end_time] > 0).item()
            outcomes.append(int(event))
            
        elif horizon_name == '1yr':
            # 1-year dynamic
            if t_enroll + 1 >= pi_subset.shape[2]:
                continue
            pi_ascvd = pi_subset[i, disease_indices, t_enroll + 1]
            yearly_risk = 1 - torch.prod(1 - pi_ascvd)
            risks.append(yearly_risk.item())
            # Check for events in 1 year
            if t_enroll + 1 < Y_subset.shape[2]:
                event = torch.any(Y_subset[i, disease_indices, t_enroll + 1] > 0).item()
                outcomes.append(int(event))
            else:
                outcomes.append(0)
                
        else:
            # Dynamic multi-year (10yr, 30yr)
            if t_enroll + horizon_years >= pi_subset.shape[2]:
                continue
            # Collect yearly risks
            yearly_risks = []
            for t in range(1, horizon_years + 1):
                if t_enroll + t >= pi_subset.shape[2]:
                    break
                pi_ascvd = pi_subset[i, disease_indices, t_enroll + t]
                yearly_risk = 1 - torch.prod(1 - pi_ascvd)
                yearly_risks.append(yearly_risk.item())
            # Cumulative risk
            if yearly_risks:
                survival_prob = np.prod([1 - r for r in yearly_risks])
                cumulative_risk = 1 - survival_prob
                risks.append(cumulative_risk)
                # Check for events
                end_time = min(t_enroll + horizon_years, Y_subset.shape[2])
                event = torch.any(Y_subset[i, disease_indices, t_enroll:end_time] > 0).item()
                outcomes.append(int(event))
    
    risks = np.array(risks)
    outcomes = np.array(outcomes)
    
    if len(risks) == 0 or len(np.unique(outcomes)) < 2:
        return None, None, np.nan
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(outcomes, risks)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score

def main():
    parser = argparse.ArgumentParser(description='Generate age-stratified ROC curves for ASCVD')
    parser.add_argument('--approach', type=str, required=True,
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to use')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/age_stratified/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set up paths
    if args.approach == 'pooled_enrollment':
        pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_FULL.pt'
        approach_name = 'pooled_enrollment'
    elif args.approach == 'pooled_retrospective':
        pi_path = '/Users/sarahurbut/Downloads/pi_full_400k.pt'
        approach_name = 'pooled_retrospective'
    
    # Create output directory
    output_dir = Path(args.output_dir) / approach_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"GENERATING AGE-STRATIFIED ROC CURVES FOR ASCVD: {approach_name.upper()}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    pi_full = torch.load(pi_path, weights_only=False)
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    baseline_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Cap at 400K
    MAX_PATIENTS = 400000
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    pce_df_full = pce_df_full.iloc[:MAX_PATIENTS].reset_index(drop=True)
    baseline_df = baseline_df.iloc[:MAX_PATIENTS].reset_index(drop=True)
    
    # Get enrollment ages
    age_cols = [col for col in baseline_df.columns if 'age' in col.lower() and 'enroll' not in col.lower()]
    if age_cols:
        enrollment_ages = baseline_df[age_cols[0]].values
    else:
        enrollment_ages = pce_df_full['age'].values
    
    # Define age groups
    age_groups = [
        (40, 50, '40-50'),
        (50, 60, '50-60'),
        (60, 70, '60-70')
    ]
    
    # Define time horizons (just 1-year for now, as requested)
    time_horizons = [
        ('1yr', 1, 'dynamic')
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for age_idx, (age_min, age_max, age_group_name) in enumerate(age_groups):
        print(f"\nProcessing age group {age_group_name}...")
        
        # Get age group mask
        age_mask = (enrollment_ages >= age_min) & (enrollment_ages < age_max)
        age_group_indices = np.where(age_mask)[0].tolist()
        
        if len(age_group_indices) == 0:
            print(f"  No patients in age group {age_group_name}")
            continue
        
        ax = axes[age_idx]
        
        # Process each time horizon
        for horizon_name, horizon_years, horizon_type in time_horizons:
            print(f"  Extracting {horizon_name} predictions...")
            
            fpr, tpr, auc_score = extract_ascvd_predictions_for_roc(
                pi_full, Y_full, E_full, pce_df_full, disease_names,
                age_group_indices, horizon_name, horizon_years, horizon_type
            )
            
            if fpr is not None:
                ax.plot(fpr, tpr, label=f'1-Year (AUC={auc_score:.3f})', linewidth=2)
                print(f"    AUC: {auc_score:.3f}")
            else:
                print(f"    No valid predictions for {horizon_name}")
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'Age {age_group_name} (n={len(age_group_indices):,})', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'roc_curves_ascvd_age_stratified.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved ROC curves to: {output_file}")
    
    # Also save as PNG
    output_file_png = output_dir / 'roc_curves_ascvd_age_stratified.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves to: {output_file_png}")
    
    plt.show()

if __name__ == '__main__':
    main()

