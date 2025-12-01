#!/usr/bin/env python3
"""
Calculate age-stratified AUC for 1-year, 10-year static, and 30-year predictions.

This script:
1. Loads predictions and enrollment ages
2. Stratifies patients into age groups (40-50, 50-60, 60-70)
3. Calculates AUC for each age group × time horizon × disease combination
4. Saves results to CSV

Usage:
    python calculate_age_stratified_auc.py --approach pooled_retrospective
"""

import argparse
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

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

def get_age_group_mask(ages, age_min, age_max):
    """Get boolean mask for patients in age range [age_min, age_max)"""
    return (ages >= age_min) & (ages < age_max)

def main():
    parser = argparse.ArgumentParser(description='Calculate age-stratified AUC')
    parser.add_argument('--approach', type=str, required=True,
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to use')
    parser.add_argument('--n_bootstraps', type=int, default=10,
                       help='Number of bootstrap iterations (default: 10 for faster computation)')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/age_stratified/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set up paths based on approach
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
    print(f"AGE-STRATIFIED AUC CALCULATION: {approach_name.upper()}")
    print("="*80)
    print(f"Pi tensor: {pi_path}")
    print(f"Output directory: {output_dir}")
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
    
    print(f"Loaded pi tensor: {pi_full.shape}")
    print(f"Loaded Y tensor: {Y_full.shape}")
    print(f"Loaded E tensor: {E_full.shape}")
    print(f"Loaded pce_df: {len(pce_df_full)} patients")
    print(f"Loaded baseline_df: {len(baseline_df)} patients")
    
    # Cap at 400K patients
    MAX_PATIENTS = 400000
    print(f"\nSubsetting to first {MAX_PATIENTS} patients...")
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    pce_df_full = pce_df_full.iloc[:MAX_PATIENTS].reset_index(drop=True)
    baseline_df = baseline_df.iloc[:MAX_PATIENTS].reset_index(drop=True)
    
    print(f"After subsetting: pi: {pi_full.shape[0]}, Y: {Y_full.shape[0]}, E: {E_full.shape[0]}, pce_df: {len(pce_df_full)}, baseline_df: {len(baseline_df)}")
    
    # Get enrollment ages
    # Try to find age column in baseline_df
    age_cols = [col for col in baseline_df.columns if 'age' in col.lower() and 'enroll' not in col.lower()]
    if not age_cols:
        # Fall back to pce_df age
        print("Warning: No age column found in baseline_df, using pce_df age")
        enrollment_ages = pce_df_full['age'].values
    else:
        # Use first age column found
        age_col = age_cols[0]
        print(f"Using age column: {age_col}")
        enrollment_ages = baseline_df[age_col].values
    
    # Define age groups
    age_groups = [
        (40, 50, '40-50'),
        (50, 60, '50-60'),
        (60, 70, '60-70')
    ]
    
    # Define time horizons
    time_horizons = [
        ('1yr', 1, 'dynamic'),
        ('10yr_static', 10, 'static'),
        ('30yr', 30, 'dynamic')
    ]
    
    # Check if results already exist
    output_file = output_dir / 'age_stratified_auc_results.csv'
    if output_file.exists():
        print("\n" + "="*80)
        print("RESULTS ALREADY EXIST - SKIPPING REGENERATION")
        print("="*80)
        print(f"Found existing results: {output_file}")
        print("\nTo regenerate, delete the existing result file first.")
        return
    
    # Store all results
    all_results = []
    
    # Process each age group
    for age_min, age_max, age_group_name in age_groups:
        print(f"\n{'='*80}")
        print(f"PROCESSING AGE GROUP: {age_group_name}")
        print(f"{'='*80}")
        
        # Get mask for this age group
        age_mask = get_age_group_mask(enrollment_ages, age_min, age_max)
        patient_indices = np.where(age_mask)[0].tolist()
        
        n_patients = len(patient_indices)
        print(f"Patients in age group {age_group_name}: {n_patients}")
        
        if n_patients == 0:
            print(f"Warning: No patients in age group {age_group_name}, skipping...")
            continue
        
        # Process each time horizon
        for horizon_name, horizon_years, horizon_type in time_horizons:
            print(f"\n  Processing {horizon_name} ({horizon_type})...")
            
            try:
                if horizon_type == 'static':
                    # Static 10-year (1-year score for 10-year outcome)
                    # Filter data for this age group
                    pi_subset = pi_full[patient_indices]
                    Y_subset = Y_full[patient_indices]
                    E_subset = E_full[patient_indices]
                    pce_df_subset = pce_df_full.iloc[patient_indices].reset_index(drop=True)
                    
                    results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
                        pi=pi_subset,
                        Y_100k=Y_subset,
                        E_100k=E_subset,
                        disease_names=disease_names,
                        pce_df=pce_df_subset,
                        n_bootstraps=args.n_bootstraps,
                        follow_up_duration_years=horizon_years
                    )
                elif horizon_name == '1yr':
                    # 1-year dynamic
                    results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year(
                        pi=pi_full,
                        Y_100k=Y_full,
                        E_100k=E_full,
                        disease_names=disease_names,
                        pce_df=pce_df_full,
                        n_bootstraps=args.n_bootstraps,
                        follow_up_duration_years=1,
                        patient_indices=patient_indices
                    )
                else:
                    # Dynamic predictions (30yr)
                    results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
                        pi=pi_full,
                        Y_100k=Y_full,
                        E_100k=E_full,
                        disease_names=disease_names,
                        pce_df=pce_df_full,
                        n_bootstraps=args.n_bootstraps,
                        follow_up_duration_years=horizon_years,
                        patient_indices=patient_indices
                    )
                
                # Store results
                for disease, metrics in results.items():
                    all_results.append({
                        'Age_Group': age_group_name,
                        'Time_Horizon': horizon_name,
                        'Disease': disease,
                        'AUC': metrics.get('auc', np.nan),
                        'CI_Lower': metrics.get('ci_lower', np.nan),
                        'CI_Upper': metrics.get('ci_upper', np.nan),
                        'N_Events': metrics.get('n_events', 0),
                        'Event_Rate': metrics.get('event_rate', 0.0),
                        'N_Patients': n_patients
                    })
                
                print(f"  ✓ Completed {horizon_name} for age group {age_group_name}")
                
            except Exception as e:
                print(f"  ✗ Error processing {horizon_name} for age group {age_group_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(['Age_Group', 'Time_Horizon', 'Disease'])
        
        # Save results
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print("COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  Age groups: {len(age_groups)}")
        print(f"  Time horizons: {len(time_horizons)}")
        print(f"  Total disease-horizon-age combinations: {len(results_df)}")
        print(f"\nSample results:")
        print(results_df.head(10).to_string())
    else:
        print("\n⚠️  WARNING: No results generated!")

if __name__ == '__main__':
    main()

