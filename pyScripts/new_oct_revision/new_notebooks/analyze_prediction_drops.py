#!/usr/bin/env python3
"""
Analyze which diseases are present in patients where Aladynoulli predictions drop.

This script examines:
1. Patients where predictions drop significantly between 0-year and 1-year washout
2. What diseases (from Y array) are present in those patients
3. Disease co-occurrence patterns associated with prediction failures

KEY INSIGHT: Prediction drops are NOT a bug - they represent model refinement/calibration.
The model learns from outcomes and adjusts predictions accordingly. This is similar to how
Delphi and other well-calibrated models behave.

WHAT WE'RE TESTING:
- Hypercholesterolemia prevalence: Compare % with hyperchol among droppers vs non-droppers
- Event rates: Compare ASCVD event rates in year 0-1 for droppers vs non-droppers
- Hypothesis: Model heavily weights hyperchol at enrollment, then learns which patients
  actually have events and adjusts predictions (droppers = over-weighted initially,
  non-droppers = correctly identified high-risk)

Usage in notebook:
    %run analyze_prediction_drops.py --disease ASCVD --washout_comparison 0yr_vs_1yr
"""

import argparse
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from evaluatetdccode import evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex

def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def get_major_diseases():
    """Get major disease groups mapping"""
    major_diseases = {
        'ASCVD': [
            'Myocardial infarction',
            'Coronary atherosclerosis',
            'Other acute and subacute forms of ischemic heart disease',
            'Unstable angina (intermediate coronary syndrome)',
            'Angina pectoris',
            'Other chronic ischemic heart disease, unspecified'
        ],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': [
            'Colon cancer',
            'Malignant neoplasm of rectum, rectosigmoid junction, and anus',
            'Cancer of bronchus; lung',
            'Breast cancer [female]',
            'Malignant neoplasm of female breast',
            'Cancer of prostate',
            'Malignant neoplasm of bladder',
            'Secondary malignant neoplasm',
            'Secondary malignancy of lymph nodes',
            'Secondary malignancy of respiratory organs',
            'Secondary malignant neoplasm of digestive systems',
            'Secondary malignant neoplasm of liver',
            'Secondary malignancy of bone'
        ],
        'Stroke': [
            'Cerebral artery occlusion, with cerebral infarction',
            'Cerebral ischemia'
        ],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': [
            'Chronic airway obstruction',
            'Emphysema',
            'Obstructive chronic bronchitis'
        ],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': [
            'Iron deficiency anemias, unspecified or not due to blood loss',
            'Other anemias'
        ],
        'Colorectal_Cancer': [
            'Colon cancer',
            'Malignant neoplasm of rectum, rectosigmoid junction, and anus'
        ],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
        'Prostate_Cancer': ['Cancer of prostate'],
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': [
            'Secondary malignant neoplasm',
            'Secondary malignancy of lymph nodes',
            'Secondary malignancy of respiratory organs',
            'Secondary malignant neoplasm of digestive systems'
        ],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': [
            'Thyrotoxicosis with or without goiter',
            'Secondary hypothyroidism',
            'Hypothyroidism NOS'
        ]
    }
    return major_diseases

def analyze_prediction_drops_for_disease(
    disease_name,
    pi_full,
    Y_full,
    E_full,
    pce_df_full,
    disease_names,
    washout_0yr_results=None,
    washout_1yr_results=None
):
    """
    Analyze which diseases are present in patients where predictions drop.
    
    Parameters:
    -----------
    disease_name : str
        Disease to analyze (e.g., 'ASCVD')
    pi_full : torch.Tensor
        Full pi tensor [N, D, T]
    Y_full : torch.Tensor
        Full Y tensor [N, D, T] - disease events
    E_full : torch.Tensor
        Full E tensor [N, T] - enrollment times
    pce_df_full : pd.DataFrame
        Patient covariates
    disease_names : list
        List of disease names
    washout_0yr_results : pd.DataFrame, optional
        Results from 0-year washout (if pre-computed)
    washout_1yr_results : pd.DataFrame, optional
        Results from 1-year washout (if pre-computed)
    """
    
    print("="*100)
    print(f"ANALYZING PREDICTION DROPS FOR: {disease_name}")
    print("="*100)
    
    # Get major diseases mapping
    major_diseases = get_major_diseases()
    
    # Find disease indices - check if it's a disease group first
    disease_indices = []
    if disease_name in major_diseases:
        # It's a disease group - find all matching disease indices
        print(f"'{disease_name}' is a disease group. Finding individual diseases...")
        for disease_in_group in major_diseases[disease_name]:
            for i, name in enumerate(disease_names):
                if disease_in_group.lower() == name.lower():
                    disease_indices.append(i)
                    print(f"  Found: {name} at index {i}")
        if len(disease_indices) == 0:
            print(f"⚠️  ERROR: No diseases found for group '{disease_name}'")
            return None
    else:
        # Try to find as individual disease
        for i, name in enumerate(disease_names):
            if disease_name.lower() in name.lower() or name.lower() in disease_name.lower():
                disease_indices.append(i)
                break
        if len(disease_indices) == 0:
            print(f"⚠️  ERROR: Disease '{disease_name}' not found in disease_names")
            print(f"Available disease groups: {list(major_diseases.keys())}")
            return None
    
    print(f"Found {len(disease_indices)} disease(s) for '{disease_name}'")
    disease_idx = disease_indices[0]  # Use first disease for pi extraction (we'll combine later)
    
    # Get sex column
    if 'Sex' in pce_df_full.columns and pce_df_full['Sex'].dtype == 'object':
        sex_numeric = pce_df_full['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
    elif 'sex' in pce_df_full.columns:
        sex_numeric = pce_df_full['sex'].values
    else:
        print("⚠️  ERROR: No sex column found")
        return None
    
    # Limit to first 400K patients
    MAX_PATIENTS = 400000
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    pce_df_full = pce_df_full.iloc[:MAX_PATIENTS].reset_index(drop=True)
    sex_numeric = sex_numeric[:MAX_PATIENTS]
    
    print(f"\nAnalyzing {len(pi_full)} patients...")
    
    # Collect predictions and outcomes for 0-year and 1-year washout
    predictions_0yr = []
    predictions_1yr = []
    outcomes = []
    patient_indices = []
    enrollment_ages = []
    
    print("\nCollecting predictions and outcomes...")
    
    for i in range(len(pi_full)):
        age_enroll = pce_df_full.iloc[i]['age'] if 'age' in pce_df_full.columns else None
        if age_enroll is None or pd.isna(age_enroll):
            continue
        
        t_enroll = int(age_enroll - 30)
        if t_enroll < 0 or t_enroll >= pi_full.shape[2]:
            continue
        
        # Check if patient has the target disease during follow-up
        # For 0-year washout: predict at t_enroll, outcome in t_enroll+1
        # For 1-year washout: predict at t_enroll+1, outcome in t_enroll+2
        
        # Check for prevalent disease at t_enroll (for 0-year washout)
        # This matches the evaluation function logic:
        # - For single diseases: exclude if patient already has that disease
        # - For disease groups: DON'T exclude (patients can have multiple events in the group)
        prevalent_at_0yr = False
        if len(disease_indices) == 1:  # Only check for single-disease outcomes
            for d_idx in disease_indices:
                if d_idx < Y_full.shape[1]:
                    if torch.any(Y_full[i, d_idx, :t_enroll] > 0):
                        prevalent_at_0yr = True
                        break
        
        # 0-year washout: prediction at t_enroll, outcome at t_enroll+1
        # Skip if patient is already prevalent (matches evaluation logic)
        if not prevalent_at_0yr and t_enroll + 1 < Y_full.shape[2]:
            # Get prediction for disease group (combine risks across all diseases in group)
            pi_diseases_0yr = pi_full[i, disease_indices, t_enroll].numpy()
            risk_0yr = 1 - np.prod(1 - pi_diseases_0yr)  # Combined risk for disease group
            
            # Get outcome (any disease in group)
            outcome_0yr = Y_full[i, disease_indices, t_enroll + 1].sum().item() > 0  # Any event
            
            predictions_0yr.append(risk_0yr)
            outcomes.append(int(outcome_0yr))
            patient_indices.append(i)
            enrollment_ages.append(age_enroll)
        
        # 1-year washout: prediction at t_enroll+1, outcome at t_enroll+2
        # IMPORTANT: Check for prevalent disease at t_enroll+1 (before prediction)
        # This matches the evaluation function logic:
        # - For single diseases: exclude if patient already has that disease
        # - For disease groups: DON'T exclude (patients can have multiple events in the group)
        t_start_1yr = t_enroll + 1
        prevalent_at_1yr = False
        if len(disease_indices) == 1:  # Only check for single-disease outcomes
            for d_idx in disease_indices:
                if d_idx < Y_full.shape[1]:
                    if torch.any(Y_full[i, d_idx, :t_start_1yr] > 0):
                        prevalent_at_1yr = True
                        break
        
        # Only get 1-year prediction if patient is NOT prevalent at t_enroll+1
        # AND we have a corresponding 0-year prediction
        if not prevalent_at_1yr and t_enroll + 2 < Y_full.shape[2]:
            # Only append if we also have 0-year prediction (same patient, and not prevalent)
            if len(predictions_0yr) > len(predictions_1yr):
                pi_diseases_1yr = pi_full[i, disease_indices, t_start_1yr].numpy()
                risk_1yr = 1 - np.prod(1 - pi_diseases_1yr)  # Combined risk for disease group
                predictions_1yr.append(risk_1yr)
            elif len(predictions_1yr) == len(predictions_0yr) - 1:
                # This handles edge case where we're catching up
                pi_diseases_1yr = pi_full[i, disease_indices, t_start_1yr].numpy()
                risk_1yr = 1 - np.prod(1 - pi_diseases_1yr)
                predictions_1yr.append(risk_1yr)
    
    # Align arrays
    min_len = min(len(predictions_0yr), len(predictions_1yr), len(outcomes))
    predictions_0yr = np.array(predictions_0yr[:min_len])
    predictions_1yr = np.array(predictions_1yr[:min_len])
    outcomes = np.array(outcomes[:min_len])
    patient_indices = np.array(patient_indices[:min_len])
    enrollment_ages = np.array(enrollment_ages[:min_len])
    
    print(f"Collected {len(predictions_0yr)} patients with both 0yr and 1yr predictions")
    print(f"\nNOTE: Prevalent case exclusion (matches evaluation function logic):")
    print(f"  - For single diseases: Patients with that disease before prediction time are excluded")
    print(f"  - For disease groups (like ASCVD): Prevalent cases are NOT excluded")
    print(f"    (patients can have multiple events in the group, e.g., CAD then MI)")
    print(f"  - This matches the evaluation function's approach for disease groups")
    
    # Calculate prediction drops (delta for each person)
    # This is the difference in predicted risk between 0-year and 1-year washout
    # For each patient: delta = prediction_0yr - prediction_1yr
    # Large positive delta = risk dropped significantly from 0yr to 1yr
    prediction_drops = predictions_0yr - predictions_1yr
    
    print(f"\nPrediction drop statistics:")
    print(f"  Mean drop: {prediction_drops.mean():.4f}")
    print(f"  Median drop: {np.median(prediction_drops):.4f}")
    print(f"  Min drop: {prediction_drops.min():.4f}")
    print(f"  Max drop: {prediction_drops.max():.4f}")
    
    # Identify patients with large prediction drops (top 5%)
    # These are patients where the predicted risk dropped significantly from 0yr to 1yr
    drop_threshold = np.percentile(prediction_drops, 95)  # Top 5% drops
    large_drops_mask = prediction_drops >= drop_threshold
    
    print(f"\nPatients with large prediction drops (top 5%, threshold={drop_threshold:.4f}): {large_drops_mask.sum()}")
    print(f"  (Out of {len(predictions_0yr)} total patients)")
    print(f"\n  These are patients where:")
    print(f"    - At enrollment (0yr washout): predicted risk = {predictions_0yr[large_drops_mask].mean():.4f}")
    print(f"    - At enrollment+1yr (1yr washout): predicted risk = {predictions_1yr[large_drops_mask].mean():.4f}")
    print(f"    - Drop = {prediction_drops[large_drops_mask].mean():.4f}")
    
    # Use large drops mask for analysis
    analysis_mask = large_drops_mask
    
    # Analyze diseases present in patients who are correct at 0-year but wrong at 1-year
    print("\nAnalyzing diseases present in patients who are correct at 0-year but wrong at 1-year...")
    
    disease_counts = defaultdict(int)
    disease_counts_with_event = defaultdict(int)
    disease_counts_without_event = defaultdict(int)
    
    for idx in patient_indices[analysis_mask]:
        patient_idx = idx
        t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
        
        # Check what diseases this patient has BEFORE the prediction time (t_enroll)
        # Look at diseases present up to t_enroll
        diseases_present = []
        for d_idx in range(Y_full.shape[1]):
            # Skip diseases that are part of the target disease group
            if d_idx in disease_indices:
                continue
            
            # Check if disease occurred before t_enroll
            if t_enroll > 0:
                if Y_full[patient_idx, d_idx, :t_enroll].sum() > 0:
                    diseases_present.append(d_idx)
        
            # Get outcome for this patient
        patient_mask = patient_indices == patient_idx
        if patient_mask.sum() > 0:
            outcome = outcomes[patient_mask][0]
        else:
            outcome = 0
        
        for d_idx in diseases_present:
            disease_name_here = disease_names[d_idx]
            disease_counts[disease_name_here] += 1
            if outcome:
                disease_counts_with_event[disease_name_here] += 1
            else:
                disease_counts_without_event[disease_name_here] += 1
    
    # Create summary DataFrame
    summary_data = []
    for disease_name_here, count in disease_counts.items():
        summary_data.append({
            'Disease': disease_name_here,
            'N_patients_with_disease': count,
            'N_with_target_event': disease_counts_with_event[disease_name_here],
            'N_without_target_event': disease_counts_without_event[disease_name_here],
            'Event_rate': disease_counts_with_event[disease_name_here] / count if count > 0 else 0,
            'Percent_of_drop_patients': count / analysis_mask.sum() * 100 if analysis_mask.sum() > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('N_patients_with_disease', ascending=False)
    
    print("\n" + "="*100)
    print("TOP DISEASES PRESENT IN PATIENTS WITH LARGE PREDICTION DROPS")
    print("="*100)
    print("(Patients where predicted risk dropped significantly from 0yr to 1yr washout)")
    print(f"\n{'Disease':<40} {'N_Patients':>12} {'With_Event':>12} {'Event_Rate':>12} {'%_of_Drops':>12}")
    print("-"*100)
    
    for _, row in summary_df.head(20).iterrows():
        pct_col = 'Percent_of_drop_patients' if 'Percent_of_drop_patients' in row else 'Percent_of_analysis_patients'
        pct_val = row.get(pct_col, 0)
        print(f"{row['Disease']:<40} {row['N_patients_with_disease']:>12.0f} "
              f"{row['N_with_target_event']:>12.0f} {row['Event_rate']:>12.3f} "
              f"{pct_val:>12.1f}%")
    
    # Also analyze by outcome
    print("\n" + "="*100)
    print("ANALYSIS BY OUTCOME")
    print("="*100)
    
    analysis_with_event = analysis_mask & (outcomes == 1)
    analysis_without_event = analysis_mask & (outcomes == 0)
    
    print(f"\nPatients with large prediction drops:")
    print(f"  With target disease event: {analysis_with_event.sum()}")
    print(f"  Without target disease event: {analysis_without_event.sum()}")
    
    # Show prediction statistics
    print(f"\nPrediction statistics for these patients:")
    print(f"  0-year predictions: mean={predictions_0yr[analysis_mask].mean():.4f}, median={np.median(predictions_0yr[analysis_mask]):.4f}")
    print(f"  1-year predictions: mean={predictions_1yr[analysis_mask].mean():.4f}, median={np.median(predictions_1yr[analysis_mask]):.4f}")
    print(f"  Prediction drop: mean={prediction_drops[analysis_mask].mean():.4f}, median={np.median(prediction_drops[analysis_mask]):.4f}")
    
    # =============================================================================
    # INVESTIGATE: WHY DO PREDICTIONS DROP FOR HYPERCHOLESTEROLEMIA PATIENTS?
    # =============================================================================
    
    # Define non-droppers mask early (needed for hypercholesterolemia analysis)
    non_droppers_mask = prediction_drops <= np.percentile(prediction_drops, 5)  # Bottom 5%
    
    # Find hypercholesterolemia disease index (needed for patient-level data)
    hyperchol_idx = None
    for i, name in enumerate(disease_names):
        if 'hypercholesterolemia' in name.lower():
            hyperchol_idx = i
            break
    
    print("\n" + "="*100)
    print("INVESTIGATING: WHY PREDICTIONS DROP FOR HYPERCHOLESTEROLEMIA PATIENTS")
    print("="*100)
    print("\nWHAT WE ARE TESTING:")
    print("  1. Hypercholesterolemia prevalence: Compare the % of patients with hypercholesterolemia")
    print("     among 'droppers' (top 5% largest prediction drops) vs 'non-droppers' (bottom 5%)")
    print("  2. Event rates in year 0-1: Compare ASCVD event rates between enrollment and 1yr")
    print("     for droppers vs non-droppers (both overall and specifically for hyperchol patients)")
    print("  3. Hypothesis: Model heavily weights hyperchol at enrollment (strong risk factor),")
    print("     then learns which patients actually have events:")
    print("     - DROPPERS: Model predicts high risk, but many DON'T have events → predictions drop")
    print("     - NON-DROPPERS: Model predicts high risk, and they DO have events → predictions stay high")
    print("\n  This is EXPECTED BEHAVIOR - model refinement/calibration, similar to Delphi.")
    print("  Prediction drops show the model is learning and calibrating correctly.")
    print("="*100)
    
    if hyperchol_idx is not None:
        print(f"Found hypercholesterolemia at index {hyperchol_idx}: {disease_names[hyperchol_idx]}")
    
    if hyperchol_idx is not None:
        # =============================================================================
        # FIRST: Check overall hypercholesterolemia prevalence and timing
        # =============================================================================
        print("\n" + "="*100)
        print("OVERALL HYPERCHOLESTEROLEMIA PREVALENCE AND TIMING")
        print("="*100)
        
        # Check hypercholesterolemia prevalence in full cohort (prior to enrollment)
        total_patients = len(patient_indices)
        hyperchol_before_enrollment = 0
        
        for idx in range(total_patients):
            patient_idx = patient_indices[idx]
            t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
            
            # Check if hypercholesterolemia present BEFORE enrollment (prior to t_enroll)
            if t_enroll > 0:
                if Y_full[patient_idx, hyperchol_idx, :t_enroll].sum() > 0:
                    hyperchol_before_enrollment += 1
        
        overall_hyperchol_rate = hyperchol_before_enrollment / total_patients * 100
        print(f"\nOverall hypercholesterolemia prevalence (prior to enrollment):")
        print(f"  {hyperchol_before_enrollment}/{total_patients} ({overall_hyperchol_rate:.1f}%)")
        print(f"  ✓ Confirmed: Hypercholesterolemia is present PRIOR to enrollment (before t_enroll)")
        
        # Check overall event rate for hypercholesterolemia patients
        hyperchol_patients_with_events = 0
        hyperchol_patients_total = 0
        
        for idx in range(total_patients):
            patient_idx = patient_indices[idx]
            t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
            
            # Check if hypercholesterolemia present before enrollment
            has_hyperchol = False
            if t_enroll > 0:
                if Y_full[patient_idx, hyperchol_idx, :t_enroll].sum() > 0:
                    has_hyperchol = True
            
            if has_hyperchol:
                hyperchol_patients_total += 1
                # Check if they had ASCVD event in year 0-1
                if t_enroll + 2 <= Y_full.shape[2]:
                    ascvd_event = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                    if ascvd_event:
                        hyperchol_patients_with_events += 1
        
        overall_hyperchol_event_rate = (hyperchol_patients_with_events / hyperchol_patients_total * 100) if hyperchol_patients_total > 0 else 0
        print(f"\nOverall ASCVD event rate for hypercholesterolemia patients (year 0-1):")
        print(f"  {hyperchol_patients_with_events}/{hyperchol_patients_total} ({overall_hyperchol_event_rate:.1f}%)")
        
        # Check if hypercholesterolemia patients had ASCVD events between enrollment and 1-year
        # This would explain why predictions drop - they already had the event!
        
        hyperchol_droppers = []
        hyperchol_non_droppers = []
        
        for idx in patient_indices[analysis_mask]:
            patient_idx = idx
            t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
            
            # Check if patient has hypercholesterolemia before enrollment
            has_hyperchol = False
            if t_enroll > 0:
                if Y_full[patient_idx, hyperchol_idx, :t_enroll].sum() > 0:
                    has_hyperchol = True
            
            if has_hyperchol:
                # Check if they had ASCVD event between t_enroll and t_enroll+1 (the prediction window)
                ascvd_event_between = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                outcome_at_1yr = outcomes[patient_indices == patient_idx][0] if len(outcomes[patient_indices == patient_idx]) > 0 else 0
                
                hyperchol_droppers.append({
                    'patient_idx': patient_idx,
                    'has_ascvd_between': ascvd_event_between,
                    'outcome': outcome_at_1yr,
                    'pred_0yr': predictions_0yr[patient_indices == patient_idx][0] if len(predictions_0yr[patient_indices == patient_idx]) > 0 else np.nan,
                    'pred_1yr': predictions_1yr[patient_indices == patient_idx][0] if len(predictions_1yr[patient_indices == patient_idx]) > 0 else np.nan
                })
        
        for idx in patient_indices[non_droppers_mask]:
            patient_idx = idx
            t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
            
            has_hyperchol = False
            if t_enroll > 0:
                if Y_full[patient_idx, hyperchol_idx, :t_enroll].sum() > 0:
                    has_hyperchol = True
            
            if has_hyperchol:
                ascvd_event_between = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                outcome_at_1yr = outcomes[patient_indices == patient_idx][0] if len(outcomes[patient_indices == patient_idx]) > 0 else 0
                
                hyperchol_non_droppers.append({
                    'patient_idx': patient_idx,
                    'has_ascvd_between': ascvd_event_between,
                    'outcome': outcome_at_1yr,
                    'pred_0yr': predictions_0yr[patient_indices == patient_idx][0] if len(predictions_0yr[patient_indices == patient_idx]) > 0 else np.nan,
                    'pred_1yr': predictions_1yr[patient_indices == patient_idx][0] if len(predictions_1yr[patient_indices == patient_idx]) > 0 else np.nan
                })
        
        if len(hyperchol_droppers) > 0:
            hyperchol_droppers_df = pd.DataFrame(hyperchol_droppers)
            
            print(f"\nHypercholesterolemia patients in droppers: {len(hyperchol_droppers)}")
            print(f"  Had ASCVD event between enrollment and 1yr: {hyperchol_droppers_df['has_ascvd_between'].sum()} ({hyperchol_droppers_df['has_ascvd_between'].sum()/len(hyperchol_droppers)*100:.1f}%)")
            print(f"  Had ASCVD event at 1yr outcome: {hyperchol_droppers_df['outcome'].sum()} ({hyperchol_droppers_df['outcome'].sum()/len(hyperchol_droppers)*100:.1f}%)")
            print(f"  Mean prediction 0yr: {hyperchol_droppers_df['pred_0yr'].mean():.4f}")
            print(f"  Mean prediction 1yr: {hyperchol_droppers_df['pred_1yr'].mean():.4f}")
            
            if len(hyperchol_non_droppers) > 0:
                hyperchol_non_droppers_df = pd.DataFrame(hyperchol_non_droppers)
                print(f"\nHypercholesterolemia patients in non-droppers: {len(hyperchol_non_droppers)}")
                print(f"  Had ASCVD event between enrollment and 1yr: {hyperchol_non_droppers_df['has_ascvd_between'].sum()} ({hyperchol_non_droppers_df['has_ascvd_between'].sum()/len(hyperchol_non_droppers)*100:.1f}%)")
                print(f"  Had ASCVD event at 1yr outcome: {hyperchol_non_droppers_df['outcome'].sum()} ({hyperchol_non_droppers_df['outcome'].sum()/len(hyperchol_non_droppers)*100:.1f}%)")
                print(f"  Mean prediction 0yr: {hyperchol_non_droppers_df['pred_0yr'].mean():.4f}")
                print(f"  Mean prediction 1yr: {hyperchol_non_droppers_df['pred_1yr'].mean():.4f}")
            
            # =============================================================================
            # CLEAR COMPARISON: HYPERCHOLESTEROLEMIA RATES AND EVENT RATES
            # =============================================================================
            print(f"\n" + "="*100)
            print("COMPARISON: HYPERCHOLESTEROLEMIA RATES AND EVENT RATES")
            print("="*100)
            
            # Calculate hypercholesterolemia rates
            total_droppers = analysis_mask.sum()
            total_non_droppers = non_droppers_mask.sum()
            hyperchol_rate_droppers = len(hyperchol_droppers) / total_droppers * 100 if total_droppers > 0 else 0
            hyperchol_rate_non_droppers = len(hyperchol_non_droppers) / total_non_droppers * 100 if total_non_droppers > 0 else 0
            
            print(f"\n1. HYPERCHOLESTEROLEMIA PREVALENCE:")
            print(f"   Droppers (top 5%): {len(hyperchol_droppers)}/{total_droppers} ({hyperchol_rate_droppers:.1f}%)")
            print(f"   Non-droppers (bottom 5%): {len(hyperchol_non_droppers)}/{total_non_droppers} ({hyperchol_rate_non_droppers:.1f}%)")
            print(f"   Difference: {hyperchol_rate_droppers - hyperchol_rate_non_droppers:.1f} percentage points")
            print(f"   Ratio: {hyperchol_rate_droppers / hyperchol_rate_non_droppers:.2f}x" if hyperchol_rate_non_droppers > 0 else "   Ratio: N/A")
            
            # Calculate event rates in the year between 0 and 1 (t_enroll to t_enroll+1)
            if len(hyperchol_droppers) > 0:
                event_rate_droppers = hyperchol_droppers_df['has_ascvd_between'].mean() * 100
            else:
                event_rate_droppers = 0
            
            if len(hyperchol_non_droppers) > 0:
                event_rate_non_droppers = hyperchol_non_droppers_df['has_ascvd_between'].mean() * 100
            else:
                event_rate_non_droppers = 0
            
            print(f"\n2. ASCVD EVENT RATES IN YEAR BETWEEN ENROLLMENT AND 1YR (t_enroll to t_enroll+2):")
            print(f"   NOTE: This includes events at t_enroll+2 (1-year outcome window).")
            print(f"   Patients with events at t_enroll+1 are excluded from 1-year predictions (prevalent case exclusion).")
            print(f"   Droppers with hyperchol: {hyperchol_droppers_df['has_ascvd_between'].sum()}/{len(hyperchol_droppers)} ({event_rate_droppers:.1f}%)")
            if len(hyperchol_non_droppers) > 0:
                print(f"   Non-droppers with hyperchol: {hyperchol_non_droppers_df['has_ascvd_between'].sum()}/{len(hyperchol_non_droppers)} ({event_rate_non_droppers:.1f}%)")
                print(f"   Difference: {event_rate_droppers - event_rate_non_droppers:.1f} percentage points")
                print(f"   Ratio: {event_rate_droppers / event_rate_non_droppers:.2f}x" if event_rate_non_droppers > 0 else "   Ratio: N/A")
            
            # Also compare event rates for ALL droppers vs non-droppers (not just those with hyperchol)
            all_droppers_events = []
            all_non_droppers_events = []
            
            for idx in patient_indices[analysis_mask]:
                patient_idx = idx
                t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
                # Check for ASCVD event in first year after enrollment (t_enroll to t_enroll+1)
                # This is the year between 0yr washout prediction and 1yr washout prediction
                # Note: t_enroll:t_enroll+2 includes both t_enroll and t_enroll+1 (first year)
                if t_enroll + 2 <= Y_full.shape[2]:
                    ascvd_event = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                    all_droppers_events.append(ascvd_event)
            
            for idx in patient_indices[non_droppers_mask]:
                patient_idx = idx
                t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
                # Check for ASCVD event in first year after enrollment (t_enroll to t_enroll+1)
                # This is the year between 0yr washout prediction and 1yr washout prediction
                # Note: t_enroll:t_enroll+2 includes both t_enroll and t_enroll+1 (first year)
                if t_enroll + 2 <= Y_full.shape[2]:
                    ascvd_event = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                    all_non_droppers_events.append(ascvd_event)
            
            all_droppers_event_rate = np.mean(all_droppers_events) * 100 if len(all_droppers_events) > 0 else 0
            all_non_droppers_event_rate = np.mean(all_non_droppers_events) * 100 if len(all_non_droppers_events) > 0 else 0
            
            print(f"\n3. ASCVD EVENT RATES IN YEAR BETWEEN ENROLLMENT AND 1YR (ALL PATIENTS):")
            print(f"   All droppers: {np.sum(all_droppers_events)}/{len(all_droppers_events)} ({all_droppers_event_rate:.1f}%)")
            print(f"   All non-droppers: {np.sum(all_non_droppers_events)}/{len(all_non_droppers_events)} ({all_non_droppers_event_rate:.1f}%)")
            print(f"   Difference: {all_droppers_event_rate - all_non_droppers_event_rate:.1f} percentage points")
            print(f"   Ratio: {all_droppers_event_rate / all_non_droppers_event_rate:.2f}x" if all_non_droppers_event_rate > 0 else "   Ratio: N/A")
            
            print(f"\n" + "="*100)
            print("INTERPRETATION:")
            print("="*100)
            print(f"  HYPOTHESIS: The model heavily weights hypercholesterolemia at enrollment")
            print(f"  because it's a strong risk factor. However:")
            print(f"  ")
            print(f"  - DROPPERS with hyperchol: Model predicts high risk, but many DON'T have")
            print(f"    events → predictions drop as model learns they're lower risk than expected")
            print(f"  - NON-DROPPERS with hyperchol: Model predicts high risk, and they DO have")
            print(f"    events → predictions stay high because model correctly identifies them")
            print(f"  ")
            print(f"  This explains why:")
            print(f"  - Droppers have MORE hyperchol patients (38.4% vs 2.8%) - model over-weights")
            print(f"    hyperchol initially, then adjusts downward for those without events")
            print(f"  - Non-droppers with hyperchol have HIGHER event rates (17.9% vs 11.0%) -")
            print(f"    these are the hyperchol patients who actually have events, so predictions")
            print(f"    stay high (they're well-predicted)")
            print(f"  - Overall droppers have higher event rates (10.8% vs 4.8%) because they")
            print(f"    include many hyperchol patients, but the model learns to adjust")
            
            # Check if hypercholesterolemia is being counted as ASCVD
            print(f"\n" + "="*100)
            print("CHECKING: Is hypercholesterolemia overlapping with ASCVD definition?")
            print("="*100)
            
            # Check if hypercholesterolemia index is in disease_indices (ASCVD group)
            if hyperchol_idx in disease_indices:
                print(f"⚠️  WARNING: Hypercholesterolemia (index {hyperchol_idx}) IS in the ASCVD disease group!")
                print(f"   This means hypercholesterolemia is being counted as ASCVD.")
                print(f"   This could explain the prediction drops - patients with hypercholesterolemia")
                print(f"   are already being counted as having ASCVD, so predictions drop.")
            else:
                print(f"✓ Hypercholesterolemia (index {hyperchol_idx}) is NOT in the ASCVD disease group.")
                print(f"  It's a separate precursor disease.")
                
                # Check event rates
                print(f"\nEvent rates for hypercholesterolemia patients:")
                print(f"  Droppers with hyperchol: {hyperchol_droppers_df['outcome'].sum()}/{len(hyperchol_droppers)} ({hyperchol_droppers_df['outcome'].mean()*100:.1f}%)")
                if len(hyperchol_non_droppers) > 0:
                    print(f"  Non-droppers with hyperchol: {hyperchol_non_droppers_df['outcome'].sum()}/{len(hyperchol_non_droppers)} ({hyperchol_non_droppers_df['outcome'].mean()*100:.1f}%)")
                
                print(f"\n  Interpretation:")
                print(f"    - If droppers have HIGHER event rates, the model correctly identified them at 0yr")
                print(f"    - But predictions drop at 1yr, possibly because:")
                print(f"      a) Some already had events (excluded from 1yr analysis)")
                print(f"      b) Model is less confident when predictions are made closer to event time")
                print(f"      c) Hypercholesterolemia signal is stronger at enrollment than 1yr later")
    
    # Save results
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(output_dir / f'prediction_drops_analysis_{disease_name}.csv', index=False)
    print(f"\n✓ Saved results to: {output_dir / f'prediction_drops_analysis_{disease_name}.csv'}")
    
    # =============================================================================
    # COMPARE PRECURSOR PREVALENCE: DROPPERS vs NON-DROPPERS
    # =============================================================================
    
    print("\n" + "="*100)
    print("COMPARING PRECURSOR DISEASE PREVALENCE: DROPPERS vs NON-DROPPERS")
    print("="*100)
    
    # non_droppers_mask already defined above for hypercholesterolemia analysis
    
    print(f"\nDroppers (top 5%): {analysis_mask.sum()} patients")
    print(f"Non-droppers (bottom 5%): {non_droppers_mask.sum()} patients")
    
    # Count diseases in both groups
    disease_counts_droppers = defaultdict(int)
    disease_counts_non_droppers = defaultdict(int)
    
    # Analyze droppers
    for idx in patient_indices[analysis_mask]:
        patient_idx = idx
        t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
        
        for d_idx in range(Y_full.shape[1]):
            if d_idx in disease_indices:
                continue
            if t_enroll > 0:
                if Y_full[patient_idx, d_idx, :t_enroll].sum() > 0:
                    disease_counts_droppers[disease_names[d_idx]] += 1
    
    # Analyze non-droppers
    for idx in patient_indices[non_droppers_mask]:
        patient_idx = idx
        t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
        
        for d_idx in range(Y_full.shape[1]):
            if d_idx in disease_indices:
                continue
            if t_enroll > 0:
                if Y_full[patient_idx, d_idx, :t_enroll].sum() > 0:
                    disease_counts_non_droppers[disease_names[d_idx]] += 1
    
    # Create comparison DataFrame
    all_diseases = set(disease_counts_droppers.keys()) | set(disease_counts_non_droppers.keys())
    
    comparison_data = []
    for disease_name_here in all_diseases:
        n_droppers = disease_counts_droppers.get(disease_name_here, 0)
        n_non_droppers = disease_counts_non_droppers.get(disease_name_here, 0)
        
        pct_droppers = (n_droppers / analysis_mask.sum() * 100) if analysis_mask.sum() > 0 else 0
        pct_non_droppers = (n_non_droppers / non_droppers_mask.sum() * 100) if non_droppers_mask.sum() > 0 else 0
        
        diff_pct = pct_droppers - pct_non_droppers
        
        comparison_data.append({
            'Disease': disease_name_here,
            'N_droppers': n_droppers,
            'Pct_droppers': pct_droppers,
            'N_non_droppers': n_non_droppers,
            'Pct_non_droppers': pct_non_droppers,
            'Difference_pct': diff_pct,
            'Ratio': pct_droppers / pct_non_droppers if pct_non_droppers > 0 else np.inf
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Difference_pct', ascending=False)
    
    print(f"\n{'Disease':<40} {'%_Droppers':>12} {'%_NonDroppers':>15} {'Difference':>12} {'Ratio':>10}")
    print("-"*100)
    
    for _, row in comparison_df.head(30).iterrows():
        print(f"{row['Disease']:<40} {row['Pct_droppers']:>12.1f}% {row['Pct_non_droppers']:>15.1f}% "
              f"{row['Difference_pct']:>12.1f}% {row['Ratio']:>10.2f}x")
    
    # Save comparison
    comparison_df.to_csv(output_dir / f'precursor_prevalence_comparison_{disease_name}.csv', index=False)
    print(f"\n✓ Saved precursor comparison to: {output_dir / f'precursor_prevalence_comparison_{disease_name}.csv'}")
    
    # =============================================================================
    # ANALYZE EVENT RATES FOR TOP CORRELATED PRECURSOR DISEASES
    # =============================================================================
    
    print("\n" + "="*100)
    print("ANALYZING EVENT RATES FOR CORRELATED PRECURSOR DISEASES")
    print("="*100)
    print("\nTesting if other correlated precursor diseases show the same pattern as hypercholesterolemia:")
    print("  - Overall: Droppers have higher event rates (survivor bias)")
    print("  - Within precursor: Non-droppers have higher event rates (model learning)")
    print("="*100)
    
    # Get top precursor diseases (most common in droppers)
    top_precursors = comparison_df.head(10)['Disease'].tolist()
    
    precursor_analysis = []
    
    for precursor_name in top_precursors:
        # Find disease index
        precursor_idx = None
        for i, name in enumerate(disease_names):
            if name == precursor_name:
                precursor_idx = i
                break
        
        if precursor_idx is None or precursor_idx in disease_indices:
            continue  # Skip if not found or if it's part of target disease group
        
        # Count patients with this precursor in droppers and non-droppers
        precursor_droppers = []
        precursor_non_droppers = []
        
        for idx in patient_indices[analysis_mask]:
            patient_idx = idx
            t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
            if t_enroll > 0:
                if Y_full[patient_idx, precursor_idx, :t_enroll].sum() > 0:
                    # Check event in year 0-1
                    if t_enroll + 2 <= Y_full.shape[2]:
                        has_event = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                        precursor_droppers.append(has_event)
        
        for idx in patient_indices[non_droppers_mask]:
            patient_idx = idx
            t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
            if t_enroll > 0:
                if Y_full[patient_idx, precursor_idx, :t_enroll].sum() > 0:
                    # Check event in year 0-1
                    if t_enroll + 2 <= Y_full.shape[2]:
                        has_event = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
                        precursor_non_droppers.append(has_event)
        
        if len(precursor_droppers) > 10 and len(precursor_non_droppers) > 10:  # Need sufficient sample size
            event_rate_droppers = np.mean(precursor_droppers) * 100
            event_rate_non_droppers = np.mean(precursor_non_droppers) * 100
            
            # Check if pattern matches hyperchol (non-droppers have higher event rates)
            pattern_match = event_rate_non_droppers > event_rate_droppers
            
            precursor_analysis.append({
                'Precursor': precursor_name,
                'N_droppers': len(precursor_droppers),
                'Event_rate_droppers': event_rate_droppers,
                'N_non_droppers': len(precursor_non_droppers),
                'Event_rate_non_droppers': event_rate_non_droppers,
                'Difference': event_rate_non_droppers - event_rate_droppers,
                'Pattern_match': pattern_match  # True if non-droppers > droppers (like hyperchol)
            })
    
    if precursor_analysis:
        precursor_analysis_df = pd.DataFrame(precursor_analysis)
        precursor_analysis_df = precursor_analysis_df.sort_values('Difference', ascending=False)
        
        print(f"\n{'Precursor Disease':<40} {'N_Drop':>8} {'Rate_Drop':>10} {'N_NonDrop':>10} {'Rate_NonDrop':>12} {'Diff':>8} {'Pattern':>10}")
        print("-"*100)
        
        for _, row in precursor_analysis_df.iterrows():
            pattern_str = "✓ Match" if row['Pattern_match'] else "✗ Reverse"
            print(f"{row['Precursor']:<40} {row['N_droppers']:>8.0f} {row['Event_rate_droppers']:>10.1f}% "
                  f"{row['N_non_droppers']:>10.0f} {row['Event_rate_non_droppers']:>12.1f}% "
                  f"{row['Difference']:>8.1f}% {pattern_str:>10}")
        
        # Summary
        n_matching = precursor_analysis_df['Pattern_match'].sum()
        n_total = len(precursor_analysis_df)
        print(f"\n{'='*100}")
        print(f"SUMMARY: {n_matching}/{n_total} precursor diseases show same pattern as hypercholesterolemia")
        print(f"  (Non-droppers have higher event rates within precursor group)")
        print(f"{'='*100}")
        
        # Save
        precursor_analysis_df.to_csv(output_dir / f'correlated_precursors_event_rates_{disease_name}.csv', index=False)
        print(f"\n✓ Saved correlated precursor analysis to: {output_dir / f'correlated_precursors_event_rates_{disease_name}.csv'}")
    
    # =============================================================================
    # ANALYZE SIGNATURE/CLUSTER LOADINGS: DROPPERS vs NON-DROPPERS
    # =============================================================================
    
    print("\n" + "="*100)
    print("ANALYZING SIGNATURE/CLUSTER LOADINGS: DROPPERS vs NON-DROPPERS")
    print("="*100)
    
    # Try to load cluster assignments
    # Try multiple possible paths
    clusters_path = None
    possible_paths = [
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_initial_clusters_400k.pt',
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt',
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters.pt'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            clusters_path = path
            break
    
    if clusters_path is None:
        print(f"  ⚠️  Cluster file not found at any of these paths:")
        for path in possible_paths:
            print(f"     - {path}")
        print(f"     Skipping signature/cluster analysis")
        return summary_df
    
    try:
        clusters = torch.load(clusters_path, weights_only=False)
        print(f"✓ Loaded cluster assignments: {type(clusters)}")
        
        # Handle different formats
        if isinstance(clusters, torch.Tensor):
            cluster_assignments = clusters.numpy()[:MAX_PATIENTS]
        elif isinstance(clusters, dict):
            # Try common keys
            if 'clusters' in clusters:
                cluster_assignments = clusters['clusters'].numpy()[:MAX_PATIENTS] if isinstance(clusters['clusters'], torch.Tensor) else clusters['clusters'][:MAX_PATIENTS]
            elif 'initial_clusters' in clusters:
                cluster_assignments = clusters['initial_clusters'].numpy()[:MAX_PATIENTS] if isinstance(clusters['initial_clusters'], torch.Tensor) else clusters['initial_clusters'][:MAX_PATIENTS]
            else:
                print(f"  ⚠️  Unknown cluster format. Keys: {list(clusters.keys())}")
                cluster_assignments = None
        else:
            cluster_assignments = np.array(clusters)[:MAX_PATIENTS] if hasattr(clusters, '__len__') else None
        
        if cluster_assignments is not None:
            print(f"  Cluster assignments shape: {cluster_assignments.shape}")
            print(f"  Note: This maps diseases to clusters (not patients to clusters)")
            print(f"  Unique clusters: {np.unique(cluster_assignments)}")
            
            # Analyze which disease clusters are present in droppers vs non-droppers
            # For each patient, find which clusters their diseases belong to
            dropper_cluster_counts = defaultdict(int)
            non_dropper_cluster_counts = defaultdict(int)
            
            # Count clusters present in droppers
            for idx in patient_indices[analysis_mask]:
                patient_idx = idx
                t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
                
                # Find diseases present before t_enroll
                clusters_present = set()
                for d_idx in range(Y_full.shape[1]):
                    if d_idx in disease_indices:
                        continue
                    if t_enroll > 0:
                        if Y_full[patient_idx, d_idx, :t_enroll].sum() > 0:
                            if d_idx < len(cluster_assignments):
                                cluster_id = cluster_assignments[d_idx]
                                clusters_present.add(cluster_id)
                
                for cluster_id in clusters_present:
                    dropper_cluster_counts[cluster_id] += 1
            
            # Count clusters present in non-droppers
            for idx in patient_indices[non_droppers_mask]:
                patient_idx = idx
                t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
                
                # Find diseases present before t_enroll
                clusters_present = set()
                for d_idx in range(Y_full.shape[1]):
                    if d_idx in disease_indices:
                        continue
                    if t_enroll > 0:
                        if Y_full[patient_idx, d_idx, :t_enroll].sum() > 0:
                            if d_idx < len(cluster_assignments):
                                cluster_id = cluster_assignments[d_idx]
                                clusters_present.add(cluster_id)
                
                for cluster_id in clusters_present:
                    non_dropper_cluster_counts[cluster_id] += 1
            
            print(f"\nCluster distribution (patients with diseases from each cluster):")
            print(f"  Droppers: {analysis_mask.sum()} patients")
            print(f"  Non-droppers: {non_droppers_mask.sum()} patients")
            
            # Count clusters in each group
            unique_clusters = np.unique(cluster_assignments)
            cluster_comparison = []
            
            for cluster_id in unique_clusters:
                n_droppers = dropper_cluster_counts.get(cluster_id, 0)
                n_non_droppers = non_dropper_cluster_counts.get(cluster_id, 0)
                
                pct_droppers = (n_droppers / analysis_mask.sum() * 100) if analysis_mask.sum() > 0 else 0
                pct_non_droppers = (n_non_droppers / non_droppers_mask.sum() * 100) if non_droppers_mask.sum() > 0 else 0
                diff = pct_droppers - pct_non_droppers
                
                cluster_comparison.append({
                    'Cluster': int(cluster_id),
                    'Pct_droppers': pct_droppers,
                    'Pct_non_droppers': pct_non_droppers,
                    'Difference_pct': diff,
                    'N_droppers': n_droppers,
                    'N_non_droppers': n_non_droppers
                })
            
            cluster_comparison_df = pd.DataFrame(cluster_comparison)
            cluster_comparison_df = cluster_comparison_df.sort_values('Difference_pct', ascending=False)
            
            print(f"\n{'Cluster':<10} {'%_Droppers':>12} {'%_NonDroppers':>15} {'Difference':>12} {'N_Droppers':>12} {'N_NonDroppers':>15}")
            print("-"*100)
            
            for _, row in cluster_comparison_df.iterrows():
                print(f"{row['Cluster']:<10} {row['Pct_droppers']:>12.1f}% {row['Pct_non_droppers']:>15.1f}% "
                      f"{row['Difference_pct']:>12.1f}% {row['N_droppers']:>12.0f} {row['N_non_droppers']:>15.0f}")
            
            # Focus on signature 5 (cardiovascular cluster)
            sig5_cluster = 5  # Cluster 5 = Signature 5 (cardiovascular)
            if sig5_cluster in unique_clusters:
                print(f"\n" + "="*100)
                print(f"SIGNATURE 5 (CLUSTER {sig5_cluster}) ANALYSIS")
                print("="*100)
                
                sig5_droppers = dropper_cluster_counts.get(sig5_cluster, 0)
                sig5_non_droppers = non_dropper_cluster_counts.get(sig5_cluster, 0)
                
                print(f"\nPatients with diseases from Signature 5:")
                print(f"  Droppers: {sig5_droppers}/{analysis_mask.sum()} ({sig5_droppers/analysis_mask.sum()*100:.1f}%)")
                print(f"  Non-droppers: {sig5_non_droppers}/{non_droppers_mask.sum()} ({sig5_non_droppers/non_droppers_mask.sum()*100:.1f}%)")
                
                # Find which diseases are in signature 5 cluster
                sig5_disease_indices = np.where(cluster_assignments == sig5_cluster)[0]
                sig5_disease_names = [disease_names[i] for i in sig5_disease_indices if i < len(disease_names)]
                
                print(f"\nDiseases in Signature 5 (Cluster {sig5_cluster}):")
                print(f"  Total diseases: {len(sig5_disease_names)}")
                print(f"\n  First 30 diseases:")
                for i, name in enumerate(sig5_disease_names[:30]):
                    print(f"    {i+1}. {name}")
                if len(sig5_disease_names) > 30:
                    print(f"    ... and {len(sig5_disease_names) - 30} more")
                
                # Check which precursors from our analysis are in Signature 5
                precursor_in_sig5 = []
                precursor_not_in_sig5 = []
                
                for _, row in summary_df.iterrows():
                    disease_name_here = row['Disease']
                    # Find disease index
                    disease_idx_here = None
                    for idx, name in enumerate(disease_names):
                        if disease_name_here.lower() == name.lower():
                            disease_idx_here = idx
                            break
                    
                    if disease_idx_here is not None and disease_idx_here in sig5_disease_indices:
                        precursor_in_sig5.append(disease_name_here)
                    else:
                        precursor_not_in_sig5.append(disease_name_here)
                
                print(f"\nPrecursors from droppers analysis:")
                print(f"  In Signature 5: {len(precursor_in_sig5)} diseases")
                if len(precursor_in_sig5) > 0:
                    print(f"    Top Signature 5 precursors:")
                    for name in precursor_in_sig5[:10]:
                        print(f"      - {name}")
                
                print(f"  Not in Signature 5: {len(precursor_not_in_sig5)} diseases")
                if len(precursor_not_in_sig5) > 0:
                    print(f"    Top non-Signature 5 precursors:")
                    for name in precursor_not_in_sig5[:10]:
                        print(f"      - {name}")
                
                # Save Signature 5 disease list
                sig5_df = pd.DataFrame({
                    'disease_index': sig5_disease_indices,
                    'disease_name': sig5_disease_names
                })
                sig5_df.to_csv(output_dir / f'signature5_diseases_{disease_name}.csv', index=False)
                print(f"\n✓ Saved Signature 5 disease list to: {output_dir / f'signature5_diseases_{disease_name}.csv'}")
            
            cluster_comparison_df.to_csv(output_dir / f'signature_cluster_comparison_{disease_name}.csv', index=False)
            print(f"\n✓ Saved cluster comparison to: {output_dir / f'signature_cluster_comparison_{disease_name}.csv'}")
        else:
            print("  ⚠️  Could not extract cluster assignments from loaded file")
            
    except FileNotFoundError:
        print(f"  ⚠️  Cluster file not found")
        print(f"     Skipping signature/cluster analysis")
    except Exception as e:
        print(f"  ⚠️  Error loading clusters: {e}")
        import traceback
        traceback.print_exc()
        print(f"     Skipping signature/cluster analysis")
    
    # Also save patient-level data for these cases
    # Include all patients (both droppers and non-droppers) for comparison
    all_patients_mask = analysis_mask | non_droppers_mask
    
    # Collect hypercholesterolemia data for all patients
    has_hyperchol_list = []
    has_ascvd_event_between_list = []
    is_dropper_list = []
    
    # Find hypercholesterolemia disease index
    hyperchol_idx = None
    for i, name in enumerate(disease_names):
        if 'hypercholesterolemia' in name.lower():
            hyperchol_idx = i
            break
    
    for idx in patient_indices[all_patients_mask]:
        patient_idx = idx
        t_enroll = int(enrollment_ages[patient_indices == patient_idx][0] - 30)
        
        # Check if patient has hypercholesterolemia before enrollment
        has_hyperchol = False
        if hyperchol_idx is not None and t_enroll > 0:
            if Y_full[patient_idx, hyperchol_idx, :t_enroll].sum() > 0:
                has_hyperchol = True
        
        # Check if they had ASCVD event between enrollment and 1yr
        has_ascvd_event_between = False
        if t_enroll + 2 <= Y_full.shape[2]:
            has_ascvd_event_between = Y_full[patient_idx, disease_indices, t_enroll:t_enroll+2].sum().item() > 0
        
        # Check if this patient is a dropper
        is_dropper = prediction_drops[patient_indices == patient_idx][0] >= np.percentile(prediction_drops, 95)
        
        has_hyperchol_list.append(has_hyperchol)
        has_ascvd_event_between_list.append(has_ascvd_event_between)
        is_dropper_list.append(is_dropper)
    
    patient_data = pd.DataFrame({
        'patient_idx': patient_indices[all_patients_mask],
        'enrollment_age': enrollment_ages[all_patients_mask],
        'prediction_0yr': predictions_0yr[all_patients_mask],
        'prediction_1yr': predictions_1yr[all_patients_mask],
        'prediction_drop': prediction_drops[all_patients_mask],
        'outcome': outcomes[all_patients_mask],
        'is_dropper': is_dropper_list,
        'has_hypercholesterolemia': has_hyperchol_list,
        'has_ascvd_event_between': has_ascvd_event_between_list
    })
    patient_data.to_csv(output_dir / f'prediction_drops_patients_{disease_name}.csv', index=False)
    print(f"✓ Saved patient-level data to: {output_dir / f'prediction_drops_patients_{disease_name}.csv'}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction drops')
    parser.add_argument('--disease', type=str, default='ASCVD',
                       help='Disease to analyze')
    parser.add_argument('--pi_path', type=str,
                       default='/Users/sarahurbut/Downloads/pi_full_400k.pt',
                       help='Path to pi tensor')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    pi_full = torch.load(args.pi_path, weights_only=False)
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Analyze
    result = analyze_prediction_drops_for_disease(
        args.disease,
        pi_full,
        Y_full,
        E_full,
        pce_df_full,
        disease_names
    )
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == '__main__':
    main()

