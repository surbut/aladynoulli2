"""
Code snippets to verify Y and E arrays are correct for different scenarios.
Paste these into your notebook to check specific cases.
"""

import numpy as np
import pandas as pd
import torch

# ============================================================================
# 1. Person who had an event - verify Y=1 at correct timepoint and E is appropriate
# ============================================================================

def verify_event_patient(Y, E, E_corrected, patient_idx, disease_idx, age_offset=30):
    """
    Verify a patient who had a disease event.
    
    Parameters:
    -----------
    Y : np.ndarray, shape (N, D, T)
    E : np.ndarray, shape (N, D, T) 
    E_corrected : np.ndarray, shape (N, D)
    patient_idx : int - patient index to check
    disease_idx : int - disease index to check
    age_offset : int - age offset (default 30)
    """
    print("="*60)
    print(f"1. VERIFYING EVENT PATIENT")
    print("="*60)
    print(f"Patient index: {patient_idx}")
    print(f"Disease index: {disease_idx}")
    print()
    
    # Check Y - should have 1 at the event timepoint
    Y_patient_disease = Y[patient_idx, disease_idx, :]
    event_timepoints = np.where(Y_patient_disease == 1)[0]
    
    print("Y values (patient, disease, timepoints):")
    print(f"  Event occurred at timepoints: {event_timepoints}")
    if len(event_timepoints) > 0:
        for t in event_timepoints:
            age_at_event = age_offset + t
            print(f"    Timepoint {t} = Age {age_at_event}: Y = {Y_patient_disease[t]}")
    else:
        print("  ⚠️ No events found for this patient-disease pair!")
    print()
    
    # Check E - should be >= event timepoint for all timepoints before/at event
    E_patient_disease = E[patient_idx, disease_idx, :]
    E_corrected_patient_disease = E_corrected[patient_idx, disease_idx]
    
    print("E values (patient, disease, timepoints):")
    print(f"  E_corrected (max censor age): {E_corrected_patient_disease}")
    print(f"  E at first 10 timepoints: {E_patient_disease[:10]}")
    print(f"  E at event timepoint(s): {[E_patient_disease[t] for t in event_timepoints] if len(event_timepoints) > 0 else 'N/A'}")
    print()
    
    # Verify logic: E should be >= event timepoint for all timepoints up to event
    if len(event_timepoints) > 0:
        first_event_tp = event_timepoints[0]
        print("Verification:")
        print(f"  First event at timepoint {first_event_tp} (age {age_offset + first_event_tp})")
        print(f"  E_corrected = {E_corrected_patient_disease} (age {E_corrected_patient_disease})")
        print(f"  E_corrected >= first_event_age? {E_corrected_patient_disease >= (age_offset + first_event_tp)}")
        
        # Check E values before event
        if first_event_tp > 0:
            E_before_event = E_patient_disease[:first_event_tp]
            print(f"  E values before event (should all be >= {first_event_tp+1}): {E_before_event[:5]}...")
            print(f"  All E >= event timepoint? {np.all(E_before_event >= first_event_tp+1)}")
    
    return event_timepoints


# ============================================================================
# 2. Person diagnosed before/after time period - verify Y and E are appropriate
# ============================================================================

def verify_early_late_diagnosis_patient(Y, E, E_corrected, patient_idx, disease_idx, 
                                       icd10_df, patient_names, age_offset=30):
    """
    Verify a patient with diagnosis before or after the time period.
    
    Parameters:
    -----------
    Y : np.ndarray, shape (N, D, T)
    E : np.ndarray, shape (N, D, T)
    E_corrected : np.ndarray, shape (N, D)
    patient_idx : int
    disease_idx : int
    icd10_df : pd.DataFrame - original ICD10 data
    patient_names : list - patient IDs
    age_offset : int
    """
    print("="*60)
    print(f"2. VERIFYING EARLY/LATE DIAGNOSIS PATIENT")
    print("="*60)
    print(f"Patient index: {patient_idx}")
    print(f"Patient ID: {patient_names[patient_idx]}")
    print(f"Disease index: {disease_idx}")
    print()
    
    # Get patient's diagnoses from original data
    patient_id = patient_names[patient_idx]
    patient_diagnoses = icd10_df[icd10_df['eid'] == patient_id].copy()
    
    print("Patient's diagnoses from original data:")
    print(patient_diagnoses.head(10))
    print()
    
    # Check for diagnoses before age_offset
    early_diagnoses = patient_diagnoses[patient_diagnoses['age_diag'] < age_offset]
    print(f"Diagnoses BEFORE age {age_offset}: {len(early_diagnoses)}")
    if len(early_diagnoses) > 0:
        print(early_diagnoses[['diag_icd10', 'age_diag']].head())
        print(f"  Earliest diagnosis age: {early_diagnoses['age_diag'].min()}")
    print()
    
    # Check for diagnoses after max timepoint
    max_age = age_offset + Y.shape[2] - 1
    late_diagnoses = patient_diagnoses[patient_diagnoses['age_diag'] > max_age]
    print(f"Diagnoses AFTER age {max_age}: {len(late_diagnoses)}")
    if len(late_diagnoses) > 0:
        print(late_diagnoses[['diag_icd10', 'age_diag']].head())
    print()
    
    # Check Y - should be 0 for all timepoints if diagnosis was before/after
    Y_patient_disease = Y[patient_idx, disease_idx, :]
    disease_code = icd10_df['diag_icd10'].unique()[disease_idx] if hasattr(icd10_df['diag_icd10'], 'unique') else None
    
    print(f"Y values for disease {disease_code}:")
    print(f"  Y sum (should be 0 if before/after): {Y_patient_disease.sum()}")
    print(f"  Y values: {Y_patient_disease[:10]}...")
    print()
    
    # Check E - if max_censor < age_offset, E should be 0 or set to age_offset
    E_corrected_patient_disease = E_corrected[patient_idx, disease_idx]
    max_censor_age = patient_diagnoses['age_diag'].max() if len(patient_diagnoses) > 0 else None
    
    print("E values:")
    print(f"  Max censor age from data: {max_censor_age}")
    print(f"  E_corrected: {E_corrected_patient_disease}")
    
    if max_censor_age is not None:
        if max_censor_age < age_offset:
            print(f"  ⚠️ Max censor ({max_censor_age}) < age_offset ({age_offset})")
            print(f"  E should be 0 or set to {age_offset}")
        elif max_censor_age > max_age:
            print(f"  ⚠️ Max censor ({max_censor_age}) > max age ({max_age})")
            print(f"  E should be capped at {max_age}")
        else:
            print(f"  ✓ Max censor ({max_censor_age}) is within observation window")
    
    E_patient_disease = E[patient_idx, disease_idx, :]
    print(f"  E array (first 10): {E_patient_disease[:10]}")
    print(f"  E array sum: {E_patient_disease.sum()}")
    
    return early_diagnoses, late_diagnoses


# ============================================================================
# 3. Person who left before end of follow-up - verify E is set to censor time
# ============================================================================

def verify_censored_patient(Y, E, E_corrected, patient_idx, disease_idx,
                            max_censor_df, patient_names, age_offset=30, T=None):
    """
    Verify a patient who was censored (left before end of follow-up).
    
    Parameters:
    -----------
    Y : np.ndarray, shape (N, D, T)
    E : np.ndarray, shape (N, D, T)
    E_corrected : np.ndarray, shape (N, D)
    patient_idx : int
    disease_idx : int
    max_censor_df : pd.DataFrame - max censor dataframe
    patient_names : list
    age_offset : int
    T : int - number of timepoints
    """
    print("="*60)
    print(f"3. VERIFYING CENSORED PATIENT (LEFT BEFORE END)")
    print("="*60)
    print(f"Patient index: {patient_idx}")
    print(f"Patient ID: {patient_names[patient_idx]}")
    print(f"Disease index: {disease_idx}")
    print()
    
    # Get max censor for this patient
    patient_id = patient_names[patient_idx]
    max_censor_row = max_censor_df[max_censor_df['eid'] == patient_id]
    
    if len(max_censor_row) > 0:
        max_censor_age = max_censor_row['max_censor'].values[0]
        print(f"Max censor age: {max_censor_age}")
        
        # Convert to timepoint
        max_censor_timepoint = int(max_censor_age - age_offset)
        if T is None:
            T = E.shape[2]
        max_timepoint_in_array = min(max_censor_timepoint, T - 1)
        
        print(f"Max censor timepoint: {max_censor_timepoint}")
        print(f"Max timepoint in array: {max_timepoint_in_array}")
        print(f"Total timepoints T: {T}")
        print()
        
        # Check E_corrected - should equal max_censor_age
        E_corrected_patient_disease = E_corrected[patient_idx, disease_idx]
        print("E_corrected:")
        print(f"  Value: {E_corrected_patient_disease}")
        print(f"  Should equal max_censor_age ({max_censor_age}): {E_corrected_patient_disease == max_censor_age}")
        print()
        
        # Check E array - should be max_censor_timepoint+1 up to max_timepoint, then 0
        E_patient_disease = E[patient_idx, disease_idx, :]
        print("E array:")
        print(f"  E[0:5]: {E_patient_disease[:5]}")
        print(f"  E[{max_timepoint_in_array-2}:{max_timepoint_in_array+3}]: {E_patient_disease[max_timepoint_in_array-2:max_timepoint_in_array+3]}")
        if max_timepoint_in_array < T - 1:
            print(f"  E[{max_timepoint_in_array+1}:{max_timepoint_in_array+5}]: {E_patient_disease[max_timepoint_in_array+1:max_timepoint_in_array+5]}")
        print()
        
        # Verify: E should be constant (max_timepoint+1) up to max_timepoint, then 0
        if max_timepoint_in_array >= 0:
            E_before_censor = E_patient_disease[:max_timepoint_in_array+1]
            expected_value = max_timepoint_in_array + 1
            print("Verification:")
            print(f"  E values up to timepoint {max_timepoint_in_array} should all be {expected_value}")
            print(f"  Actual E values: {E_before_censor[:5]}... (last: {E_before_censor[-1]})")
            print(f"  All equal to {expected_value}? {np.all(E_before_censor == expected_value)}")
            
            if max_timepoint_in_array < T - 1:
                E_after_censor = E_patient_disease[max_timepoint_in_array+1:]
                print(f"  E values after timepoint {max_timepoint_in_array} should all be 0")
                print(f"  Actual E values: {E_after_censor[:5]}...")
                print(f"  All equal to 0? {np.all(E_after_censor == 0)}")
        else:
            print(f"  ⚠️ Max censor timepoint ({max_censor_timepoint}) is negative!")
            print(f"  Patient was censored before observation window starts")
            print(f"  E should be all 0s: {np.all(E_patient_disease == 0)}")
    
    return max_censor_age


# ============================================================================
# Helper function to find example patients for each scenario
# ============================================================================

def find_example_patients(Y, E, E_corrected, icd10_df, patient_names, max_censor_df, 
                         age_offset=30, disease_idx=0):
    """
    Find example patients for each scenario.
    
    Returns:
    --------
    examples : dict
        - 'event_patient': (patient_idx, disease_idx) - patient with event
        - 'early_diagnosis': (patient_idx, disease_idx) - diagnosis before age_offset
        - 'late_diagnosis': (patient_idx, disease_idx) - diagnosis after max age
        - 'censored': (patient_idx, disease_idx) - censored before end
    """
    T = Y.shape[2]
    max_age = age_offset + T - 1
    
    examples = {}
    
    # 1. Find patient with event
    event_mask = Y[:, disease_idx, :].sum(axis=1) > 0
    if event_mask.any():
        event_patient_idx = np.where(event_mask)[0][0]
        examples['event_patient'] = (event_patient_idx, disease_idx)
    
    # 2. Find patient with early diagnosis
    early_patients = icd10_df[icd10_df['age_diag'] < age_offset]['eid'].unique()
    if len(early_patients) > 0:
        early_patient_id = early_patients[0]
        early_patient_idx = patient_names.index(early_patient_id) if early_patient_id in patient_names else None
        if early_patient_idx is not None:
            examples['early_diagnosis'] = (early_patient_idx, disease_idx)
    
    # 3. Find patient with late diagnosis
    late_patients = icd10_df[icd10_df['age_diag'] > max_age]['eid'].unique()
    if len(late_patients) > 0:
        late_patient_id = late_patients[0]
        late_patient_idx = patient_names.index(late_patient_id) if late_patient_id in patient_names else None
        if late_patient_idx is not None:
            examples['late_diagnosis'] = (late_patient_idx, disease_idx)
    
    # 4. Find censored patient (max_censor < max_age)
    censored_patients = max_censor_df[max_censor_df['max_censor'] < max_age]
    if len(censored_patients) > 0:
        censored_patient_id = censored_patients.iloc[0]['eid']
        censored_patient_idx = patient_names.index(censored_patient_id) if censored_patient_id in patient_names else None
        if censored_patient_idx is not None:
            examples['censored'] = (censored_patient_idx, disease_idx)
    
    return examples

