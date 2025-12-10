#!/usr/bin/env python3
"""
Apply 6-month washout window to E matrix to address reverse causation concerns.

The washout window excludes events that occur within 6 months of enrollment,
as these may be due to diagnostic cascades rather than true disease associations.
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path

def load_enrollment_dates(enrollment_df_path, eid_col='f.eid'):
    """
    Load enrollment dates from UK Biobank enrollment data.
    
    Enrollment dates are in fields f.53.0.0, f.53.1.0, f.53.2.0, f.53.3.0
    (assessment center dates for visits 0-3).
    
    Returns:
        dict: {eid: enrollment_date}
    """
    print("Loading enrollment dates...")
    df = pd.read_csv(enrollment_df_path)
    
    # Get enrollment date (first non-NA date from f.53.0.0, f.53.1.0, f.53.2.0, f.53.3.0)
    enrollment_dates = {}
    for idx, row in df.iterrows():
        eid = row[eid_col]
        # Try to get first non-NA enrollment date
        for col in ['f.53.0.0', 'f.53.1.0', 'f.53.2.0', 'f.53.3.0']:
            if col in row and pd.notna(row[col]):
                try:
                    # Parse date (assuming format like '2008-07-23')
                    enroll_date = pd.to_datetime(row[col])
                    enrollment_dates[eid] = enroll_date
                    break
                except:
                    continue
    
    print(f"Loaded enrollment dates for {len(enrollment_dates):,} patients")
    return enrollment_dates


def load_hesin_data(hesin_path):
    """
    Load HESIN data with event dates.
    
    Expected columns:
    - f.eid: patient ID
    - event_start: event date
    - code: ICD code
    - age: age at event
    """
    print("Loading HESIN data...")
    df = pd.read_csv(hesin_path)
    
    # Parse event_start dates
    if 'event_start' in df.columns:
        df['event_start'] = pd.to_datetime(df['event_start'])
    
    print(f"Loaded {len(df):,} HESIN records")
    return df


def create_washout_mask(hesin_df, enrollment_dates, washout_months=6, eid_col='f.eid', 
                        event_date_col='event_start', code_col='code'):
    """
    Create a mask indicating which events should be excluded due to washout window.
    
    Args:
        hesin_df: DataFrame with HESIN events
        enrollment_dates: dict of {eid: enrollment_date}
        washout_months: Number of months to exclude after enrollment
        eid_col: Column name for patient ID
        event_date_col: Column name for event date
        code_col: Column name for ICD code
    
    Returns:
        DataFrame with washout_mask column (True = exclude, False = keep)
    """
    print(f"\nCreating washout mask (excluding events within {washout_months} months of enrollment)...")
    
    # Add enrollment date to hesin_df
    hesin_df['enrollment_date'] = hesin_df[eid_col].map(enrollment_dates)
    
    # Calculate days between enrollment and event
    hesin_df['days_from_enrollment'] = (hesin_df[event_date_col] - hesin_df['enrollment_date']).dt.days
    
    # Events within washout window (0 to washout_months*30 days)
    washout_days = washout_months * 30
    hesin_df['washout_mask'] = (hesin_df['days_from_enrollment'] >= 0) & \
                               (hesin_df['days_from_enrollment'] <= washout_days)
    
    # Also exclude events that occur before enrollment (negative days) - these are pre-existing
    # Actually, wait - we might want to keep pre-enrollment events, just exclude post-enrollment washout
    
    n_excluded = hesin_df['washout_mask'].sum()
    n_total = len(hesin_df)
    print(f"Excluding {n_excluded:,} / {n_total:,} events ({100*n_excluded/n_total:.1f}%) within washout window")
    
    return hesin_df


def apply_washout_to_E_matrix(E_original, Y, hesin_df, enrollment_dates, disease_mapping,
                              washout_months=6, eid_order=None, min_age=30):
    """
    Apply washout window to E matrix by censoring events within washout period.
    
    Args:
        E_original: Original E matrix (N, D) with event times (age indices)
        Y: Y matrix (N, D, T) - kept unchanged
        hesin_df: DataFrame with HESIN events and washout_mask
        enrollment_dates: dict of {eid: enrollment_date}
        disease_mapping: dict mapping ICD codes to disease indices
        washout_months: Number of months to exclude
        eid_order: List of eids in same order as E_original rows
        min_age: Minimum age (for converting dates to age indices)
    
    Returns:
        E_washout: Modified E matrix with events in washout window censored
    """
    print(f"\nApplying {washout_months}-month washout to E matrix...")
    
    E_washout = E_original.clone()
    
    # Calculate enrollment age for each patient
    if eid_order is None:
        raise ValueError("eid_order must be provided to map patients to E matrix rows")
    
    eid_to_idx = {eid: idx for idx, eid in enumerate(eid_order)}
    
    # For each event in washout window, censor it in E matrix
    washout_events = hesin_df[hesin_df['washout_mask'] == True].copy()
    
    print(f"Processing {len(washout_events):,} events to censor...")
    
    n_censored = 0
    for idx, row in washout_events.iterrows():
        eid = row['f.eid']
        code = row['code']
        
        # Get patient index
        if eid not in eid_to_idx:
            continue
        
        patient_idx = eid_to_idx[eid]
        
        # Get disease index
        if code not in disease_mapping:
            continue
        
        disease_idx = disease_mapping[code]
        
        # Get enrollment age
        if eid not in enrollment_dates:
            continue
        
        enroll_date = enrollment_dates[eid]
        enroll_age = row.get('age', None)  # If age is in the row
        
        # If we don't have age, calculate from enrollment date and birthdate
        if pd.isna(enroll_age):
            # Try to get birthdate from hesin_df or calculate from age at event
            # For now, use the event age minus days_from_enrollment converted to years
            days_from_enroll = row.get('days_from_enrollment', 0)
            if 'age' in row and pd.notna(row['age']):
                enroll_age = row['age'] - (days_from_enroll / 365.25)
            else:
                continue
        
        # Convert enrollment age to time index
        enroll_time_idx = int(enroll_age - min_age)
        
        # Censor: set E to enrollment time (or enrollment time - washout period in age units)
        # Actually, we want to set it to just before the washout window starts
        # So if enrollment is at age 50, and washout is 6 months (0.5 years),
        # we set E to 50 - 0.5 = 49.5, which rounds to time index 19.5 -> 19
        washout_years = washout_months / 12.0
        censor_age = enroll_age - washout_years
        censor_time_idx = max(0, int(censor_age - min_age))
        
        # Only censor if the original event time is within the washout window
        original_event_time = E_original[patient_idx, disease_idx].item()
        if original_event_time >= enroll_time_idx:  # Event is at or after enrollment
            E_washout[patient_idx, disease_idx] = censor_time_idx
            n_censored += 1
    
    print(f"Censored {n_censored:,} events in E matrix")
    
    return E_washout


def main():
    """
    Example usage for applying washout window.
    """
    # Paths (update these to your actual paths)
    hesin_path = "path/to/hesin_data.csv"
    enrollment_path = "path/to/enrollment_data.csv"
    E_path = "path/to/E_matrix.pt"
    Y_path = "path/to/Y_matrix.pt"
    eid_order_path = "path/to/eid_order.txt"  # List of eids in same order as E rows
    
    # Load data
    hesin_df = load_hesin_data(hesin_path)
    enrollment_dates = load_enrollment_dates(enrollment_path)
    
    # Create washout mask
    hesin_df = create_washout_mask(hesin_df, enrollment_dates, washout_months=6)
    
    # Load E and Y matrices
    E_original = torch.load(E_path)
    Y = torch.load(Y_path)
    
    # Load eid order
    with open(eid_order_path, 'r') as f:
        eid_order = [int(line.strip()) for line in f]
    
    # Create disease mapping (you'll need to provide this based on your phecode mapping)
    # disease_mapping = {code: idx for idx, code in enumerate(disease_codes)}
    
    # Apply washout
    # E_washout = apply_washout_to_E_matrix(
    #     E_original, Y, hesin_df, enrollment_dates, disease_mapping,
    #     washout_months=6, eid_order=eid_order, min_age=30
    # )
    
    # Save
    # torch.save(E_washout, "path/to/E_washout_6month.pt")
    
    print("\nDone! Use this script as a template and update paths.")


if __name__ == "__main__":
    main()

