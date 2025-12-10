#!/usr/bin/env python3
"""
Apply 6-month washout window to E matrix.

For events within 6 months of enrollment, censor them in E matrix
(but keep Y unchanged - we still want to predict those outcomes).

This addresses reverse causation where diagnostic procedures lead to cascades.
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Optional: if you want to read RDS files directly
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    print("Note: pyreadr not installed. Install with: pip install pyreadr")
    print("      Or save bd.censor as CSV from R instead.")

# For loading sparse_array.rds to get person_ids and disease_codes order
try:
    import rpy2.robjects as robjects
    import scipy.sparse as sp
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("Note: rpy2 not installed. Install with: pip install rpy2")
    print("      Or load person_ids and disease_codes from saved files.")

def load_bd_censor_from_rds(rds_path):
    """
    Load bd.censor from R RDS file.
    
    Args:
        rds_path: Path to .rds file saved from R
    
    Returns:
        DataFrame with columns: f.eid, enroll_date, birthdate, etc.
    """
    if HAS_PYREADR:
        result = pyreadr.read_r(rds_path)
        # pyreadr returns a dict, get the first (and usually only) dataframe
        df = list(result.values())[0]
        return df
    else:
        raise ImportError("pyreadr not available. Save bd.censor as CSV from R instead.")


def load_bd_censor_from_csv(csv_path):
    """
    Load bd.censor from CSV file (saved from R).
    
    Args:
        csv_path: Path to .csv file
    
    Returns:
        DataFrame with columns: f.eid, enroll_date, birthdate, etc.
    """
    df = pd.read_csv(csv_path)
    # Convert date columns
    if 'enroll_date' in df.columns:
        df['enroll_date'] = pd.to_datetime(df['enroll_date'])
    if 'birthdate' in df.columns:
        df['birthdate'] = pd.to_datetime(df['birthdate'])
    return df


def get_order_from_sparse_array(sparse_array_path):
    """
    Extract person_ids and disease_codes order from sparse_array.rds.
    
    This matches the order used when creating Y and E matrices.
    
    Args:
        sparse_array_path: Path to sparse_array.rds file
    
    Returns:
        tuple: (person_ids, disease_codes) as lists in the correct order
    """
    if not HAS_RPY2:
        raise ImportError("rpy2 required to read sparse_array.rds")
    
    print("Loading sparse_array.rds to extract person_ids and disease_codes order...")
    r_sparse = robjects.r.readRDS(sparse_array_path)
    
    # Get first matrix to extract dimnames
    matrix = r_sparse[0]
    dimnames = matrix.slots['Dimnames']
    
    # Extract person_ids (row names) and disease_codes (column names)
    person_ids = [int(x) for x in list(dimnames[0])]  # Row names = person IDs
    disease_codes = list(dimnames[1])  # Column names = disease codes (phecodes)
    
    print(f"Extracted {len(person_ids):,} person_ids and {len(disease_codes):,} disease_codes")
    
    return person_ids, disease_codes


def get_person_ids_from_csv(csv_path):
    """
    Load person_ids from processed_ids.csv.
    
    Args:
        csv_path: Path to processed_ids.csv
    
    Returns:
        list: person_ids in order
    """
    df = pd.read_csv(csv_path)
    person_ids = df['eid'].tolist()
    print(f"Loaded {len(person_ids):,} person_ids from {csv_path}")
    return person_ids


def load_enrollment_ages(baselinage_path):
    """
    Load enrollment ages from baselinagefamh.csv.
    
    Args:
        baselinage_path: Path to baselinagefamh.csv
    
    Returns:
        dict: {eid: enrollment_age}
    """
    print(f"Loading enrollment ages from {baselinage_path}...")
    df = pd.read_csv(baselinage_path)
    enrollment_ages = dict(zip(df['identifier'], df['age']))
    print(f"Loaded enrollment ages for {len(enrollment_ages):,} patients")
    return enrollment_ages


def create_icd_to_phecode_mapping(icdlab_df):
    """
    Create mapping from (eid, icd_code, age) to phecode from icdlab.
    
    Since icdlab already has the Phecode mappings from icd2phecode(),
    we can use it to map ICD codes to Phecodes.
    
    Args:
        icdlab_df: DataFrame with columns: eid, diag_icd10 (phecode), age_diag
                   This should have the Phecode mappings
    
    Returns:
        dict: {(eid, icd_code, age): phecode} or we can create a simpler mapping
    """
    # Actually, icdlab already has Phecodes, not ICD codes
    # So we need the original HESIN data with ICD codes
    # For now, return empty - we'll handle this differently
    return {}


def apply_washout_to_E(hesin_df, icdlab_df, bd_censor_df, enrollment_ages_dict, E_original, eid_order, 
                       disease_mapping, washout_months=6, min_age=30):
    """
    Apply 6-month washout window to E matrix.
    
    Args:
        hesin_df: DataFrame with columns: f.eid, event_start, code (ICD code), age
                  This is the HESIN data (the 'd' dataframe from R)
        icdlab_df: DataFrame with columns: eid, diag_icd10 (phecode), age_diag
                   This is the output from icd2phecode() - used for Phecode mapping
        bd_censor_df: DataFrame with columns: f.eid, enroll_date, birthdate
                     This is your bd.censor from R
        enrollment_ages_dict: dict mapping {eid: enrollment_age}
                             From baselinagefamh.csv
        E_original: Original E matrix (N, D) - event times as age indices
        eid_order: List of eids in same order as E_original rows
        disease_mapping: dict mapping {phecode: disease_index} for E matrix
        washout_months: Number of months to exclude (default 6)
        min_age: Minimum age for time indexing (default 30)
    
    Returns:
        E_washout: Modified E matrix with events in washout window censored
        hesin_annotated: DataFrame with washout annotations
    """
    print("="*80)
    print(f"APPLYING {washout_months}-MONTH WASHOUT WINDOW")
    print("="*80)
    
    # Create mapping from eid to row index
    eid_to_idx = {eid: idx for idx, eid in enumerate(eid_order)}
    print(f"Mapping {len(eid_to_idx):,} patients to E matrix rows")
    
    # Prepare HESIN data
    print("\nPreparing HESIN data...")
    hesin = hesin_df.copy()
    
    # Parse event_start dates
    hesin['event_start'] = pd.to_datetime(hesin['event_start'])
    
    # Join with bd.censor to get enrollment dates
    print("Joining with bd.censor for enrollment dates...")
    hesin = hesin.merge(
        bd_censor_df[['f.eid', 'enroll_date', 'birthdate']],
        on='f.eid',
        how='left'
    )
    
    # Map ICD codes to Phecodes using icdlab
    # The icdlab has (eid, phecode, age_diag) - we need to match on eid and age
    print("Mapping ICD codes to Phecodes...")
    # Create a mapping from icdlab: (eid, age_diag) -> phecode
    # Round ages to match: d$age is rounded to 1 decimal in R, icdlab$age_diag is rounded to integer
    # So we round both to integer for the join
    icdlab_mapping = icdlab_df.copy()
    icdlab_mapping['age_rounded'] = icdlab_mapping['age_diag'].round(0).astype(int)
    hesin['age_rounded'] = hesin['age'].round(0).astype(int)
    
    # Join to get Phecodes
    # Note: hesin has 'f.eid', icdlab has 'eid' - they're the same, just different column names
    hesin = hesin.merge(
        icdlab_mapping[['eid', 'age_rounded', 'diag_icd10']].rename(columns={'diag_icd10': 'phecode', 'eid': 'f.eid'}),
        on=['f.eid', 'age_rounded'],
        how='left'
    )
    
    # Convert Phecodes to strings to match disease_mapping keys
    # disease_mapping keys are strings (from R's as.character()), but Phecodes from icdlab are numeric
    # Handle NaN values - keep them as NaN (not convert to string "nan")
    hesin['phecode'] = hesin['phecode'].apply(lambda x: str(x) if pd.notna(x) else x)
    
    # Filter to only events that mapped to Phecodes in our disease list
    valid_phecodes = set(disease_mapping.keys())
    hesin = hesin[hesin['phecode'].isin(valid_phecodes)].copy()
    
    print(f"After Phecode mapping: {len(hesin):,} events with valid Phecodes")
    
    # Calculate days from enrollment
    hesin['days_from_enroll'] = (hesin['event_start'] - hesin['enroll_date']).dt.days
    
    # Add enrollment age
    hesin['enroll_age'] = hesin['f.eid'].map(enrollment_ages_dict)
    
    # Identify events in washout window (0 to 6 months = 0 to ~180 days)
    washout_days = washout_months * 30.44  # Average days per month
    hesin['in_washout'] = (hesin['days_from_enroll'] >= 0) & \
                          (hesin['days_from_enroll'] <= washout_days) & \
                          (pd.notna(hesin['enroll_date']))
    
    n_washout = hesin['in_washout'].sum()
    n_total = len(hesin)
    print(f"Found {n_washout:,} / {n_total:,} events ({100*n_washout/n_total:.2f}%) in washout window")
    
    # Clone E matrix
    E_washout = E_original.clone()
    
    # Censor events in washout window
    print("\nCensoring events in washout window...")
    washout_events = hesin[hesin['in_washout'] == True].copy()
    
    n_censored = 0
    n_skipped = 0
    
    for idx, row in washout_events.iterrows():
        eid = row['f.eid']  # Use f.eid from hesin
        phecode = row['phecode']  # This is the phecode from the join
        event_age = row['age']  # Use age from hesin
        enroll_age = row.get('enroll_age', None)
        
        # Skip if patient not in E matrix
        if eid not in eid_to_idx:
            n_skipped += 1
            continue
        
        patient_idx = eid_to_idx[eid]
        
        # Skip if no enrollment date
        if pd.isna(row['enroll_date']):
            n_skipped += 1
            continue
        
        # Get disease index from mapping
        if phecode not in disease_mapping:
            n_skipped += 1
            continue
        
        disease_idx = disease_mapping[phecode]
        
        # Get enrollment age (from baselinagefamh.csv)
        if pd.isna(enroll_age):
            # Fallback: calculate from days_from_enroll if enrollment_age not available
            days_from_enroll = row['days_from_enroll']
            if pd.notna(days_from_enroll):
                enroll_age = event_age - (days_from_enroll / 365.25)
            else:
                n_skipped += 1
                continue
        
        # Convert to time indices
        enroll_time_idx = int(enroll_age - min_age)
        event_time_idx = int(event_age - min_age)
        
        # Censor time: enrollment age minus washout period (in years)
        washout_years = washout_months / 12.0
        censor_age = enroll_age - washout_years
        censor_time_idx = max(0, int(censor_age - min_age))
        
        # Only censor if the event actually occurred at or after enrollment
        original_time = E_original[patient_idx, disease_idx].item()
        if original_time >= enroll_time_idx:
            E_washout[patient_idx, disease_idx] = censor_time_idx
            n_censored += 1
    
    print(f"\nCensored {n_censored:,} events in E matrix")
    print(f"Skipped {n_skipped:,} events (patient not in E matrix, missing data, or no disease mapping)")
    
    return E_washout, hesin


# Example usage function
def example_usage():
    """
    Example of how to use this function with R data.
    """
    # Load bd.censor (enrollment dates)
    # In R, you saved: write.csv(bd.censor, "~/Library/CloudStorage/Dropbox-Personal/bd_censor.csv", row.names=FALSE)
    bd_censor_df = load_bd_censor_from_csv("~/Library/CloudStorage/Dropbox-Personal/bd_censor.csv")
    
    # OPTION 2: Load from RDS directly (requires pyreadr)
    # bd_censor_df = load_bd_censor_from_rds("path/to/bd_censor.rds")
    
    # Load HESIN data with dates (the 'd' dataframe from R)
    # In R, you saved: saveRDS(d, "~/Library/CloudStorage/Dropbox-Personal/hesin_with_dates.rds")
    if HAS_PYREADR:
        hesin_result = pyreadr.read_r("~/Library/CloudStorage/Dropbox-Personal/hesin_with_dates.rds")
        hesin_df = list(hesin_result.values())[0]
    else:
        hesin_df = pd.read_csv("path/to/hesin_with_dates.csv")
    
    # Load icdlab (the filtered Phecode-mapped version from R)
    # In R, you saved: saveRDS(icdlab, "~/Library/CloudStorage/Dropbox-Personal/hesin_phecoded.rds")
    if HAS_PYREADR:
        icdlab_result = pyreadr.read_r("~/Library/CloudStorage/Dropbox-Personal/hesin_phecoded.rds")
        icdlab_df = list(icdlab_result.values())[0]
    else:
        icdlab_df = pd.read_csv("path/to/hesin_phecoded.csv")
    
    # Load enrollment ages from baselinagefamh.csv
    enrollment_ages = load_enrollment_ages("path/to/baselinagefamh.csv")
    
    # Load E matrix
    E_original = torch.load("path/to/E_matrix.pt")
    
    # Load eid order (list of eids in same order as E matrix rows)
    # OPTION 1: From processed_ids.csv
    processed_ids_df = pd.read_csv("pyScripts/csv/processed_ids.csv")
    eid_order = processed_ids_df['eid'].tolist()
    
    # OPTION 2: Or from sparse_array.rds (see below)
    
    # Get person_ids and disease_codes order
    # OPTION 1: From sparse_array.rds (source of truth)
    sparse_array_path = "path/to/sparse_array.rds"  # Update this path
    try:
        person_ids, disease_codes = get_order_from_sparse_array(sparse_array_path)
    except:
        # OPTION 2: From processed_ids.csv (if available)
        processed_ids_path = "pyScripts/csv/processed_ids.csv"  # Update if needed
        person_ids = get_person_ids_from_csv(processed_ids_path)
        # For disease_codes, you'll need to load from R or another source
        # disease_codes = ...  # You need to provide this
        print("WARNING: Could not load from sparse_array.rds, using processed_ids.csv")
        print("You still need to provide disease_codes order!")
        disease_codes = []  # You need to provide this
    
    # Create disease mapping: {phecode: disease_index}
    # This uses the exact order from sparse_array.rds
    disease_mapping = {code: idx for idx, code in enumerate(disease_codes)}
    
    # Verify eid_order matches person_ids
    if eid_order != person_ids:
        print("WARNING: eid_order doesn't match person_ids!")
        print("Using person_ids from sparse_array/processed_ids instead.")
        eid_order = person_ids
    
    # Apply washout
    E_washout, hesin_annotated = apply_washout_to_E(
        hesin_df=hesin_df,
        icdlab_df=icdlab_df,
        bd_censor_df=bd_censor_df,
        enrollment_ages_dict=enrollment_ages,
        E_original=E_original,
        eid_order=eid_order,
        disease_mapping=disease_mapping,
        washout_months=6,
        min_age=30
    )
    
    # Save results
    torch.save(E_washout, "path/to/E_washout_6month.pt")
    hesin_annotated.to_csv("path/to/hesin_with_washout_annotation.csv", index=False)
    
    print("\nDone! E matrix with washout applied saved.")


if __name__ == "__main__":
    print("This script provides functions to apply 6-month washout window.")
    print("See example_usage() for how to use it.")
    print("\nTo use with R data:")
    print("  1. In R, save bd.censor: write.csv(bd.censor, 'bd_censor.csv', row.names=FALSE)")
    print("  2. In R, save icdlab (from icd2phecode): saveRDS(i, 'icdlab_phecoded.rds')")
    print("  3. Load both in Python using the functions above")
    print("  4. Provide disease_mapping: {phecode: disease_index} matching your E matrix")

