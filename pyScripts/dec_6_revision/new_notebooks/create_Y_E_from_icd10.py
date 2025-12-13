#!/usr/bin/env python3
"""
Create Y and E Arrays from ICD10 Data

This script converts a dataframe with patient diagnoses (eid, diag_icd10, age_diag) into:
- Y array: Binary array of shape (N, D, T) where Y[n, d, t] = 1 if patient n had disease d at age (30+t)
- E array: Exposure/censor array of shape (N, D, T) indicating when patients are at risk

Then computes:
- Max censor replacement
- Corrected E matrix
- Prevalence estimates
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri

pandas2ri.activate()
numpy2ri.activate()


def create_Y_E_arrays(icd10_df, age_offset=30, output_dir=None, disease_order_csv=None):
    """
    Convert ICD10 dataframe into Y and E arrays.
    
    Parameters:
    -----------
    icd10_df : pd.DataFrame
        Dataframe with columns: eid, diag_icd10, age_diag
    age_offset : int
        Age offset (timepoint 0 = age_offset). Default 30.
    output_dir : str or Path, optional
        Directory to save outputs. If None, returns arrays without saving.
    disease_order_csv : str or Path, optional
        Path to CSV file with disease order. CSV should have one column 'x' with disease codes.
        If provided, diseases will be ordered to match this list. Missing diseases will be added at the end.
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - Y: np.ndarray, shape (N, D, T) - binary disease indicators
        - E: np.ndarray, shape (N, D, T) - exposure/censor matrix
        - E_corrected: np.ndarray, shape (N, D) - corrected E using max censor
        - prevalence_t: np.ndarray, shape (D, T) - disease prevalence
        - patient_names: list - patient IDs in order
        - disease_names: list - disease codes in order
        - max_censor_df: pd.DataFrame - max censor for each patient
    """
    print("="*60)
    print("Creating Y and E arrays from ICD10 data")
    print("="*60)
    
    # Step 1: Get unique patients and diseases
    print("\nStep 1: Creating mappings...")
    unique_patients = sorted(icd10_df['eid'].unique())
    
    # Determine disease order
    if disease_order_csv is not None:
        # Load disease order from CSV file
        print(f"  Loading disease order from CSV: {disease_order_csv}")
        disease_order_df = pd.read_csv(disease_order_csv)
        reference_disease_order = disease_order_df['x'].tolist()
        print(f"  Reference order has {len(reference_disease_order)} diseases")
        
        # Use reference order, but add any new diseases from the data at the end
        diseases_in_data = set(icd10_df['diag_icd10'].unique())
        diseases_in_ref = set(reference_disease_order)
        
        # Start with reference order
        unique_diseases = list(reference_disease_order)
        
        # Add any diseases in data but not in reference
        missing_diseases = sorted(diseases_in_data - diseases_in_ref)
        if missing_diseases:
            print(f"  Warning: {len(missing_diseases)} diseases in data but not in reference, adding at end")
            unique_diseases.extend(missing_diseases)
    else:
        # Default: sort diseases
        unique_diseases = sorted(icd10_df['diag_icd10'].unique())
        print(f"  Using sorted disease order ({len(unique_diseases)} diseases)")
    
    patient_to_idx = {patient: idx for idx, patient in enumerate(unique_patients)}
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    N = len(unique_patients)
    D = len(unique_diseases)
    
    print(f"  Patients: {N}")
    print(f"  Diseases: {D}")
    
    # Step 2: Determine time dimension
    min_age = icd10_df['age_diag'].min()
    max_age = icd10_df['age_diag'].max()
    T = max_age - age_offset + 1
    
    print(f"\nStep 2: Time dimension")
    print(f"  Age range: {min_age} to {max_age}")
    print(f"  Timepoints T: {T} (ages {age_offset} to {max_age})")
    
    # Step 3: Create Y array
    print(f"\nStep 3: Creating Y array ({N}, {D}, {T})...")
    Y = np.zeros((N, D, T), dtype=np.int32)
    
    for idx, row in icd10_df.iterrows():
        patient_id = row['eid']
        disease_code = row['diag_icd10']
        age = row['age_diag']
        
        patient_idx = patient_to_idx[patient_id]
        disease_idx = disease_to_idx[disease_code]
        
        # Convert age to timepoint (age 30 = timepoint 0)
        if age >= age_offset:
            timepoint = int(age - age_offset)
            if timepoint < T:
                Y[patient_idx, disease_idx, timepoint] = 1
        
        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1:,} records...")
    
    print(f"  ✓ Y array created")
    print(f"    Total diagnoses: {Y.sum():,}")
    print(f"    Patients with diagnoses: {(Y.sum(axis=(1,2)) > 0).sum():,}")
    print(f"    Diseases with diagnoses: {(Y.sum(axis=(0,2)) > 0).sum():,}")
    
    # Step 4: Compute max censor
    print(f"\nStep 4: Computing max censor...")
    max_censor = icd10_df.groupby('eid')['age_diag'].max().reset_index()
    max_censor.columns = ['eid', 'max_censor']
    
    max_age_default = max_censor['max_censor'].max() if len(max_censor) > 0 else 81
    
    # Create full max_censor dataframe for all patients
    all_patients_df = pd.DataFrame({'eid': unique_patients})
    max_censor_full = all_patients_df.merge(max_censor, on='eid', how='left')
    max_censor_full['max_censor'] = max_censor_full['max_censor'].fillna(max_age_default)
    max_censor_full = max_censor_full.set_index('eid').reindex(unique_patients).reset_index()
    
    print(f"  ✓ Max censor computed")
    print(f"    Range: {max_censor_full['max_censor'].min():.1f} - {max_censor_full['max_censor'].max():.1f}")
    print(f"    Mean: {max_censor_full['max_censor'].mean():.1f}")
    
    # Step 5: Create E array
    print(f"\nStep 5: Creating E array ({N}, {D}, {T})...")
    E = np.zeros((N, D, T), dtype=np.int32)
    
    # First, initialize E with max censor timepoint for all patients
    for patient_idx, patient_id in enumerate(unique_patients):
        max_age_patient = max_censor_full.loc[max_censor_full['eid'] == patient_id, 'max_censor'].values[0]
        max_timepoint = int(max_age_patient - age_offset)
        max_timepoint = min(max_timepoint, T - 1)
        
        # E[n, d, t] = max_timepoint if patient is at risk at timepoint t
        # (timepoint indexed from age 30, so age 64 = timepoint 34)
        for t in range(max_timepoint + 1):
            E[patient_idx, :, t] = max_timepoint
    
    # Then, update E for diseases where patient had an event (earlier than max censor)
    for idx, row in icd10_df.iterrows():
        patient_id = row['eid']
        disease_code = row['diag_icd10']
        age = row['age_diag']
        
        if disease_code not in disease_to_idx:
            continue
            
        patient_idx = patient_to_idx[patient_id]
        disease_idx = disease_to_idx[disease_code]
        
        # Convert age to timepoint (age 30 = timepoint 0)
        if age >= age_offset:
            event_timepoint = int(age - age_offset)
            if event_timepoint < T:
                # Update E: event occurred at this timepoint, so E should be event_timepoint
                # for all timepoints up to and including the event
                for t in range(event_timepoint + 1):
                    if E[patient_idx, disease_idx, t] > event_timepoint:
                        E[patient_idx, disease_idx, t] = event_timepoint
    
    print(f"  ✓ E array created")
    print(f"    Mean E: {E[E > 0].mean():.2f}")
    print(f"    E range: {E[E > 0].min()} - {E[E > 0].max()}")
    
    # Step 6: Create corrected E matrix (stores timepoints, 0-indexed from age 30)
    print(f"\nStep 6: Creating E_corrected ({N}, {D})...")
    E_corrected = np.zeros((N, D), dtype=np.int32)
    
    # Initialize with max censor timepoint for all patients and diseases
    for patient_idx, patient_id in enumerate(unique_patients):
        max_age_patient = max_censor_full.loc[max_censor_full['eid'] == patient_id, 'max_censor'].values[0]
        max_timepoint = int(max_age_patient - age_offset)
        max_timepoint = min(max_timepoint, T - 1)
        E_corrected[patient_idx, :] = max_timepoint
    
    # Update for diseases where patient had an event (event timepoint < max censor timepoint)
    for idx, row in icd10_df.iterrows():
        patient_id = row['eid']
        disease_code = row['diag_icd10']
        age = row['age_diag']
        
        if disease_code not in disease_to_idx:
            continue
            
        patient_idx = patient_to_idx[patient_id]
        disease_idx = disease_to_idx[disease_code]
        
        # Convert age to timepoint (age 30 = timepoint 0)
        if age >= age_offset:
            event_timepoint = int(age - age_offset)
            if event_timepoint < T:
                # E_corrected stores the event timepoint (or max censor timepoint if no event)
                if event_timepoint < E_corrected[patient_idx, disease_idx]:
                    E_corrected[patient_idx, disease_idx] = event_timepoint
    
    print(f"  ✓ E_corrected created")
    print(f"    Mean: {E_corrected.mean():.2f}")
    print(f"    Range: {E_corrected.min()} - {E_corrected.max()}")
    
    # Step 7: Compute prevalence
    print(f"\nStep 7: Computing prevalence ({D}, {T})...")
    prevalence_t = np.zeros((D, T))
    
    for d in range(D):
        for t in range(T):
            age_at_t = age_offset + t
            at_risk_mask = E_corrected[:, d] >= age_at_t
            
            if at_risk_mask.sum() > 0:
                prevalence_t[d, t] = Y[at_risk_mask, d, t].mean()
            else:
                prevalence_t[d, t] = np.nan
        
        if (d + 1) % 50 == 0:
            print(f"  Processed {d + 1}/{D} diseases...")
    
    print(f"  ✓ Prevalence computed")
    print(f"    Mean: {np.nanmean(prevalence_t):.6f}")
    print(f"    Range: {np.nanmin(prevalence_t):.6f} - {np.nanmax(prevalence_t):.6f}")
    
    # Prepare results
    results = {
        'Y': Y,
        'E': E,
        'E_corrected': E_corrected,
        'prevalence_t': prevalence_t,
        'patient_names': unique_patients,
        'disease_names': unique_diseases,
        'max_censor_df': max_censor_full,
        'patient_to_idx': patient_to_idx,
        'disease_to_idx': disease_to_idx
    }
    
    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStep 8: Saving results to {output_dir}...")
        
        # Convert to tensors and save
        torch.save(torch.FloatTensor(Y), output_dir / 'Y_binary.pt')
        torch.save(torch.FloatTensor(E), output_dir / 'E_binary.pt')
        torch.save(torch.FloatTensor(E_corrected), output_dir / 'E_corrected.pt')
        torch.save(torch.FloatTensor(prevalence_t), output_dir / 'prevalence_t_corrected.pt')
        
        # Save CSVs
        pd.DataFrame({'eid': unique_patients}).to_csv(output_dir / 'patient_names.csv', index=False)
        pd.DataFrame({'disease_code': unique_diseases, 'disease_idx': range(D)}).to_csv(
            output_dir / 'disease_names.csv', index=False)
        max_censor_full.to_csv(output_dir / 'max_censor.csv', index=False)
        
        print(f"  ✓ All files saved")
    
    print(f"\n{'='*60}")
    print("✓ Complete!")
    print(f"{'='*60}")
    
    return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Y and E arrays from ICD10 data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to ICD10 CSV file (columns: eid, diag_icd10, age_diag)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for saved arrays')
    parser.add_argument('--age_offset', type=int, default=30,
                       help='Age offset (timepoint 0 = this age). Default: 30')
    parser.add_argument('--disease_order_csv', type=str, default=None,
                       help='Path to CSV file with disease order (column "x" with disease codes)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ICD10 data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} diagnosis records")
    
    # Create arrays
    results = create_Y_E_arrays(df, age_offset=args.age_offset, output_dir=args.output_dir,
                                disease_order_csv=args.disease_order_csv)
    
    print(f"\nSummary:")
    print(f"  Y shape: {results['Y'].shape}")
    print(f"  E shape: {results['E'].shape}")
    print(f"  E_corrected shape: {results['E_corrected'].shape}")
    print(f"  Prevalence shape: {results['prevalence_t'].shape}")


if __name__ == "__main__":
    main()

