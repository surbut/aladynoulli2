#!/usr/bin/env python3
"""
Transform MGB Medication Data to Pathway Analysis Format

Converts MGB medication data (with EMPI, Medication, Medication_Date) 
into the format expected by pathway analysis code.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def transform_mgb_medications(mgb_med_file, patient_birth_dates_file=None, 
                              output_file=None, patient_id_col='EMPI',
                              medication_col='Medication',
                              date_col='Medication_Date',
                              clinic_col='Clinic',
                              inpatient_col='Inpatient_Outpatient'):
    """
    Transform MGB medication data to pathway analysis format
    
    Parameters:
    -----------
    mgb_med_file : str
        Path to MGB medication data file (CSV or similar)
    patient_birth_dates_file : str, optional
        Path to file with patient birth dates (EMPI, birth_date columns)
        If None, age calculation will be skipped
    output_file : str, optional
        Path to save transformed data (default: adds '_transformed' to input filename)
    patient_id_col : str
        Name of patient ID column in MGB data (default: 'EMPI')
    medication_col : str
        Name of medication name column (default: 'Medication')
    date_col : str
        Name of medication date column (default: 'Medication_Date')
    clinic_col : str
        Name of clinic column (default: 'Clinic')
    inpatient_col : str
        Name of inpatient/outpatient column (default: 'Inpatient_Outpatient')
    
    Returns:
    --------
    pd.DataFrame : Transformed medication data
    """
    print("="*80)
    print("TRANSFORMING MGB MEDICATION DATA")
    print("="*80)
    
    # Load MGB medication data
    print(f"\n1. Loading MGB medication data from: {mgb_med_file}")
    try:
        # Try different separators
        for sep in [',', '\t', '|']:
            try:
                mgb_meds = pd.read_csv(mgb_med_file, sep=sep, low_memory=False)
                if len(mgb_meds.columns) > 1:
                    print(f"   ✅ Loaded with separator: '{sep}'")
                    break
            except:
                continue
        else:
            # Try without specifying separator (pandas will auto-detect)
            mgb_meds = pd.read_csv(mgb_med_file, low_memory=False)
            print(f"   ✅ Loaded with auto-detected separator")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return None
    
    print(f"   Loaded {len(mgb_meds):,} rows")
    print(f"   Columns: {list(mgb_meds.columns)}")
    
    # Check required columns
    required_cols = [patient_id_col, medication_col, date_col]
    missing_cols = [col for col in required_cols if col not in mgb_meds.columns]
    if missing_cols:
        print(f"   ❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(mgb_meds.columns)}")
        return None
    
    # Rename columns to standard format
    print(f"\n2. Renaming columns to standard format")
    rename_dict = {
        patient_id_col: 'eid',
        medication_col: 'Medication_Name',
        date_col: 'Medication_Date'
    }
    
    # Keep optional columns if they exist
    if clinic_col in mgb_meds.columns:
        rename_dict[clinic_col] = 'Clinic'
    if inpatient_col in mgb_meds.columns:
        rename_dict[inpatient_col] = 'Inpatient_Outpatient'
    
    mgb_meds = mgb_meds.rename(columns=rename_dict)
    
    # Create required columns
    print(f"\n3. Creating required columns for pathway analysis")
    
    # drug_name: Use medication name directly (normalize for consistency)
    mgb_meds['drug_name'] = mgb_meds['Medication_Name'].str.strip().str.lower()
    
    # read_2: Use medication name as code (since we don't have READ codes)
    # Alternative: could create hash or use medication name directly
    mgb_meds['read_2'] = mgb_meds['drug_name']
    
    # bnf_code: Try to infer from medication name (will be mostly empty)
    # We'll create a simple mapping for common medications
    mgb_meds['bnf_code'] = infer_bnf_category(mgb_meds['drug_name'])
    
    # Convert dates
    print(f"\n4. Processing dates")
    mgb_meds['Medication_Date'] = pd.to_datetime(mgb_meds['Medication_Date'], 
                                                  errors='coerce', 
                                                  infer_datetime_format=True)
    
    # Calculate age at prescription if birth dates available
    if patient_birth_dates_file:
        print(f"   Loading patient birth dates from: {patient_birth_dates_file}")
        try:
            birth_dates = pd.read_csv(patient_birth_dates_file)
            if 'EMPI' in birth_dates.columns and 'birth_date' in birth_dates.columns:
                birth_dates['birth_date'] = pd.to_datetime(birth_dates['birth_date'], errors='coerce')
                birth_dates = birth_dates.set_index('EMPI')['birth_date']
                
                mgb_meds['age_at_prescription'] = (
                    mgb_meds['eid'].map(birth_dates)
                    .apply(lambda x: None if pd.isna(x) else x)
                )
                mgb_meds['age_at_prescription'] = (
                    (mgb_meds['Medication_Date'] - mgb_meds['age_at_prescription']).dt.days / 365.25
                )
                print(f"   ✅ Calculated age at prescription for {mgb_meds['age_at_prescription'].notna().sum():,} records")
            else:
                print(f"   ⚠️  Birth date file missing required columns (EMPI, birth_date)")
        except Exception as e:
            print(f"   ⚠️  Could not load birth dates: {e}")
            print(f"   Age calculation skipped")
    else:
        print(f"   ⚠️  No birth date file provided - age calculation skipped")
        mgb_meds['age_at_prescription'] = None
    
    # Clean up: Remove rows with missing essential data
    initial_rows = len(mgb_meds)
    mgb_meds = mgb_meds.dropna(subset=['eid', 'drug_name', 'Medication_Date'])
    final_rows = len(mgb_meds)
    if initial_rows != final_rows:
        print(f"\n5. Removed {initial_rows - final_rows:,} rows with missing essential data")
    
    # Select columns for output (matching expected format)
    output_cols = ['eid', 'drug_name', 'read_2', 'bnf_code', 'Medication_Date', 'Medication_Name']
    if 'Clinic' in mgb_meds.columns:
        output_cols.append('Clinic')
    if 'Inpatient_Outpatient' in mgb_meds.columns:
        output_cols.append('Inpatient_Outpatient')
    if 'age_at_prescription' in mgb_meds.columns:
        output_cols.append('age_at_prescription')
    
    output_df = mgb_meds[output_cols].copy()
    
    # Summary statistics
    print(f"\n6. TRANSFORMATION SUMMARY")
    print(f"   Total prescription records: {len(output_df):,}")
    print(f"   Unique patients: {output_df['eid'].nunique():,}")
    print(f"   Unique medications: {output_df['drug_name'].nunique():,}")
    print(f"   Date range: {output_df['Medication_Date'].min()} to {output_df['Medication_Date'].max()}")
    
    if 'age_at_prescription' in output_df.columns:
        valid_ages = output_df['age_at_prescription'].notna()
        if valid_ages.sum() > 0:
            print(f"   Age range: {output_df.loc[valid_ages, 'age_at_prescription'].min():.1f} to "
                  f"{output_df.loc[valid_ages, 'age_at_prescription'].max():.1f} years")
    
    # Save output
    if output_file is None:
        output_file = mgb_med_file.replace('.csv', '_transformed.csv')
        output_file = output_file.replace('.txt', '_transformed.txt')
    
    print(f"\n7. Saving transformed data to: {output_file}")
    output_df.to_csv(output_file, index=False, sep='\t')
    print(f"   ✅ Saved {len(output_df):,} rows")
    
    return output_df


def infer_bnf_category(drug_names):
    """
    Infer BNF category from medication name (simple keyword matching)
    
    This is a basic implementation. For full BNF mapping, you'd need
    a proper medication dictionary or API.
    """
    bnf_categories = {
        '01': ['omeprazole', 'lansoprazole', 'pantoprazole', 'ranitidine', 'metoclopramide', 'domperidone'],
        '02': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'lisinopril', 'ramipril', 'metoprolol', 
               'bisoprolol', 'atenolol', 'amlodipine', 'losartan', 'valsartan', 'aspirin', 'clopidogrel',
               'warfarin', 'nitroglycerin', 'nitrostat'],
        '03': ['salbutamol', 'albuterol', 'budesonide', 'fluticasone', 'montelukast', 'prednisone'],
        '04': ['paracetamol', 'acetaminophen', 'ibuprofen', 'naproxen', 'tramadol', 'morphine',
               'gabapentin', 'pregabalin', 'sertraline', 'citalopram', 'fluoxetine'],
        '05': ['amoxicillin', 'azithromycin', 'cephalexin', 'ciprofloxacin', 'metronidazole'],
        '06': ['metformin', 'insulin', 'glipizide', 'glyburide', 'levothyroxine', 'synthroid'],
        '10': ['naproxen', 'ibuprofen', 'diclofenac', 'celecoxib', 'methotrexate'],
    }
    
    # Create reverse mapping: drug keyword -> BNF code
    keyword_to_bnf = {}
    for bnf_code, keywords in bnf_categories.items():
        for keyword in keywords:
            keyword_to_bnf[keyword.lower()] = bnf_code
    
    # Match drugs to BNF categories
    bnf_codes = []
    for drug_name in drug_names:
        drug_lower = str(drug_name).lower()
        matched_bnf = None
        
        # Try to find matching keyword
        for keyword, bnf_code in keyword_to_bnf.items():
            if keyword in drug_lower:
                matched_bnf = bnf_code
                break
        
        bnf_codes.append(matched_bnf if matched_bnf else None)
    
    return pd.Series(bnf_codes, index=drug_names.index)


def create_sample_mgb_med_format():
    """
    Create a sample file showing the expected MGB input format
    """
    sample_data = {
        'EMPI': [100035476, 100035476, 100035476, 100035476, 100035476],
        'Medication': [
            'Nitrostat 0.4mg tablet',
            'Morphine sulft 10mg inj',
            'Not otherwise classified, antineoplastic drugs',
            'Midrin 325mg-100mg-65mg capsule',
            'Tylenol extra strength 500mg tablet'
        ],
        'Medication_Date': ['6/7/1998', '6/7/1998', '6/7/1998', '6/19/2000', '11/2/2002'],
        'Clinic': [
            'BPG At 850 Boylston (100)',
            'BPG At 850 Boylston (100)',
            'BPG At 850 Boylston (100)',
            'Neurophysiology (101)',
            'Day Surgery (8)'
        ],
        'Inpatient_Outpatient': ['Outpatient', 'Outpatient', 'Outpatient', 'Outpatient', 'Outpatient']
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = 'sample_mgb_medication_format.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"Created sample file: {sample_file}")
    return sample_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform MGB medication data for pathway analysis')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to MGB medication data file')
    parser.add_argument('--birth_dates', type=str, default=None,
                        help='Path to patient birth dates file (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: adds _transformed to input)')
    parser.add_argument('--patient_id_col', type=str, default='EMPI',
                        help='Patient ID column name (default: EMPI)')
    parser.add_argument('--medication_col', type=str, default='Medication',
                        help='Medication name column (default: Medication)')
    parser.add_argument('--date_col', type=str, default='Medication_Date',
                        help='Date column name (default: Medication_Date)')
    
    args = parser.parse_args()
    
    transformed_data = transform_mgb_medications(
        mgb_med_file=args.input,
        patient_birth_dates_file=args.birth_dates,
        output_file=args.output,
        patient_id_col=args.patient_id_col,
        medication_col=args.medication_col,
        date_col=args.date_col
    )
    
    if transformed_data is not None:
        print(f"\n✅ Transformation complete!")
        print(f"\nFirst few rows of transformed data:")
        print(transformed_data.head())
    else:
        print(f"\n❌ Transformation failed")

