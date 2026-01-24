#!/usr/bin/env python3
"""
Check and fix sex column alignment between SexNumeric, Sex, and sex columns.
This ensures both washout (needs 'sex' numeric) and rolling (needs 'Sex' string) functions work.
"""

import pandas as pd

def align_sex_columns(pce_df):
    """
    Ensure pce_df has both 'sex' (numeric: 0=Female, 1=Male) and 'Sex' (string: 'Female'/'Male')
    to support both washout and rolling evaluation functions.
    
    Priority:
    1. If 'SexNumeric' exists, use it to create both 'sex' and 'Sex'
    2. If 'Sex' (string) exists, use it to create 'sex'
    3. If 'sex' (numeric) exists, use it to create 'Sex'
    """
    pce_df = pce_df.copy()
    
    # Check what columns exist
    has_sex_numeric = 'SexNumeric' in pce_df.columns
    has_sex_string = 'Sex' in pce_df.columns
    has_sex_numeric_lower = 'sex' in pce_df.columns
    
    print("="*80)
    print("SEX COLUMN ALIGNMENT CHECK")
    print("="*80)
    print(f"Has 'SexNumeric': {has_sex_numeric}")
    print(f"Has 'Sex' (string): {has_sex_string}")
    print(f"Has 'sex' (numeric): {has_sex_numeric_lower}")
    
    # Priority 1: SexNumeric exists
    if has_sex_numeric:
        print("\n✓ Found 'SexNumeric' column")
        if not has_sex_numeric_lower:
            print("  → Creating 'sex' (numeric) from SexNumeric")
            pce_df['sex'] = pce_df['SexNumeric'].astype(int)
        if not has_sex_string:
            print("  → Creating 'Sex' (string) from SexNumeric")
            pce_df['Sex'] = pce_df['SexNumeric'].map({0: 'Female', 1: 'Male'}).astype(str)
    
    # Priority 2: Sex (string) exists
    elif has_sex_string:
        print("\n✓ Found 'Sex' (string) column")
        if pce_df['Sex'].dtype == 'object':
            if not has_sex_numeric_lower:
                print("  → Creating 'sex' (numeric) from Sex")
                pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
        else:
            # Sex is numeric, convert to string
            print("  → Converting 'Sex' to string format")
            pce_df['Sex'] = pce_df['Sex'].map({0: 'Female', 1: 'Male'}).astype(str)
            if not has_sex_numeric_lower:
                pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
    
    # Priority 3: sex (numeric) exists
    elif has_sex_numeric_lower:
        print("\n✓ Found 'sex' (numeric) column")
        if not has_sex_string:
            print("  → Creating 'Sex' (string) from sex")
            pce_df['Sex'] = pce_df['sex'].map({0: 'Female', 1: 'Male'}).astype(str)
    
    else:
        raise ValueError("No sex column found! Need 'SexNumeric', 'Sex', or 'sex'")
    
    # Verify alignment
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    if 'sex' in pce_df.columns and 'Sex' in pce_df.columns:
        # Check alignment
        sex_from_numeric = pce_df['sex'].map({0: 'Female', 1: 'Male'})
        sex_from_string = pce_df['Sex']
        
        mismatches = (sex_from_numeric != sex_from_string).sum()
        if mismatches == 0:
            print("✓ 'sex' and 'Sex' columns are aligned!")
        else:
            print(f"⚠️  WARNING: {mismatches} mismatches between 'sex' and 'Sex' columns")
            print("   First few mismatches:")
            mismatch_indices = (sex_from_numeric != sex_from_string).head(10)
            for idx in mismatch_indices[mismatch_indices].index:
                print(f"     Index {idx}: sex={pce_df.loc[idx, 'sex']}, Sex={pce_df.loc[idx, 'Sex']}")
    
    # Show value counts
    if 'sex' in pce_df.columns:
        print(f"\n'sex' (numeric) value counts:")
        print(pce_df['sex'].value_counts().sort_index())
    if 'Sex' in pce_df.columns:
        print(f"\n'Sex' (string) value counts:")
        print(pce_df['Sex'].value_counts())
    
    return pce_df

if __name__ == "__main__":
    # Test with actual data
    pce_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    print(f"Original columns: {[c for c in pce_df.columns if 'sex' in c.lower() or 'Sex' in c]}")
    
    pce_df_aligned = align_sex_columns(pce_df)
    
    print(f"\nFinal columns: {[c for c in pce_df_aligned.columns if 'sex' in c.lower() or 'Sex' in c]}")
    print("\n✓ Alignment complete!")



















