#!/usr/bin/env python3
"""
Helper function to find disease indices by name.
"""

import pandas as pd
import numpy as np

def find_disease_indices(disease_names, disease_list):
    """
    Find indices for diseases by name.
    
    Parameters:
    -----------
    disease_names : pd.Series or list
        List of all disease names
    disease_list : list
        List of disease names to find
        
    Returns:
    --------
    indices : list
        List of disease indices (0-based)
    """
    if isinstance(disease_names, pd.Series):
        disease_names = disease_names.tolist()
    
    indices = []
    for disease_name in disease_list:
        # Try exact match first
        matches = [i for i, name in enumerate(disease_names) if name == disease_name]
        if not matches:
            # Try case-insensitive partial match
            matches = [i for i, name in enumerate(disease_names) 
                      if disease_name.lower() in name.lower() or name.lower() in disease_name.lower()]
        
        if matches:
            indices.extend(matches)
        else:
            print(f"⚠️  Warning: Disease '{disease_name}' not found")
    
    return sorted(list(set(indices)))


def get_major_disease_indices(disease_names):
    """
    Get disease indices for major disease groups.
    
    Uses actual disease names from disease_names.csv (0-indexed: indices 0-347).
    
    Returns dict mapping disease group name to list of indices.
    """
    # Note: disease_names.csv is 1-indexed in the file, but we use 0-indexed for arrays
    # So disease at row 1 (index 0) is "Bacterial enteritis"
    # Disease at row 113 (index 112) is "Unstable angina"
    
    major_diseases = {
        'ASCVD': [
            "Unstable angina (intermediate coronary syndrome)",      # Should be index 112
            "Myocardial infarction",                                  # Should be index 113
            "Angina pectoris",                                        # Should be index 114
            "Coronary atherosclerosis",                              # Should be index 115
            "Other chronic ischemic heart disease, unspecified",      # Should be index 116
            "Other acute and subacute forms of ischemic heart disease" # Should be index 117
        ],
        'Stroke': [
            "Cerebral artery occlusion, with cerebral infarction",    # Should be index 136
            "Cerebral ischemia"                                       # Should be index 137
        ],
        'Heart_Failure': [
            "Congestive heart failure (CHF) NOS",                     # Should be index 131
            "Heart failure NOS"                                       # Should be index 132
        ],
        'Atrial_Fib': [
            "Atrial fibrillation and flutter"                        # Should be index 128
        ],
        'Diabetes': [
            "Type 2 diabetes"                                         # Should be index 48
        ],
        'All_Cancers': [
            "Colon cancer",                                           # Should be index 11
            "Malignant neoplasm of rectum, rectosigmoid junction, and anus", # Should be index 12
            "Cancer of bronchus; lung",                               # Should be index 14
            "Breast cancer [female]",                                 # Should be index 17
            "Malignant neoplasm of female breast",                    # Should be index 18
            "Cancer of prostate",                                     # Should be index 22
            "Malignant neoplasm of bladder",                         # Should be index 24
            "Secondary malignant neoplasm",                          # Should be index 26
            "Secondary malignancy of lymph nodes",                   # Should be index 27
            "Secondary malignancy of respiratory organs",            # Should be index 28
            "Secondary malignant neoplasm of digestive systems",     # Should be index 29
            "Secondary malignant neoplasm of liver",                 # Should be index 30
            "Secondary malignancy of bone"                           # Should be index 31
        ],
        'Leukemia_MDS': [
            # Note: These may not be in the disease list - CHIP can progress to these
            # but they might be rare or coded differently
            "Non-Hodgkins lymphoma",                                  # Should be index 33 (closest)
            # Leukemia/MDS might not be in the 348 diseases
        ],
        'Anemia': [
            "Iron deficiency anemias, unspecified or not due to blood loss", # Should be index 62
            "Other anemias"                                          # Should be index 63
        ],
        'Pneumonia': [
            "Pneumonia",                                              # Should be index 159
            "Bacterial pneumonia",                                   # Should be index 160
            "Pneumococcal pneumonia"                                 # Should be index 161
        ],
        'COPD': [
            "Chronic airway obstruction",                            # Should be index 163
            "Emphysema",                                             # Should be index 164
            "Obstructive chronic bronchitis"                        # Should be index 165
        ],
        'Sepsis': [
            "Sepsis"                                                 # Should be index 347
        ],
        'Acute_Renal_Failure': [
            "Acute renal failure"                                    # Should be index 229
        ],
    }
    
    disease_indices = {}
    for group_name, disease_list in major_diseases.items():
        indices = find_disease_indices(disease_names, disease_list)
        disease_indices[group_name] = indices
        if len(indices) > 0:
            print(f"{group_name}: {len(indices)} diseases found: {indices}")
            # Show which diseases were matched
            matched_names = [disease_names.iloc[i] if isinstance(disease_names, pd.Series) else disease_names[i] 
                           for i in indices]
            print(f"  Matched: {matched_names}")
        else:
            print(f"{group_name}: No diseases found")
    
    return disease_indices

