#!/usr/bin/env python3
"""
Run Transition-Based Signature Analysis

This script analyzes signature patterns for specific disease transitions,
focusing on the "patient morphs" narrative.
"""

import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data
from transition_signature_analysis import run_transition_analysis

def main():
    print("Loading data...")
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Define transition diseases to analyze
    # These are diseases that might lead to MI through different pathways
    transition_diseases = [
        'rheumatoid arthritis',
        'diabetes', 
        'hypertension',
        'hypercholesterolemia',
        'obesity',
        'Major depressive disorder',
        'Malignant neoplasm of breast',
        'Maligant neoplasm of prostate',
        'Maligant neoplasm of bladder',
        'Maligant neoplasm of colon',
        'Maligant neoplasm of lung',
        'Maligant neoplasm of esophagus',
        'Maligant neoplasm of rectum',
        #'Maligant neoplasm of pancreas',
        'anxiety disorder',
    ]
    
    print(f"Analyzing transitions to Myocardial Infarction...")
    print(f"Looking for patients who had these diseases before MI:")
    for disease in transition_diseases:
        print(f"  - {disease}")
    
    # Run the analysis
    results = run_transition_analysis(
        target_disease='myocardial infarction',
        transition_diseases=transition_diseases,
        Y=Y,
        thetas=thetas,
        disease_names=disease_names,
        processed_ids=processed_ids
    )
    
    if results:
        print("\n✅ Transition analysis completed!")
        print(f"Found {len(results['transition_data']['transition_groups'])} transition groups")
        
        # Print summary
        for group_name, patients in results['transition_data']['transition_groups'].items():
            print(f"  {group_name}: {len(patients)} patients")
    else:
        print("❌ Transition analysis failed")

if __name__ == "__main__":
    main()
