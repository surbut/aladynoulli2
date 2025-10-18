#!/usr/bin/env python3
"""
Run Complete Pathway Analysis
This script runs the complete pathway discovery and analysis pipeline
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data, discover_disease_pathways, compare_clustering_methods
from pathway_interrogation import interrogate_disease_pathways, compare_pathway_methods, analyze_prs_by_pathway, analyze_granular_diseases_by_pathway
from medication_integration import integrate_medications_with_pathways, visualize_medication_pathway_integration
from disease_sequence_analysis import analyze_disease_sequences_for_target

def run_complete_pathway_analysis(target_disease, n_pathways=4):
    """
    Run the complete pathway analysis pipeline
    
    Parameters:
    - target_disease: Name of disease to analyze (e.g., "myocardial infarction", "breast cancer")
    - n_pathways: Number of pathways to discover
    """
    print("="*80)
    print(f"COMPLETE PATHWAY ANALYSIS: {target_disease.upper()}")
    print("="*80)
    
    # Step 1: Load full data
    print("\n1. LOADING FULL DATASET")
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Step 2: Discover pathways using both methods
    print(f"\n2. DISCOVERING PATHWAYS TO {target_disease.upper()}")
    pathway_data_avg, pathway_data_traj = compare_clustering_methods(
        target_disease, Y, thetas, disease_names, n_pathways=n_pathways
    )
    
    if pathway_data_avg is None or pathway_data_traj is None:
        print(f"❌ Could not discover pathways for {target_disease}")
        return None
    
    # Step 3: Interrogate both pathway methods
    print(f"\n3. INTERROGATING PATHWAYS")
    results_avg, results_traj = compare_pathway_methods(
        pathway_data_avg, pathway_data_traj, Y, thetas, disease_names
    )
    
    # Step 4: Integrate medications with pathways
    print(f"\n4. INTEGRATING MEDICATIONS WITH PATHWAYS")
    
    # Try with average loading method first
    print("\n4a. Average Loading Method:")
    medication_results_avg = integrate_medications_with_pathways(
        pathway_data_avg, Y, thetas, disease_names, processed_ids
    )
    
    if medication_results_avg is not None:
        visualize_medication_pathway_integration(medication_results_avg, pathway_data_avg)
    
    # Try with trajectory similarity method
    print("\n4b. Trajectory Similarity Method:")
    medication_results_traj = integrate_medications_with_pathways(
        pathway_data_traj, Y, thetas, disease_names, processed_ids
    )
    
    if medication_results_traj is not None:
        visualize_medication_pathway_integration(medication_results_traj, pathway_data_traj)
    
    # Step 5: Analyze PRS differences by pathway
    print(f"\n5. ANALYZING PRS DIFFERENCES BY PATHWAY")
    
    # Analyze PRS for average loading method
    print("\n5a. Average Loading Method:")
    prs_results_avg = analyze_prs_by_pathway(pathway_data_avg, processed_ids)
    
    # Analyze PRS for trajectory similarity method
    print("\n5b. Trajectory Similarity Method:")
    prs_results_traj = analyze_prs_by_pathway(pathway_data_traj, processed_ids)
    
    # Step 6: Analyze granular disease patterns
    print(f"\n6. ANALYZING GRANULAR DISEASE PATTERNS")
    
    # Analyze granular diseases for average loading method
    print("\n6a. Average Loading Method:")
    granular_results_avg = analyze_granular_diseases_by_pathway(pathway_data_avg, Y, disease_names)
    
    # Analyze granular diseases for trajectory similarity method
    print("\n6b. Trajectory Similarity Method:")
    granular_results_traj = analyze_granular_diseases_by_pathway(pathway_data_traj, Y, disease_names)
    
    # Step 7: Analyze disease sequences using granular ICD-10 data (optional)
    print(f"\n7. ANALYZING DISEASE SEQUENCES FROM GRANULAR ICD-10 DATA")
    print("(This step requires ICD-10 diagnosis data and pyreadr package)")
    
    sequence_results = None
    try:
        # Map disease names to ICD-10 codes
        # This is a simple mapping - you may need to expand this
        icd10_mapping = {
            'myocardial infarction': ['I21', 'I22'],
            'breast cancer': ['C50'],
            'diabetes': ['E10', 'E11'],
            'stroke': ['I63', 'I64'],
            'coronary atherosclerosis': ['I25'],
            'depression': ['F32', 'F33']
        }
        
        target_lower = target_disease.lower()
        target_icd_codes = None
        
        # Find matching ICD-10 codes
        for disease_key, icd_codes in icd10_mapping.items():
            if disease_key in target_lower or target_lower in disease_key:
                target_icd_codes = icd_codes
                break
        
        if target_icd_codes:
            print(f"Found ICD-10 codes for {target_disease}: {target_icd_codes}")
            sequence_results = analyze_disease_sequences_for_target(
                target_disease, target_icd_codes, thetas, processed_ids
            )
        else:
            print(f"⚠️  No ICD-10 mapping found for {target_disease}")
            print(f"   Add mapping to run sequence analysis")
    
    except Exception as e:
        print(f"⚠️  Could not run sequence analysis: {e}")
        print(f"   This is optional - main pathway analysis is complete")
    
    print(f"\n✅ Complete pathway analysis for {target_disease} finished!")
    
    return {
        'pathway_data_avg': pathway_data_avg,
        'pathway_data_traj': pathway_data_traj,
        'results_avg': results_avg,
        'results_traj': results_traj,
        'medication_results_avg': medication_results_avg,
        'medication_results_traj': medication_results_traj,
        'prs_results_avg': prs_results_avg,
        'prs_results_traj': prs_results_traj,
        'granular_results_avg': granular_results_avg,
        'granular_results_traj': granular_results_traj,
        'sequence_results': sequence_results
    }

def run_multiple_diseases(diseases, n_pathways=4):
    """Run pathway analysis for multiple diseases"""
    print("="*80)
    print("MULTIPLE DISEASE PATHWAY ANALYSIS")
    print("="*80)
    
    results = {}
    
    for disease in diseases:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {disease.upper()}")
        print(f"{'='*60}")
        
        try:
            disease_results = run_complete_pathway_analysis(disease, n_pathways=n_pathways)
            results[disease] = disease_results
        except Exception as e:
            print(f"❌ Error analyzing {disease}: {e}")
            results[disease] = None
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Pathway Analysis Pipeline")
    print("Available functions:")
    print("1. run_complete_pathway_analysis(target_disease, n_pathways=4)")
    print("2. run_multiple_diseases(diseases, n_pathways=4)")
    print("\nExample diseases to try:")
    print("- 'myocardial infarction'")
    print("- 'breast cancer'")
    print("- 'diabetes'")
    print("- 'stroke'")
    print("- 'depression'")
    
    # Uncomment to run a specific analysis:
    #results = run_complete_pathway_analysis("myocardial infarction", n_pathways=4)
