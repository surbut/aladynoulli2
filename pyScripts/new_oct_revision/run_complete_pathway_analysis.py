#!/usr/bin/env python3
"""
Complete Pathway Analysis Pipeline

This script runs the full analysis framework:
1. Deviation-based pathway discovery (UKB)
2. Transition analysis (precursor → target disease)
3. Cross-cohort reproducibility validation (UKB vs MGB)

Usage:
    python run_complete_pathway_analysis.py
    OR
    from run_complete_pathway_analysis import run_complete_analysis
    results = run_complete_analysis()
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from run_complete_pathway_analysis_deviation_only import run_deviation_only_analysis
from run_transition_analysis_ukb_mgb import run_transition_analysis_both_cohorts
from show_pathway_reproducibility import main as show_reproducibility_main
from analyze_sig5_by_pathway import analyze_signature5_by_pathway


def run_complete_analysis(
    target_disease="myocardial infarction",
    transition_disease="Rheumatoid arthritis",
    n_pathways=4,
    lookback_years=10,
    output_dir='complete_pathway_analysis_output',
    mgb_model_path='/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'
):
    """
    Run complete pathway analysis pipeline
    
    Parameters:
    -----------
    target_disease : str
        Target disease (e.g., "myocardial infarction")
    transition_disease : str
        Precursor disease for transition analysis (e.g., "Rheumatoid arthritis")
    n_pathways : int
        Number of pathways to discover (default: 4)
    lookback_years : int
        Years before disease to analyze (default: 10)
    output_dir : str
        Output directory for all results
    mgb_model_path : str
        Path to MGB model file
    
    Returns:
    --------
    dict with all analysis results
    """
    print("="*80)
    print("COMPLETE PATHWAY ANALYSIS PIPELINE")
    print("="*80)
    print(f"Target disease: {target_disease}")
    print(f"Transition analysis: {transition_disease} → {target_disease}")
    print(f"Number of pathways: {n_pathways}")
    print(f"Lookback years: {lookback_years}")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # ============================================================================
    # STEP 1: DEVIATION-BASED PATHWAY DISCOVERY (UKB)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: DEVIATION-BASED PATHWAY DISCOVERY (UKB)")
    print("="*80)
    
    ukb_results = run_deviation_only_analysis(
        target_disease=target_disease,
        n_pathways=n_pathways,
        output_dir=os.path.join(output_dir, 'ukb_pathway_discovery'),
        lookback_years=lookback_years
    )
    
    results['ukb_pathway_discovery'] = ukb_results
    
    print("\n✅ UKB pathway discovery complete")
    
    # ============================================================================
    # STEP 2: TRANSITION ANALYSIS (UKB)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: TRANSITION ANALYSIS (UKB)")
    print("="*80)
    print(f"Analyzing: {transition_disease} → {target_disease}")
    
    try:
        transition_results = run_transition_analysis_both_cohorts(
            transition_disease_name=transition_disease,
            target_disease_name=target_disease,
            years_before=lookback_years,
            age_tolerance=5,
            min_followup=5,
            mgb_model_path=mgb_model_path,
            output_dir=os.path.join(output_dir, 'transition_analysis')
        )
        results['transition_analysis'] = transition_results
        print("\n✅ Transition analysis complete")
    except Exception as e:
        print(f"\n⚠️  Transition analysis failed: {e}")
        results['transition_analysis'] = None
    
    # ============================================================================
    # STEP 3: SIGNATURE 5 ANALYSIS BY PATHWAY (UKB)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 3: SIGNATURE 5 ANALYSIS BY PATHWAY (UKB)")
    print("="*80)
    
    try:
        sig5_results = analyze_signature5_by_pathway(
            target_disease=target_disease,
            output_dir=os.path.join(output_dir, 'ukb_pathway_discovery'),
            fh_carrier_path='/Users/sarahurbut/Downloads/out/ukb_exome_450k_fh.carrier.txt'
        )
        results['signature5_analysis'] = sig5_results
        print("\n✅ Signature 5 analysis complete")
    except Exception as e:
        print(f"\n⚠️  Signature 5 analysis failed: {e}")
        results['signature5_analysis'] = None
    
    # ============================================================================
    # STEP 4: CROSS-COHORT REPRODUCIBILITY (UKB vs MGB)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 4: CROSS-COHORT REPRODUCIBILITY VALIDATION")
    print("="*80)
    print("Comparing pathways between UKB and MGB...")
    
    try:
        # Run reproducibility analysis (uses existing results)
        print("   Note: This will use existing UKB and MGB pathway results")
        print("   If MGB results don't exist, run MGB analysis first")
        reproducibility_results = show_reproducibility_main(force_rerun_mgb=False)
        results['reproducibility'] = reproducibility_results
        print("\n✅ Reproducibility analysis complete")
    except Exception as e:
        print(f"\n⚠️  Reproducibility analysis failed: {e}")
        print("   This may require MGB results to be generated first")
        import traceback
        traceback.print_exc()
        results['reproducibility'] = None
    
    # ============================================================================
    # STEP 5: SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n✅ Completed analyses:")
    print(f"   1. UKB pathway discovery: {'✓' if results.get('ukb_pathway_discovery') else '✗'}")
    print(f"   2. Transition analysis: {'✓' if results.get('transition_analysis') else '✗'}")
    print(f"   3. Signature 5 analysis: {'✓' if results.get('signature5_analysis') else '✗'}")
    print(f"   4. Reproducibility validation: {'✓' if results.get('reproducibility') else '✗'}")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"\n   - UKB pathway discovery: {output_dir}/ukb_pathway_discovery/")
    print(f"   - Transition analysis: {output_dir}/transition_analysis/")
    print(f"   - Reproducibility: {output_dir}/reproducibility_analysis/")
    
    print("\n" + "="*80)
    print("✅ COMPLETE PATHWAY ANALYSIS PIPELINE FINISHED")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run complete analysis
    results = run_complete_analysis(
        target_disease="myocardial infarction",
        transition_disease="Rheumatoid arthritis",
        n_pathways=4,
        lookback_years=10
    )
    
    print("\n✅ All analyses complete!")
    print(f"\nResults dictionary keys: {list(results.keys())}")

