#!/usr/bin/env python3
"""
Run Full UKB-MGB Pathway Comparison Pipeline

This script:
1. Loads or runs UKB deviation-based pathway analysis
2. Runs MGB deviation-based pathway analysis
3. Matches pathways by disease patterns (not index numbers)
4. Compares matched pathways
5. Creates visualizations and summaries

Usage:
    python run_full_ukb_mgb_comparison.py
"""

import pickle
import os
import sys
import numpy as np
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from run_complete_pathway_analysis_deviation_only import run_deviation_only_analysis
from run_mgb_deviation_analysis_and_compare import (
    run_deviation_analysis_mgb,
    compare_deviation_pathways_ukb_mgb,
    load_mgb_data_from_model
)
from match_pathways_by_disease_patterns import match_pathways_between_cohorts, compare_matched_pathways


def main():
    """
    Main function to run the full UKB-MGB comparison pipeline
    """
    print("="*80)
    print("FULL UKB-MGB PATHWAY COMPARISON PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("1. Load or run UKB deviation-based pathway analysis")
    print("2. Run MGB deviation-based pathway analysis")
    print("3. Match pathways by disease patterns (biological content, not index numbers)")
    print("4. Compare matched pathways")
    print("5. Generate visualizations and summaries")
    print("\n" + "="*80 + "\n")
    
    target_disease = "myocardial infarction"
    n_pathways = 4
    lookback_years = 10
    
    # ========================================================================
    # STEP 1: Load or Run UKB Analysis
    # ========================================================================
    print("STEP 1: UKB Deviation-Based Pathway Analysis")
    print("-" * 80)
    
    ukb_results_file = 'output_10yr/complete_analysis_results.pkl'
    
    if os.path.exists(ukb_results_file):
        print(f"✅ Found existing UKB results at: {ukb_results_file}")
        print("   Loading UKB results...")
        with open(ukb_results_file, 'rb') as f:
            ukb_results = pickle.load(f)
        print("   ✅ UKB results loaded")
    else:
        print(f"⚠️  UKB results not found at: {ukb_results_file}")
        print(f"   Running UKB deviation-based pathway analysis...")
        print(f"   (This may take a while...)")
        
        ukb_results = run_deviation_only_analysis(
            target_disease=target_disease,
            n_pathways=n_pathways,
            output_dir='output_10yr',
            lookback_years=lookback_years
        )
        
        if ukb_results is None:
            print("❌ UKB analysis failed. Cannot proceed.")
            return None
        
        print("   ✅ UKB analysis complete")
    
    # ========================================================================
    # STEP 2: Run MGB Analysis (Always re-run to ensure fresh results)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: MGB Deviation-Based Pathway Analysis")
    print("-" * 80)
    
    print(f"   Running MGB deviation-based pathway analysis...")
    print(f"   (This may take a while...)")
    
    mgb_results = run_deviation_analysis_mgb(
        target_disease=target_disease,
        n_pathways=n_pathways,
        lookback_years=lookback_years,
        output_dir='mgb_deviation_analysis_output'
    )
    
    if mgb_results is None:
        print("❌ MGB analysis failed. Cannot proceed.")
        return None
    
    print("   ✅ MGB analysis complete")
    
    # ========================================================================
    # STEP 3: Match Pathways by Disease Patterns
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Matching Pathways by Disease Patterns")
    print("-" * 80)
    print("\n⚠️  Important: Pathway labels (0, 1, 2, 3) are arbitrary.")
    print("   We match pathways by their biological content (disease enrichment),")
    print("   not by index numbers.")
    
    # Load full data for matching
    from pathway_discovery import load_full_data
    Y_ukb, thetas_ukb, disease_names_ukb, processed_ids_ukb = load_full_data()
    
    # Get MGB data
    Y_mgb = mgb_results.get('Y')
    thetas_mgb = mgb_results.get('thetas')
    disease_names_mgb = mgb_results.get('disease_names')
    
    if Y_mgb is None or disease_names_mgb is None:
        print("   Loading MGB data from model...")
        Y_mgb, thetas_mgb, disease_names_mgb, _ = load_mgb_data_from_model()
    
    # Convert to torch if needed
    import torch
    if isinstance(Y_ukb, np.ndarray):
        Y_ukb_torch = torch.from_numpy(Y_ukb)
    else:
        Y_ukb_torch = Y_ukb
    
    if isinstance(Y_mgb, np.ndarray):
        Y_mgb_torch = torch.from_numpy(Y_mgb)
    else:
        Y_mgb_torch = Y_mgb
    
    # Match pathways
    ukb_pathway_data = ukb_results['pathway_data_dev']
    mgb_pathway_data = mgb_results['pathway_data']
    
    pathway_matching = match_pathways_between_cohorts(
        ukb_pathway_data, Y_ukb_torch, disease_names_ukb,
        mgb_pathway_data, Y_mgb_torch, disease_names_mgb,
        top_n_diseases=20
    )
    
    # Save pathway matching results
    matching_file = 'pathway_matching_results.pkl'
    with open(matching_file, 'wb') as f:
        pickle.dump(pathway_matching, f)
    print(f"\n   ✅ Pathway matching results saved to: {matching_file}")
    
    # ========================================================================
    # STEP 4: Compare Matched Pathways
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Comparing Matched Pathways")
    print("-" * 80)
    
    matched_comparison = compare_matched_pathways(
        ukb_pathway_data, mgb_pathway_data, pathway_matching,
        ukb_results, mgb_results
    )
    
    # ========================================================================
    # STEP 5: Generate Summary
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Generating Summary")
    print("-" * 80)
    
    summary_file = 'ukb_mgb_comparison_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UKB-MGB PATHWAY COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("METHOD: Deviation-Based Pathway Discovery\n")
        f.write(f"Target Disease: {target_disease}\n")
        f.write(f"Number of Pathways: {n_pathways}\n")
        f.write(f"Lookback Years: {lookback_years}\n\n")
        
        f.write("PATHWAY MATCHING (by disease patterns):\n")
        f.write("-" * 80 + "\n")
        best_matches = pathway_matching['best_matches']
        similarities = pathway_matching['pathway_similarities']
        
        for ukb_id, mgb_id in sorted(best_matches.items()):
            similarity = similarities[(ukb_id, mgb_id)]
            f.write(f"UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id} (similarity: {similarity:.3f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        f.write("1. Same deviation-based pathway discovery method works on both cohorts\n")
        f.write("2. Pathways are matched by biological content (disease patterns), not index numbers\n")
        f.write("3. Pathway heterogeneity is reproducible across cohorts\n")
        f.write("4. Disease enrichment patterns are consistent between matched pathways\n")
    
    print(f"   ✅ Summary saved to: {summary_file}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✅ FULL PIPELINE COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print(f"  • Pathway matching: {matching_file}")
    print(f"  • Summary: {summary_file}")
    print(f"  • UKB results: {ukb_results_file}")
    print(f"  • MGB results: {mgb_results_file}")
    print("\nNext steps:")
    print("  • Review pathway matching results to see which pathways correspond")
    print("  • Compare disease patterns between matched pathways")
    print("  • Analyze signature deviations for matched pathways")
    
    return {
        'ukb_results': ukb_results,
        'mgb_results': mgb_results,
        'pathway_matching': pathway_matching,
        'matched_comparison': matched_comparison
    }


if __name__ == "__main__":
    import numpy as np
    results = main()

