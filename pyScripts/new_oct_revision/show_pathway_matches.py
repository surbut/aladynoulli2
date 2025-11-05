#!/usr/bin/env python3
"""
Simple script to show which pathways match between UKB and MGB

Just shows the matching - UKB Pathway X ↔ MGB Pathway Y
"""

import pickle
import os
import sys
import numpy as np
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from match_pathways_by_disease_patterns import match_pathways_between_cohorts
from pathway_discovery import load_full_data
from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
import torch


def show_pathway_matches(force_rerun_mgb=False):
    """
    Show which pathways match between UKB and MGB
    """
    print("="*80)
    print("PATHWAY MATCHING: UKB ↔ MGB")
    print("="*80)
    
    # Load UKB results
    ukb_results_file = 'output_10yr/complete_analysis_results.pkl'
    if not os.path.exists(ukb_results_file):
        print(f"❌ UKB results not found at: {ukb_results_file}")
        print("   Run UKB analysis first!")
        return None
    
    print(f"\n1. Loading UKB results from: {ukb_results_file}")
    with open(ukb_results_file, 'rb') as f:
        ukb_results = pickle.load(f)
    print("   ✅ UKB results loaded")
    
    # Load or run MGB analysis
    mgb_results_file = 'mgb_deviation_analysis_output/mgb_deviation_analysis_results.pkl'
    
    if force_rerun_mgb or not os.path.exists(mgb_results_file):
        print(f"\n2. Running MGB analysis (force_rerun={force_rerun_mgb})...")
        from run_mgb_deviation_analysis_and_compare import run_deviation_analysis_mgb
        
        mgb_results = run_deviation_analysis_mgb(
            target_disease="myocardial infarction",
            n_pathways=4,
            lookback_years=10,
            output_dir='mgb_deviation_analysis_output'
        )
        
        if mgb_results is None:
            print("❌ MGB analysis failed")
            return None
        print("   ✅ MGB analysis complete")
    else:
        print(f"\n2. Loading MGB results from: {mgb_results_file}")
        with open(mgb_results_file, 'rb') as f:
            mgb_results = pickle.load(f)
        print("   ✅ MGB results loaded")
    
    # Load data for matching
    print(f"\n3. Loading data for pathway matching...")
    Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
    
    Y_mgb = mgb_results.get('Y')
    thetas_mgb = mgb_results.get('thetas')
    disease_names_mgb = mgb_results.get('disease_names')
    
    if Y_mgb is None or disease_names_mgb is None:
        print("   Loading MGB data from model...")
        Y_mgb, thetas_mgb, disease_names_mgb, _ = load_mgb_data_from_model()
    
    # Convert to torch
    if isinstance(Y_ukb, np.ndarray):
        Y_ukb = torch.from_numpy(Y_ukb)
    if isinstance(Y_mgb, np.ndarray):
        Y_mgb = torch.from_numpy(Y_mgb)
    
    # Match pathways
    print(f"\n4. Matching pathways by disease patterns...")
    ukb_pathway_data = ukb_results['pathway_data_dev']
    mgb_pathway_data = mgb_results['pathway_data']
    
    pathway_matching = match_pathways_between_cohorts(
        ukb_pathway_data, Y_ukb, disease_names_ukb,
        mgb_pathway_data, Y_mgb, disease_names_mgb,
        top_n_diseases=30
    )
    
    # Show simple matching table
    print("\n" + "="*80)
    print("PATHWAY MATCHES")
    print("="*80)
    print(f"\n{'UKB Pathway':<15} {'MGB Pathway':<15} {'Similarity':<15} {'Diseases Matched':<20}")
    print("-" * 80)
    
    best_matches = pathway_matching['best_matches']
    similarities = pathway_matching['pathway_similarities']
    disease_mappings = pathway_matching['disease_mappings']
    
    for ukb_id in sorted(best_matches.keys()):
        mgb_id = best_matches[ukb_id]
        similarity = similarities[(ukb_id, mgb_id)]
        n_diseases = len(disease_mappings[(ukb_id, mgb_id)])
        
        print(f"Pathway {ukb_id:<12} Pathway {mgb_id:<12} {similarity:<15.3f} {n_diseases:<20}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Found {len(best_matches)} pathway matches")
    
    avg_similarity = np.mean([similarities[(u, m)] for u, m in best_matches.items()])
    print(f"   Average similarity: {avg_similarity:.3f}")
    
    high_sim = sum(1 for u, m in best_matches.items() if similarities[(u, m)] > 0.5)
    print(f"   High similarity matches (>0.5): {high_sim}/{len(best_matches)}")
    
    return {
        'pathway_matching': pathway_matching,
        'ukb_results': ukb_results,
        'mgb_results': mgb_results
    }


if __name__ == "__main__":
    # Force re-run MGB analysis
    results = show_pathway_matches(force_rerun_mgb=True)

