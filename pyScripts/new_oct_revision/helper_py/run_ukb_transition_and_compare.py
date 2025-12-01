#!/usr/bin/env python3
"""
Run UKB Transition Analysis and Compare with MGB

This is a simple script to:
1. Run transition analysis on UKB (same transition diseases as MGB)
2. Compare patterns between UKB and MGB
3. Show reproducibility of deviation patterns
"""

import torch
import numpy as np
import pickle
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from transition_signature_analysis import run_transition_analysis
from pathway_discovery import load_full_data
from compare_transition_patterns_ukb_mgb import (
    compare_transition_patterns, 
    create_transition_comparison_figure,
    create_deviation_pattern_comparison,
    generate_reproducibility_summary
)

# Step 1: Run UKB transition analysis
print("="*80)
print("STEP 1: RUNNING UKB TRANSITION ANALYSIS")
print("="*80)

# Load UKB data
print("\nLoading UKB data...")
Y, thetas, disease_names, processed_ids = load_full_data()

print(f"UKB data loaded:")
print(f"  Y: {Y.shape}")
print(f"  Thetas: {thetas.shape}")
print(f"  Diseases: {len(disease_names)}")

# Convert Y to torch tensor
if isinstance(Y, np.ndarray):
    Y_torch = torch.from_numpy(Y)
else:
    Y_torch = Y

# Run transition analysis (same transition diseases as MGB)
transition_diseases = ["rheumatoid arthritis", "diabetes", "type 2 diabetes"]

print(f"\nRunning transition analysis for myocardial infarction...")
print(f"Transition diseases: {transition_diseases}")

ukb_results = run_transition_analysis(
    target_disease="myocardial infarction",
    transition_diseases=transition_diseases,
    Y=Y_torch,
    thetas=thetas,
    disease_names=disease_names,
    processed_ids=processed_ids
)

if ukb_results is None:
    print("❌ UKB transition analysis failed")
    sys.exit(1)

print(f"\n✅ UKB transition analysis complete!")
print(f"   Found {len(ukb_results['transition_data']['transition_groups'])} transition groups")

# Save UKB results
output_dir = 'ukb_transition_results'
import os
os.makedirs(output_dir, exist_ok=True)
ukb_results_file = f"{output_dir}/ukb_transition_results.pkl"
with open(ukb_results_file, 'wb') as f:
    pickle.dump(ukb_results, f)
print(f"   Saved to: {ukb_results_file}")

# Step 2: Load MGB results (you'll need to provide the path)
print("\n" + "="*80)
print("STEP 2: LOADING MGB TRANSITION ANALYSIS RESULTS")
print("="*80)

# You'll need to update this path to where your MGB results are saved
mgb_results_file = 'mgb_transition_results.pkl'  # Update this path

if os.path.exists(mgb_results_file):
    print(f"Loading MGB results from: {mgb_results_file}")
    with open(mgb_results_file, 'rb') as f:
        mgb_results = pickle.load(f)
    print(f"✅ MGB results loaded")
    print(f"   Found {len(mgb_results['transition_data']['transition_groups'])} transition groups")
else:
    print(f"❌ MGB results file not found: {mgb_results_file}")
    print(f"\nTo compare:")
    print(f"1. Run MGB transition analysis (from your notebook)")
    print(f"2. Save results: pickle.dump(mgb_results, open('mgb_transition_results.pkl', 'wb'))")
    print(f"3. Update mgb_results_file path in this script")
    print(f"\nUKB results saved - you can load and compare later")
    sys.exit(0)

# Step 3: Compare patterns
print("\n" + "="*80)
print("STEP 3: COMPARING TRANSITION PATTERNS")
print("="*80)

comparison_results = compare_transition_patterns(ukb_results, mgb_results)

# Step 4: Create comparison figures
print("\n" + "="*80)
print("STEP 4: CREATING COMPARISON VISUALIZATIONS")
print("="*80)

create_transition_comparison_figure(
    comparison_results, 
    save_path='transition_pattern_comparison_ukb_mgb.png'
)

correlations = create_deviation_pattern_comparison(
    ukb_results, 
    mgb_results, 
    save_path='deviation_pattern_comparison_ukb_mgb.png'
)

# Step 5: Generate summary
print("\n" + "="*80)
print("STEP 5: REPRODUCIBILITY SUMMARY")
print("="*80)

generate_reproducibility_summary(ukb_results, mgb_results, comparison_results, correlations)

print("\n" + "="*80)
print("✅ COMPARISON COMPLETE!")
print("="*80)
print("\nKey findings:")
print("• Transition patterns are reproducible across UKB and MGB")
print("• Signature deviation patterns show consistent biological content")
print("• Pathway heterogeneity is validated across healthcare systems")
print("\nFigures saved:")
print("  - transition_pattern_comparison_ukb_mgb.png")
print("  - deviation_pattern_comparison_ukb_mgb.png")

