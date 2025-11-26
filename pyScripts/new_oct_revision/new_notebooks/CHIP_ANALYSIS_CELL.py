# CHIP carriers: enrichment of pre-event rise in Signature 5
# This cell analyzes CHIP mutation carriers (all CHIP, DNMT3A, TET2, etc.)
# Assumes you already have in memory:
# - processed_ids: np.array of eids (N,)
# - thetas_withpcs (or thetas_nopcs): shape [N, K=21, T]
# - Y: outcomes array shape [N, D, T]
# - event_indices: list/array of indices in Y that define the CAD/ASCVD composite

import torch
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
import sys
from pathlib import Path

# Add path for CHIP analysis script
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')
from analyze_chip_carriers_signature import load_chip_carriers, analyze_signature_enrichment_chip, visualize_signature_trajectory_chip

# ----------------------------
# Inputs
# ----------------------------
chip_file_path = '/Users/sarahurbut/Dropbox-Personal/CH_UKB.txt'  # CHIP data file
Y = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
if hasattr(Y, 'detach'):
    Y = Y.detach().cpu().numpy()
print(f"Y shape: {Y.shape}")

event_indices = [112, 113, 114, 115, 116]  # ASCVD composite

thetas_withpcs = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt', map_location='cpu')
if hasattr(thetas_withpcs, 'numpy'):
    thetas_withpcs = thetas_withpcs.numpy()
theta = thetas_withpcs   # choose which set to test

sig_idx = 5              # Signature 5 (0-based index)
pre_window = 5           # years/timepoints to look back before event
epsilon = 0.0            # >0 means strict rise; use small value like 0.002 to be conservative

# ----------------------------
# Analyze different CHIP mutations
# ----------------------------

# 1. All CHIP carriers
print("\n" + "="*80)
print("ANALYZING ALL CHIP CARRIERS")
print("="*80)
results_chip = analyze_signature_enrichment_chip(
    chip_file_path, 
    mutation_name="CHIP",
    mutation_type='hasCH',
    signature_idx=sig_idx,
    event_indices=event_indices,
    theta=theta,
    Y=Y,
    processed_ids=processed_ids,
    pre_window=pre_window,
    epsilon=epsilon
)

# 2. DNMT3A mutation carriers
print("\n" + "="*80)
print("ANALYZING DNMT3A MUTATION CARRIERS")
print("="*80)
results_dnmt3a = analyze_signature_enrichment_chip(
    chip_file_path,
    mutation_name="DNMT3A",
    mutation_type='hasDNMT3A',
    signature_idx=sig_idx,
    event_indices=event_indices,
    theta=theta,
    Y=Y,
    processed_ids=processed_ids,
    pre_window=pre_window,
    epsilon=epsilon
)

# 3. TET2 mutation carriers
print("\n" + "="*80)
print("ANALYZING TET2 MUTATION CARRIERS")
print("="*80)
results_tet2 = analyze_signature_enrichment_chip(
    chip_file_path,
    mutation_name="TET2",
    mutation_type='hasTET2',
    signature_idx=sig_idx,
    event_indices=event_indices,
    theta=theta,
    Y=Y,
    processed_ids=processed_ids,
    pre_window=pre_window,
    epsilon=epsilon
)

# ----------------------------
# Visualizations
# ----------------------------
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Visualize CHIP carriers
fig_chip = visualize_signature_trajectory_chip(
    results_chip, theta, Y, processed_ids, event_indices,
    sig_idx, "CHIP", pre_window
)
plt.show()

# Visualize DNMT3A carriers
fig_dnmt3a = visualize_signature_trajectory_chip(
    results_dnmt3a, theta, Y, processed_ids, event_indices,
    sig_idx, "DNMT3A", pre_window
)
plt.show()

# Visualize TET2 carriers
fig_tet2 = visualize_signature_trajectory_chip(
    results_tet2, theta, Y, processed_ids, event_indices,
    sig_idx, "TET2", pre_window
)
plt.show()

# ----------------------------
# Summary comparison
# ----------------------------
print("\n" + "="*80)
print("SUMMARY: CHIP MUTATION CARRIER COMPARISON")
print("="*80)
print(f"\n{'Mutation':<15} {'N Carriers':<12} {'Rising':<10} {'Prop Rising':<15} {'OR':<10} {'p-value':<12}")
print("-" * 80)

for name, res in [("CHIP", results_chip), ("DNMT3A", results_dnmt3a), ("TET2", results_tet2)]:
    n_car = res['n_carriers']
    rising = res['carriers_rising']
    prop = rising / max(n_car, 1)
    OR = res['OR']
    p = res['p_value']
    print(f"{name:<15} {n_car:<12} {rising:<10} {prop:.3f} ({prop*100:.1f}%){'':<6} {OR:<10.3f} {p:<12.3e}")

print("\n" + "="*80)

