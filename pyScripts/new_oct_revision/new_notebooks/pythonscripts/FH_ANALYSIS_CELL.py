# FH carriers: enrichment of pre-event rise in Signature 5
# This cell analyzes FH mutation carriers
# Assumes you already have in memory:
# - processed_ids: np.array of eids (N,)
# - thetas_withpcs (or thetas_nopcs): shape [N, K=21, T]
# - Y: outcomes array shape [N, D, T]
# - event_indices: list/array of indices in Y that define the CAD/ASCVD composite

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
import sys
from pathlib import Path

# Add path for FH analysis script
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')
from analyze_fh_carriers_signature import load_fh_carriers, analyze_signature_enrichment_fh, visualize_signature_trajectory_fh

# ----------------------------
# Inputs
# ----------------------------
fh_file_path = '/Users/sarahurbut/Downloads/out/ukb_exome_450k_fh.carrier.txt'  # FH carrier file
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

# Output directory for results
output_dir = Path('results/fh_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Run FH analysis
# ----------------------------
print("\n" + "="*80)
print("ANALYZING FH CARRIERS")
print("="*80)

results_fh = analyze_signature_enrichment_fh(
    fh_file_path, 
    signature_idx=sig_idx,
    event_indices=event_indices,
    theta=theta,
    Y=Y,
    processed_ids=processed_ids,
    pre_window=pre_window,
    epsilon=epsilon,
    output_dir=output_dir
)

# ----------------------------
# Generate visualization
# ----------------------------
fig = visualize_signature_trajectory_fh(
    results_fh, theta, Y, processed_ids, event_indices,
    signature_idx=sig_idx,
    pre_window=pre_window
)

# Save figure
output_path = output_dir / f'FH_signature{sig_idx}_trajectory.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")

plt.show()

