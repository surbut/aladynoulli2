"""
Simple example: How to use preprocessing functions

Just copy and paste this into your notebook or script!
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Import preprocessing utilities
from preprocessing_utils import (
    compute_smoothed_prevalence,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)

# Load your data
data_path = Path('/Users/sarahurbut/Dropbox-Personal/data_for_running/')
Y = torch.load(data_path / 'Y_tensor.pt', weights_only=False)
K = 20  # Number of signatures

# ============================================================================
# Step 1: Compute prevalence
# ============================================================================
prevalence_t = compute_smoothed_prevalence(Y, window_size=5, smooth_on_logit=True)
print(f"✓ Prevalence shape: {prevalence_t.shape}")

# ============================================================================
# Step 2: Create clusters and psi
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

clusters, psi = create_initial_clusters_and_psi(
    Y=Y,
    K=K,
    psi_config=None,  # Uses defaults, or pass dict with 'in_cluster', 'out_cluster', etc.
    healthy_reference=None,  # Set to True if you want healthy reference
    random_state=42
)
print(f"✓ Clusters shape: {clusters.shape}")
print(f"✓ Psi shape: {psi.shape}")

# ============================================================================
# Step 3: Create reference trajectories
# ============================================================================
signature_refs, healthy_ref = create_reference_trajectories(
    Y=Y,
    initial_clusters=clusters,
    K=K,
    healthy_prop=0.01,
    frac=0.3
)
print(f"✓ Signature refs shape: {signature_refs.shape}")
print(f"✓ Healthy ref shape: {healthy_ref.shape}")

# ============================================================================
# Save files (optional)
# ============================================================================
# output_dir = Path('/Users/sarahurbut/Dropbox-Personal/data_for_running/')
# torch.save(clusters, output_dir / 'initial_clusters_400k.pt')
# torch.save(psi, output_dir / 'initial_psi_400k.pt')
# torch.save({
#     'signature_refs': signature_refs,
#     'healthy_ref': healthy_ref
# }, output_dir / 'reference_trajectories.pt')
# print("✓ Saved all files!")

