
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering  # Add this import
from utils import *
from clust_huge_amp_fixedPhi import *
import sys
import os
import gc


start_index = 0 
end_index = 10000

def load_model_essentials(base_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'):
    """
    Load all essential components
    """
    print("Loading components...")
    
    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_matrix_corrected.pt', weights_only=False)  # Use enrollment-capped E
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)
    
    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)
    
    print("Loaded all components successfully!")
    
    return Y, E, G, essentials

# Load and initialize model:
Y, E, G, essentials = load_model_essentials()

# Subset the data
Y_100k, E_100k, G_100k, indices = subset_data(Y, E, G, start_index=start_index, end_index=end_index)

del Y

# Load references (signatures only, no healthy)
refs = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/reference_trajectories.pt', weights_only=False)
signature_refs = refs['signature_refs']

# Load the RDS file
import pandas as pd
fh_processed = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
len(fh_processed)

pce_df_subset = fh_processed.iloc[start_index:end_index].reset_index(drop=True)
sex = pce_df_subset['sex'].values
G_with_sex = np.column_stack([G_100k, sex])

# Add PCs
pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
              'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
pcs = pce_df_subset[pc_columns].values
G_with_sex = np.column_stack([G_with_sex, pcs])
print(f"G_with_sex shape: {G_with_sex.shape} (should be [N, 36 PRS + 1 sex + 10 PCs = 47])") 

import torch
import numpy as np
import cProfile
import pstats
from pstats import SortKey

# Path to your total fit model
from clust_huge_amp_fixedPhi import *
total_fit_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt'
total_checkpoint = torch.load(total_fit_path, map_location='cpu', weights_only=False)
phi_total = total_checkpoint['model_state_dict']['phi'].cpu().numpy()  # shape: (K, D, T)
psi_total = total_checkpoint['model_state_dict']['psi'].cpu().numpy()  # shape: (K, D, T)

print("\n" + "="*80)
print("WASHOUT ANALYSIS: Looping over 1, 2, 3 year washout periods")
print("="*80)
print("This removes all events within N years of enrollment")
print("="*80 + "\n")

# Loop over washout periods: 1, 2, 3 years
washout_years_list = [1, 2, 3]

for washout_years in washout_years_list:
    print("\n" + "="*80)
    print(f"{washout_years}-YEAR WASHOUT: Capping events at enrollment_age - {washout_years}")
    print("="*80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    model = AladynSurvivalFixedPhi(
        N=Y_100k.shape[0],
        D=Y_100k.shape[1],
        T=Y_100k.shape[2],
        K=20,
        P=G_with_sex.shape[1],
        G=G_with_sex,
        Y=Y_100k,
        R=0,
        W=0.0001,
        prevalence_t=essentials['prevalence_t'],
        init_sd_scaler=1e-1,
        genetic_scale=1,
        pretrained_phi=phi_total,
        pretrained_psi=psi_total,
        signature_references=signature_refs,
        healthy_reference=True,
        disease_names=essentials['disease_names']
    )

    if np.allclose(model.phi.cpu().numpy(), phi_total):
        print("✓ phi matches phi_total!")
    else:
        print("✗ WARNING: phi does NOT match phi_total!")

    if np.allclose(model.psi.cpu().numpy(), psi_total):
        print("✓ psi matches psi_total!")
    else:
        print("✗ WARNING: psi does NOT match psi_total!")

    # Reinitialize with psi_total for consistent gamma initialization
    print("Reinitializing gamma with psi_total for consistency...")
    model.initialize_params(init_psi=torch.tensor(psi_total, dtype=torch.float32))
    print("✓ Gamma reinitialized with psi_total")

    # Create washout event times: cap at enrollment_age - washout_years
    E_washout = E_100k.clone()
    pce_df_subset = fh_processed.iloc[start_index:end_index].reset_index(drop=True)

    # Initialize tracking variables
    total_times_changed = 0
    max_cap_applied = 0
    min_cap_applied = float('inf')

    print(f"\nApplying {washout_years}-year washout (capping at enrollment_age - {washout_years})...")

    for patient_idx, row in enumerate(pce_df_subset.itertuples()):
        if patient_idx >= E_washout.shape[0]:
            break
            
        # Enrollment age
        enrollment_age = row.age
        
        # Cap at enrollment_age - washout_years
        washout_age = enrollment_age - washout_years
        
        # Time since age 30 for washout age
        time_since_30 = max(0, washout_age - 30)

        max_cap_applied = max(max_cap_applied, time_since_30)
        min_cap_applied = min(min_cap_applied, time_since_30)
        
        # Store original times for this patient
        original_times = E_washout[patient_idx, :].clone()
        
        # Cap event times to washout age
        E_washout[patient_idx, :] = torch.minimum(
            E_washout[patient_idx, :],
            torch.full_like(E_washout[patient_idx, :], time_since_30)
        )

        times_changed = torch.sum(E_washout[patient_idx, :] != original_times).item()
        total_times_changed += times_changed

    # Print censoring verification
    print(f"\n{washout_years}-year washout censoring summary:")
    print(f"  Total event times changed: {total_times_changed:,}")
    print(f"  Max cap applied: {max_cap_applied:.1f}")
    print(f"  Min cap applied: {min_cap_applied:.1f}")

    # Check a few specific patients
    test_patients = [0, 1, 100]  # Check patients 0, 1, and 100
    for test_idx in test_patients:
        if test_idx < len(pce_df_subset):
            row = pce_df_subset.iloc[test_idx]
            enrollment_age = row.age
            washout_age = enrollment_age - washout_years
            expected_cap = max(0, washout_age - 30)
            
            # Check max value in this patient's event times
            max_time = torch.max(E_washout[test_idx, :]).item()
            
            print(f"  Patient {test_idx}: enrollment={enrollment_age:.0f}, washout_cap={washout_age:.0f}, "
                  f"time_cap={expected_cap:.1f}, max_event_time={max_time:.1f}")
            
            # Verify cap was applied correctly
            if max_time > expected_cap + 0.01:  # Small tolerance
                print(f"    WARNING: Max time {max_time:.1f} exceeds cap {expected_cap:.1f}!")

    # Train model with washout
    print(f"\nTraining model with {washout_years}-year washout...")
    profiler = cProfile.Profile()
    profiler.enable()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    history_new = model.fit(
        E_washout, 
        num_epochs=200, 
        learning_rate=1e-1, 
        lambda_reg=1e-2
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

    # Get predictions
    with torch.no_grad():
        pi, _, _ = model.forward()
        
        # Save predictions
        filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/washout_{washout_years}yr_local/pi_washout_{washout_years}yr_fixedphi_sex_{start_index}_{end_index}_withpcs.pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(pi, filename)
       
        print(f"\n✓ Saved predictions to {filename}")

    filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/washout_{washout_years}yr_local/model_washout_{washout_years}yr_fixedphi_sex_{start_index}_{end_index}_withpcs.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'E': E_washout,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
    }, filename)
    print(f"✓ Saved model to {filename}")

    # Clean up to free memory
    del pi
    del model
    del E_washout
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{washout_years}-YEAR WASHOUT COMPLETE!")

print("\n" + "="*80)
print("ALL WASHOUT PERIODS COMPLETE!")
print("="*80)

