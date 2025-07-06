
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

def load_model_essentials(base_path='/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/'):
    """
    Load all essential components
    """
    print("Loading components...")
    
    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt')
    E = torch.load(base_path + 'E_matrix.pt')
    G = torch.load(base_path + 'G_matrix.pt')
    
    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt')
    
    print("Loaded all components successfully!")
    
    return Y, E, G, essentials

# Load and initialize model:
Y, E, G, essentials = load_model_essentials()

# Subset the data

# Subset the data
Y_100k, E_100k, G_100k, indices = subset_data(Y, E, G, start_index=start_index, end_index=end_index)


del Y

# Load references (signatures only, no healthy)
refs = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/reference_trajectories.pt')
signature_refs = refs['signature_refs']
# When initializing the model:


# Load the RDS file

import pandas as pd
fh_processed=pd.read_csv('/Users/sarahurbut/Library/Cloudstorage/Dropbox/baselinagefamh.csv')
len(fh_processed)


pce_df_subset = fh_processed.iloc[start_index:end_index].reset_index(drop=True)
sex=pce_df_subset['sex'].values
G_with_sex = np.column_stack([G_100k, sex]) 

import torch
import numpy as np
import cProfile
import pstats
from pstats import SortKey


    
    # Path to your total fit model
from clust_huge_amp_fixedPhi import *
total_fit_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_model_W0.0001_fulldata_sexspecific.pt'
total_checkpoint = torch.load(total_fit_path, map_location='cpu')
phi_total = total_checkpoint['model_state_dict']['phi'].cpu().numpy()  # shape: (K, D, T)
psi_total = total_checkpoint['model_state_dict']['psi'].cpu().numpy()  # shape: (K, D, T)



# Store predictions for each age
age_predictions = {}

# At the top of your script, define your batch indices
start_index = start_index
end_index = end_index

for age_offset in range(0, 10):  # Ages 0-10 years after enrollment
    print(f"\n=== Predicting for age offset {age_offset} years ===")

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
        print("phi matches phi_total!")
    else:
        print("phi does NOT match phi_total!")

    if np.allclose(model.psi.cpu().numpy(), psi_total):
        print("psi matches psi_total!")
    else:
        print("psi does NOT match psi_total!")



     # Create age-specific event times
    E_age_specific = E_100k.clone()
    pce_df_subset = fh_processed.iloc[start_index:end_index].reset_index(drop=True)

     
    # Initialize tracking variables for this age offset
    total_times_changed = 0
    max_cap_applied = 0
    min_cap_applied = float('inf')

    
    for patient_idx, row in enumerate(pce_df_subset.itertuples()):
        if patient_idx >= E_age_specific.shape[0]:
            break
            
        # Current age = enrollment age + age_offset
        current_age = row.age + age_offset
        
        # Time since age 30 for this current age
        time_since_30 = max(0, current_age - 30)

        max_cap_applied = max(max_cap_applied, time_since_30)
        min_cap_applied = min(min_cap_applied, time_since_30)
        
        # Store original times for this patient
        original_times = E_age_specific[patient_idx, :].clone()
        
        # Cap event times to current age
        E_age_specific[patient_idx, :] = torch.minimum(
            E_age_specific[patient_idx, :],
            torch.full_like(E_age_specific[patient_idx, :], time_since_30)
        )

        times_changed = torch.sum(E_age_specific[patient_idx, :] != original_times).item()
        total_times_changed += times_changed
    
    # Print censoring verification
    print(f"Censoring verification for age offset {age_offset}:")
    print(f"  Total event times changed: {total_times_changed}")
    print(f"  Max cap applied: {max_cap_applied:.1f}")
    print(f"  Min cap applied: {min_cap_applied:.1f}")
    
    # Check a few specific patients
    test_patients = [0, 1, 100]  # Check patients 0, 1, and 100
    for test_idx in test_patients:
        if test_idx < len(pce_df_subset):
            row = pce_df_subset.iloc[test_idx]
            enrollment_age = row.age
            current_age = enrollment_age + age_offset
            expected_cap = max(0, current_age - 30)
            
            # Check max value in this patient's event times
            max_time = torch.max(E_age_specific[test_idx, :]).item()
            
            print(f"  Patient {test_idx}: enrollment={enrollment_age:.0f}, current={current_age:.0f}, "
                  f"cap={expected_cap:.1f}, max_event_time={max_time:.1f}")
            
            # Verify cap was applied correctly
            if max_time > expected_cap + 0.01:  # Small tolerance
                print(f"    WARNING: Max time {max_time:.1f} exceeds cap {expected_cap:.1f}!")
    
  
    
    # Train model for this specific age
    print(f"Training model for age offset {age_offset}...")
    profiler = cProfile.Profile()
    profiler.enable()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    history_new = model.fit(
        E_age_specific, 
        num_epochs=200, 
        learning_rate=1e-1, 
        lambda_reg=1e-2
    )
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    
    

    plot_training_evolution(history_new)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    
    # Get predictions for this age
    with torch.no_grad():
        pi, _, _ = model.forward()
        
        # Save age-specific predictions
        filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_enroll_fixedphi_age_offset_{age_offset}_sex_{start_index}_{end_index}_try2.pt"
        torch.save(pi, filename)
       
        print(f"Saved predictions to {filename}")

    filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox/model_enroll_fixedphi_age_offset_{age_offset}_sex_{start_index}_{end_index}_try2.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'E': E_age_specific,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
    }, filename)
    print(f"Saved model to {filename}")
        # Store in dictionary for potential analysis
        
    
    # Clean up to free memory
    del pi
    del model
    del E_age_specific
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
