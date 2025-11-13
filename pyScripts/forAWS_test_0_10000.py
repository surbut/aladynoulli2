
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering  # Add this import
import sys
import os
import gc
from utils import *
class DummyFile(object):
    def write(self, x): pass

def suppress_stdout():
    sys.stdout = DummyFile()

def enable_stdout():
    sys.stdout = sys.__stdout__



def plot_training_evolution(history, plot_dir):
    """Plot and save training metrics."""
    losses, gradient_history = history
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.yscale('log')
    plt.legend()
    
    # Plot lambda gradients
    plt.subplot(1, 3, 2)
    lambda_norms = [torch.norm(g).item() for g in gradient_history['lambda_grad']]
    plt.plot(lambda_norms, label='Lambda gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Lambda Gradient Evolution')
    plt.legend()
    
    # Plot phi gradients
    plt.subplot(1, 3, 3)
    phi_norms = [torch.norm(g).item() for g in gradient_history['phi_grad']]
    plt.plot(phi_norms, label='Phi gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Phi Gradient Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_evolution.png"))
    plt.close()



def load_model_essentials(base_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'):
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

from clust_huge_amp import *
# Subset the data

# Subset the data
Y_100k, E_100k, G_100k, indices = subset_data(Y, E, G, start_index=0, end_index=10000)


del Y

# Load references (signatures only, no healthy)
refs = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/reference_trajectories.pt')
signature_refs = refs['signature_refs']


# Load the RDS file

import pandas as pd
fh_processed=pd.read_csv('/Users/sarahurbut/Library/Cloudstorage/Dropbox-Personal/baselinagefamh_withpcs.csv')


pce_df_subset = fh_processed.iloc[0:10000].reset_index(drop=True)
sex=pce_df_subset['sex'].values
G_with_sex = np.column_stack([G_100k, sex]) 
pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
pcs = fh_processed.iloc[0:10000][pc_columns].values
G_with_sex = np.column_stack([G_100k, sex, pcs])



import torch
import numpy as np
import cProfile
import pstats
from pstats import SortKey

# Store predictions for each age
age_predictions = {}

for age_offset in range(0, 11):  # Ages 0-10 years after enrollment
    print(f"\n=== Predicting for age offset {age_offset} years ===")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Initialize fresh model for this age
    suppress_stdout()
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_100k.shape[0],
        D=Y_100k.shape[1],
        T=Y_100k.shape[2],
        K=20,
        P=G_with_sex.shape[1],
        init_sd_scaler=1e-1,
        G=G_with_sex,
        Y=Y_100k,
        genetic_scale=1,
        W=0.0001,
        R=0,
        prevalence_t=essentials['prevalence_t'],
        signature_references=signature_refs,
        healthy_reference=True,
        disease_names=essentials['disease_names']
    )
   
    # Reset seeds for parameter initialization
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Load and set initial parameters
    initial_psi = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_psi_400k.pt')
    initial_clusters = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt')
    model.initialize_params(true_psi=initial_psi)
    enable_stdout()
    model.clusters = initial_clusters
    
    # Verify clusters match
    clusters_match = np.array_equal(initial_clusters, model.clusters)
    print(f"Clusters match exactly: {clusters_match}")
    
    # Create age-specific event times
    E_age_specific = E_100k.clone()
    pce_df_subset = fh_processed.iloc[0:10000].reset_index(drop=True)

     
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
    
    suppress_stdout()
    history_new = model.fit(
        E_age_specific, 
        num_epochs=200, 
        learning_rate=1e-1, 
        lambda_reg=1e-2
    )
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    
    enable_stdout()

    plot_dir = f"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/plots_age_offsetwithpcs_{age_offset}"
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_evolution(history_new, plot_dir)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    
    # Get predictions for this age
    model.eval()
    with torch.no_grad():
        # Compute pi predictions
        theta = torch.softmax(model.lambda_, dim=1)  # [N, K, T]
        phi_prob = torch.sigmoid(model.phi)  # [K, D, T]
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * model.kappa  # [N, D, T]
    
    print(f"Generated predictions: pi shape = {pi.shape}")
    
    # Save age-specific predictions
    pi_filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_enroll_age_offset_{age_offset}_sex_0_10000_try2_withpcs.pt"
    torch.save(pi, pi_filename)
    print(f"Saved predictions to: {pi_filename}")
    
    # Save model
    filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_enroll_age_offset_{age_offset}_sex_0_10000_try2_withpcs.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'E': E_age_specific,
        'phi': model.phi,
        'Y': model.Y,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'G': model.G,
        'indices': indices,  # Save indices for reference
        'age_offset': age_offset,  # Save which age offset this is
    }, filename)
    print(f"Saved model to: {filename}")
        
 
        
