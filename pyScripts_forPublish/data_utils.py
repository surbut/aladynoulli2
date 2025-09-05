"""
Data utilities for Aladynoulli
Provides functions to load data or generate synthetic data for demonstration purposes.
"""

import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.special import softmax, expit
import os

def generate_synthetic_data(N=1000, D=50, T=50, K=5, P=20, seed=42):
    """
    Generate synthetic data matching the Aladynoulli model structure.
    This function creates realistic data for demonstration purposes.
    
    Parameters:
    N: Number of individuals
    D: Number of diseases  
    T: Number of time points
    K: Number of signatures
    P: Number of genetic features
    seed: Random seed for reproducibility
    
    Returns:
    Dictionary containing all data components needed for the model
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Fixed kernel parameters as in the fitted model
    lambda_length = T/4
    phi_length = T/3
    amplitude = 1.0
    
    # Setup time grid
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = amplitude**2 * np.exp(-0.5 * (time_diff**2) / lambda_length**2)
    K_phi = amplitude**2 * np.exp(-0.5 * (time_diff**2) / phi_length**2)
    
    # 1. Generate baseline disease prevalence trajectories
    logit_prev_t = np.zeros((D, T))
    for d in range(D):
        # More diverse base rates
        base_rate = np.random.choice([
            np.random.uniform(-14, -12),  # Uncommon
            np.random.uniform(-12, -10),  # Moderate
            np.random.uniform(-10, -8),   # Common
            np.random.uniform(-8, -6)     # Very common
        ], p=[0.40, 0.40, 0.15, 0.05])
        
        # More diverse trajectory shapes
        peak_age = np.random.uniform(20, 40)
        slope = np.random.uniform(0.10, 0.4)
        decay = np.random.uniform(0.002, 0.01)
        
        # Add possibility of early vs late onset patterns
        onset_shift = np.random.uniform(-10, 10)
        time_points_shifted = time_points - onset_shift
        
        # Generate trajectory with more complex patterns
        logit_prev_t[d, :] = base_rate + \
                            slope * time_points_shifted - \
                            decay * (time_points_shifted - peak_age)**2
    
    # 2. Generate cluster assignments
    clusters = np.zeros(D)
    diseases_per_cluster = D // K
    for k in range(K):
        clusters[k*diseases_per_cluster:(k+1)*diseases_per_cluster] = k
    
    # 3. Generate lambda (individual trajectories)
    G = np.random.randn(N, P)  # Genetic covariates
    Gamma_k = np.random.randn(P, K)  # Genetic effects
    lambda_ik = np.zeros((N, K, T))
    
    for i in range(N):
        mean_lambda = G[i] @ Gamma_k  # Individual-specific means
        for k in range(K):
            lambda_ik[i,k,:] = multivariate_normal.rvs(
                mean=mean_lambda[k] * np.ones(T), 
                cov=K_lambda
            )
    
    # 4. Generate phi with cluster structure
    phi_kd = np.zeros((K, D, T))
    psi = np.zeros((K, D))
    
    for k in range(K):
        for d in range(D):
            # Set cluster-specific offsets
            if clusters[d] == k:
                psi[k,d] = 1.0  # In-cluster
            else:
                psi[k,d] = -3.0  # Out-cluster
                
            # Generate phi around mu_d + psi
            mean_phi = logit_prev_t[d,:] + psi[k,d]
            phi_kd[k,d,:] = multivariate_normal.rvs(mean=mean_phi, cov=K_phi)
    
    # 5. Compute probabilities
    theta = softmax(lambda_ik, axis=1)
    eta = expit(phi_kd)
    pi = np.einsum('nkt,kdt->ndt', theta, eta)
    
    # 6. Generate events
    Y = np.zeros((N, D, T))
    event_times = np.full((N, D), T)
    
    for n in range(N):
        for d in range(D):
            for t in range(T):
                if Y[n,d,:t].sum() == 0:  # Still at risk
                    if np.random.rand() < pi[n,d,t]:
                        Y[n,d,t] = 1
                        event_times[n,d] = t
                        break
    
    # 7. Create signature references (for compatibility)
    signature_refs = np.zeros((K, D, T))
    for k in range(K):
        for d in range(D):
            if clusters[d] == k:
                signature_refs[k, d, :] = logit_prev_t[d, :] + psi[k, d]
            else:
                signature_refs[k, d, :] = logit_prev_t[d, :]
    
    # 8. Create disease names
    disease_names = [f"Disease_{i}" for i in range(D)]
    
    # 9. Create essentials dictionary
    essentials = {
        'prevalence_t': expit(logit_prev_t),
        'disease_names': disease_names,
        'clusters': clusters
    }
    
    return {
        'Y': torch.tensor(Y, dtype=torch.float32),
        'E': torch.tensor(event_times, dtype=torch.float32),
        'G': torch.tensor(G, dtype=torch.float32),
        'essentials': essentials,
        'signature_refs': torch.tensor(signature_refs, dtype=torch.float32),
        'initial_psi': torch.tensor(psi, dtype=torch.float32),
        'initial_clusters': torch.tensor(clusters, dtype=torch.long),
        'logit_prev_t': torch.tensor(logit_prev_t, dtype=torch.float32),
        'theta': torch.tensor(theta, dtype=torch.float32),
        'phi': torch.tensor(phi_kd, dtype=torch.float32),
        'lambda': torch.tensor(lambda_ik, dtype=torch.float32),
        'pi': torch.tensor(pi, dtype=torch.float32)
    }

def load_model_essentials(base_path=None, use_synthetic=True, **kwargs):
    """
    Load model essentials from files or generate synthetic data.
    
    Parameters:
    base_path: Path to data files (if None and use_synthetic=False, will try default paths)
    use_synthetic: If True, generate synthetic data instead of loading from files
    **kwargs: Additional arguments passed to generate_synthetic_data
    
    Returns:
    Y, E, G, essentials: Data components needed for the model
    """
    if use_synthetic:
        print("Generating synthetic data for demonstration...")
        data = generate_synthetic_data(**kwargs)
        return data['Y'], data['E'], data['G'], data['essentials']
    
    # Try to load from files
    if base_path is None:
        # Try common paths
        possible_paths = [
            '/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/',
            './data_for_running/',
            '../data_for_running/'
        ]
        
        for path in possible_paths:
            if os.path.exists(path + 'Y_tensor.pt'):
                base_path = path
                break
        
        if base_path is None:
            print("Could not find data files. Generating synthetic data instead...")
            data = generate_synthetic_data(**kwargs)
            return data['Y'], data['E'], data['G'], data['essentials']
    
    print(f"Loading data from {base_path}...")
    
    try:
        # Load large matrices
        Y = torch.load(base_path + 'Y_tensor.pt')
        E = torch.load(base_path + 'E_matrix.pt')
        G = torch.load(base_path + 'G_matrix.pt')
        
        # Load other components
        essentials = torch.load(base_path + 'model_essentials.pt')
        
        print("Loaded all components successfully!")
        return Y, E, G, essentials
        
    except FileNotFoundError as e:
        print(f"Could not load data from {base_path}: {e}")
        print("Generating synthetic data instead...")
        data = generate_synthetic_data(**kwargs)
        return data['Y'], data['E'], data['G'], data['essentials']

def load_reference_trajectories(base_path=None, use_synthetic=True, **kwargs):
    """
    Load reference trajectories from files or generate synthetic data.
    
    Parameters:
    base_path: Path to data files
    use_synthetic: If True, generate synthetic data instead of loading from files
    **kwargs: Additional arguments passed to generate_synthetic_data
    
    Returns:
    Dictionary containing reference trajectories
    """
    if use_synthetic:
        print("Generating synthetic reference trajectories...")
        data = generate_synthetic_data(**kwargs)
        return {
            'signature_refs': data['signature_refs'],
            'initial_psi': data['initial_psi'],
            'initial_clusters': data['initial_clusters']
        }
    
    # Try to load from files
    if base_path is None:
        possible_paths = [
            '/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/',
            './data_for_running/',
            '../data_for_running/'
        ]
        
        for path in possible_paths:
            if os.path.exists(path + 'reference_trajectories.pt'):
                base_path = path
                break
        
        if base_path is None:
            print("Could not find reference trajectory files. Generating synthetic data instead...")
            data = generate_synthetic_data(**kwargs)
            return {
                'signature_refs': data['signature_refs'],
                'initial_psi': data['initial_psi'],
                'initial_clusters': data['initial_clusters']
            }
    
    try:
        refs = torch.load(base_path + 'reference_trajectories.pt')
        initial_psi = torch.load(base_path + 'initial_psi_400k.pt')
        initial_clusters = torch.load(base_path + 'initial_clusters_400k.pt')
        
        return {
            'signature_refs': refs['signature_refs'],
            'initial_psi': initial_psi,
            'initial_clusters': initial_clusters
        }
        
    except FileNotFoundError as e:
        print(f"Could not load reference trajectories from {base_path}: {e}")
        print("Generating synthetic data instead...")
        data = generate_synthetic_data(**kwargs)
        return {
            'signature_refs': data['signature_refs'],
            'initial_psi': data['initial_psi'],
            'initial_clusters': data['initial_clusters']
        }

def subset_data(Y, E, G, start_index=0, end_index=10000):
    """
    Subset the data to a smaller sample for faster processing.
    
    Parameters:
    Y: Disease outcome tensor
    E: Censoring matrix
    G: Genetic data matrix
    start_index: Starting index for subset
    end_index: Ending index for subset
    
    Returns:
    Y_subset, E_subset, G_subset, indices: Subsetted data and indices
    """
    indices = list(range(start_index, min(end_index, Y.shape[0])))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices
