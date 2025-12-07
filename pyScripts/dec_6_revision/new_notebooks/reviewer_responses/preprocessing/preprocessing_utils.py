"""
Preprocessing utilities for Aladynoulli model initialization.

This module provides standalone functions to create initialization files without
needing to initialize the full model. This is much faster and avoids double initialization.

Functions:
    - compute_smoothed_prevalence: Compute disease prevalence over time
    - create_initial_clusters_and_psi: Create clusters and psi via spectral clustering
    - create_reference_trajectories: Create signature reference trajectories
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.special import logit
from sklearn.cluster import SpectralClustering
from statsmodels.nonparametric.smoothers_lowess import lowess


def compute_smoothed_prevalence(Y, window_size=5, smooth_on_logit=True):
    """
    Compute smoothed disease prevalence over time.
    
    STANDALONE FUNCTION - No model initialization required.
    
    Parameters:
    -----------
    Y : torch.Tensor or np.ndarray
        Disease outcome tensor (N × D × T)
    window_size : int
        Gaussian smoothing window size (sigma parameter)
    smooth_on_logit : bool
        If True: smooth on logit scale (matches original cluster_g.py)
        If False: smooth on probability scale (more intuitive)
        
    Returns:
    --------
    prevalence_t : np.ndarray
        Prevalence matrix (D × T) on probability scale
    """
    # Convert to numpy if needed
    if torch.is_tensor(Y):
        Y = Y.numpy()
    
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    
    for d in range(D):
        # Compute raw prevalence at each time point
        raw_prev = Y[:, d, :].mean(axis=0)  # (T,)
        
        if smooth_on_logit:
            # Smooth on logit scale (matches original cluster_g.py)
            # Better for rare events, preserves relative differences
            epsilon = 1e-8
            logit_prev = np.log((raw_prev + epsilon) / (1 - raw_prev + epsilon))
            smoothed_logit = gaussian_filter1d(logit_prev, sigma=window_size)
            prevalence_t[d, :] = 1 / (1 + np.exp(-smoothed_logit))
        else:
            # Smooth on probability scale (more intuitive)
            # Clamp to avoid negative values
            raw_prev = np.clip(raw_prev, 1e-8, 1 - 1e-8)
            smoothed_prev = gaussian_filter1d(raw_prev, sigma=window_size)
            prevalence_t[d, :] = np.clip(smoothed_prev, 0, 1)
    
    return prevalence_t


def create_initial_clusters_and_psi(Y, K, psi_config=None, healthy_reference=None, random_state=42):
    """
    Create initial clusters and psi WITHOUT initializing the full model.
    
    STANDALONE FUNCTION - No model initialization required.
    
    This function replicates the exact logic from clust_huge_amp.py initialize_params(),
    but without needing to create the full model object.
    
    Parameters:
    -----------
    Y : torch.Tensor or np.ndarray
        Disease outcome tensor (N × D × T)
    K : int
        Number of signatures/clusters
    psi_config : dict, optional
        Configuration for psi initialization:
        - 'in_cluster': value for diseases in cluster (default: 1.0)
        - 'out_cluster': value for diseases outside cluster (default: -2.0)
        - 'noise_in': noise for in-cluster (default: 0.1)
        - 'noise_out': noise for out-cluster (default: 0.01)
    healthy_reference : bool, optional
        If True, adds an extra healthy cluster (K+1 total). Default: None
    random_state : int
        Random seed for spectral clustering (default: 42)
        
    Returns:
    --------
    clusters : np.ndarray
        Cluster assignments for each disease (D,)
    psi : torch.Tensor
        Initial psi matrix (K × D) or (K+1 × D) if healthy_reference=True
    """
    # Convert to torch if needed
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float32)
    
    # Compute Y_avg: average over time dimension, then logit transform
    # This matches the exact logic from clust_huge_amp.py
    Y_avg = torch.mean(Y, dim=2)  # (N × D)
    epsilon = 1e-6
    Y_avg = torch.clamp(Y_avg, epsilon, 1.0 - epsilon)
    Y_avg = torch.log(Y_avg / (1 - Y_avg))  # Logit transform: (N × D)
    
    # Compute correlation matrix and similarity
    Y_corr = torch.corrcoef(Y_avg.T)  # (D × D)
    Y_corr = torch.nan_to_num(Y_corr, nan=0.0)
    similarity = (Y_corr + 1) / 2  # Convert correlation [-1,1] to similarity [0,1]
    
    # Spectral clustering
    spectral = SpectralClustering(
        n_clusters=K,
        assign_labels='kmeans',
        affinity='precomputed',
        n_init=10,
        random_state=random_state
    ).fit(similarity.numpy())
    
    clusters = spectral.labels_  # (D,)
    
    # Determine K_total (with or without healthy reference)
    K_total = K + (1 if healthy_reference else 0)
    
    # Initialize psi
    if psi_config is None:
        # Default values (matches clust_huge_amp.py)
        in_cluster = 1.0
        out_cluster = -2.0
        noise_in = 0.1
        noise_out = 0.01
    else:
        in_cluster = psi_config.get('in_cluster', 1.0)
        out_cluster = psi_config.get('out_cluster', -2.0)
        noise_in = psi_config.get('noise_in', 0.1)
        noise_out = psi_config.get('noise_out', 0.01)
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    psi = torch.zeros((K_total, clusters.shape[0]))
    
    # Initialize psi for disease clusters
    for k in range(K):
        cluster_mask = (clusters == k)
        psi[k, cluster_mask] = in_cluster + noise_in * torch.randn(cluster_mask.sum())
        psi[k, ~cluster_mask] = out_cluster + noise_out * torch.randn((~cluster_mask).sum())
    
    # Add healthy reference if requested
    if healthy_reference:
        psi[K, :] = -5.0 + 0.01 * torch.randn(clusters.shape[0])
    
    # Print cluster sizes (matches original output)
    print("\nCluster Sizes:")
    unique, counts = np.unique(clusters, return_counts=True)
    for k, count in zip(unique, counts):
        print(f"Cluster {k}: {count} diseases")
    
    return clusters, psi


def create_reference_trajectories(Y, initial_clusters, K, healthy_prop=0.01, frac=0.3):
    """
    Create reference trajectories using LOWESS smoothing on logit scale.
    
    STANDALONE FUNCTION - No model initialization required.
    
    Parameters:
    -----------
    Y : torch.Tensor
        Disease outcome tensor (N × D × T)
    initial_clusters : torch.Tensor or np.ndarray
        Cluster assignments for each disease [D] (from create_initial_clusters_and_psi)
    K : int
        Number of signatures
    healthy_prop : float
        Proportion of healthy state (default: 0.01)
    frac : float
        LOWESS smoothing fraction (default: 0.3)
        
    Returns:
    --------
    signature_refs : torch.Tensor
        Reference trajectories (K × T) on logit scale
    healthy_ref : torch.Tensor
        Healthy reference trajectory (T) on logit scale
    """
    # Convert to torch if needed
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float32)
    if not torch.is_tensor(initial_clusters):
        initial_clusters = torch.tensor(initial_clusters, dtype=torch.long)
    
    T = Y.shape[2]
    
    # Get raw counts and proportions
    Y_counts = Y.sum(dim=0)  # D × T
    signature_props = torch.zeros(K, T)
    total_counts = Y_counts.sum(dim=0) + 1e-8
    
    # For each signature, compute proportion of diseases in that signature over time
    for k in range(K):
        cluster_mask = (initial_clusters == k)
        signature_props[k] = Y_counts[cluster_mask].sum(dim=0) / total_counts
    
    # Normalize and clamp
    signature_props = torch.clamp(signature_props, min=1e-8, max=1-1e-8)
    signature_props = signature_props / signature_props.sum(dim=0, keepdim=True)
    signature_props *= (1 - healthy_prop)
    
    # Convert to logit and smooth
    logit_props = torch.tensor(logit(signature_props.numpy()))
    signature_refs = torch.zeros_like(logit_props)
    
    times = np.arange(T)
    for k in range(K):
        smoothed = lowess(
            logit_props[k].numpy(), 
            times,
            frac=frac,
            it=3,
            delta=0.0,
            return_sorted=False
        )
        signature_refs[k] = torch.tensor(smoothed)
    
    healthy_ref = torch.ones(T) * logit(torch.tensor(healthy_prop))
    
    return signature_refs, healthy_ref

