import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering  # Add this import
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns

from scipy.stats import norm

def generate_synthetic_data(N=100, D=5, T=50, K=3, P=5, return_true_params=False):
    """
    Generate synthetic survival data for testing the model.
    """
    np.random.seed(123)

    # Genetic covariates G (N x P)
    G = np.random.randn(N, P)

    # Prevalence of diseases (D)
    prevalence = np.random.uniform(0.01, 0.05, D)

    # Length scales and amplitudes for GP kernels
    length_scales = np.random.uniform(T / 4, T / 3, K)
    amplitudes = np.random.uniform(0.8, 1.2, K)

    # Generate time differences for covariance matrices
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]

    # Simulate mu_d (average disease prevalence trajectories)
    mu_d = np.zeros((D, T))
    for d in range(D):
        base_trend = np.log(prevalence[d]) * np.ones(T)
        mu_d[d, :] = base_trend

    # Simulate lambda (individual-topic trajectories)
    Gamma_k = np.random.randn(P, K)
    lambda_ik = np.zeros((N, K, T))
    for k in range(K):
        cov_matrix = amplitudes[k] ** 2 * np.exp(-0.5 * (time_diff ** 2) / length_scales[k] ** 2)
        for i in range(N):
            mean_lambda = G[i] @ Gamma_k[:, k]
            lambda_ik[i, k, :] = multivariate_normal.rvs(mean=mean_lambda * np.ones(T), cov=cov_matrix)

    # Compute theta
    exp_lambda = np.exp(lambda_ik)
    theta = exp_lambda / np.sum(exp_lambda, axis=1, keepdims=True)  # N x K x T

    # Simulate phi (topic-disease trajectories)
    phi_kd = np.zeros((K, D, T))
    for k in range(K):
        cov_matrix = amplitudes[k] ** 2 * np.exp(-0.5 * (time_diff ** 2) / length_scales[k] ** 2)
        for d in range(D):
            mean_phi = np.log(prevalence[d]) * np.ones(T)
            phi_kd[k, d, :] = multivariate_normal.rvs(mean=mean_phi, cov=cov_matrix)

    # Compute eta
    eta = expit(phi_kd)  # K x D x T

    # Compute pi
    pi = np.einsum('nkt,kdt->ndt', theta, eta)

    # Generate survival data Y
    Y = np.zeros((N, D, T), dtype=int)
    event_times = np.full((N, D), T)
    for n in range(N):
        for d in range(D):
            for t in range(T):
                if Y[n, d, :t].sum() == 0:
                    if np.random.rand() < pi[n, d, t]:
                        Y[n, d, t] = 1
                        event_times[n, d] = t
                        break

    if return_true_params:
        return {
            'Y': Y,
            'G': G,
            'prevalence': prevalence,
            'length_scales': length_scales,
            'amplitudes': amplitudes,
            'event_times': event_times,
            'theta': theta,
            'phi': phi_kd,
            'lambda': lambda_ik,
            'gamma': Gamma_k,
            'pi': pi
        }
    else:
        return Y, G, prevalence, event_times




def plot_model_fit(model, sim_data, n_samples=5, n_diseases=3):
    """
    Plot model fit against true synthetic data for selected individuals and diseases
    
    Parameters:
    model: trained model
    sim_data: dictionary with true synthetic data
    n_samples: number of individuals to plot
    n_diseases: number of diseases to plot
    """
    # Get model predictions
    with torch.no_grad():
        pi_pred = model.forward().cpu().numpy()
    
    # Get true pi from synthetic data
    pi_true = sim_data['pi']
    
    N, D, T = pi_pred.shape
    time_points = np.arange(T)
    
    # Select individuals with varying predictions
    pi_var = np.var(pi_pred, axis=(1,2))  # Variance across diseases and time
    sample_idx = np.quantile(np.arange(N), np.linspace(0, 1, n_samples)).astype(int)
    
    # Select most variable diseases
    disease_var = np.var(pi_pred, axis=(0,2))  # Variance across individuals and time
    disease_idx = np.argsort(-disease_var)[:n_diseases]
    
    # Create plots
    fig, axes = plt.subplots(n_samples, n_diseases, figsize=(4*n_diseases, 4*n_samples))
    
    for i, ind in enumerate(sample_idx):
        for j, dis in enumerate(disease_idx):
            ax = axes[i,j] if n_samples > 1 and n_diseases > 1 else axes
            
            # Plot predicted and true pi
            ax.plot(time_points, pi_pred[ind, dis, :], 
                   'b-', label='Predicted', linewidth=2)
            ax.plot(time_points, pi_true[ind, dis, :], 
                   'r--', label='True', linewidth=2)
            
            ax.set_title(f'Individual {ind}, Disease {dis}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Probability')
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.show()



def plot_random_comparisons(true_pi, pred_pi, n_samples=10, n_cols=2):
    """
    Plot true vs predicted pi for random individuals and diseases
    
    Parameters:
    true_pi: numpy array (N×D×T)
    pred_pi: torch tensor (N×D×T)
    n_samples: number of random comparisons to show
    n_cols: number of columns in subplot grid
    """
    N, D, T = true_pi.shape
    pred_pi = pred_pi.detach().numpy()
    
    # Generate random indices
    random_inds = np.random.randint(0, N, n_samples)
    random_diseases = np.random.randint(0, D, n_samples)
    
    # Calculate number of rows needed
    n_rows = int(np.ceil(n_samples / n_cols))
    
    plt.figure(figsize=(6*n_cols, 4*n_rows))
    
    for idx in range(n_samples):
        i = random_inds[idx]
        d = random_diseases[idx]
        
        plt.subplot(n_rows, n_cols, idx+1)
        
        # Plot true and predicted
        plt.plot(true_pi[i,d,:], 'b-', label='True π', linewidth=2)
        plt.plot(pred_pi[i,d,:], 'r--', label='Predicted π', linewidth=2)
        
        plt.title(f'Individual {i}, Disease {d}')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_best_matches(true_pi, pred_pi, n_samples=10, n_cols=2):
    """
    Plot cases where model predictions best match true values
    
    Parameters:
    true_pi: numpy array (N×D×T)
    pred_pi: torch tensor (N×D×T)
    """
    N, D, T = true_pi.shape
    pred_pi = pred_pi.detach().numpy()
    
    # Compute MSE for each individual-disease pair
    mse = np.mean((true_pi - pred_pi)**2, axis=2)  # N×D
    
    # Get indices of best matches
    best_indices = np.argsort(mse.flatten())[:n_samples]
    best_pairs = [(idx // D, idx % D) for idx in best_indices]
    
    # Plot
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=(6*n_cols, 4*n_rows))
    
    for idx, (i, d) in enumerate(best_pairs):
        plt.subplot(n_rows, n_cols, idx+1)
        
        # Plot true and predicted
        plt.plot(true_pi[i,d,:], 'b-', label='True π', linewidth=2)
        plt.plot(pred_pi[i,d,:], 'r--', label='Predicted π', linewidth=2)
        
        mse_val = mse[i,d]
        plt.title(f'Individual {i}, Disease {d}\nMSE = {mse_val:.6f}')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Use after model fitting:
# Example of preparing smoothed time-dependent prevalence


def compute_smoothed_prevalence(Y, window_size=5):
    """Compute smoothed time-dependent prevalence on logit scale"""
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    logit_prev_t = np.zeros((D, T))
    
    for d in range(D):
        # Compute raw prevalence at each time point
        raw_prev = Y[:, d, :].mean(axis=0)
        
        # Convert to logit scale
        epsilon = 1e-8
        logit_prev = np.log((raw_prev + epsilon) / (1 - raw_prev + epsilon))
        
        # Smooth on logit scale
        from scipy.ndimage import gaussian_filter1d
        smoothed_logit = gaussian_filter1d(logit_prev, sigma=window_size)
        
        # Store both versions
        logit_prev_t[d, :] = smoothed_logit
        prevalence_t[d, :] = 1 / (1 + np.exp(-smoothed_logit))
    
    return prevalence_t




from scipy.stats import norm

def generate_clustered_survival_data_from_real(N, D, T, K, P, 
                                             real_signature_refs,
                                             real_logit_prev_t,
                                             real_gamma,
                                             real_psi,  # Still needed to get cluster structure
                                             init_sd_scaler=1e-2,
                                             G=None,
                                             use_fixed_psi=False):  # New parameter
    """
    Generate synthetic data using real patterns with option for fixed psi values
    """
    # Convert PyTorch tensors to numpy if needed
    if torch.is_tensor(real_signature_refs):
        real_signature_refs = real_signature_refs.detach().numpy()
    if torch.is_tensor(real_logit_prev_t):
        real_logit_prev_t = real_logit_prev_t.detach().numpy()
    if torch.is_tensor(real_gamma):
        real_gamma = real_gamma.detach().numpy()
    if torch.is_tensor(real_psi):
        real_psi = real_psi.detach().numpy()

    # Generate or use provided genetic components
    if G is None:
        G = np.random.randn(N, P)
        G = G - G.mean(axis=0, keepdims=True)
        G = G / G.std(axis=0, keepdims=True)

    # Setup kernel parameters
    lambda_length = T/4
    phi_length = T/3
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    
    # Scale kernels with init_sd_scaler
    K_lambda_init = (init_sd_scaler**2) * np.exp(-0.5 * (time_diff**2) / (lambda_length**2))
    K_phi_init = (init_sd_scaler**2) * np.exp(-0.5 * (time_diff**2) / (phi_length**2))

    # Generate lambda using real signatures and gamma
    lambda_ik = np.zeros((N, K, T))
    for i in range(N):
        lambda_means = G[i] @ real_gamma
        for k in range(K):
            eps = np.random.multivariate_normal(np.zeros(T), K_lambda_init)
            lambda_ik[i,k,:] = real_signature_refs[k,:T] + lambda_means[k] + eps

    # Get cluster assignments from real psi
    clusters = np.argmax(real_psi, axis=0)
    
    # Generate new psi with fixed values if requested
    if use_fixed_psi:
        psi = np.full((K, D), -2.0)  # Initialize all to out-cluster value
        for k in range(K):
            psi[k, clusters == k] = 1.0  # Set in-cluster value
    else:
        psi = real_psi

    # Generate phi using real mu_d and psi
    phi_kd = np.zeros((K, D, T))
    for k in range(K):
        for d in range(D):
            mean_phi = real_logit_prev_t[d,:T] + psi[k,d]
            eps = np.random.multivariate_normal(np.zeros(T), K_phi_init)
            phi_kd[k,d,:] = mean_phi + eps

    # Compute probabilities and generate events
    theta = softmax(lambda_ik, axis=1)
    eta = expit(phi_kd)
    pi = np.einsum('nkt,kdt->ndt', theta, eta)
    
    Y = np.zeros((N, D, T))
    event_times = np.full((N, D), T)
    
    for n in range(N):
        for d in range(D):
            for t in range(T):
                if Y[n,d,:t].sum() == 0:
                    if np.random.rand() < pi[n,d,t]:
                        Y[n,d,t] = 1
                        event_times[n,d] = t
                        break

    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'logit_prev_t': real_logit_prev_t[:D, :T],
        'theta': theta,
        'phi': phi_kd,
        'lambda': lambda_ik,
        'psi': psi,
        'pi': pi,
        'signature_refs': real_signature_refs[:K, :T],
        'gamma': real_gamma,
        'clusters': clusters
    }




def reorder_clusters(true_clusters, initial_clusters, K):
    """
    Reorder cluster labels to maximize diagonal alignment in confusion matrix
    
    Args:
        true_clusters: array of true cluster labels
        initial_clusters: array of initial cluster labels
        K: number of clusters
    
    Returns:
        remapped_clusters: array of remapped initial cluster labels
    """
    from scipy.optimize import linear_sum_assignment
    
    # Create confusion matrix
    conf_mat = np.zeros((K, K))
    for i in range(len(true_clusters)):
        conf_mat[true_clusters[i], initial_clusters[i]] += 1
    
    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    
    # Create mapping dictionary
    mapping = {old: new for old, new in zip(col_ind, row_ind)}
    
    # Remap clusters
    remapped_clusters = np.array([mapping[c] for c in initial_clusters])
    
    return remapped_clusters


def generate_clustered_survival_data(N=1000, D=20, T=50, K=5, P=5):
    """
    Generate synthetic data with fixed cluster structure
    """
    # Fixed kernel parameters
    lambda_length = T/4
    phi_length = T/3
    amplitude = 1.0
    
    # Setup time grid
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = amplitude**2 * np.exp(-0.5 * (time_diff**2) / lambda_length**2)
    K_phi = amplitude**2 * np.exp(-0.5 * (time_diff**2) / phi_length**2)
    
    # 1. Generate baseline trajectories
    logit_prev_t = np.zeros((D, T))
    for d in range(D):
        # Base rates
        base_rate = np.random.choice([
            np.random.uniform(-14, -12),  # Uncommon
            np.random.uniform(-12, -10),  # Moderate
            np.random.uniform(-10, -8),   # Common
            np.random.uniform(-8, -6)     # Very common
        ], p=[0.40, 0.40, 0.15, 0.05])
        
        # Trajectory shapes
        peak_age = np.random.uniform(20, 40)
        slope = np.random.uniform(0.10, 0.4)
        decay = np.random.uniform(0.002, 0.01)
        onset_shift = np.random.uniform(-10, 10)
        time_points_shifted = time_points - onset_shift
        
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
                psi[k,d] = -4.0  # Out-cluster
                
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
    
    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'clusters': clusters,
        'logit_prev_t': logit_prev_t,
        'theta': theta,
        'phi': phi_kd,
        'lambda': lambda_ik,
        'psi': psi,
        'pi': pi
    }




def generate_clustered_survival_data_with_refs(N, D, T, K, P, signature_refs=None):
    """
    Generate synthetic data with fixed cluster structure and optional signature references
    """
    # Fixed kernel parameters
    lambda_length = T/4
    phi_length = T/3
    amplitude = 1.0
    
    # Setup time grid
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = amplitude**2 * np.exp(-0.5 * (time_diff**2) / lambda_length**2)
    K_phi = amplitude**2 * np.exp(-0.5 * (time_diff**2) / phi_length**2)
    
    # 1. Generate baseline trajectories (simpler version)
    logit_prev_t = np.zeros((D, T))
    for d in range(D):
        base_rate = np.random.choice([
            np.random.uniform(-14, -12),  # Uncommon
            np.random.uniform(-12, -10),  # Moderate
            np.random.uniform(-10, -8),   # Common
            np.random.uniform(-8, -6)     # Very common
        ], p=[0.40, 0.40, 0.15, 0.05])
    
    peak_age = np.random.uniform(20, 40)
    slope = np.random.uniform(0.10, 0.4)
    decay = np.random.uniform(0.002, 0.01)
    onset_shift = np.random.uniform(-10, 10)
    time_points_shifted = time_points - onset_shift
    
    logit_prev_t[d, :] = base_rate + \
                        slope * time_points_shifted - \
                        decay * (time_points_shifted - peak_age)**2
    # 2. Generate cluster assignments (same as original)
    clusters = np.zeros(D)
    diseases_per_cluster = D // K
    for k in range(K):
        clusters[k*diseases_per_cluster:(k+1)*diseases_per_cluster] = k
    
    # 3. Generate lambda with signature references
    G = np.random.randn(N, P)  # Genetic covariates
    G = G - G.mean(axis=0, keepdims=True)  # Center
    G = G / G.std(axis=0, keepdims=True)   # Scale
    
    Gamma_k = np.random.randn(P, K)  # Genetic effects
    lambda_ik = np.zeros((N, K, T))
    
    if signature_refs is None:
        signature_refs = np.zeros((K, T))  # Flat references if none provided
        
    for i in range(N):
        mean_lambda = G[i] @ Gamma_k  # Individual-specific means
        for k in range(K):
            lambda_ik[i,k,:] = signature_refs[k,:] + mean_lambda[k] + \
                              multivariate_normal.rvs(mean=np.zeros(T), cov=K_lambda)
    
    # 4. Generate phi with strong cluster structure
    phi_kd = np.zeros((K, D, T))
    psi = np.zeros((K, D))
    
    for k in range(K):
        for d in range(D):
            # Set cluster-specific offsets with strong separation
            if clusters[d] == k:
                psi[k,d] = 1.0  # In-cluster
            else:
                psi[k,d] = -4.0  # Out-cluster
                
            # Generate phi around mu_d + psi
            mean_phi = logit_prev_t[d,:] + psi[k,d]
            phi_kd[k,d,:] = mean_phi + \
                           multivariate_normal.rvs(mean=np.zeros(T), cov=K_phi)
    
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
    
    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'clusters': clusters,
        'logit_prev_t': logit_prev_t,
        'theta': theta,
        'phi': phi_kd,
        'lambda': lambda_ik,
        'psi': psi,
        'pi': pi,
        'signature_refs': signature_refs
    }

sim_data = generate_clustered_survival_data_with_refs(
    N, D, T, K, P,
    signature_refs=None  # This will use the default flat references
)