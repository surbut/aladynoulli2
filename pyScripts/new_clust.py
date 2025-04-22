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
                                             use_fixed_psi=False,
                                             signature_scale=0.1):  # Control signature strength
    """
    Generate synthetic data using real patterns with scaled signature effects
    
    Parameters:
    -----------
    signature_scale : float
        Scale factor for signature references (0 = no effect, 1 = full effect)
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

    # Generate lambda using scaled signature references and gamma
    lambda_ik = np.zeros((N, K, T))
    for i in range(N):
        lambda_means = G[i] @ real_gamma  # This should be shape (K,)
        for k in range(K):
            eps = np.random.multivariate_normal(np.zeros(T), K_lambda_init)
            # Scale the signature reference contribution
            sig_ref = real_signature_refs[k,:T] if real_signature_refs.shape[1] > T else real_signature_refs[k]
            mean_shift = np.full(T, lambda_means[k])  # Broadcast mean to shape (T,)
            lambda_ik[i,k,:] = signature_scale * sig_ref + mean_shift + eps

    # Get cluster assignments from real psi
    clusters = np.argmax(real_psi, axis=0)
    
    # Generate new psi with fixed values if requested
    if use_fixed_psi:
        psi = np.full((K, D), -4.0)  # Initialize all to out-cluster value
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

""" 
sim_data = generate_clustered_survival_data_with_refs(
    N, D, T, K, P,
    signature_refs=None  # This will use the default flat references
)

"""


def generate_state_driven_data(N, D, T, K, P, init_sd_scaler=1e-2):
    """
    Generate synthetic data where disease clustering emerges from shared state activation.
    People with similar state patterns get similar diseases.
    
    Parameters:
    -----------
    N : int
        Number of individuals
    D : int
        Number of diseases
    T : int
        Number of time points
    K : int
        Number of states
    P : int
        Number of genetic components
    init_sd_scaler : float
        Scale for initial noise
    """
    # Generate genetic components
    G = np.random.randn(N, P)
    G = G - G.mean(axis=0, keepdims=True)
    G = G / G.std(axis=0, keepdims=True)
    
    # Generate gamma (genetic effects on states)
    gamma = np.random.randn(P, K)
    
    # Setup kernel parameters
    lambda_length = T/4
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = (init_sd_scaler**2) * np.exp(-0.5 * (time_diff**2) / (lambda_length**2))
    
    # Generate lambda (individual state trajectories)
    lambda_ik = np.zeros((N, K, T))
    for i in range(N):
        # Each person gets a base state pattern from their genetics
        state_means = G[i] @ gamma
        for k in range(K):
            # Add temporal variation with slightly larger scale
            eps = np.random.multivariate_normal(np.zeros(T), K_lambda * 2.0)  # Just increase variation a bit
            lambda_ik[i,k,:] = state_means[k] + eps
    
    # Convert to state proportions
    theta = softmax(lambda_ik, axis=1)  # N x K x T
    
    # Now create disease-state associations
    # Each disease will be strongly associated with 1-2 states
    disease_state_weights = np.zeros((K, D))
    for d in range(D):
        # Pick 1 or 2 states for this disease
        n_states = np.random.choice([1, 2], p=[0.7, 0.3])
        primary_states = np.random.choice(K, size=n_states, replace=False)
        disease_state_weights[primary_states, d] = np.random.uniform(1, 2, size=n_states)
        # Small weights for other states
        other_states = np.setdiff1d(np.arange(K), primary_states)
        disease_state_weights[other_states, d] = np.random.uniform(-3, -2, size=len(other_states))
    
    # Generate base disease trajectories (simpler than real_logit_prev_t)
    base_trajectories = np.zeros((D, T))
    for d in range(D):
        # Simple increasing risk with age
        base_rate = np.random.uniform(-6, -4)  # Base rate
        slope = np.random.uniform(0.02, 0.05)  # Age effect
        base_trajectories[d] = base_rate + slope * time_points
    
    # Combine everything to get probabilities
    # pi[n,d,t] = σ(base[d,t] + Σ_k theta[n,k,t] * weights[k,d])
    pi = np.zeros((N, D, T))
    for n in range(N):
        for t in range(T):
            state_effects = theta[n, :, t, None] * disease_state_weights  # K x D
            pi[n, :, t] = expit(base_trajectories[:, t] + state_effects.sum(axis=0))
    
    # Generate events
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
    
    # Get emergent clusters from disease-state associations
    clusters = np.argmax(disease_state_weights, axis=0)
    
    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'theta': theta,
        'lambda': lambda_ik,
        'disease_state_weights': disease_state_weights,
        'pi': pi,
        'clusters': clusters,
        'gamma': gamma
    }


def generate_state_driven_data_strong_clusters(N, D, T, K, P, init_sd_scaler=1e-2):
    """
    Generate synthetic data where disease clustering emerges from shared state activation.
    Modified to create stronger, more distinct clusters.
    """
    # Generate genetic components
    G = np.random.randn(N, P)
    G = G - G.mean(axis=0, keepdims=True)
    G = G / G.std(axis=0, keepdims=True)
    
    # Generate gamma (genetic effects on states)
    gamma = np.random.randn(P, K)
    
    # Setup kernel parameters
    lambda_length = T/4
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = (init_sd_scaler**2) * np.exp(-0.5 * (time_diff**2) / (lambda_length**2))
    
    # Generate lambda (individual state trajectories)
    lambda_ik = np.zeros((N, K, T))
    for i in range(N):
        # Each person gets a base state pattern from their genetics
        state_means = G[i] @ gamma
        for k in range(K):
            # Add smooth temporal variation
            eps = np.random.multivariate_normal(np.zeros(T), K_lambda)
            lambda_ik[i,k,:] = state_means[k] + eps
    
    # Convert to state proportions
    theta = softmax(lambda_ik, axis=1)  # N x K x T
    
    # Create stronger disease-state associations
    disease_state_weights = np.full((K, D), -6.0)  # Much lower base weight for non-primary states
    diseases_per_state = D // K
    remaining = D % K
    
    # Assign diseases to states more evenly
    current_disease = 0
    for k in range(K):
        # Calculate number of diseases for this state
        n_diseases = diseases_per_state + (1 if k < remaining else 0)
        
        # Assign strong positive weights for this state's diseases
        disease_indices = range(current_disease, current_disease + n_diseases)
        disease_state_weights[k, disease_indices] = np.random.uniform(2.0, 3.0)  # Stronger positive weights
        
        current_disease += n_diseases
    
    # Generate base disease trajectories (simpler than real_logit_prev_t)
    base_trajectories = np.zeros((D, T))
    for d in range(D):
        # Simple increasing risk with age
        base_rate = np.random.uniform(-6, -4)  # Base rate
        slope = np.random.uniform(0.02, 0.05)  # Age effect
        base_trajectories[d] = base_rate + slope * time_points
    
    # Combine everything to get probabilities
    pi = np.zeros((N, D, T))
    for n in range(N):
        for t in range(T):
            state_effects = theta[n, :, t, None] * disease_state_weights  # K x D
            pi[n, :, t] = expit(base_trajectories[:, t] + state_effects.sum(axis=0))
    
    # Generate events
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
    
    # Get emergent clusters from disease-state associations
    clusters = np.argmax(disease_state_weights, axis=0)
    
    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'theta': theta,
        'lambda': lambda_ik,
        'disease_state_weights': disease_state_weights,
        'pi': pi,
        'clusters': clusters,
        'gamma': gamma
    }

def extract_signature_refs_from_sim(sim_data, window_size=5):
    """
    Extract signature references from simulated data by looking at average lambda 
    trajectories for each state's primary diseases.
    
    Parameters:
    -----------
    sim_data : dict
        Output from generate_state_driven_data_strong_clusters
    window_size : int
        Window size for smoothing
    """
    N, K, T = sim_data['lambda'].shape
    disease_state_weights = sim_data['disease_state_weights']
    clusters = sim_data['clusters']
    
    # Initialize signature references
    signature_refs = np.zeros((K, T))
    
    # For each state
    for k in range(K):
        # Find diseases primarily associated with this state
        primary_diseases = (clusters == k)
        
        # Get average lambda trajectory for people who get these diseases
        has_disease = (sim_data['Y'][:, primary_diseases, :].sum(axis=(1,2)) > 0)
        if has_disease.sum() > 0:
            state_lambdas = sim_data['lambda'][has_disease, k, :]
            signature_refs[k] = state_lambdas.mean(axis=0)
        
        # Smooth the reference
        from scipy.ndimage import gaussian_filter1d
        signature_refs[k] = gaussian_filter1d(signature_refs[k], sigma=window_size)
    
    # Center and scale the references
    signature_refs = signature_refs - signature_refs.mean(axis=1, keepdims=True)
    signature_refs = signature_refs / np.maximum(signature_refs.std(axis=1, keepdims=True), 1e-6)
    
    return signature_refs

def generate_and_get_refs(N=1000, D=20, T=50, K=5, P=5):
    """
    Generate data and extract signature references in one step.
    """
    # Generate data with strong clusters
    sim_data = generate_state_driven_data_strong_clusters(N, D, T, K, P)
    
    # Extract signature references
    signature_refs = extract_signature_refs_from_sim(sim_data)
    
    return sim_data, signature_refs



def create_reference_trajectories(Y, initial_clusters, K, healthy_prop=0, frac=0.3):
    """Create reference trajectories using LOWESS smoothing on logit scale"""
    from scipy.special import logit
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import numpy as np
    
    # Convert everything to torch tensors
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float32)
    if not torch.is_tensor(initial_clusters):
        initial_clusters = torch.tensor(initial_clusters)
        
    T = Y.shape[2]
    
    # Get raw counts and proportions
    Y_counts = Y.sum(axis=0)  # Changed dim to axis
    signature_props = torch.zeros(K, T)
    total_counts = Y_counts.sum(axis=0) + 1e-8  # Changed dim to axis
    
    for k in range(K):
        cluster_mask = (initial_clusters == k)
        signature_props[k] = Y_counts[cluster_mask].sum(axis=0) / total_counts
    
    # Normalize and clamp
    signature_props = torch.clamp(signature_props, min=1e-8, max=1-1e-8)
    signature_props = signature_props / signature_props.sum(axis=0, keepdim=True)  # Changed dim to axis
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

def plot_sim_components(sim_data, n_samples=3):
    """
    Visualize components from simulated data
    
    Parameters:
    -----------
    sim_data : dict
        Output from generate_state_driven_data_strong_clusters
    n_samples : int
        Number of samples to show for each component
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    N, K, T = sim_data['lambda'].shape
    D = sim_data['disease_state_weights'].shape[1]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Plot lambdas for a few individuals and states
    ax = axes[0,0]
    sample_inds = np.random.choice(N, n_samples)
    sample_states = np.random.choice(K, n_samples)
    
    for i, n in enumerate(sample_inds):
        for j, k in enumerate(sample_states):
            ax.plot(sim_data['lambda'][n, k, :], 
                   label=f'Individual {n}, State {k}')
    ax.set_title('λ Trajectories (Sample Individual-State Pairs)')
    ax.set_xlabel('Time')
    ax.set_ylabel('λ Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Plot thetas (state proportions) for same individuals
    ax = axes[0,1]
    for i, n in enumerate(sample_inds):
        for j, k in enumerate(sample_states):
            ax.plot(sim_data['theta'][n, k, :], 
                   label=f'Individual {n}, State {k}')
    ax.set_title('θ (State Proportions) for Same Individuals')
    ax.set_xlabel('Time')
    ax.set_ylabel('θ Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Plot base trajectories for a few diseases
    ax = axes[1,0]
    sample_diseases = np.random.choice(D, n_samples)
    time_points = np.arange(T)
    
    for d in sample_diseases:
        base_rate = np.random.uniform(-6, -4)
        slope = np.random.uniform(0.02, 0.05)
        traj = base_rate + slope * time_points
        ax.plot(traj, label=f'Disease {d}')
    ax.set_title('Base Disease Trajectories (Sample Diseases)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Logit Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Plot psi heatmap
    ax = axes[1,1]
    im = ax.imshow(sim_data['disease_state_weights'], aspect='auto', cmap='RdBu_r')
    ax.set_title('ψ (Disease-State Weights)')
    ax.set_xlabel('Disease')
    ax.set_ylabel('State')
    plt.colorbar(im, ax=ax)
    
    # Add some stats as text
    stats_text = (
        f"λ range: [{sim_data['lambda'].min():.2f}, {sim_data['lambda'].max():.2f}]\n"
        f"θ range: [{sim_data['theta'].min():.2f}, {sim_data['theta'].max():.2f}]\n"
        f"ψ range: [{sim_data['disease_state_weights'].min():.2f}, "
        f"{sim_data['disease_state_weights'].max():.2f}]"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def compute_equivalent_phi(sim_data):
    """
    Compute the equivalent of phi (logit disease trajectories per state) 
    from simulation components.
    """
    N, K, T = sim_data['lambda'].shape
    D = sim_data['disease_state_weights'].shape[1]
    
    # Reconstruct base trajectories
    time_points = np.arange(T)
    base_trajectories = np.zeros((D, T))
    for d in range(D):
        base_rate = np.random.uniform(-6, -4)
        slope = np.random.uniform(0.02, 0.05)
        base_trajectories[d] = base_rate + slope * time_points
    
    # For each state-disease pair, compute effective phi
    phi_equivalent = np.zeros((K, D, T))
    for k in range(K):
        for d in range(D):
            # Base trajectory plus state effect
            phi_equivalent[k, d, :] = base_trajectories[d] + sim_data['disease_state_weights'][k, d]
    
    return phi_equivalent

def plot_sim_components_with_phi(sim_data, n_samples=3):
    """
    Visualize components including equivalent phi
    """
    import matplotlib.pyplot as plt
    
    N, K, T = sim_data['lambda'].shape
    D = sim_data['disease_state_weights'].shape[1]
    
    # Compute equivalent phi
    phi_equiv = compute_equivalent_phi(sim_data)
    
    # Create figure with 5 subplots (2x3 grid)
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(2, 3)
    
    # 1. Plot lambdas
    ax = fig.add_subplot(gs[0, 0])
    sample_inds = np.random.choice(N, n_samples)
    sample_states = np.random.choice(K, n_samples)
    
    for i, n in enumerate(sample_inds):
        for j, k in enumerate(sample_states):
            ax.plot(sim_data['lambda'][n, k, :], 
                   label=f'Individual {n}, State {k}')
    ax.set_title('λ Trajectories')
    ax.set_xlabel('Time')
    ax.set_ylabel('λ Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Plot thetas
    ax = fig.add_subplot(gs[0, 1])
    for i, n in enumerate(sample_inds):
        for j, k in enumerate(sample_states):
            ax.plot(sim_data['theta'][n, k, :], 
                   label=f'Individual {n}, State {k}')
    ax.set_title('θ (State Proportions)')
    ax.set_xlabel('Time')
    ax.set_ylabel('θ Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Plot equivalent phis
    ax = fig.add_subplot(gs[0, 2])
    sample_diseases = np.random.choice(D, n_samples)
    
    for d in sample_diseases:
        # Get primary state
        primary_state = np.argmax(sim_data['disease_state_weights'][:, d])
        for k in sample_states:
            linestyle = '-' if k == primary_state else '--'
            alpha = 1.0 if k == primary_state else 0.5
            ax.plot(phi_equiv[k, d, :], linestyle=linestyle, alpha=alpha,
                   label=f'Disease {d}, State {k}{"(Primary)" if k == primary_state else ""}')
    ax.set_title('Equivalent φ Trajectories')
    ax.set_xlabel('Time')
    ax.set_ylabel('Logit Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Plot psi heatmap
    ax = fig.add_subplot(gs[1, 0:2])
    im = ax.imshow(sim_data['disease_state_weights'], aspect='auto', cmap='RdBu_r')
    ax.set_title('ψ (Disease-State Weights)')
    ax.set_xlabel('Disease')
    ax.set_ylabel('State')
    plt.colorbar(im, ax=ax)
    
    # Add stats
    stats_text = (
        f"λ range: [{sim_data['lambda'].min():.2f}, {sim_data['lambda'].max():.2f}]\n"
        f"θ range: [{sim_data['theta'].min():.2f}, {sim_data['theta'].max():.2f}]\n"
        f"ψ range: [{sim_data['disease_state_weights'].min():.2f}, "
        f"{sim_data['disease_state_weights'].max():.2f}]\n"
        f"φ range: [{phi_equiv.min():.2f}, {phi_equiv.max():.2f}]"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def compare_model_runs(run1, run2, K):
    """
    Compare two model runs by matching clusters and comparing parameters
    
    Parameters:
    -----------
    run1, run2 : dict
        Dictionaries containing model parameters including:
        - clusters
        - lambda
        - psi
        - logit_prev_t (base trajectories)
    K : int
        Number of clusters
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    
    # First match clusters using psi similarity
    psi1, psi2 = run1['psi'], run2['psi']
    similarity_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            similarity_matrix[i,j] = np.corrcoef(psi1[i,:], psi2[j,:])[0,1]
    
    # Use Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    
    # Create figure for comparisons
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Plot psi correlation heatmap
    ax = axes[0,0]
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Psi Correlations Between Runs')
    ax.set_xlabel('Run 2 Clusters')
    ax.set_ylabel('Run 1 Clusters')
    plt.colorbar(im, ax=ax)
    
    # 2. Plot matched effective phis
    ax = axes[0,1]
    for k in range(K):
        k2 = col_ind[k]  # Matched cluster in run2
        
        # Compute effective phi (psi + base trajectory) for a sample disease
        d = np.argmax(psi1[k,:])  # Get most associated disease
        phi1 = run1['logit_prev_t'][d,:] + psi1[k,d]
        phi2 = run2['logit_prev_t'][d,:] + psi2[k2,d]
        
        ax.plot(phi1, '-', label=f'Run1 Cluster {k}')
        ax.plot(phi2, '--', label=f'Run2 Cluster {k2}')
    
    ax.set_title('Effective Phi for Matched Clusters\n(Most Associated Disease)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Logit Probability')
    ax.legend()
    
    # 3. Plot lambda comparisons for a few individuals
    ax = axes[1,0]
    n_samples = 3
    sample_inds = np.random.choice(run1['lambda'].shape[0], n_samples)
    
    for i, n in enumerate(sample_inds):
        for k in range(K):
            k2 = col_ind[k]
            ax.plot(run1['lambda'][n,k,:], '-', label=f'Run1 Ind{n} Clust{k}')
            ax.plot(run2['lambda'][n,k2,:], '--', label=f'Run2 Ind{n} Clust{k2}')
    
    ax.set_title('Lambda Trajectories for Sample Individuals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Lambda Value')
    ax.legend()
    
    # 4. Plot cluster assignment confusion matrix
    ax = axes[1,1]
    clusters1, clusters2 = run1['clusters'], run2['clusters']
    conf_mat = np.zeros((K, K))
    for i in range(len(clusters1)):
        conf_mat[clusters1[i], clusters2[i]] += 1
    
    im = ax.imshow(conf_mat / conf_mat.sum(axis=1, keepdims=True), 
                   cmap='Blues')
    ax.set_title('Normalized Cluster Assignment Confusion Matrix')
    ax.set_xlabel('Run 2 Clusters')
    ax.set_ylabel('Run 1 Clusters')
    plt.colorbar(im, ax=ax)
    
    # Add summary statistics as text
    stats_text = (
        f"Average Psi Correlation: {np.mean([similarity_matrix[i,col_ind[i]] for i in range(K)]):.3f}\n"
        f"Cluster Assignment Agreement: {np.sum(clusters1 == clusters2) / len(clusters1):.3f}"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, row_ind, col_ind

def compare_fit_to_truth(sim_data, model_fit, K, n_diseases=4):
    """
    Compare model fit against simulation truth with simpler disease-specific plots
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    
    # First match clusters using disease-state weight patterns
    true_weights = sim_data['disease_state_weights']
    fit_weights = model_fit['psi']
    similarity_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            similarity_matrix[i,j] = np.corrcoef(true_weights[i,:], fit_weights[j,:])[0,1]
    
    # Use Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    
    # Create figure for comparisons
    fig = plt.figure(figsize=(20, 5*n_diseases))
    
    # Find diseases with most events
    Y = sim_data['Y']
    event_counts = Y.sum(axis=(0,2))  # Sum over individuals and time
    top_diseases = np.argsort(event_counts)[-n_diseases:]  # Top n_diseases by event count
    
    # Plot each disease comparison
    for idx, d in enumerate(top_diseases):
        ax = plt.subplot(n_diseases, 1, idx+1)
        
        # Get predicted probabilities
        pi_pred = model_fit['pi']
        mean_pred = pi_pred[:, d, :].mean(axis=0)  # Average across people
        
        # Get actual incidence
        actual_incidence = Y[:, d, :].mean(axis=0)  # Average across people
        
        # Plot
        ax.plot(mean_pred, label='Model Predicted average π')
        ax.plot(actual_incidence, label='Actual Incidence')
        ax.set_title(f'Disease {d}: Predicted vs Actual Prevalence')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
    
    # Add summary statistics as text
    matched_corrs = [similarity_matrix[i,col_ind[i]] for i in range(K)]
    stats_text = (
        f"Average Disease-State Weight Correlation: {np.mean(matched_corrs):.3f}\n"
        f"Min Correlation: {np.min(matched_corrs):.3f}\n"
        f"Max Correlation: {np.max(matched_corrs):.3f}"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, row_ind, col_ind

def extract_model_dict(model):
    """
    Extract parameters from an AladynSurvivalFixedKernelsAvgLoss model into dictionary format
    
    Parameters:
    -----------
    model : AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
        The fitted model
        
    Returns:
    --------
    dict : Dictionary containing model parameters in numpy format
    """
    with torch.no_grad():
        # Get basic parameters
        lambda_ik = model.lambda_.detach().cpu().numpy()
        psi = model.psi.detach().cpu().numpy()
        
        # Compute theta (state proportions)
        theta = torch.softmax(model.lambda_, dim=1).detach().cpu().numpy()
        
        # Get clusters
        clusters = torch.argmax(model.psi, dim=0).detach().cpu().numpy()
        
        # Compute pi (event probabilities)
        # Handle case where forward returns a tuple
        forward_output = model.forward()
        if isinstance(forward_output, tuple):
            pi = forward_output[0].detach().cpu().numpy()  # Assume pi is first element
        else:
            pi = forward_output.detach().cpu().numpy()
        
        # Extract other parameters if they exist
        params = {
            'lambda': lambda_ik,
            'psi': psi,
            'theta': theta,
            'clusters': clusters,
            'pi': pi
        }
        
        # Add optional parameters if they exist
        if hasattr(model, 'logit_prev_t'):
            params['logit_prev_t'] = model.logit_prev_t.detach().cpu().numpy()
            
    return params

# Now you can use it like:
# model_dict = extract_model_dict(model)
# fig, row_ind, col_ind = compare_fit_to_truth(sim_data, model_dict, K)

def print_model_params(model):
    """
    Print key parameters from the model for diagnostics
    """
    print("\nModel Parameter Diagnostics:")
    print("-" * 30)
    
    with torch.no_grad():
        # Check lambda
        if hasattr(model, 'lambda_'):
            lambda_param = model.lambda_
        elif hasattr(model, 'lambda_ik'):
            lambda_param = model.lambda_ik
        else:
            lambda_param = None
            
        if lambda_param is not None:
            print(f"\nLambda shape: {lambda_param.shape}")
            print(f"Lambda range: [{lambda_param.min().item():.3f}, {lambda_param.max().item():.3f}]")
            print(f"Lambda mean: {lambda_param.mean().item():.3f}")
            print(f"Lambda std: {lambda_param.std().item():.3f}")
        
        # Check psi
        if hasattr(model, 'psi'):
            print(f"\nPsi shape: {model.psi.shape}")
            print(f"Psi range: [{model.psi.min().item():.3f}, {model.psi.max().item():.3f}]")
            print(f"Psi mean: {model.psi.mean().item():.3f}")
            print(f"Psi std: {model.psi.std().item():.3f}")
        
        # Check if model can compute pi
        try:
            pi = model.forward()
            if isinstance(pi, tuple):
                pi = pi[0]
            print(f"\nPi shape: {pi.shape}")
            print(f"Pi range: [{pi.min().item():.3f}, {pi.max().item():.3f}]")
            print(f"Pi mean: {pi.mean().item():.3f}")
            print(f"Pi std: {pi.std().item():.3f}")
        except:
            print("\nCouldn't compute pi")
            
        # Print model attributes
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                try:
                    val = getattr(model, attr)
                    if isinstance(val, torch.Tensor):
                        print(f"{attr}: Tensor of shape {val.shape}")
                    else:
                        print(f"{attr}: {type(val)}")
                except:
                    pass

# Usage:
# print_model_params(model_with_refs)

def compare_phi_patterns(sim_data, model_fit, K, n_states_to_show=5):
    """
    Compare phi patterns between simulation and fitted model
    
    Parameters:
    -----------
    sim_data : dict
        Dictionary containing simulation parameters including:
        - disease_state_weights
        - base_trajectories or logit_prev_t
    model_fit : dict
        Dictionary containing fitted parameters including:
        - psi
        - phi
        - logit_prev_t
    K : int
        Number of clusters
    n_states_to_show : int
        Number of states to show in comparison
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    
    # First match states using disease-state weight patterns
    true_weights = sim_data['disease_state_weights']
    fit_weights = model_fit['psi']
    similarity_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            similarity_matrix[i,j] = np.corrcoef(true_weights[i,:], fit_weights[j,:])[0,1]
    
    # Use Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    
    # Get base patterns
    if 'base_trajectories' in sim_data:
        true_base = sim_data['base_trajectories']
    else:
        true_base = sim_data['logit_prev_t']
    
    fit_base = model_fit['logit_prev_t']
    
    # Reconstruct true phi
    D = true_weights.shape[1]
    T = true_base.shape[1] if len(true_base.shape) > 1 else len(true_base)
    true_phi = np.zeros((K, D, T))
    
    for k in range(K):
        for d in range(D):
            true_phi[k,d,:] = true_base[d] + true_weights[k,d]
    
    # Get fitted phi
    fit_phi = model_fit['phi']
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 2)
    
    # 1. Plot correlation matrix
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('State Correlation Matrix', fontsize=12)
    ax.set_xlabel('Fitted States', fontsize=10)
    ax.set_ylabel('True States', fontsize=10)
    plt.colorbar(im, ax=ax)
    
    # 2. Plot base patterns for a few diseases
    ax = fig.add_subplot(gs[0, 1])
    n_diseases = 5
    diseases = np.random.choice(D, n_diseases)
    for d in diseases:
        ax.plot(true_base[d], '-', label=f'True Base D{d}', alpha=0.7)
        ax.plot(fit_base[d], '--', label=f'Fitted Base D{d}', alpha=0.7)
    ax.set_title('Base Disease Patterns\n(Before State Effects)', fontsize=12)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Logit Probability', fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Plot phi patterns for top correlating states
    ax = fig.add_subplot(gs[1, :])
    correlations = [similarity_matrix[i,col_ind[i]] for i in range(K)]
    top_states = np.argsort(correlations)[-n_states_to_show:]
    
    # For each top state, plot phi for its most associated diseases
    for k in top_states:
        k_fit = col_ind[k]
        # Find most associated diseases for this state
        top_diseases = np.argsort(true_weights[k])[-3:]  # Top 3 diseases
        
        for d in top_diseases:
            # Plot true phi
            ax.plot(true_phi[k,d,:], '-', 
                   label=f'True S{k}D{d}', alpha=0.7)
            # Plot fitted phi
            ax.plot(fit_phi[k_fit,d,:], '--',
                   label=f'Fitted S{k_fit}D{d}', alpha=0.7)
    
    ax.set_title('φ Patterns (Base + State Effects)\nTop States with Their Most Associated Diseases', 
                 fontsize=12)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Logit Probability', fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add summary statistics
    matched_corrs = [similarity_matrix[i,col_ind[i]] for i in range(K)]
    stats_text = (
        f"Average State Correlation: {np.mean(matched_corrs):.3f}\n"
        f"Min Correlation: {np.min(matched_corrs):.3f}\n"
        f"Max Correlation: {np.max(matched_corrs):.3f}"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, row_ind, col_ind

# Usage:
# fig, row_ind, col_ind = compare_phi_patterns(sim_data, model_dict, K=20)
# plt.show()

def plot_smoothed_prevalence_check(smoothed_prev, base_trajectories, sample_diseases=None):
    """
    Plot smoothed prevalence against base trajectories for comparison
    
    Args:
        smoothed_prev: D x T tensor of smoothed prevalence
        base_trajectories: D x T tensor of base trajectories
        sample_diseases: list of disease indices to plot (default: first 3)
    """
    if sample_diseases is None:
        sample_diseases = [0, 1, 2]
    
    plt.figure(figsize=(12, 6))
    
    # Plot both on logit scale for fair comparison
    epsilon = 1e-8
    logit_prev = np.log(smoothed_prev / (1 - smoothed_prev + epsilon))
    
    for d in sample_diseases:
        plt.plot(logit_prev[d], '--', label=f'Smoothed Prev (Disease {d})')
        plt.plot(base_trajectories[d], '-', label=f'Base Trajectory (Disease {d})')
    
    plt.title('Smoothed Prevalence vs Base Trajectories (Logit Scale)')
    plt.xlabel('Time')
    plt.ylabel('Logit Probability')
    plt.legend()
    plt.grid(True)

def plot_event_distribution(Y, n_diseases=5):
    """
    Plot the distribution of events over time for selected diseases
    
    Args:
        Y: N x D x T array of events
        n_diseases: number of diseases to plot
    """
    N, D, T = Y.shape
    
    # Get total events per disease
    total_events = Y.sum(axis=0).sum(axis=1)  # Sum over individuals and time
    # Select top diseases by event count
    top_diseases = np.argsort(-total_events)[:n_diseases]
    
    plt.figure(figsize=(15, 8))
    
    # Plot event counts over time
    for d in top_diseases:
        events_over_time = Y[:, d, :].sum(axis=0)  # Sum over individuals
        plt.plot(events_over_time, label=f'Disease {d} (total={total_events[d]:.0f})')
    
    plt.title('Event Distribution Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.legend()
    plt.grid(True)
    
    # Add summary statistics
    stats_text = (
        f"Total events: {Y.sum():.0f}\n"
        f"Average events per disease: {total_events.mean():.1f}\n"
        f"Max events per disease: {total_events.max():.0f}\n"
        f"Diseases with no events: {(total_events == 0).sum()}"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    
    # Also show cumulative distribution
    plt.figure(figsize=(15, 8))
    for d in top_diseases:
        events_over_time = Y[:, d, :].sum(axis=0)
        cumulative = np.cumsum(events_over_time)
        plt.plot(cumulative / cumulative[-1], label=f'Disease {d}')
    
    plt.title('Cumulative Event Distribution (Normalized)')
    plt.xlabel('Time')
    plt.ylabel('Fraction of Total Events')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_raw_events(Y, n_diseases=5):
    """
    Plot raw event counts without any smoothing
    
    Args:
        Y: N x D x T array of events
        n_diseases: number of diseases to plot
    """
    N, D, T = Y.shape
    
    # Get total events per disease
    total_events = Y.sum(axis=0).sum(axis=1)  # Sum over individuals and time
    # Select top diseases by event count
    top_diseases = np.argsort(-total_events)[:n_diseases]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Raw event counts
    for d in top_diseases:
        events = Y[:, d, :].sum(axis=0)  # Sum over individuals
        ax1.scatter(np.arange(T), events, alpha=0.5, label=f'Disease {d}')
        ax1.plot(events, alpha=0.3)  # Line to connect points
    
    ax1.set_title('Raw Event Counts (No Smoothing)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Events')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Event timing distribution
    event_times = []
    disease_ids = []
    for d in top_diseases:
        # Get times where events occurred
        times = np.where(Y[:, d, :].sum(axis=0) > 0)[0]
        event_times.extend(times)
        disease_ids.extend([d] * len(times))
    
    ax2.hist(event_times, bins=T//2, alpha=0.5)
    ax2.set_title('Distribution of Event Timing (All Selected Diseases)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Time Points with Events')
    ax2.grid(True)
    
    # Add summary stats
    stats_text = (
        f"Total events in dataset: {Y.sum():.0f}\n"
        f"Diseases shown: {n_diseases} most common\n"
        f"Max events at any timepoint: {max(Y.sum(axis=(0,1))):.0f}\n"
        f"Percent timepoints with no events: {100 * (Y.sum(axis=(0,1)) == 0).mean():.1f}%"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_diseases_per_timepoint(Y):
    """
    Plot number of diseases that have events at each time point
    
    Args:
        Y: N x D x T array of events
    """
    N, D, T = Y.shape
    
    # For each time point, count how many diseases had at least one event
    diseases_per_time = (Y.sum(axis=0) > 0).sum(axis=0)  # First sum over N (patients), then count diseases
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(T), diseases_per_time, alpha=0.6)
    plt.title('Number of Diseases with Events at Each Time Point')
    plt.xlabel('Time')
    plt.ylabel('Number of Diseases with Events')
    
    # Add summary stats
    stats_text = (
        f"Average diseases per timepoint: {diseases_per_time.mean():.1f}\n"
        f"Max diseases at one time: {diseases_per_time.max():.0f}\n"
        f"Time points with no events: {(diseases_per_time == 0).sum()}"
    )
    plt.figtext(0.02, 0.98, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_events_comparison(Y):
    """
    Compare first events vs total events at each time point
    
    Args:
        Y: N x D x T array of events
    """
    N, D, T = Y.shape
    
    # Get first events (current plot)
    first_events = (Y.sum(axis=0) > 0).sum(axis=0)  # Number of diseases with first event
    
    # Get total events
    total_events = Y.sum(axis=(0,1))  # Sum over both N and D for total events at each time
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot first events
    ax1.bar(np.arange(T), first_events, alpha=0.6, color='skyblue')
    ax1.set_title('Number of Diseases with First Events at Each Time Point')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Diseases')
    ax1.grid(True, alpha=0.3)
    
    # Add stats for first events
    stats_text1 = (
        f"Average diseases with first events per timepoint: {first_events.mean():.1f}\n"
        f"Max diseases with first events at one time: {first_events.max():.0f}"
    )
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    # Plot total events
    ax2.bar(np.arange(T), total_events, alpha=0.6, color='lightgreen')
    ax2.set_title('Total Number of Events at Each Time Point')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Events')
    ax2.grid(True, alpha=0.3)
    
    # Add stats for total events
    stats_text2 = (
        f"Average total events per timepoint: {total_events.mean():.1f}\n"
        f"Max total events at one time: {total_events.max():.0f}\n"
        f"Total events in dataset: {total_events.sum():.0f}"
    )
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def evaluate_survival_predictions(Y, pi_pred):
    """
    Evaluate model predictions accounting for survival structure
    
    Args:
        Y: N x D x T array of observed events
        pi_pred: N x D x T array of predicted hazards
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    N, D, T = Y.shape
    
    # Create at_risk mask (1 if no event has occurred yet)
    at_risk = np.ones_like(Y)
    for n in range(N):
        for d in range(D):
            # Find first event
            event_times = np.where(Y[n,d,:])[0]
            if len(event_times) > 0:
                first_event = event_times[0]
                # Mask out all times after first event
                at_risk[n,d,(first_event+1):] = 0
    
    # Flatten arrays and keep only at-risk predictions
    y_true = Y[at_risk == 1]
    y_pred = pi_pred[at_risk == 1]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Model Predictions ROC Curve\n(Only At-Risk Times)')
    ax1.legend()
    
    # 2. Discrimination plot
    ax2.hist(y_pred[y_true == 0], bins=50, alpha=0.5, density=True, label='No Event')
    ax2.hist(y_pred[y_true == 1], bins=50, alpha=0.5, density=True, label='Event')
    ax2.set_xlabel('Predicted Hazard')
    ax2.set_ylabel('Density')
    ax2.set_title('Hazard Distribution by Event Status\n(Only At-Risk Times)')
    ax2.legend()
    
    # 3. Time-specific ROC AUCs
    aucs = []
    times = []
    for t in range(T):
        mask_t = at_risk[:,:,t].flatten() == 1
        if mask_t.sum() > 0:  # If anyone still at risk
            y_true_t = Y[:,:,t].flatten()[mask_t]
            y_pred_t = pi_pred[:,:,t].flatten()[mask_t]
            if len(np.unique(y_true_t)) > 1:  # Need both classes for ROC
                fpr, tpr, _ = roc_curve(y_true_t, y_pred_t)
                aucs.append(auc(fpr, tpr))
                times.append(t)
    
    ax3.plot(times, aucs)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('AUC')
    ax3.set_title('AUC Over Time\n(Only At-Risk Individuals)')
    
    # 4. Calibration plot with risk groups
    n_groups = 10
    pred_probs = np.linspace(0, y_pred.max(), n_groups+1)
    obs_props = []
    mean_preds = []
    
    for i in range(n_groups):
        mask = (y_pred >= pred_probs[i]) & (y_pred < pred_probs[i+1])
        if mask.sum() > 0:
            obs_props.append(y_true[mask].mean())
            mean_preds.append(y_pred[mask].mean())
    
    ax4.scatter(mean_preds, obs_props)
    ax4.plot([0, max(mean_preds)], [0, max(mean_preds)], 'k--')
    ax4.set_xlabel('Predicted Hazard')
    ax4.set_ylabel('Observed Event Proportion')
    ax4.set_title('Calibration Plot\n(Only At-Risk Times)')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nSummary Statistics (Only At-Risk Times):")
    print(f"Overall AUC: {roc_auc:.3f}")
    print(f"Mean predicted hazard for non-events: {y_pred[y_true == 0].mean():.4f}")
    print(f"Mean predicted hazard for events: {y_pred[y_true == 1].mean():.4f}")
    print(f"Number of valid predictions: {len(y_true)}")
    print(f"Event rate in valid predictions: {y_true.mean():.4f}")
    
    return fig