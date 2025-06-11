import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import softmax, expit



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

def generate_clustered_survival_data(N=1000, D=20, T=50, K=5, P=5):
    """
    Generate synthetic data matching our fitted model structure
    """
    # Fixed kernel parameters as in the fit
    lambda_length = T/4
    phi_length = T/3
    amplitude = 1.0
    
    # Setup time grid
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = amplitude**2 * np.exp(-0.5 * (time_diff**2) / lambda_length**2)
    K_phi = amplitude**2 * np.exp(-0.5 * (time_diff**2) / phi_length**2)
    
        # 1. Generate more realistic baseline trajectories
    logit_prev_t = np.zeros((D, T))
    for d in range(D):
        # More diverse base rates
        base_rate = np.random.choice([
            #np.random.uniform(-18, -16),  # Very rare
            #np.random.uniform(-16, -14),  # Rare
            np.random.uniform(-14, -12),  # Uncommon
            np.random.uniform(-12, -10),  # Moderate
            np.random.uniform(-10, -8),   # Common
            np.random.uniform(-8, -6)     # Very common
        ], p=[0.40, 0.40, 0.15, 0.05])
        
        # More diverse trajectory shapes
        peak_age = np.random.uniform(20, 40)  # Wider range for peak
        # Reduce slope range
        slope = np.random.uniform(0.10, 0.4)  # More modest increase

# Increase decay to ensure prevalence plateaus
        decay = np.random.uniform(0.002, 0.01)
        
        # Add possibility of early vs late onset patterns
        onset_shift = np.random.uniform(-10, 10)
        time_points_shifted = time_points - onset_shift
        
        # Generate trajectory with more complex patterns
        logit_prev_t[d, :] = base_rate + \
                            slope * time_points_shifted - \
                            decay * (time_points_shifted - peak_age)**2 #+ \
                           # np.random.normal(0, 0.5, T)  # Add some noise
        
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

def generate_clustered_survival_data_warp(N=1000, D=20, T=50, K=5, P=5, warping=False, beta_warp_sd=0.5):
    """
    Generate synthetic data matching our fitted model structure
    """
    # Fixed kernel parameters as in the fit
    lambda_length = T/4
    phi_length = T/3
    amplitude = 1.0
    
    # Setup time grid
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = amplitude**2 * np.exp(-0.5 * (time_diff**2) / lambda_length**2)
    K_phi = amplitude**2 * np.exp(-0.5 * (time_diff**2) / phi_length**2)
    
        # 1. Generate more realistic baseline trajectories
    logit_prev_t = np.zeros((D, T))
    for d in range(D):
        # More diverse base rates
        base_rate = np.random.choice([
            #np.random.uniform(-18, -16),  # Very rare
            #np.random.uniform(-16, -14),  # Rare
            np.random.uniform(-14, -12),  # Uncommon
            np.random.uniform(-12, -10),  # Moderate
            np.random.uniform(-10, -8),   # Common
            np.random.uniform(-8, -6)     # Very common
        ], p=[0.40, 0.40, 0.15, 0.05])
        
        # More diverse trajectory shapes
        peak_age = np.random.uniform(20, 40)  # Wider range for peak
        # Reduce slope range
        slope = np.random.uniform(0.10, 0.4)  # More modest increase

# Increase decay to ensure prevalence plateaus
        decay = np.random.uniform(0.002, 0.01)
        
        # Add possibility of early vs late onset patterns
        onset_shift = np.random.uniform(-10, 10)
        time_points_shifted = time_points - onset_shift
        
        # Generate trajectory with more complex patterns
        logit_prev_t[d, :] = base_rate + \
                            slope * time_points_shifted - \
                            decay * (time_points_shifted - peak_age)**2 #+ \
                           # np.random.normal(0, 0.5, T)  # Add some noise
        
    # 2. Generate cluster assignments
    clusters = np.zeros(D)
    diseases_per_cluster = D // K
    for k in range(K):
        clusters[k*diseases_per_cluster:(k+1)*diseases_per_cluster] = k
    
    # 3. Generate lambda (individual trajectories)
    G = np.random.randn(N, P)  # Genetic covariates
    Gamma_k = np.random.randn(P, K)  # Genetic effects

    # --- Warping parameters ---
    if warping:
        beta_warp = np.random.normal(0, beta_warp_sd, size=(P, K))
        eta = G @ beta_warp  # [N, K]
        rho = np.exp(eta)    # [N, K]
    else:
        rho = np.ones((N, K))

    lambda_ik = np.zeros((N, K, T))
    for i in range(N):
        mean_lambda = G[i] @ Gamma_k
        for k in range(K):
            lambda_ik[i, k, :] = multivariate_normal.rvs(
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
    
    # 5. Compute probabilities with warping
    theta = softmax(lambda_ik, axis=1)
    eta_phi = expit(phi_kd)
    pi = np.zeros((N, D, T))
    for n in range(N):
        for k in range(K):
            # Warped time axis for this individual/signature
            t = np.arange(T)
            t_warped = ((t / (T-1)) ** (1.0 / rho[n, k])) * (T-1)
            for d in range(D):
                # Interpolate phi_kd[k, d, :] at t_warped
                phi_warped = np.interp(t_warped, np.arange(T), eta_phi[k, d, :])
                pi[n, d, :] += theta[n, k, :] * phi_warped

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
        'rho': rho,
        'beta_warp': beta_warp if warping else None
    }