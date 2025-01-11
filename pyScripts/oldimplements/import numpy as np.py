import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

"""
# Set the random seed for reproducibility
np.random.seed(123)
torch.manual_seed(123)

class AladynSurvivalModel(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, length_scales, amplitudes, prevalence):
        super().__init__()
        self.N = N
        self.D = D
        self.T = T
        self.K = K
        self.P = P
        self.G = torch.tensor(G, dtype=torch.float32)

        # Prevalence vector for diseases
        self.prevalence = torch.tensor(prevalence, dtype=torch.float32)
        self.logit_prev = torch.log(self.prevalence / (1 - self.prevalence))

        # Create time grid and squared distances for kernel computation
        times = np.arange(T)
        sq_dists = squareform(pdist(times.reshape(-1, 1))**2)
        self.times = torch.tensor(times, dtype=torch.float32)

        # Initialize GP kernels
        self.K_lambda = []
        self.K_phi = []
        for k in range(K):
            K_lambda_k = amplitudes[k] * np.exp(-0.5 * sq_dists / (length_scales[k]**2))
            self.K_lambda.append(torch.tensor(K_lambda_k + 1e-6 * np.eye(T), dtype=torch.float32))
            self.K_phi.append(torch.tensor(K_lambda_k + 1e-6 * np.eye(T), dtype=torch.float32))

        # Initialize parameters using SVD
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        Y_avg = torch.mean(Y_tensor, dim=2)  # N x D
        U, S, Vh = torch.linalg.svd(Y_avg)
        
        # Initialize gamma using genetic covariates
        lambda_init = U[:, :K] @ torch.diag(torch.sqrt(S[:K]))  # N x K
        gamma_init = torch.linalg.lstsq(self.G, lambda_init).solution  # P x K
        self.gamma = nn.Parameter(gamma_init)
        
        # Initialize lambda as GP (mean + deviation)
        lambda_means = self.G @ gamma_init
        lambda_init = torch.zeros((N, K, T))
        for k in range(K):
            for i in range(N):
                mean = lambda_means[i, k]
                eps = torch.randn(T) @ torch.linalg.cholesky(self.K_lambda[k])
                lambda_init[i, k, :] = mean + eps  # Complete GP
        
        self.lambda_ = nn.Parameter(lambda_init)
        
        # Initialize phi as GP (mean + deviation)
        phi_init = torch.zeros((K, D, T))
        for k in range(K):
            for d in range(D):
                mean = self.logit_prev[d]
                eps = torch.randn(T) @ torch.linalg.cholesky(self.K_phi[k])
                phi_init[k, d, :] = mean + eps  # Complete GP
        
        self.phi = nn.Parameter(phi_init)

    def forward(self):
        # Use GPs directly (they already include mean + deviation)
        theta = torch.softmax(self.lambda_, dim=1)
        phi_prob = torch.sigmoid(self.phi)
        
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        
        Compute the negative log-likelihood loss for survival data.
        
        pi, theta, phi_prob = self.forward()

        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)

        loss = 0.0
        N, D, T = self.Y.shape

        for n in range(N):
            for d in range(D):
                t_event = event_times[n, d]
                if t_event < T:
                    # Sum over times before the event
                    loss -= torch.sum(torch.log(1 - pi[n, d, :t_event]))
                    # Add the event time
                    if self.Y[n, d, t_event] == 1:
                        loss -= torch.log(pi[n, d, t_event])
                    else:
                        loss -= torch.log(1 - pi[n, d, t_event])
                else:
                    # Censored observation
                    loss -= torch.sum(torch.log(1 - pi[n, d, :]))

        # Add GP prior losses for lambda and phi
        gp_loss = self.compute_gp_prior_loss()

        total_loss = loss + gp_loss
        return total_loss

    def compute_gp_prior_loss(self):
       
        gp_loss = 0.0
        N, K, T = self.lambda_.shape
        K, D, T = self.phi.shape

        for k in range(K):
            K_lambda_inv = torch.inverse(self.K_lambda[k])
            K_phi_inv = torch.inverse(self.K_phi[k])

            # Lambda GP prior
            for i in range(N):
                lambda_i_k = self.lambda_[i, k, :]  # (T)
                mean_lambda_i_k = self.G[i] @ self.gamma[:, k]  # Scalar
                deviation_lambda = lambda_i_k - mean_lambda_i_k
                gp_loss += 0.5 * deviation_lambda @ K_lambda_inv @ deviation_lambda

            # Phi GP prior
            for d in range(D):
                phi_k_d = self.phi[k, d, :]  # (T)
                mean_phi_k_d = self.logit_prev[d]
                deviation_phi = phi_k_d - mean_phi_k_d
                gp_loss += 0.5 * deviation_phi @ K_phi_inv @ deviation_phi

        return gp_loss

    def initialize_params(self):
        Initialize parameters using SVD and GP priors
        # Compute average disease occurrence matrix (N x D)
        Y_avg = torch.mean(self.Y, dim=2)
        
        # Perform SVD
        U, S, Vh = torch.linalg.svd(Y_avg)
        
        # Initialize gamma using genetic covariates
        lambda_init = U[:, :self.K] @ torch.diag(torch.sqrt(S[:self.K]))  # N x K
        gamma_init = torch.linalg.lstsq(self.G, lambda_init).solution  # P x K
        self.gamma = nn.Parameter(gamma_init)
        
        # Initialize lambda using GP prior
        lambda_means = self.G @ gamma_init  # N x K
        lambda_init = torch.zeros((self.N, self.K, self.T))
        
        for k in range(self.K):
            for i in range(self.N):
                mean = lambda_means[i, k]
                eps = torch.randn(self.T) @ torch.linalg.cholesky(self.K_lambda[k])
                lambda_init[i, k, :] = mean + eps
        
        self.lambda_ = nn.Parameter(lambda_init)
        
        # Initialize phi using prevalence information and GP prior
        phi_init = torch.zeros((self.K, self.D, self.T))
        for k in range(self.K):
            for d in range(self.D):
                mean = self.logit_prev[d]
                eps = torch.randn(self.T) @ torch.linalg.cholesky(self.K_phi[k])
                phi_init[k, d, :] = mean + eps
        
        self.phi = nn.Parameter(phi_init)

def generate_synthetic_data(N=100, D=5, T=50, K=3, P=5, return_true_params=False):
    """
    
    """
    # Genetic covariates G (N x P)
    G = np.random.randn(N, P)

    # Prevalence of diseases (D)
    prevalence = np.random.uniform(0.01, 0.05, D)

    # Length scales and amplitudes for GP kernels
    length_scales = np.random.uniform(T / 3, T / 2, K)
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
        cov_matrix = amplitudes[k] * np.exp(-0.5 * (time_diff ** 2) / length_scales[k] ** 2)
        for i in range(N):
            mean_lambda = G[i] @ Gamma_k[:, k]
            lambda_ik[i, k, :] = multivariate_normal.rvs(mean=mean_lambda * np.ones(T), cov=cov_matrix)

    # Compute theta
    exp_lambda = np.exp(lambda_ik)
    theta = exp_lambda / np.sum(exp_lambda, axis=1, keepdims=True)  # (N x K x T)

    # Simulate phi (topic-disease trajectories)
    phi_kd = np.zeros((K, D, T))
    for k in range(K):
        cov_matrix = amplitudes[k] * np.exp(-0.5 * (time_diff ** 2) / length_scales[k] ** 2)
        for d in range(D):
            mean_phi = np.log(prevalence[d])
            phi_kd[k, d, :] = multivariate_normal.rvs(mean=mean_phi * np.ones(T), cov=cov_matrix)

    # Compute eta
    eta = expit(phi_kd)  # (K x D x T)

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
        return Y, G, prevalence, length_scales, amplitudes, event_times
"""

   

    # After training, you can access the learned parameters
    # For example, model.lambda_.detach().numpy(), model.phi.detach().numpy(), etc.



