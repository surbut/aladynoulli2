import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class AladynSurvivalModel(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, length_scales, amplitudes, prevalence):
        super().__init__()
        self.N = N  # Number of individuals
        self.D = D  # Number of diseases
        self.T = T  # Number of time points
        self.K = K  # Number of topics
        self.P = P  # Number of genetic covariates

        # Convert inputs to tensors
        self.G = torch.tensor(G, dtype=torch.float32)            # Genetic covariates (N x P)
        self.Y = torch.tensor(Y, dtype=torch.float32)            # Disease occurrences (N x D x T)
        self.prevalence = torch.tensor(prevalence, dtype=torch.float32)  # Disease prevalence (D)

        # Compute logit of prevalence for centering phi
        self.logit_prev = torch.log(self.prevalence / (1 - self.prevalence))  # (D)

        # Create time grid and squared distances for kernel computation
        times = np.arange(T)
        sq_dists = squareform(pdist(times.reshape(-1, 1)) ** 2)
        self.times = torch.tensor(times, dtype=torch.float32)

        # Initialize GP kernels for lambda and phi
        self.K_lambda = []
        self.K_phi = []
        for k in range(K):
            # RBF kernel for lambda
            K_lambda_k = amplitudes[k] * np.exp(-0.5 * sq_dists / (length_scales[k] ** 2))
            self.K_lambda.append(torch.tensor(K_lambda_k + 1e-6 * np.eye(T), dtype=torch.float32))

            # RBF kernel for phi (using same length scales and amplitudes)
            self.K_phi.append(torch.tensor(K_lambda_k + 1e-6 * np.eye(T), dtype=torch.float32))

        # Initialize parameters
        self.initialize_params()

    def initialize_params(self):
        """Initialize parameters using SVD and GP priors"""
        # Compute average disease occurrence matrix (N x D)
        Y_avg = torch.mean(self.Y, dim=2)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(Y_avg, full_matrices=False)

        # Initialize gamma using genetic covariates
        lambda_init = U[:, :self.K] @ torch.diag(torch.sqrt(S[:self.K]))  # N x K
        # Fix the lstsq call
        gamma_init = torch.linalg.lstsq(self.G, lambda_init).solution  # P x K
        self.gamma = nn.Parameter(gamma_init)  # P x K

        # Initialize lambda using GP prior
        lambda_means = self.G @ self.gamma  # N x K
        lambda_init = torch.zeros((self.N, self.K, self.T))
        for k in range(self.K):
            L_k = torch.linalg.cholesky(self.K_lambda[k])
            for i in range(self.N):
                mean = lambda_means[i, k]
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = mean + eps
        self.lambda_ = nn.Parameter(lambda_init)  # N x K x T

        # Initialize phi using GP prior
        phi_init = torch.zeros((self.K, self.D, self.T))
        for k in range(self.K):
            L_k_phi = torch.linalg.cholesky(self.K_phi[k])
            for d in range(self.D):
                mean = self.logit_prev[d]
                eps = L_k_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean + eps
        self.phi = nn.Parameter(phi_init)  # K x D x T

    def forward(self):
        # Compute theta using softmax over topics (N x K x T)
        theta = torch.softmax(self.lambda_, dim=1)  # N x K x T

        # Compute phi_prob using sigmoid function (K x D x T)
        phi_prob = torch.sigmoid(self.phi)  # K x D x T

        # Compute pi (N x D x T)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob)

        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        """
        Compute the negative log-likelihood loss for survival data.
        """
        pi, theta, phi_prob = self.forward()

        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)

        # Initialize loss
        loss = 0.0

        # Convert event_times to tensor
        event_times_tensor = torch.tensor(event_times, dtype=torch.long)

        # Vectorized computation of loss
        N, D, T = self.Y.shape

        # Create time indices
        time_indices = torch.arange(T)

        # Compute loss for each individual and disease
        for n in range(N):
            for d in range(D):
                t_event = event_times_tensor[n, d]
                if t_event < T:
                    # Times before the event
                    pi_censored = pi[n, d, :t_event]
                    loss -= torch.sum(torch.log(1 - pi_censored))

                    # At the event time
                    if self.Y[n, d, t_event] == 1:
                        pi_event = pi[n, d, t_event]
                        loss -= torch.log(pi_event)
                    else:
                        # If no event occurred at t_event, treat as censored
                        pi_censored = pi[n, d, t_event]
                        loss -= torch.log(1 - pi_censored)
                else:
                    # Censored observation
                    pi_censored = pi[n, d, :]
                    loss -= torch.sum(torch.log(1 - pi_censored))

        # Add GP prior losses for lambda and phi
        gp_loss = self.compute_gp_prior_loss()

        total_loss = loss + gp_loss
        return total_loss

    def compute_gp_prior_loss(self):
        """
        Compute the GP prior loss for lambda and phi.
        """
        gp_loss = 0.0
        N, K, T = self.lambda_.shape
        K, D, T = self.phi.shape

        for k in range(K):
            # Inverse of covariance matrices with regularization
            K_lambda_inv = torch.inverse(self.K_lambda[k] + 1e-6 * torch.eye(self.T))
            K_phi_inv = torch.inverse(self.K_phi[k] + 1e-6 * torch.eye(self.T))

            # Lambda GP prior
            lambda_k = self.lambda_[:, k, :]  # N x T
            mean_lambda_k = (self.G @ self.gamma[:, k]).unsqueeze(1)  # N x 1
            deviations_lambda = lambda_k - mean_lambda_k  # N x T
            gp_loss += 0.5 * torch.sum(deviations_lambda @ K_lambda_inv * deviations_lambda)

            # Phi GP prior
            phi_k = self.phi[k, :, :]  # D x T
            mean_phi_k = self.logit_prev.unsqueeze(1)  # D x 1
            deviations_phi = phi_k - mean_phi_k  # D x T
            gp_loss += 0.5 * torch.sum(deviations_phi @ K_phi_inv * deviations_phi)

        return gp_loss

    def fit(self, event_times, num_epochs=100, learning_rate=1e-3):
        """
        Fit the model using gradient descent.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        losses = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        return losses


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
    theta = exp_lambda / np.sum(exp_lambda, axis=1, keepdims=True)  # N x K x T

    # Simulate phi (topic-disease trajectories)
    phi_kd = np.zeros((K, D, T))
    for k in range(K):
        cov_matrix = amplitudes[k] * np.exp(-0.5 * (time_diff ** 2) / length_scales[k] ** 2)
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
        return Y, G, prevalence, length_scales, amplitudes, event_times

