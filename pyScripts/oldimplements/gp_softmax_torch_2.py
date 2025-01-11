import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class AladynSurvivalModel(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, prevalence):
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

        # Initialize GP kernel hyperparameters with more stable values
        self.length_scales = nn.Parameter(torch.tensor(np.full(K, T/10), dtype=torch.float32))  # Shorter length scales
        self.log_amplitudes = nn.Parameter(torch.log(torch.tensor(np.full(K, 2.0), dtype=torch.float32)))  # Larger amplitudes
        
        # Initialize parameters with smaller values
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
        self.lambda_ = nn.Parameter(torch.zeros((self.N, self.K, self.T)))  # N x K x T

        # Initialize phi using GP prior
        self.phi = nn.Parameter(torch.zeros((self.K, self.D, self.T)))  # K x D x T

        # Initialize covariance matrices
        self.update_kernels()

        # Sample initial values for lambda and phi from the GPs
        for k in range(self.K):
            # Cholesky decomposition
            L_k = torch.linalg.cholesky(self.K_lambda[k])
            L_k_phi = torch.linalg.cholesky(self.K_phi[k])

            # Sample lambda
            for i in range(self.N):
                mean = lambda_means[i, k]
                eps = L_k @ torch.randn(self.T)
                self.lambda_.data[i, k, :] = mean + eps

            # Sample phi
            for d in range(self.D):
                mean = self.logit_prev[d]
                eps = L_k_phi @ torch.randn(self.T)
                self.phi.data[k, d, :] = mean + eps

    def update_kernels(self):
        """Update covariance matrices with robust numerical handling"""
        times = torch.arange(self.T, dtype=torch.float32)
        sq_dists = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2  # T x T

        self.K_lambda = []
        self.K_phi = []
        
        # Increased jitter and minimum length scale
        min_length_scale = 1.0
        jitter = 1e-2
        
        for k in range(self.K):
            # Ensure positive length scales with minimum value
            length_scale = torch.abs(self.length_scales[k]) + min_length_scale
            amplitude = torch.exp(self.log_amplitudes[k])

            # Compute base kernel
            K_base = amplitude ** 2 * torch.exp(-0.5 * sq_dists / (length_scale ** 2))
            
            # Add stronger diagonal regularization
            K_reg = K_base + (jitter + amplitude ** 2 * 1e-3) * torch.eye(self.T)
            
            # Ensure symmetry
            K_reg = 0.5 * (K_reg + K_reg.T)
            
            # Add to lists
            self.K_lambda.append(K_reg)
            self.K_phi.append(K_reg.clone())

    def forward(self):
        # Update kernels (in case hyperparameters have changed)
        self.update_kernels()

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

        N, D, T = self.Y.shape
        event_times_tensor = torch.tensor(event_times, dtype=torch.long)

        # Create masks for events and censoring
        time_grid = torch.arange(T).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T
        event_times_expanded = event_times_tensor.unsqueeze(-1)  # N x D x 1

        # Mask for times before the event
        mask_before_event = (time_grid < event_times_expanded).float()  # N x D x T
        # Mask for event time
        mask_at_event = (time_grid == event_times_expanded).float()  # N x D x T

        # Compute loss components
        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))

        total_data_loss = loss_censored + loss_event + loss_no_event

        # GP prior loss remains the same
        gp_loss = self.compute_gp_prior_loss()
        reg_loss = 0.01 * (torch.norm(self.gamma) + torch.norm(self.lambda_) + torch.norm(self.phi))
        total_loss = total_data_loss + gp_loss + reg_loss
        return total_loss

    def compute_gp_prior_loss(self):
        """Compute GP prior loss with robust numerical handling"""
        gp_loss = 0.0
        N, K, T = self.lambda_.shape
        K, D, T = self.phi.shape

        for k in range(K):
            try:
                # Try Cholesky decomposition with increased jitter if needed
                try:
                    L_lambda = torch.linalg.cholesky(self.K_lambda[k])
                    L_phi = torch.linalg.cholesky(self.K_phi[k])
                except RuntimeError:
                    # Add more jitter if Cholesky fails
                    extra_jitter = torch.eye(T) * 1e-1
                    L_lambda = torch.linalg.cholesky(self.K_lambda[k] + extra_jitter)
                    L_phi = torch.linalg.cholesky(self.K_phi[k] + extra_jitter)

                # Lambda GP prior
                lambda_k = self.lambda_[:, k, :]  # N x T
                mean_lambda_k = (self.G @ self.gamma[:, k]).unsqueeze(1)  # N x 1
                deviations_lambda = lambda_k - mean_lambda_k  # N x T

                # Process each individual
                for i in range(N):
                    dev_i = deviations_lambda[i:i+1].T  # T x 1
                    v_i = torch.cholesky_solve(dev_i, L_lambda)  # T x 1
                    gp_loss += 0.5 * torch.sum(v_i.T @ dev_i)

                # Phi GP prior
                phi_k = self.phi[k, :, :]  # D x T
                mean_phi_k = self.logit_prev.unsqueeze(1)  # D x 1
                deviations_phi = phi_k - mean_phi_k  # D x T

                # Process each disease
                for d in range(D):
                    dev_d = deviations_phi[d:d+1].T  # T x 1
                    v_d = torch.cholesky_solve(dev_d, L_phi)  # T x 1
                    gp_loss += 0.5 * torch.sum(v_d.T @ dev_d)

            except RuntimeError as e:
                print(f"Warning: GP prior computation failed for topic {k}. Using fallback.")
                # Fallback to simple squared error
                gp_loss += torch.sum(deviations_lambda ** 2) + torch.sum(deviations_phi ** 2)

        return gp_loss

    def fit(self, event_times, num_epochs=100, learning_rate=1e-3, lambda_reg=1e-2):
        """
        Fit the model using gradient descent with L2 regularization on gamma.
        """
        optimizer = optim.Adam([
            {'params': [self.lambda_, self.phi, self.length_scales, self.log_amplitudes]},
            {'params': [self.gamma], 'weight_decay': lambda_reg}
        ], lr=learning_rate)

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





