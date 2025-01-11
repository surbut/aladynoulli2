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
        self.length_scales = nn.Parameter(torch.tensor(np.full(K, T / 3), dtype=torch.float32))
        self.log_amplitudes = nn.Parameter(torch.zeros(K, dtype=torch.float32))  # Start at amplitude = 1.0

        """
        # Initialize kernel parameters in log space with informative priors
        self.length_scales = nn.Parameter(torch.log(torch.tensor(np.full(K, T/3), dtype=torch.float32)))
        self.log_amplitudes = nn.Parameter(torch.zeros(K, dtype=torch.float32))
        
        # Prior hyperparameters
        self.length_scale_prior_mean = np.log(T/3)  # Center around T/3
        self.length_scale_prior_std = 0.5           # Relatively tight prior
        self.amplitude_prior_mean = 0.0             # Log space, so exp(0) = 1
        self.amplitude_prior_std = 0.5              # Allow reasonable variation
        """

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
        """Update covariance matrices with condition number control"""
        times = torch.arange(self.T, dtype=torch.float32)
        sq_dists = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2
        
        # Target condition number
        max_condition = 1e4
        
        self.K_lambda = []
        self.K_phi = []
        
        for k in range(self.K):
            # More restrictive bounds on parameters
            length_scale = torch.clamp(self.length_scales[k], min=max(1.0, self.T/20), max=self.T/2)
            amplitude = torch.exp(torch.clamp(self.log_amplitudes[k], min=-2.0, max=1.0))
            
            # Compute base kernel
            K = amplitude ** 2 * torch.exp(-0.5 * sq_dists / (length_scale ** 2))
            
            # Add adaptive jitter based on condition number
            jitter = 1e-4
            while True:
                K_test = K + jitter * torch.eye(self.T)
                cond = torch.linalg.cond(K_test)
                if cond < max_condition:
                    break
                jitter *= 2
                if jitter > 0.1:  # Emergency break
                    print(f"Warning: Large jitter needed for topic {k}")
                    break
            
            K = K + jitter * torch.eye(self.T)
            self.K_lambda.append(K)
            self.K_phi.append(K.clone())

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

        # Mask for times before the event, # Masks automatically handle right-censoring because event_times = T
        mask_before_event = (time_grid < event_times_expanded).float()  # N x D x T
        # Mask for event time
        mask_at_event = (time_grid == event_times_expanded).float()  # N x D x T

        # Compute loss components
         # Loss components work automatically because:
    # 1. Right-censored (E=T-1, Y=0): contributes to survival up to T-1 and no-event at T-1
    # 2. Events (E<T-1, Y=1): contributes to survival up to E and event at E
    # 3. Early censoring (E<T-1, Y=0): contributes to survival up to E and no-event at E

        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))

        total_data_loss = loss_censored + loss_event + loss_no_event

        # GP prior loss remains the same
        gp_loss = self.compute_gp_prior_loss()
        total_loss = total_data_loss + gp_loss
        return total_loss

    def compute_gp_prior_loss(self):
        """
        Compute the GP prior loss for lambda and phi using Cholesky decomposition.
        """
        print("Shape checks:")
        print(f"lambda_: {self.lambda_.shape}")  # Should be [N, K, T]
        print(f"phi: {self.phi.shape}")         # Should be [K, D, T]
        print(f"K_lambda[0]: {self.K_lambda[0].shape}")  # Should be [T, T]
        print(f"K_phi[0]: {self.K_phi[0].shape}")       # Should be [T, T]
        print(f"G: {self.G.shape}")             # Should be [N, P]
        print(f"gamma: {self.gamma.shape}")      # Should be [P, K]
        print(f"logit_prev: {self.logit_prev.shape}")  # Should be [D]
        gp_loss = 0.0
        N, K, T = self.lambda_.shape
        K, D, T = self.phi.shape

        for k in range(K):
            # Cholesky decomposition
            L_lambda = torch.linalg.cholesky(self.K_lambda[k])  # T x T
            L_phi = torch.linalg.cholesky(self.K_phi[k])       # T x T

            # Lambda GP prior
            lambda_k = self.lambda_[:, k, :]  # N x T
            mean_lambda_k = (self.G @ self.gamma[:, k]).unsqueeze(1)  # N x 1
            deviations_lambda = lambda_k - mean_lambda_k  # N x T

            # Process each individual separately to maintain correct dimensions
            for i in range(N):
                dev_i = deviations_lambda[i:i+1].T  # T x 1
                v_i = torch.cholesky_solve(dev_i, L_lambda)  # T x 1
                gp_loss += 0.5 * torch.sum(v_i.T @ dev_i)

            # Phi GP prior
            phi_k = self.phi[k, :, :]  # D x T
            mean_phi_k = self.logit_prev.unsqueeze(1)  # D x 1
            deviations_phi = phi_k - mean_phi_k  # D x T

            # Process each disease separately
            for d in range(D):
                dev_d = deviations_phi[d:d+1].T  # T x 1
                v_d = torch.cholesky_solve(dev_d, L_phi)  # T x 1
                gp_loss += 0.5 * torch.sum(v_d.T @ dev_d)

        return gp_loss

    def fit(self, event_times, num_epochs=1000,  learning_rate=1e-3, lambda_reg=1e-2, tol=1e-4, patience=5):
        """
        Fit model with detailed monitoring of parameters and stability
        """
        optimizer = optim.Adam([
            {'params': [self.lambda_, self.phi, self.length_scales, self.log_amplitudes]},
            {'params': [self.gamma], 'weight_decay': lambda_reg}
        ], lr=learning_rate)
        
        # Initialize tracking
        history = {
            'loss': [],
            'length_scales': [],
            'amplitudes': [],
            'max_grad_lambda': [],
            'max_grad_phi': [],
            'max_grad_gamma': [],
            'condition_number': []  # Track matrix conditioning
        }
        
        best_loss = float('inf')
        no_improve_count = 0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # 1. Monitor GP parameters before update
            history['length_scales'].append(self.length_scales.detach().numpy().copy())
            history['amplitudes'].append(torch.exp(self.log_amplitudes).detach().numpy().copy())
            
            # 2. Compute loss and update
            loss = self.compute_loss(event_times)
            loss.backward()
            
            # 3. Monitor gradients
            history['max_grad_lambda'].append(self.lambda_.grad.abs().max().item())
            history['max_grad_phi'].append(self.phi.grad.abs().max().item())
            history['max_grad_gamma'].append(self.gamma.grad.abs().max().item())
            
            # 4. Monitor kernel condition numbers
            cond_nums = []
            for k in range(self.K):
                cond = torch.linalg.cond(self.K_lambda[k]).item()
                cond_nums.append(cond)
            history['condition_number'].append(np.mean(cond_nums))
            
            # 5. Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            current_loss = loss.item()
            history['loss'].append(loss.item())

            rel_change = abs(best_loss - current_loss) / (abs(best_loss) + 1e-10)

            if current_loss < best_loss:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 6. Print detailed monitoring every N epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Relative change: {rel_change:.6f}")
                print(f"No improvement count: {no_improve_count}")
                print(f"Length scales: {self.length_scales.detach().numpy()}")
                print(f"Amplitudes: {torch.exp(self.log_amplitudes).detach().numpy()}")
                print(f"Max gradients - λ: {history['max_grad_lambda'][-1]:.4f}, "
                    f"φ: {history['max_grad_phi'][-1]:.4f}, "
                    f"γ: {history['max_grad_gamma'][-1]:.4f}")
                print(f"Mean condition number: {history['condition_number'][-1]:.2f}")
             # Stop if converged
            if rel_change < tol:
                print(f"\nConverged at epoch {epoch} - Relative change {rel_change:.6f} < {tol}")
            break
            
        # Early stopping if no improvement
            if no_improve_count >= patience:
                print(f"\nStopping early - No improvement for {patience} epochs")
            break
        
        return history

def plot_training_diagnostics(history):
    """Plot training diagnostics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0,0].plot(history['loss'])
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_yscale('log')
    
    # GP Parameters
    length_scales = np.array(history['length_scales'])
    for k in range(length_scales.shape[1]):
        axes[0,1].plot(length_scales[:,k], label=f'Topic {k}')
    axes[0,1].set_title('Length Scales')
    axes[0,1].legend()
    
    # Gradients
    axes[1,0].plot(history['max_grad_lambda'], label='λ')
    axes[1,0].plot(history['max_grad_phi'], label='φ')
    axes[1,0].plot(history['max_grad_gamma'], label='γ')
    axes[1,0].set_title('Max Gradients')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    
    # Condition Numbers
    axes[1,1].plot(history['condition_number'])
    axes[1,1].set_title('Kernel Condition Numbers')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

# Usage:
"""
history = model.fit(event_times, num_epochs=1000)
plot_training_diagnostics(history)
"""


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
"""
pi_pred, theta_pred, phi_pred = model.forward()
plot_best_matches(true_pi, pi_pred, n_samples=10, n_cols=2)
"""