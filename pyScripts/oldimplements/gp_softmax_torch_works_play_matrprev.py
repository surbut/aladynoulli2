import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time

class AladynSurvivalModel(nn.Module):
    def __init__(self, N, D, T, P, G, Y, n_lengths=2, n_amplitudes=2):
        super().__init__()
        self.N, self.D, self.T = N, D, T
        self.K = n_lengths * n_amplitudes  # 12 total topics
        self.P = P

        # Convert inputs to tensors
        self.G = torch.tensor(G, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
        # Compute smoothed prevalence
        prevalence_t = compute_smoothed_prevalence(Y, window_size=5)
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        
        # Compute logit of prevalence for centering phi
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )

        # Create grid
        length_scales = np.array([T/8, T/4, T/2, T])    
        log_amplitudes = np.array([-1, 0, 1])           
        length_grid, amp_grid = np.meshgrid(length_scales, log_amplitudes)
        
        # Store initial values for tracking
        self.initial_lengths = length_grid.flatten()
        self.initial_amps = np.exp(amp_grid.flatten())
        
        # Initialize kernel parameters
        self.lambda_length_scales = nn.Parameter(
            torch.tensor(length_grid.flatten(), dtype=torch.float32)
        )
        self.lambda_log_amplitudes = nn.Parameter(
            torch.tensor(amp_grid.flatten(), dtype=torch.float32)
        )
        self.phi_length_scales = nn.Parameter(
            torch.tensor(length_grid.flatten(), dtype=torch.float32)
        )
        self.phi_log_amplitudes = nn.Parameter(
            torch.tensor(amp_grid.flatten(), dtype=torch.float32)
        )

        # Add timing tracking
        self.kernel_compute_times = []
        
        # Initialize model parameters
        self.initialize_params()

    def initialize_params(self):
        """Initialize parameters using SVD and GP priors"""
        # Compute average disease occurrence matrix (N x D)
        Y_avg = torch.mean(self.Y, dim=2)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(Y_avg, full_matrices=False)

        """Initialize pwith centered data? try?
        ## try with centering?

        disease_means = torch.mean(Y_avg, dim=0)  # D
        Y_centered = Y_avg - disease_means[None, :]  # N x D
    
        # Perform SVD on centered data
        U, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
       unclear if necessary"""
        
        # Initialize gamma using genetic covariates
        lambda_init = U[:, :self.K] @ torch.diag(torch.sqrt(S[:self.K]))
        gamma_init = torch.linalg.lstsq(self.G, lambda_init).solution
        self.gamma = nn.Parameter(gamma_init)

        # Initialize lambda using GP prior
        lambda_means = self.G @ self.gamma
        self.lambda_ = nn.Parameter(torch.zeros((self.N, self.K, self.T)))

        # Initialize phi using GP prior with time-dependent mean
        self.phi = nn.Parameter(torch.zeros((self.K, self.D, self.T)))

        # Initialize covariance matrices
        self.update_kernels()

        # Sample initial values for lambda and phi from the GPs
        for k in range(self.K):
            L_k = torch.linalg.cholesky(self.K_lambda[k])
            L_k_phi = torch.linalg.cholesky(self.K_phi[k])

            # Sample lambda
            for i in range(self.N):
                mean = lambda_means[i, k]
                eps = L_k @ torch.randn(self.T)
                self.lambda_.data[i, k, :] = mean + eps

            # Sample phi with time-dependent mean
            for d in range(self.D):
                mean = self.logit_prev_t[d, :]  # Use time-dependent mean
                eps = L_k_phi @ torch.randn(self.T)
                self.phi.data[k, d, :] = mean + eps

    def update_kernels(self):
        """Update covariance matrices with timing"""
        import time
        start_time = time.time()
        
        times = torch.arange(self.T, dtype=torch.float32)
        sq_dists = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2
        
        self.K_lambda = []
        self.K_phi = []
        
        for k in range(self.K):
            # Lambda kernels
            length_lambda = self.lambda_length_scales[k]
            amp_lambda = torch.exp(self.lambda_log_amplitudes[k])
            K_lambda = amp_lambda ** 2 * torch.exp(-0.5 * sq_dists / (length_lambda ** 2))
            K_lambda += 1e-6 * torch.eye(self.T)  # Small jitter
            self.K_lambda.append(K_lambda)
            
            # Phi kernels
            length_phi = self.phi_length_scales[k]
            amp_phi = torch.exp(self.phi_log_amplitudes[k])
            K_phi = amp_phi ** 2 * torch.exp(-0.5 * sq_dists / (length_phi ** 2))
            K_phi += 1e-6 * torch.eye(self.T)
            self.K_phi.append(K_phi)
        
        compute_time = time.time() - start_time
        self.kernel_compute_times.append(compute_time)

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
    # For times before event/censoring: contribute to survival
        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        # At event time:
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)

        # Example:
# For a patient censored at t=5 (Y[n,d,5] = 0):
#mask_at_event[n,d,:] = [0,0,0,0,0,1,0,0]  # 1 at t=5
#(1 - Y[n,d,:])       = [1,1,1,1,1,1,1,1]  # All 1s because no event
# Result: contributes -log(1-pi[n,d,5]) to loss
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))

        total_data_loss = loss_censored + loss_event + loss_no_event

        # GP prior loss remains the same
        gp_loss = self.compute_gp_prior_loss()
        total_loss = total_data_loss + gp_loss
        return total_loss

    def compute_gp_prior_loss(self):
        """Compute the GP prior loss with time-dependent mean"""
        gp_loss = 0.0
        N, K, T = self.lambda_.shape
        K, D, T = self.phi.shape

        for k in range(K):
            L_lambda = torch.linalg.cholesky(self.K_lambda[k])
            L_phi = torch.linalg.cholesky(self.K_phi[k])

            # Lambda GP prior (unchanged)
            lambda_k = self.lambda_[:, k, :]
            mean_lambda_k = (self.G @ self.gamma[:, k]).unsqueeze(1)
            deviations_lambda = lambda_k - mean_lambda_k

            for i in range(N):
                dev_i = deviations_lambda[i:i+1].T
                v_i = torch.cholesky_solve(dev_i, L_lambda)
                gp_loss += 0.5 * torch.sum(v_i.T @ dev_i)

            # Phi GP prior with time-dependent mean
            phi_k = self.phi[k, :, :]  # D x T
            for d in range(D):
                mean_phi_d = self.logit_prev_t[d, :]  # Use time-dependent mean
                dev_d = (phi_k[d:d+1, :] - mean_phi_d).T  # T x 1
                v_d = torch.cholesky_solve(dev_d, L_phi)
                gp_loss += 0.5 * torch.sum(v_d.T @ dev_d)

        return gp_loss

    def fit(self, event_times, num_epochs=5000, learning_rate=1e-3, lambda_reg=1e-2,
            convergence_threshold=1.0, patience=50):
        
        # Initialize optimizer with different learning rates for different parameter groups
        optimizer = optim.Adam([
            {'params': [self.lambda_], 'lr': learning_rate},
            {'params': [self.phi], 'lr': learning_rate * 0.5},
            {'params': [self.lambda_length_scales, self.lambda_log_amplitudes], 
             'lr': learning_rate * 0.1},
            {'params': [self.phi_length_scales, self.phi_log_amplitudes], 
             'lr': learning_rate * 0.1},
            {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': learning_rate}
        ])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,      # Halve LR when plateauing
            patience=20,     # Wait 20 epochs before reducing
            verbose=True,    # Print LR changes
            min_lr=1e-6     # Minimum LR threshold
        )
        
        # Initialize tracking
        history = {
            'loss': [],
            'learning_rates': [],
            'phi_deviations': [],
            'gradient_norms': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        prev_loss = float('inf')
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward and backward pass
            loss = self.compute_loss(event_times)
            loss_val = loss.item()
            loss.backward()
            
            # Track metrics
            deviations = self.phi - self.logit_prev_t
            history['phi_deviations'].append({
                'mean': deviations.mean().item(),
                'std': deviations.std().item()
            })
            history['gradient_norms'].append({
                'phi': self.phi.grad.abs().mean().item(),
                'lambda': self.lambda_.grad.abs().mean().item()
            })
            history['learning_rates'].append(
                [group['lr'] for group in optimizer.param_groups]
            )
            history['loss'].append(loss_val)
            
            # Update parameters
            optimizer.step()
            scheduler.step(loss_val)
            
            # Check convergence
            loss_change = abs(prev_loss - loss_val)
            if loss_change < convergence_threshold:
                print(f"\nConverged at epoch {epoch}. Loss change: {loss_change:.4f}")
                break
            
            # Early stopping check
            if loss_val < best_loss:
                patience_counter = 0
                best_loss = loss_val
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            prev_loss = loss_val
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss_val:.4f}")
                print(f"Loss change: {loss_change:.4f}")
                print(f"Mean phi deviation: {deviations.mean():.4f}")
                print(f"Phi gradient mean: {self.phi.grad.abs().mean():.4f}")
                print(f"Learning rates: {[f'{lr:.2e}' for lr in history['learning_rates'][-1]]}")
                # Track parameter evolution
                self._print_parameter_evolution()
                # Print timing stats
                avg_kernel_time = np.mean(self.kernel_compute_times[-100:])
                print(f"Average kernel computation time: {avg_kernel_time:.4f}s")
        
        return history

    def _print_parameter_evolution(self):
        """Track how parameters evolve from initialization"""
        current_lengths_lambda = self.lambda_length_scales.detach()
        current_amps_lambda = torch.exp(self.lambda_log_amplitudes.detach())
        current_lengths_phi = self.phi_length_scales.detach()
        current_amps_phi = torch.exp(self.phi_log_amplitudes.detach())
        
        print("\nParameter evolution from grid points:")
        print("\nLambda kernels:")
        for k in range(self.K):
            print(f"Topic {k}:")
            print(f"  Length: {self.initial_lengths[k]:.1f} → {current_lengths_lambda[k]:.1f}")
            print(f"  Amplitude: {self.initial_amps[k]:.1f} → {current_amps_lambda[k]:.1f}")
        
        print("\nPhi kernels:")
        for k in range(self.K):
            print(f"Topic {k}:")
            print(f"  Length: {self.initial_lengths[k]:.1f} → {current_lengths_phi[k]:.1f}")
            print(f"  Amplitude: {self.initial_amps[k]:.1f} → {current_amps_phi[k]:.1f}")

## plotting code from here down
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
# Example of preparing smoothed time-dependent prevalence
def compute_smoothed_prevalence(Y, window_size=5):
    """Compute smoothed time-dependent prevalence"""
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    
    for d in range(D):
        # Compute raw prevalence at each time point
        raw_prev = Y[:, d, :].mean(axis=0)
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        prevalence_t[d, :] = gaussian_filter1d(raw_prev, sigma=window_size)
    
    return prevalence_t

# When initializing the model:
#prevalence_t = compute_smoothed_prevalence(Y, window_size=5)
#model = AladynSurvivalModel(N, D, T, K, P, G, Y, prevalence_t)