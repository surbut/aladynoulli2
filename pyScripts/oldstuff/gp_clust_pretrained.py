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






class AladynSurvivalPretrainedModel(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, prevalence_t, 
                 pretrained_phi=None, pretrained_gamma=None, pretrained_psi=None):
        super().__init__()
        self.N = N
        self.D = D
        self.T = T
        self.K = K
        self.P = P
        self.jitter = 1e-6

        # Convert inputs to tensors
        self.G = torch.tensor(G, dtype=torch.float32)
        G_centered = G - G.mean(axis=0, keepdims=True)
        G_scaled = G_centered / G_centered.std(axis=0, keepdims=True)
        self.G = torch.tensor(G_scaled, dtype=torch.float32)
        
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        self.logit_prev_t = torch.logit(self.prevalence_t)
        
        # Store pretrained parameters
        self.pretrained_phi = pretrained_phi
        self.pretrained_gamma = pretrained_gamma
        self.pretrained_psi = pretrained_psi
        
        # Fixed kernel parameters (same as original)
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
        self.amplitude = 1

        # Initialize kernels
        self.update_kernels()
        self.initialize_params()
        
    def initialize_params(self):
        """Initialize only lambda parameters, using pretrained phi and gamma"""
        if any(param is None for param in [self.pretrained_phi, self.pretrained_gamma, self.pretrained_psi]):
            raise ValueError("All pretrained parameters must be provided")
            
        # Convert pretrained parameters to torch parameters (but frozen)
        self.phi = nn.Parameter(torch.tensor(self.pretrained_phi), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(self.pretrained_gamma), requires_grad=False)
        self.psi = nn.Parameter(torch.tensor(self.pretrained_psi), requires_grad=False)
        
        # Initialize lambda for new individuals
        lambda_init = torch.zeros((self.N, self.K, self.T))
        
        # Initialize lambda using pretrained gamma
        for k in range(self.K):
            lambda_means = self.G @ self.gamma[:, k]
            L_k = torch.linalg.cholesky(self.K_lambda[k])
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = lambda_means[i] + eps
        
        # Only lambda is trainable
        self.lambda_ = nn.Parameter(lambda_init)
        print("Pretrained model stats:")
        print(f"G stats: mean={self.G.mean():.3f}, std={self.G.std():.3f}")
        print(f"gamma stats: mean={self.gamma.mean():.3f}, std={self.gamma.std():.3f}")
        print(f"Initial lambda stats: mean={self.lambda_.mean():.3f}, std={self.lambda_.std():.3f}")
        print(f"phi stats: mean={self.phi.mean():.3f}, std={self.phi.std():.3f}")
        
        print("Initialization complete! Only lambda parameters will be updated.")

    def update_kernels(self):
        """Compute fixed covariance matrices"""
        times = torch.arange(self.T, dtype=torch.float32)
        sq_dists = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2
        
        # Target condition number
        max_condition = 1e4
        
        self.K_lambda = []
        self.K_phi = []
        
        # Compute kernels with fixed parameters
        K_lambda = self.amplitude ** 2 * torch.exp(-0.5 * sq_dists / (self.lambda_length_scale ** 2))
        K_phi = self.amplitude ** 2 * torch.exp(-0.5 * sq_dists / (self.phi_length_scale ** 2))
        
            # Use fixed small jitter
        jitter = 1e-6
        jitter_matrix = jitter * torch.eye(self.T)

        # Create kernels without adaptive jitter
        self.K_lambda = [K_lambda + jitter_matrix] * self.K
        self.K_phi = [K_phi + jitter_matrix] * self.K

        # Optional: Print condition numbers to verify they're reasonable
        print(f"Lambda kernel condition number: {torch.linalg.cond(self.K_lambda[0]):.2f}")
        print(f"Phi kernel condition number: {torch.linalg.cond(self.K_phi[0]):.2f}")

        """ 
        # Add jitter to each kernel
        for K, name in [(K_lambda, 'lambda'), (K_phi, 'phi')]:
            jitter = 1e-6
            while True:
                K_test = K + jitter * torch.eye(self.T)
                cond = torch.linalg.cond(K_test)
                if cond < max_condition:
                    break
                jitter *= 2
                if jitter > 0.1:
                    print(f"Warning: Large jitter needed for {name} kernel")
                    break
            
            if name == 'lambda':
                self.K_lambda = [K + jitter * torch.eye(self.T)] * self.K
            else:
                self.K_phi = [K + jitter * torch.eye(self.T)] * self.K
        """

    def forward(self):
        theta = torch.softmax(self.lambda_, dim=1)
        epsilon=1e-8
        phi_prob = torch.sigmoid(self.phi)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob)
        pi = torch.clamp(pi, epsilon, 1-epsilon)
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
        event_times_expanded = event_times_tensor.unsqueeze(-1)  # N x D x 1
        time_grid = torch.arange(T).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T
        
        # Mask for times before the event, # Masks automatically handle right-censoring because event_times = T
        mask_before_event = (time_grid < event_times_expanded).float()  # N x D x T
        # Mask for event time
        mask_at_event = (time_grid == event_times_expanded).float()  # N x D x T

        # Check shapes
        print(f"mask_before_event shape: {mask_before_event.shape}")
        print(f"mask_at_event shape: {mask_at_event.shape}")
       
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
          # Normalize by N (total number of individuals)
        total_data_loss = (loss_censored + loss_event + loss_no_event) / (self.N)
    
        # GP prior loss remains the same
        gp_loss = self.compute_gp_prior_loss()
        
        # Add clustering regularization
        #psi_reg = 0.1 * torch.norm(self.psi, p=1)  # L1 regularization to encourage sparsity
        
        total_loss = total_data_loss + gp_loss 
        return total_loss 
    
    def compute_gp_prior_loss(self):
        """
        Compute only the lambda GP prior loss since phi is fixed.
        """
        gp_loss_lambda = 0.0
        
        for k in range(self.K):
            L_lambda = torch.linalg.cholesky(self.K_lambda[k])
            
            # Lambda GP prior only
            lambda_k = self.lambda_[:, k, :]
            mean_lambda_k = (self.G @ self.gamma[:, k]).unsqueeze(1)
            deviations_lambda = lambda_k - mean_lambda_k
            for i in range(self.N):
                dev_i = deviations_lambda[i:i+1].T
                v_i = torch.cholesky_solve(dev_i, L_lambda)
                gp_loss_lambda += 0.5 * torch.sum(v_i.T @ dev_i)
        
        return gp_loss_lambda / self.N

    def fit(self, event_times, num_epochs=1000, learning_rate=1e-4,
            convergence_threshold=1e-3, patience=10):
        """Modified fit method that only updates lambda parameters"""
        
        # Only optimize lambda parameters
        optimizer = optim.Adam([{'params': [self.lambda_], 'lr': learning_rate}])
        
        history = {
            'loss': [],
            'max_grad_lambda': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        prev_loss = float('inf')
        
        print("Starting training (lambda parameters only)...")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss_val = loss.item()
            history['loss'].append(loss_val)
            loss.backward()
            
            # Track only lambda gradients
            grad_lambda = self.lambda_.grad.abs().max().item()
            history['max_grad_lambda'].append(grad_lambda)
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val:.4f}, "
                      f"Lambda gradient: {grad_lambda:.3e}")
            
           # Enhanced early stopping checks
            relative_change = abs(prev_loss - loss_val) / (prev_loss + 1e-10)
            
            if loss_val < best_loss:
                patience_counter = 0
                best_loss = loss_val
            elif relative_change < convergence_threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Relative change in loss: {relative_change:.2e}")
                    break
            else:
                patience_counter = 0  # Reset if we have significant change
         
            
            optimizer.step()
            prev_loss = loss_val
        
        return history

    # Other methods (forward, compute_loss, etc.) remain the same as they don't need modification

