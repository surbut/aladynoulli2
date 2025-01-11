import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class AladynoulliModel:
    def __init__(self, N, D, T, K, P, G, Y, prevalence, length_scale_prior, device='cpu'):
        # Dimensions
        self.N = N
        self.D = D
        self.T = T
        self.K = K
        self.P = P
        self.device = device
        
        # Data
        self.G = torch.tensor(G, dtype=torch.float32, device=device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=device)
        self.prevalence = torch.tensor(prevalence, dtype=torch.float32, device=device)
        
        # GP parameters
        self.jitter = 1e-6
        self.time_points = torch.linspace(0, 1, T, device=device)
        self.length_scale_alpha, self.length_scale_beta = length_scale_prior
        
        # Initialize model parameters
        self.B = torch.nn.Parameter(
            torch.randn(P, K, device=device) / np.sqrt(P)
        )
        self.U = torch.nn.Parameter(
            torch.randn(D, K, device=device) / np.sqrt(D)
        )
        self.log_length_scale = torch.nn.Parameter(
            torch.zeros(K, device=device)
        )
        self.log_variance = torch.nn.Parameter(
            torch.zeros(K, device=device)
        )
        
        # Set up GP kernels
        self.setup_gp_kernels()
        
    def setup_gp_kernels(self):
        """Setup GP kernels for each signature"""
        # Time differences matrix
        t = self.time_points.view(-1, 1)
        self.time_diffs = t - t.T
        
        # Initialize kernel matrices
        self.K_t = torch.zeros(self.K, self.T, self.T, device=self.device)
        self.update_kernels()
    
    def update_kernels(self):
        """Update GP kernel matrices"""
        for k in range(self.K):
            # RBF kernel
            length_scale = torch.exp(self.log_length_scale[k])
            variance = torch.exp(self.log_variance[k])
            
            # Compute kernel matrix
            scaled_diffs = self.time_diffs / length_scale
            self.K_t[k] = variance * torch.exp(-0.5 * scaled_diffs**2)
            
            # Add jitter for numerical stability
            self.K_t[k] += self.jitter * torch.eye(self.T, device=self.device)
    
    def forward(self):
        """Compute model predictions"""
        # Update GP kernels
        self.update_kernels()
        
        # Compute genetic effects
        genetic_effects = self.G @ self.B  # N × K
        
        # Sample GP functions
        f = torch.zeros(self.K, self.T, device=self.device)
        for k in range(self.K):
            L = torch.linalg.cholesky(self.K_t[k])
            eps = torch.randn(self.T, device=self.device)
            f[k] = L @ eps
        
        # Compute disease trajectories
        disease_trajectories = self.U @ f  # D × T
        
        # Compute final probabilities
        logits = torch.einsum('nk,kt->ndt', genetic_effects, f)
        probs = torch.sigmoid(logits)
        
        return probs
    
    def compute_loss(self):
        """Compute negative log likelihood and priors"""
        probs = self.forward()
        
        # Likelihood
        likelihood = torch.sum(
            self.Y * torch.log(probs + 1e-10) +
            (1 - self.Y) * torch.log(1 - probs + 1e-10)
        )
        
        # Priors
        length_scale_prior = torch.sum(
            -self.length_scale_alpha * self.log_length_scale +
            self.length_scale_beta * torch.exp(-self.log_length_scale)
        )
        
        # Regularization
        reg = -0.5 * (torch.sum(self.B**2) + torch.sum(self.U**2))
        
        return -(likelihood + length_scale_prior + reg)

# Initialize model
try:
    print("Initializing model...")
    model = AladynoulliModel(
        N=N,
        D=D,
        T=T,
        K=K,
        P=P,
        G=G,
        Y=Y,
        prevalence=prevalence,
        length_scale_prior=(2.0, 5.0)
    )
    print("Model initialized successfully!")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    print(f"Data shapes:")
    print(f"Y shape: {Y.shape}")
    print(f"G shape: {G.shape}")
    print(f"Prevalence shape: {prevalence.shape}")




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional

class AladynoulliModel(nn.Module):
    def __init__(
        self,
        N: int,  # Number of individuals
        D: int,  # Number of diseases
        T: int,  # Number of timepoints
        K: int,  # Number of topics
        P: int,  # Number of genetic covariates
        G: np.ndarray,  # Genetic data (N x P)
        Y: np.ndarray,  # Disease occurrence data (N x D x T)
        prevalence: np.ndarray,  # Disease prevalence (D,)
        length_scale_prior: Tuple[float, float] = (1.0, 10.0),  # Shape, scale for Gamma prior
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.N, self.D, self.T, self.K, self.P = N, D, T, K, P
        self.device = device
        
        # Convert inputs to tensors and move to device
        self.G = torch.tensor(G, dtype=torch.float32, device=device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=device)
        self.prevalence = torch.tensor(prevalence, dtype=torch.float32, device=device)
        self.logit_prev = torch.log(self.prevalence / (1 - self.prevalence))
        
        # Initialize model parameters
        self.initialize_parameters()
        
        # Set up GP kernels
        self.setup_gp_kernels()
        
        # Prior parameters
        self.length_scale_alpha, self.length_scale_beta = length_scale_prior
        
        # Constants for numerical stability
        self.eps = 1e-8
        self.jitter = 1e-6

    def initialize_parameters(self):
        """Initialize model parameters using SVD and careful scaling"""
        # Compute average disease occurrence matrix
        Y_avg = torch.mean(self.Y, dim=2)
        
        # SVD initialization for lambda and gamma
        U, S, Vh = torch.linalg.svd(Y_avg, full_matrices=False)
        lambda_init = U[:, :self.K] @ torch.diag(torch.sqrt(S[:self.K]))
        gamma_init = torch.linalg.lstsq(self.G, lambda_init).solution
        
        # Initialize parameters with careful scaling
        self.gamma = nn.Parameter(0.1 * gamma_init)
        
        # Initialize GP hyperparameters
        self.log_length_scales = nn.Parameter(torch.log(torch.full((self.K,), self.T/10.0, device=self.device)))
        self.log_amplitudes = nn.Parameter(torch.zeros(self.K, device=self.device))
        
        # Initialize latent functions
        self.lambda_ = nn.Parameter(0.1 * torch.randn(self.N, self.K, self.T, device=self.device))
        self.phi = nn.Parameter(0.1 * torch.randn(self.K, self.D, self.T, device=self.device))
        
        # Initialize means
        self.lambda_means = self.G @ self.gamma  # N x K
        self.phi_means = self.logit_prev.unsqueeze(0).expand(self.K, -1)  # K x D

    def setup_gp_kernels(self):
        """Set up GP kernel matrices with proper numerical handling"""
        times = torch.arange(self.T, dtype=torch.float32, device=self.device)
        self.sq_dists = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2
        
        # Initialize kernel storage
        self.K_lambda = []
        self.K_phi = []
        self.L_lambda = []
        self.L_phi = []
        
        self.update_kernels()

    def update_kernels(self):
        """Update kernel matrices with current hyperparameters"""
        self.K_lambda = []
        self.K_phi = []
        self.L_lambda = []
        self.L_phi = []
        
        for k in range(self.K):
            # Get hyperparameters
            length_scale = torch.exp(self.log_length_scales[k])
            amplitude = torch.exp(self.log_amplitudes[k])
            
            # Compute base kernel
            K_base = amplitude ** 2 * torch.exp(-0.5 * self.sq_dists / (length_scale ** 2))
            
            # Add jitter for numerical stability
            K = K_base + self.jitter * torch.eye(self.T, device=self.device)
            
            # Ensure symmetry
            K = 0.5 * (K + K.t())
            
            # Compute Cholesky decomposition
            try:
                L = torch.linalg.cholesky(K)
            except RuntimeError:
                # If Cholesky fails, add more jitter
                K = K + 0.1 * torch.eye(self.T, device=self.device)
                L = torch.linalg.cholesky(K)
            
            # Store matrices
            self.K_lambda.append(K)
            self.K_phi.append(K.clone())
            self.L_lambda.append(L)
            self.L_phi.append(L.clone())

    def compute_gp_prior_loss(self) -> torch.Tensor:
        """Compute GP prior loss with complete mathematical formulation"""
        gp_loss = 0.0
        
        # Update means
        self.lambda_means = self.G @ self.gamma
        
        for k in range(self.K):
            # Lambda prior
            for i in range(self.N):
                dev = self.lambda_[i, k] - self.lambda_means[i, k]
                L = self.L_lambda[k]
                
                # Solve system using Cholesky
                v = torch.triangular_solve(dev.unsqueeze(1), L, upper=False)[0]
                
                # Add quadratic term and log determinant
                gp_loss += 0.5 * (torch.sum(v ** 2) + 2 * torch.sum(torch.log(torch.diagonal(L))))
            
            # Phi prior
            for d in range(self.D):
                dev = self.phi[k, d] - self.logit_prev[d]
                L = self.L_phi[k]
                
                v = torch.triangular_solve(dev.unsqueeze(1), L, upper=False)[0]
                gp_loss += 0.5 * (torch.sum(v ** 2) + 2 * torch.sum(torch.log(torch.diagonal(L))))
            
            # Add length scale priors
            gp_loss += torch.distributions.Gamma(
                self.length_scale_alpha, 
                self.length_scale_beta
            ).log_prob(torch.exp(self.log_length_scales[k])).sum()
        
        return gp_loss

    def apply_non_recurrence_mask(self, pi: torch.Tensor, event_times: torch.Tensor) -> torch.Tensor:
        """Apply mask to enforce disease non-recurrence constraint"""
        mask = torch.ones_like(pi)
        for n in range(self.N):
            for d in range(self.D):
                t = event_times[n, d]
                if t < self.T:
                    mask[n, d, t+1:] = 0
        return pi * mask

    def forward(self, event_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing probabilities with proper constraints"""
        # Update kernels if needed
        self.update_kernels()
        
        # Compute theta using numerically stable softmax
        theta = torch.softmax(self.lambda_, dim=1)
        
        # Compute disease probabilities with sigmoid
        phi_prob = torch.sigmoid(self.phi)
        
        # Compute pi
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob)
        
        # Apply non-recurrence constraint
        pi = self.apply_non_recurrence_mask(pi, event_times)
        
        # Clip probabilities for numerical stability
        pi = torch.clamp(pi, self.eps, 1 - self.eps)
        
        return pi, theta, phi_prob

    def compute_survival_likelihood(
        self, 
        pi: torch.Tensor, 
        event_times: torch.Tensor
    ) -> torch.Tensor:
        """Compute survival likelihood with proper handling of events and censoring"""
        # Create time grid and masks
        time_grid = torch.arange(self.T, device=self.device).unsqueeze(0).unsqueeze(0)
        event_times = event_times.unsqueeze(-1)
        
        # Compute masks
        mask_before = (time_grid < event_times).float()
        mask_at = (time_grid == event_times).float()
        
        # Compute likelihood components
        censored_ll = torch.sum(torch.log(1 - pi + self.eps) * mask_before)
        event_ll = torch.sum(torch.log(pi + self.eps) * mask_at * self.Y)
        no_event_ll = torch.sum(torch.log(1 - pi + self.eps) * mask_at * (1 - self.Y))
        
        return -(censored_ll + event_ll + no_event_ll)

    def compute_loss(self, event_times: torch.Tensor) -> torch.Tensor:
        """Compute total loss with all components"""
        # Get probabilities
        pi, theta, phi_prob = self.forward(event_times)
        
        # Compute likelihood
        likelihood_loss = self.compute_survival_likelihood(pi, event_times)
        
        # Compute GP prior loss
        gp_loss = self.compute_gp_prior_loss()
        
        # Add regularization
        l2_reg = 0.01 * (
            torch.norm(self.gamma) + 
            torch.norm(self.lambda_) + 
            torch.norm(self.phi)
        )
        
        return likelihood_loss + gp_loss + l2_reg

    def fit(
        self,
        event_times: torch.Tensor,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> Dict[str, list]:
        """
        Fit model with early stopping and adaptive learning rate
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6
        )
        
        # Initialize tracking variables
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {
            'loss': [],
            'likelihood_loss': [],
            'gp_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_loss(event_times)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Track metrics
            with torch.no_grad():
                pi, _, _ = self.forward(event_times)
                likelihood_loss = self.compute_survival_likelihood(pi, event_times)
                gp_loss = self.compute_gp_prior_loss()
                
                history['loss'].append(loss.item())
                history['likelihood_loss'].append(likelihood_loss.item())
                history['gp_loss'].append(gp_loss.item())
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if loss < best_loss - min_delta:
                best_loss = loss
                best_state = {
                    k: v.cpu().clone().detach() 
                    for k, v in self.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, "
                      f"LL = {likelihood_loss.item():.4f}, "
                      f"GP = {gp_loss.item():.4f}, "
                      f"LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        # Restore best state
        if best_state is not None:
            self.load_state_dict(best_state)
        
        return history

    def predict(
        self,
        G_new: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions for new individuals or new timepoints
        """
        with torch.no_grad():
            if G_new is None:
                G_new = self.G
            if times is None:
                times = torch.arange(self.T, device=self.device)
            
            # Compute lambda means for new individuals
            lambda_means = G_new @ self.gamma
            
            # Initialize predictions
            N_new = G_new.shape[0]
            T_new = len(times)
            
            # Compute kernel for new timepoints if needed
            if T_new != self.T:
                time_diffs = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2
                K_new = []
                for k in range(self.K):
                    length_scale = torch.exp(self.log_length_scales[k])
                    amplitude = torch.exp(self.log_amplitudes[k])
                    K = amplitude ** 2 * torch.exp(-0.5 * time_diffs / (length_scale ** 2))
                    K = K + self.jitter * torch.eye(T_new, device=self.device)
                    K_new.append(K)
            else:
                K_new = self.K_lambda
            
            # Generate predictions
            lambda_pred = torch.zeros((N_new, self.K, T_new), device=self.device)
            for k in range(self.K):
                for i in range(N_new):
                    lambda_pred[i, k] = lambda_means[i, k]
            
            theta_pred = torch.softmax(lambda_pred, dim=1)
            phi_prob_pred = torch.sigmoid(self.phi)
            
            pi_pred = torch.einsum('nkt,kdt->ndt', theta_pred, phi_prob_pred)
            pi_pred = torch.clamp(pi_pred, self.eps, 1 - self.eps)
            
            return pi_pred, theta_pre

