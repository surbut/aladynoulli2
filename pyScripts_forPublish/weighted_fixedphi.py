import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns

class AladynSurvivalFixedPhi_weighted(nn.Module):
    """
    Weighted version of Aladyn with fixed phi/psi using inverse probability weights (IPW)
    to correct for selection bias in UK Biobank data.
    
    Only lambda (individual genetic effects) are estimated. Phi and psi are fixed from 
    a pretrained model.
    """
    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 pretrained_phi, pretrained_psi, signature_references=None, healthy_reference=None, 
                 disease_names=None, flat_lambda=False, weights=None):
        super().__init__()
        # Basic dimensions and settings
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.gpweight = W
        self.jitter = 1e-6
        self.lrtpen = R
        
        # ============================================================
        # HANDLE WEIGHTS (NEW!)
        # ============================================================
        if weights is not None:
            if isinstance(weights, np.ndarray):
                self.weights = torch.tensor(weights, dtype=torch.float32)
            elif isinstance(weights, torch.Tensor):
                self.weights = weights.float()
            else:
                self.weights = torch.tensor(weights, dtype=torch.float32)
            
            # Normalize weights to sum to N (for proper loss scaling)
            self.weights = self.weights * (N / self.weights.sum())
            
            print(f"\n{'='*60}")
            print(f"INVERSE PROBABILITY WEIGHTING ENABLED")
            print(f"{'='*60}")
            print(f"Weight statistics:")
            print(f"  Mean: {self.weights.mean():.3f}")
            print(f"  Std: {self.weights.std():.3f}")
            print(f"  Min: {self.weights.min():.3f}")
            print(f"  Max: {self.weights.max():.3f}")
            
            # Calculate effective sample size
            eff_n = (self.weights.sum() ** 2) / (self.weights ** 2).sum()
            print(f"\nEffective sample size: {eff_n:.0f} out of {N}")
            print(f"Efficiency: {100 * eff_n / N:.1f}%")
            print(f"{'='*60}\n")
        else:
            self.weights = torch.ones(N, dtype=torch.float32)
            print("No weights provided - using unweighted model")
        
        # Fixed kernel parameters
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
        self.init_amplitude = 1.0
        self.lambda_amplitude_init = init_sd_scaler
        self.phi_amplitude_init = init_sd_scaler
        self.kappa = nn.Parameter(torch.ones(1))
        
        # Store base kernel matrix
        time_points = torch.arange(T, dtype=torch.float32)
        time_diff = time_points[:, None] - time_points[None, :]
        self.base_K_lambda = torch.exp(-0.5 * (time_diff**2) / (self.lambda_length_scale**2))
        self.base_K_phi = torch.exp(-0.5 * (time_diff**2) / (self.phi_length_scale**2))
        
        self.K_lambda_init = (self.lambda_amplitude_init**2) * self.base_K_lambda + self.jitter * torch.eye(T)
        self.K_phi_init = (self.phi_amplitude_init**2) * self.base_K_phi + self.jitter * torch.eye(T)
        self.phi_amplitude = 1
        self.lambda_amplitude = 1
        
        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(self.T)
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(self.T)
        
        # Store prevalence and compute logit
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )
        
        # Store pretrained phi and psi (will not be updated)
        self.register_buffer('phi', torch.tensor(pretrained_phi, dtype=torch.float32))
        self.register_buffer('psi', torch.tensor(pretrained_psi, dtype=torch.float32))
        self.phi_gp_loss = self._calculate_phi_gp_loss()
        print(f"Pre-calculated phi GP loss: {self.phi_gp_loss:.4f}")
        
        # Handle signature references
        if flat_lambda:
            self.signature_refs = torch.zeros(K)
            self.genetic_scale = genetic_scale
        else:
            if signature_references is None:
                raise ValueError("signature_references must be provided when flat_lambda=False")
            self.signature_refs = torch.tensor(signature_references, dtype=torch.float32)
            self.genetic_scale = genetic_scale
            
        if healthy_reference is not None:
            self.healthy_ref = torch.tensor(-5.0, dtype=torch.float32)
        else:
            self.healthy_ref = None
            
        # Convert inputs to tensors
        self.G = torch.tensor(G, dtype=torch.float32)
        G_centered = G - G.mean(axis=0, keepdims=True)
        G_scaled = G_centered / G_centered.std(axis=0, keepdims=True)
        self.G = torch.tensor(G_scaled, dtype=torch.float32)
        
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
        # Initialize only individual-specific parameters (lambda, gamma)
        self.initialize_params()

    def _calculate_phi_gp_loss(self):
        """Calculate phi GP loss once at initialization"""
        gp_loss_phi = 0.0
        
        L_phi = torch.linalg.cholesky(self.K_phi)
        for k in range(self.K_total):
            phi_k = self.phi[k]
            for d in range(self.D):
                mean_phi_d = self.logit_prev_t[d, :] + self.psi[k, d]
                dev_d = (phi_k[d:d+1, :] - mean_phi_d).T
                v_d = torch.cholesky_solve(dev_d, L_phi)
                gp_loss_phi += 0.5 * torch.sum(v_d.T @ dev_d)
        
        return gp_loss_phi / self.D

    def initialize_params(self, **kwargs):
        """Initialize only individual-specific parameters (lambda, gamma)"""
        gamma_init = torch.zeros((self.P, self.K_total))
        lambda_init = torch.zeros((self.N, self.K_total, self.T))

        # Initialize lambda and gamma for disease clusters
        for k in range(self.K):
            Y_avg = torch.mean(self.Y, dim=2)
            strong_diseases = (self.psi[k] > 0).float()
            base_value = Y_avg[:, strong_