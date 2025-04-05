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

class AladynSurvivalFixedPhi(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 pretrained_phi, pretrained_psi, signature_references=None, healthy_reference=None, 
                 disease_names=None, flat_lambda=False):
        super().__init__()
        # Basic dimensions and settings
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.gpweight = W
        self.jitter = 1e-6
        self.lrtpen = R  # Stronger LRT penalty
        
        # Fixed kernel parameters
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
        self.init_amplitude = 1.0  # Fixed initial amplitude for both kernels
        self.lambda_amplitude_init = init_sd_scaler  # For lambda initialization
        self.phi_amplitude_init = init_sd_scaler
        self.kappa = nn.Parameter(torch.ones(1))  # Single global calibration parameter
        
        # Store base kernel matrix (structure without amplitude)
        time_points = torch.arange(T, dtype=torch.float32)
        time_diff = time_points[:, None] - time_points[None, :]
        self.base_K_lambda = torch.exp(-0.5 * (time_diff**2) / (self.lambda_length_scale**2))
        self.base_K_phi = torch.exp(-0.5 * (time_diff**2) / (self.phi_length_scale**2))
        
        # Initialize kernels with same initial amplitude
        self.K_lambda_init = (self.lambda_amplitude_init**2) * self.base_K_lambda + self.jitter * torch.eye(T)
        self.K_phi_init = (self.phi_amplitude_init**2) * self.base_K_phi + self.jitter * torch.eye(T)
        self.phi_amplitude = 1
        self.lambda_amplitude = 1

        # Add jitter and store
        jitter_matrix = self.jitter * torch.eye(T)

        # Fixed Phi kernel (not updated during training)
        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(self.T)
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(self.T)
        
        # Store pretrained phi and psi (will not be updated)
        self.register_buffer('phi', torch.tensor(pretrained_phi, dtype=torch.float32))
        self.register_buffer('psi', torch.tensor(pretrained_psi, dtype=torch.float32))
        self.disease_names = disease_names
          
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
        
        # Store prevalence and compute logit
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )  # D x T
        
        # Initialize only individual-specific parameters (lambda, gamma)
        self.initialize_params()

    def initialize_params(self, **kwargs):
        """Initialize only individual-specific parameters (lambda, gamma)"""
        # Initialize gamma for disease clusters
        gamma_init = torch.zeros((self.P, self.K_total))
        lambda_init = torch.zeros((self.N, self.K_total, self.T))

        # Initialize lambda and gamma for disease clusters
        for k in range(self.K):
            print(f"\nCalculating gamma for k={k}:")
            
            # Use average Y values as a basis for gamma initialization
            Y_avg = torch.mean(self.Y, dim=2)
            
            # Use diseases with strong psi values for this signature to initialize gamma
            strong_diseases = (self.psi[k] > 0).float()
            base_value = Y_avg[:, strong_diseases > 0].mean(dim=1)
            base_value_centered = base_value - base_value.mean()
            
            print(f"Number of diseases in signature: {strong_diseases.sum()}")
            print(f"Base value (first 5): {base_value[:5]}")
            print(f"Base value centered (first 5): {base_value_centered[:5]}")
            print(f"Base value centered mean: {base_value_centered.mean()}")
  
            gamma_init[:, k] = torch.linalg.lstsq(self.G, base_value_centered.unsqueeze(1)).solution.squeeze()
            print(f"Gamma init for k={k} (first 5): {gamma_init[:5, k]}")
            
            lambda_means = self.genetic_scale * (self.G @ gamma_init[:, k])
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = self.signature_refs[k] + lambda_means[i] + eps
   
        # Initialize healthy state if needed
        if self.healthy_ref is not None:
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, self.K, :] = self.healthy_ref + eps
            gamma_init[:, self.K] = 0.0

        # Only make lambda and gamma trainable parameters
        self.gamma = nn.Parameter(gamma_init)
        self.lambda_ = nn.Parameter(lambda_init)

        if self.healthy_ref is not None:
            print(f"Initializing with {self.K} disease states + 1 healthy state")
        else:
            print(f"Initializing with {self.K} disease states only")
        print("Initialization complete!")
    
    def forward(self):
        theta = torch.softmax(self.lambda_, dim=1)
        epsilon = 1e-8
        phi_prob = torch.sigmoid(self.phi)
        # Apply kappa scaling inside the computation of pi
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, epsilon, 1-epsilon)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        """Modified loss function that only considers lambda parameters"""
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        
        # Original survival loss components remain exactly the same
        N, D, T = self.Y.shape
        event_times_tensor = torch.tensor(event_times, dtype=torch.long)
        event_times_expanded = event_times_tensor.unsqueeze(-1)
        time_grid = torch.arange(T).unsqueeze(0).unsqueeze(0)
        mask_before_event = (time_grid < event_times_expanded).float()
        mask_at_event = (time_grid == event_times_expanded).float()
        
        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))
        total_data_loss = (loss_censored + loss_event + loss_no_event) / (self.N)

        # GP prior loss only for lambda (phi is fixed)
        if self.gpweight > 0:
            gp_loss = self.compute_gp_prior_loss()
        else:
            gp_loss = 0.0
        
        signature_update_loss = 0.0
        
        if self.lrtpen > 0:
            diagnoses = self.Y  # [N x D x T]
            phi_avg = phi_prob.mean(dim=2)  
            for d in range(self.D):
                if torch.any(diagnoses[:, d, :]):
                    spec_d = phi_avg[:, d]
                    max_sig = torch.argmax(spec_d)
                    
                    other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (self.K_total - 1)
                    lr = spec_d[max_sig] / (other_mean + epsilon)
                    
                    if lr > 2:
                        diagnosis_mask = diagnoses[:, d, :].bool()
                        patient_idx, time_idx = torch.where(diagnosis_mask)
                        lambda_at_diagnosis = self.lambda_[patient_idx, max_sig, time_idx]
                        
                        target_value = 2.0  # This should give ~0.4 theta share
                        disease_prevalence = diagnoses[:, d, :].float().mean() + epsilon
                        prevalence_scaling = min(0.1 / disease_prevalence, 10.0)
                        
                        signature_update_loss += torch.sum(
                            torch.log(lr) * prevalence_scaling * (target_value - lambda_at_diagnosis)
                        )
                
        total_loss = total_data_loss + self.gpweight*gp_loss + self.lrtpen*signature_update_loss / (self.N * self.T)
        
        return total_loss
        
    def compute_gp_prior_loss(self):
        """Compute GP prior loss only for lambda parameters (phi is fixed)"""
        # Initialize loss
        gp_loss_lambda = 0.0
        
        # Compute Cholesky once
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        
        # Lambda GP prior
        for k in range(self.K_total):
            lambda_k = self.lambda_[:, k, :]  # N x T
            
            if k == self.K and self.healthy_ref is not None:  # Healthy state
                mean_lambda_k = self.healthy_ref.unsqueeze(0)
            else:  # Disease signatures
                mean_lambda_k = self.signature_refs[k].unsqueeze(0) + \
                            self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
            
            deviations_lambda = lambda_k - mean_lambda_k
            for i in range(self.N):
                dev_i = deviations_lambda[i:i+1].T
                v_i = torch.cholesky_solve(dev_i, L_lambda)
                gp_loss_lambda += 0.5 * torch.sum(v_i.T @ dev_i)
            
        # Return loss with appropriate scaling
        return gp_loss_lambda / self.N
        
    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01):
        """Modified fit method that only updates lambda and gamma"""
        
        optimizer = optim.Adam([
            {'params': [self.lambda_], 'lr': learning_rate},      # e.g. 1e-2
            {'params': [self.kappa], 'lr': learning_rate},        # e.g. 1e-3
            {'params': [self.gamma], 'lr': learning_rate}         # Same rate as lambda
        ])

        # Initialize gradient history
        gradient_history = {
            'lambda_grad': [],
        }
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            
            if self.kappa.grad is not None:
               print(f"Kappa gradient: {self.kappa.grad.item():.3e}")

            gradient_history['lambda_grad'].append(self.lambda_.grad.clone().detach())
        
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 1 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss.item():.4f}")
                self.analyze_signature_responses()
        
        return losses, gradient_history
    
    def analyze_signature_responses(self, top_n=5):
        """Monitor theta (signature proportion) changes for top N most specific diseases"""
        with torch.no_grad():
            pi, theta, phi_prob = self.forward()  # Get thetas from softmax(lambda)
            phi_avg = phi_prob.mean(dim=2)
            
            print(f"\nMonitoring signature responses:")
            
            # Find diseases with highest LR
            disease_lrs = []
            for d in range(self.D):
                spec_d = phi_avg[:, d]
                max_sig = torch.argmax(spec_d)
                other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (self.K_total - 1)
                lr = spec_d[max_sig] / (other_mean + 1e-8)
                disease_lrs.append((d, max_sig, lr.item()))
            
            # Sort by LR and take top N
            top_diseases = sorted(disease_lrs, key=lambda x: x[2], reverse=True)[:top_n]
            
            for d, max_sig, lr in top_diseases:
                diagnosed = self.Y[:, d, :].any(dim=1)
                if diagnosed.any():
                    # Look at theta (proportion) values instead of lambda
                    theta_vals = theta[diagnosed, max_sig, :]
                    theta_others = theta[~diagnosed, max_sig, :]
                    
                    print(f"\nDisease {d} (signature {max_sig}, LR={lr:.2f}):")
                    print(f"  Theta for diagnosed: {theta_vals.mean():.3f} ± {theta_vals.std():.3f}")
                    print(f"  Theta for others: {theta_others.mean():.3f}")
                    print(f"  Proportion difference: {(theta_vals.mean() - theta_others.mean()):.3f}")

    def plot_individual_trajectories(self, individual_ids, k=None):
        """
        Plot theta trajectories for selected individuals
        
        Parameters:
        individual_ids: list of individual indices to plot
        k: if provided, only plot this signature; otherwise plot all
        """
        with torch.no_grad():
            pi, theta, phi_prob = self.forward()
            time_points = np.arange(self.T)
            
            if k is None:
                # Plot all signatures for each individual
                for i in individual_ids:
                    plt.figure(figsize=(12, 6))
                    for k in range(self.K_total):
                        plt.plot(time_points, theta[i, k, :].numpy(), label=f'Signature {k}')
                    plt.title(f'Individual {i} - Signature Proportions')
                    plt.xlabel('Time')
                    plt.ylabel('Proportion (Theta)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.show()
            else:
                # Plot one signature for all individuals
                plt.figure(figsize=(12, 6))
                for i in individual_ids:
                    plt.plot(time_points, theta[i, k, :].numpy(), label=f'Individual {i}')
                plt.title(f'Signature {k} - Multiple Individuals')
                plt.xlabel('Time')
                plt.ylabel('Proportion (Theta)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
                
    def predict(self, new_G=None, new_indices=None):
        """
        Make predictions for either new genetic data or existing indices
        
        Parameters:
        new_G: New genetic data matrix (optional)
        new_indices: Indices to use from the existing data (optional)
        
        Returns:
        pi: Disease probabilities [N_new x D x T]
        theta: Signature proportions [N_new x K x T]
        """
        with torch.no_grad():
            if new_G is not None:
                # Center and scale new genetic data
                new_G_tensor = torch.tensor(new_G, dtype=torch.float32)
                new_G_scaled = (new_G_tensor - self.G.mean(dim=0)) / self.G.std(dim=0)
                
                # Compute new lambda values
                N_new = new_G_scaled.shape[0]
                new_lambda = torch.zeros((N_new, self.K_total, self.T))
                
                # Compute lambda means for each signature
                for k in range(self.K):
                    lambda_means = self.genetic_scale * (new_G_scaled @ self.gamma[:, k])
                    for i in range(N_new):
                        new_lambda[i, k, :] = self.signature_refs[k] + lambda_means[i]
                
                # Set healthy signature if applicable
                if self.healthy_ref is not None:
                    new_lambda[:, self.K, :] = self.healthy_ref
                
                # Compute theta and pi
                new_theta = torch.softmax(new_lambda, dim=1)
                phi_prob = torch.sigmoid(self.phi)
                new_pi = torch.einsum('nkt,kdt->ndt', new_theta, phi_prob) * self.kappa
                
                return new_pi, new_theta
            
            elif new_indices is not None:
                # Extract existing lambda values for given indices
                subset_lambda = self.lambda_[new_indices]
                
                # Compute theta and pi
                subset_theta = torch.softmax(subset_lambda, dim=1)
                phi_prob = torch.sigmoid(self.phi)
                subset_pi = torch.einsum('nkt,kdt->ndt', subset_theta, phi_prob) * self.kappa
                
                return subset_pi, subset_theta
            
            else:
                raise ValueError("Either new_G or new_indices must be provided") 