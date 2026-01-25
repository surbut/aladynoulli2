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

class AladynSurvivalFixedPhiFixedGammaFixedKappa(nn.Module):
    """Version with phi, gamma, and kappa all fixed - only trains lambda for single-patient scenarios"""
    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 pretrained_phi, pretrained_psi, pretrained_gamma, pretrained_kappa, 
                 signature_references=None, healthy_reference=None, disease_names=None, flat_lambda=False):
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
       
        # Fixed Phi kernel (not updated during training)
        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(self.T)
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(self.T)
        
        # Store prevalence and compute logit
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )  # D x T
        
        # Store pretrained phi and psi (will not be updated)
        self.register_buffer('phi', torch.tensor(pretrained_phi, dtype=torch.float32))
        self.register_buffer('psi', torch.tensor(pretrained_psi, dtype=torch.float32))
        
        # Store pretrained gamma as a buffer (not trainable)
        if pretrained_gamma is None:
            raise ValueError("pretrained_gamma must be provided for fixed-gamma model")
        if torch.is_tensor(pretrained_gamma):
            self.register_buffer('gamma', pretrained_gamma.clone().detach())
        else:
            self.register_buffer('gamma', torch.tensor(pretrained_gamma, dtype=torch.float32))
        
        # Store pretrained kappa as a buffer (not trainable)
        if pretrained_kappa is None:
            raise ValueError("pretrained_kappa must be provided for fixed-kappa model")
        if isinstance(pretrained_kappa, (int, float)):
            self.register_buffer('kappa', torch.tensor(float(pretrained_kappa), dtype=torch.float32))
        elif torch.is_tensor(pretrained_kappa):
            if pretrained_kappa.numel() == 1:
                self.register_buffer('kappa', pretrained_kappa.clone().detach().squeeze())
            else:
                raise ValueError(f"kappa must be a scalar, got shape {pretrained_kappa.shape}")
        else:
            self.register_buffer('kappa', torch.tensor(float(pretrained_kappa), dtype=torch.float32))
        
        self.phi_gp_loss = self._calculate_phi_gp_loss()
        print(f"Pre-calculated phi GP loss: {self.phi_gp_loss:.4f}")
        print(f"Using fixed gamma with shape: {self.gamma.shape}")
        print(f"Using fixed kappa: {self.kappa.item():.6f}")

          
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
        # Normalize G only if we have multiple distinct patients (N > 1)
        # For single-patient refitting (N=1), skip normalization to avoid std=0
        if N > 1:
            # Convert to numpy for checking if rows are identical
            G_np = np.array(G) if not isinstance(G, np.ndarray) else G
            # Check if all rows are identical (single-patient duplicated scenario)
            if np.allclose(G_np[0:1], G_np, atol=1e-6):
                # Single patient duplicated - skip normalization, use as-is
                self.G = torch.tensor(G, dtype=torch.float32)
            else:
                # Multiple distinct patients - normalize across batch
                G_centered = G - G.mean(axis=0, keepdims=True)
                G_std = G_centered.std(axis=0, keepdims=True)
                # Avoid division by zero
                G_std = np.where(G_std < 1e-8, np.ones_like(G_std), G_std)
                G_scaled = G_centered / G_std
                self.G = torch.tensor(G_scaled, dtype=torch.float32)
        else:
            # For N=1, use G as-is (no normalization possible without population statistics)
            self.G = torch.tensor(G, dtype=torch.float32)
        
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
        # Initialize only lambda (gamma and kappa are fixed)
        self.initialize_params()

    def _calculate_phi_gp_loss(self):
        """Calculate phi GP loss once at initialization"""
        gp_loss_phi = 0.0
        
        L_phi = torch.linalg.cholesky(self.K_phi)
        for k in range(self.K):  # Only iterate over disease signatures (phi has K entries)
            phi_k = self.phi[k]  # D x T
            for d in range(self.D):
                mean_phi_d = self.logit_prev_t[d, :] + self.psi[k, d]
                dev_d = (phi_k[d:d+1, :] - mean_phi_d).T
                v_d = torch.cholesky_solve(dev_d, L_phi)
                gp_loss_phi += 0.5 * torch.sum(v_d.T @ dev_d)
        
        return gp_loss_phi / self.D


    def initialize_params(self, **kwargs):
        """Initialize only lambda (gamma and kappa are fixed from pretrained values)"""
        lambda_init = torch.zeros((self.N, self.K_total, self.T))
        
        # Initialize lambda using the fixed gamma
        for k in range(self.K):
            lambda_means = self.genetic_scale * (self.G @ self.gamma[:, k])
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

        # Only make lambda trainable (gamma and kappa are fixed as buffers)
        self.lambda_ = nn.Parameter(lambda_init)

        if self.healthy_ref is not None:
            print(f"Initializing with {self.K} disease states + 1 healthy state")
        else:
            print(f"Initializing with {self.K} disease states only")
        print("Initialization complete! (gamma and kappa are fixed)")
    
    def forward(self):
        theta = torch.softmax(self.lambda_, dim=1)
        epsilon = 1e-8
        phi_prob = torch.sigmoid(self.phi)
        # Apply fixed kappa scaling inside the computation of pi
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
        
        # Convert event_times to tensor and ensure correct shape
        if not isinstance(event_times, torch.Tensor):
            event_times_tensor = torch.tensor(event_times, dtype=torch.long)
        else:
            event_times_tensor = event_times.long()
        
        # Handle event_times: can be [N, D] (per disease) or [N] (per patient)
        if len(event_times_tensor.shape) == 2:
            # [N, D] - per disease event times
            if event_times_tensor.shape != (N, D):
                raise ValueError(f"event_times shape {event_times_tensor.shape} doesn't match expected (N={N}, D={D})")
            event_times_expanded = event_times_tensor.unsqueeze(-1)  # [N, D, 1]
        elif len(event_times_tensor.shape) == 1:
            # [N] - per patient event times, expand to [N, D]
            if event_times_tensor.shape[0] != N:
                raise ValueError(f"event_times shape {event_times_tensor.shape} doesn't match N={N}")
            event_times_expanded = event_times_tensor.unsqueeze(-1).expand(N, D).unsqueeze(-1)  # [N, D, 1]
        else:
            raise ValueError(f"event_times must be 1D [N] or 2D [N, D], got shape {event_times_tensor.shape}")
        
        # Ensure event_times_expanded is [N, D, 1]
        if len(event_times_expanded.shape) != 3 or event_times_expanded.shape != (N, D, 1):
            raise ValueError(f"event_times_expanded shape {event_times_expanded.shape} != expected ({N}, {D}, 1)")
        
        # Create time grid explicitly expanded to [N, D, T] for element-wise comparison
        time_grid = torch.arange(T, dtype=torch.long, device=event_times_expanded.device)
        time_grid = time_grid.unsqueeze(0).unsqueeze(0).expand(N, D, T)  # [N, D, T]
        
        # Element-wise comparison: [N, D, T] < [N, D, 1] -> [N, D, T]
        mask_before_event = (time_grid < event_times_expanded).float()  # [N, D, T]
        mask_at_event = (time_grid == event_times_expanded).float()  # [N, D, T]
        
        # Verify shapes match
        if pi.shape != (N, D, T):
            raise ValueError(f"pi shape {pi.shape} != expected {(N, D, T)}")
        if mask_before_event.shape != (N, D, T):
            raise ValueError(f"mask_before_event shape {mask_before_event.shape} != expected {(N, D, T)}")
        if mask_at_event.shape != (N, D, T):
            raise ValueError(f"mask_at_event shape {mask_at_event.shape} != expected {(N, D, T)}")
        
        # pi is [N, D, T], masks are [N, D, T] - shapes match
        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))
        total_data_loss = (loss_censored + loss_event + loss_no_event) / (self.N)

        # GP prior loss only for lambda (phi, gamma, and kappa are fixed)
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
        """Compute GP prior loss only for lambda parameters (phi, gamma, and kappa are fixed)
        
        VECTORIZED VERSION: Solves for all individuals simultaneously using batched operations.
        Mathematically equivalent to the sequential version but much faster on multi-core systems.
        """
        # Initialize loss
        gp_loss_lambda = 0.0
        
        # Compute Cholesky once
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        
        # Lambda GP prior - VECTORIZED
        for k in range(self.K_total):
            lambda_k = self.lambda_[:, k, :]  # N x T
            
            if k == self.K and self.healthy_ref is not None:  # Healthy state
                mean_lambda_k = self.healthy_ref.unsqueeze(0)
            else:  # Disease signatures
                mean_lambda_k = self.signature_refs[k].unsqueeze(0) + \
                            self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
            
            deviations_lambda = lambda_k - mean_lambda_k  # N x T
            
            # VECTORIZED: Solve for all individuals at once
            # deviations_lambda.T is [T x N], each column is one individual's deviation
            V = torch.cholesky_solve(deviations_lambda.T, L_lambda)  # [T x N]
            
            # Compute quadratic form for all individuals using trace
            # trace(deviations_lambda @ V) = sum_i (dev_i^T @ V[:, i])
            gp_loss_lambda += 0.5 * torch.trace(deviations_lambda @ V)
            
        # Return combined loss with appropriate scaling
        return gp_loss_lambda / self.N + self.phi_gp_loss 
    
    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01):
        """Modified fit method that only updates lambda (gamma and kappa are fixed)"""
        
        # Only lambda is trainable - gamma and kappa are fixed buffers
        optimizer = optim.Adam([
            {'params': [self.lambda_], 'lr': learning_rate},      # e.g. 1e-2
            # gamma and kappa are NOT in optimizer - they're fixed buffers
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
            
            gradient_history['lambda_grad'].append(self.lambda_.grad.clone().detach())
        
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 10 == 0:
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
                
                # Compute lambda means for each signature using fixed gamma
                for k in range(self.K):
                    lambda_means = self.genetic_scale * (new_G_scaled @ self.gamma[:, k])
                    for i in range(N_new):
                        new_lambda[i, k, :] = self.signature_refs[k] + lambda_means[i]
                
                # Set healthy signature if applicable
                if self.healthy_ref is not None:
                    new_lambda[:, self.K, :] = self.healthy_ref
                
                # Compute theta and pi using fixed kappa
                new_theta = torch.softmax(new_lambda, dim=1)
                phi_prob = torch.sigmoid(self.phi)
                new_pi = torch.einsum('nkt,kdt->ndt', new_theta, phi_prob) * self.kappa
                
                return new_pi, new_theta
            
            elif new_indices is not None:
                # Extract existing lambda values for given indices
                subset_lambda = self.lambda_[new_indices]
                
                # Compute theta and pi using fixed kappa
                subset_theta = torch.softmax(subset_lambda, dim=1)
                phi_prob = torch.sigmoid(self.phi)
                subset_pi = torch.einsum('nkt,kdt->ndt', subset_theta, phi_prob) * self.kappa
                
                return subset_pi, subset_theta
            
            else:
                raise ValueError("Either new_G or new_indices must be provided") 
            


def plot_training_evolution(history_tuple):
    losses, gradient_history = history_tuple
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.yscale('log')
    plt.legend()
    
    # Plot lambda gradients
    plt.subplot(1, 3, 2)
    lambda_norms = [torch.norm(g).item() for g in gradient_history['lambda_grad']]
    plt.plot(lambda_norms, label='Lambda gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Lambda Gradient Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]  # Changed from slice to index list
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices

