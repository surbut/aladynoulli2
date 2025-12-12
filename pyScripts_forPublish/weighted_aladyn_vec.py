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

class AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest_weighted(nn.Module):
    """
    Weighted version of Aladyn model using inverse probability weights (IPW)
    to correct for selection bias in UK Biobank data.
    """
    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 signature_references=None, healthy_reference=None, disease_names=None, 
                 flat_lambda=False, learn_kappa=True, weights=None):
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
        
        # Make kappa configurable
        if learn_kappa:
            self.kappa = nn.Parameter(torch.ones(1))
        else:
            self.kappa = torch.ones(1)
        
        # Fixed kernel parameters
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
        self.init_amplitude = 1.0
        self.lambda_amplitude_init = init_sd_scaler
        self.phi_amplitude_init = init_sd_scaler
        
        # Store base kernel matrix
        time_points = torch.arange(T, dtype=torch.float32)
        time_diff = time_points[:, None] - time_points[None, :]
        self.base_K_lambda = torch.exp(-0.5 * (time_diff**2) / (self.lambda_length_scale**2))
        self.base_K_phi = torch.exp(-0.5 * (time_diff**2) / (self.phi_length_scale**2))
        
        self.K_lambda_init = (self.lambda_amplitude_init**2) * self.base_K_lambda + self.jitter * torch.eye(T)
        self.K_phi_init = (self.phi_amplitude_init**2) * self.base_K_phi + self.jitter * torch.eye(T)
        self.phi_amplitude = 1
        self.lambda_amplitude = 1
        
        jitter_matrix = self.jitter * torch.eye(T)
        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(self.T)
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(self.T)
        
        self.psi = None 
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
        )
        
        # Initialize parameters
        self.initialize_params()

    def initialize_params(self, psi_config=None, true_psi=None, **kwargs):
        """Initialize parameters with K disease clusters plus one healthy cluster"""
        Y_avg = torch.mean(self.Y, dim=2)
        epsilon = 1e-6
        Y_avg = torch.clamp(Y_avg, epsilon, 1.0 - epsilon)
        Y_avg = torch.log(Y_avg/(1-Y_avg))

        # Initialize psi for disease clusters
        if true_psi is not None:
            psi_init = torch.zeros((self.K_total, self.D))
            psi_init[:self.K, :] = true_psi
            if self.healthy_ref is not None:
                psi_init[self.K, :] = -5.0 + 0.01 * torch.randn(self.D)
        elif psi_config is not None:
            psi_init = torch.zeros((self.K_total, self.D))
            for k in range(self.K):
                cluster_mask = (self.clusters == k)
                psi_init[k, cluster_mask] = psi_config['in_cluster'] + psi_config['noise_in'] * torch.randn(cluster_mask.sum())
                psi_init[k, ~cluster_mask] = psi_config['out_cluster'] + psi_config['noise_out'] * torch.randn((~cluster_mask).sum())
            if self.healthy_ref is not None:
                psi_init[self.K, :] = -5.0 + 0.01 * torch.randn(self.D)
        else:
            Y_corr = torch.corrcoef(Y_avg.T)
            Y_corr = torch.nan_to_num(Y_corr, nan=0.0)
            similarity = (Y_corr + 1) / 2
            
            spectral = SpectralClustering(
                n_clusters=self.K,
                assign_labels='kmeans',
                affinity='precomputed',
                n_init=10,
                random_state=42
            ).fit(similarity.numpy())
            
            self.clusters = spectral.labels_
            
            psi_init = torch.zeros((self.K_total, self.D))
            for k in range(self.K):
                cluster_mask = (self.clusters == k)
                psi_init[k, cluster_mask] = 1.0 + 0.1 * torch.randn(cluster_mask.sum())
                psi_init[k, ~cluster_mask] = -2.0 + 0.01 * torch.randn((~cluster_mask).sum())
            if self.healthy_ref is not None:
                psi_init[self.K, :] = -5.0 + 0.01 * torch.randn(self.D)

            print("\nCluster Sizes:")
            unique, counts = np.unique(self.clusters, return_counts=True)
            for k, count in zip(unique, counts):
                print(f"Cluster {k}: {count} diseases")
        
        gamma_init = torch.zeros((self.P, self.K_total))
        lambda_init = torch.zeros((self.N, self.K_total, self.T))
        phi_init = torch.zeros((self.K_total, self.D, self.T))

        # Initialize phi
        for k in range(self.K):
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps

        # Initialize lambda and gamma
        for k in range(self.K):
            if true_psi is None:
                cluster_diseases = (self.clusters == k)
                base_value = Y_avg[:, cluster_diseases].mean(dim=1)
                base_value_centered = base_value - base_value.mean()
            else:
                strong_diseases = (true_psi[k] > 0).float()
                base_value = Y_avg[:, strong_diseases > 0].mean(dim=1)
                base_value_centered = base_value - base_value.mean()
  
            gamma_init[:, k] = torch.linalg.lstsq(self.G, base_value_centered.unsqueeze(1)).solution.squeeze()
            lambda_means = self.genetic_scale * (self.G @ gamma_init[:, k])
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = self.signature_refs[k] + lambda_means[i] + eps
   
        if self.healthy_ref is not None:
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[self.K, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[self.K, d, :] = mean_phi + eps

            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, self.K, :] = self.healthy_ref + eps
            gamma_init[:, self.K] = 0.0

        self.gamma = nn.Parameter(gamma_init)
        self.lambda_ = nn.Parameter(lambda_init)
        self.phi = nn.Parameter(phi_init)
        self.psi = nn.Parameter(psi_init)

        if self.healthy_ref is not None:
            print(f"Initializing with {self.K} disease states + 1 healthy state")
        else:
            print(f"Initializing with {self.K} disease states only")
        print("Initialization complete!")

    def forward(self):
        theta = torch.softmax(self.lambda_, dim=1)
        epsilon = 1e-6
        phi_prob = torch.sigmoid(self.phi)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, epsilon, 1-epsilon)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        """
        WEIGHTED loss function using inverse probability weights
        
        Key change: Each individual's contribution to the data likelihood is 
        weighted by their IPW, correcting for selection bias.
        """
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        
        N, D, T = self.Y.shape
        event_times_tensor = torch.tensor(event_times, dtype=torch.long)
        event_times_expanded = event_times_tensor.unsqueeze(-1)
        time_grid = torch.arange(T).unsqueeze(0).unsqueeze(0)
        mask_before_event = (time_grid < event_times_expanded).float()
        mask_at_event = (time_grid == event_times_expanded).float()
        
        # ============================================================
        # WEIGHTED DATA LIKELIHOOD (MODIFIED!)
        # ============================================================
        # Compute log-likelihood components for each individual
        log_lik_censored = torch.log(1 - pi) * mask_before_event
        log_lik_event = torch.log(pi) * mask_at_event * self.Y
        log_lik_no_event = torch.log(1 - pi) * mask_at_event * (1 - self.Y)
        
        # Combine: shape [N x D x T]
        log_likelihood = log_lik_censored + log_lik_event + log_lik_no_event
        
        # Sum across diseases and time for each individual: shape [N]
        individual_ll = log_likelihood.sum(dim=(1, 2))
        
        # Weight each individual's contribution
        weighted_ll = individual_ll * self.weights
        
        # Average (negative because we minimize)
        # Divide by N since weights sum to N
        total_data_loss = -weighted_ll.sum() / self.N
        # ============================================================
        
        # GP prior loss (UNWEIGHTED - these are priors on parameters)
        if self.gpweight > 0:
            gp_loss = self.compute_gp_prior_loss()
        else:
            gp_loss = 0.0
        
        # Signature update loss
        signature_update_loss = 0.0
        if self.lrtpen > 0:
            diagnoses = self.Y
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
                        
                        target_value = 2.0
                        disease_prevalence = diagnoses[:, d, :].float().mean() + epsilon
                        prevalence_scaling = min(0.1 / disease_prevalence, 10.0)
                        
                        signature_update_loss += torch.sum(
                            torch.log(lr) * prevalence_scaling * (target_value - lambda_at_diagnosis)
                        )
        
        total_loss = total_data_loss + self.gpweight*gp_loss + self.lrtpen*signature_update_loss / (self.N * self.T)
        
        return total_loss

    def compute_gp_prior_loss(self):
        """
        Vectorized GP prior loss computation.
        This replaces the loop-based version with fully vectorized operations.
        """
        # Compute Cholesky decompositions once
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        L_phi = torch.linalg.cholesky(self.K_phi)
        
        # Lambda GP prior - VECTORIZED
        # Shape: lambda_ is [N x K_total x T]
        # Compute mean for each signature
        # For disease signatures: mean = signature_refs[k] + genetic_scale * (G @ gamma[:, k])
        # For healthy state: mean = healthy_ref
        
        # Compute mean_lambda for all signatures at once
        # Shape: [N x K_total x T]
        mean_lambda = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        
        # Disease signatures: [N x T] for each k
        for k in range(self.K):
            mean_lambda[:, k, :] = self.signature_refs[k].unsqueeze(0).unsqueeze(1) + \
                                  self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
        
        # Healthy state (if exists)
        if self.healthy_ref is not None:
            mean_lambda[:, self.K, :] = self.healthy_ref.unsqueeze(0).unsqueeze(1)
        
        # Compute deviations: [N x K_total x T]
        deviations_lambda = self.lambda_ - mean_lambda  # [N x K_total x T]
        
        # Vectorized Cholesky solve for all patients and signatures
        # Reshape to [N*K_total x T] for batch processing
        # Each row is a time series that needs to be solved
        deviations_flat = deviations_lambda.reshape(-1, self.T)  # [N*K_total x T]
        # Transpose for cholesky_solve which expects [T x batch_size]
        deviations_flat_T = deviations_flat.T  # [T x N*K_total]
        v_flat_T = torch.cholesky_solve(deviations_flat_T, L_lambda)  # [T x N*K_total]
        # Transpose back and compute quadratic forms
        v_flat = v_flat_T.T  # [N*K_total x T]
        # Compute quadratic forms: sum over T dimension for each patient-signature pair
        gp_loss_lambda = 0.5 * torch.sum(deviations_flat * v_flat)  # Scalar
        
        # Phi GP prior - VECTORIZED
        # Shape: phi is [K_total x D x T]
        # Mean: logit_prev_t[d, :] + psi[k, d] for each k, d
        # Compute mean_phi: [K_total x D x T]
        mean_phi = self.logit_prev_t.unsqueeze(0) + self.psi.unsqueeze(2)  # [K_total x D x T]
        deviations_phi = self.phi - mean_phi  # [K_total x D x T]
        
        # Vectorized Cholesky solve for all signatures and diseases
        # Reshape to [K_total*D x T] for batch processing
        deviations_phi_flat = deviations_phi.reshape(-1, self.T)  # [K_total*D x T]
        # Transpose for cholesky_solve
        deviations_phi_flat_T = deviations_phi_flat.T  # [T x K_total*D]
        v_phi_flat_T = torch.cholesky_solve(deviations_phi_flat_T, L_phi)  # [T x K_total*D]
        # Transpose back and compute quadratic forms
        v_phi_flat = v_phi_flat_T.T  # [K_total*D x T]
        # Compute quadratic forms: sum over T dimension
        gp_loss_phi = 0.5 * torch.sum(deviations_phi_flat * v_phi_flat)  # Scalar
        
        # Return combined loss with appropriate scaling
        return gp_loss_lambda / self.N + gp_loss_phi / self.D

    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01):
        """Training with weighted loss"""
        param_groups = [
            {'params': [self.lambda_], 'lr': learning_rate},
            {'params': [self.phi], 'lr': learning_rate * 0.1},
            {'params': [self.psi], 'lr': learning_rate*0.1},
            {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': learning_rate}
        ]
        
        if isinstance(self.kappa, nn.Parameter):
            param_groups.append({'params': [self.kappa], 'lr': learning_rate})
        
        optimizer = optim.Adam(param_groups)

        gradient_history = {
            'lambda_grad': [],
            'phi_grad': []
        }
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            
            gradient_history['lambda_grad'].append(self.lambda_.grad.clone().detach())
            gradient_history['phi_grad'].append(self.phi.grad.clone().detach())
        
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 1 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss.item():.4f}")
                self.analyze_signature_responses()
        
        return losses, gradient_history

    def analyze_signature_responses(self, top_n=5):
        """Monitor theta changes for top diseases"""
        with torch.no_grad():
            pi, theta, phi_prob = self.forward()
            phi_avg = phi_prob.mean(dim=2)
            
            print(f"\nMonitoring signature responses:")
            
            disease_lrs = []
            for d in range(self.D):
                spec_d = phi_avg[:, d]
                max_sig = torch.argmax(spec_d)
                other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (self.K_total - 1)
                lr = spec_d[max_sig] / (other_mean + 1e-8)
                disease_lrs.append((d, max_sig, lr.item()))
            
            top_diseases = sorted(disease_lrs, key=lambda x: x[2], reverse=True)[:top_n]
            
            for d, max_sig, lr in top_diseases:
                diagnosed = self.Y[:, d, :].any(dim=1)
                if diagnosed.any():
                    theta_vals = theta[diagnosed, max_sig, :]
                    theta_others = theta[~diagnosed, max_sig, :]
                    
                    print(f"\nDisease {d} (signature {max_sig}, LR={lr:.2f}):")
                    print(f"  Theta for diagnosed: {theta_vals.mean():.3f} ± {theta_vals.std():.3f}")
                    print(f"  Theta for others: {theta_others.mean():.3f}")
                    print(f"  Proportion difference: {(theta_vals.mean() - theta_others.mean()):.3f}")


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices
