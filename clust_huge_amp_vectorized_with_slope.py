"""
ALADYN Model with Genetic Effects on Progression Speed

This extends clust_huge_amp_vectorized.py to include genetic effects on slope:

    lambda_ik(t) ~ GP(r_k + g_i^T * gamma_level + t * g_i^T * gamma_slope, Omega_lambda)

Key changes from original:
1. Added gamma_slope parameter (P x K)
2. Modified lambda mean to include time-varying genetic effect
3. Updated initialization for gamma_slope
4. Updated GP prior loss
5. Added gamma_slope to optimizer with same regularization as gamma_level

Author: Extended from original clust_huge_amp_vectorized.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import seaborn as sns

class AladynWithGeneticSlope(nn.Module):
    """ALADYN model with genetic effects on both level AND progression speed"""

    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 signature_references=None, healthy_reference=None, disease_names=None,
                 flat_lambda=False, learn_kappa=True, learn_slope=True):
        super().__init__()
        # Basic dimensions and settings
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.gpweight = W
        self.jitter = 1e-6
        self.learn_slope = learn_slope  # NEW: control whether to learn slope

        # Store timepoints for slope calculation
        self.register_buffer('t_centered', torch.arange(T, dtype=torch.float32) - 30)  # Center at age 30

        # Make kappa configurable
        if learn_kappa:
            self.kappa = nn.Parameter(torch.ones(1))
        else:
            self.kappa = torch.ones(1)

        self.lrtpen = R

        # Fixed kernel parameters
        self.lambda_length_scale = T / 4
        self.phi_length_scale = T / 3
        self.lambda_amplitude_init = init_sd_scaler
        self.phi_amplitude_init = init_sd_scaler

        # Store base kernel matrix
        time_points = torch.arange(T, dtype=torch.float32)
        time_diff = time_points[:, None] - time_points[None, :]
        self.base_K_lambda = torch.exp(-0.5 * (time_diff ** 2) / (self.lambda_length_scale ** 2))
        self.base_K_phi = torch.exp(-0.5 * (time_diff ** 2) / (self.phi_length_scale ** 2))

        # Initialize kernels
        self.K_lambda_init = (self.lambda_amplitude_init ** 2) * self.base_K_lambda + self.jitter * torch.eye(T)
        self.K_phi_init = (self.phi_amplitude_init ** 2) * self.base_K_phi + self.jitter * torch.eye(T)
        self.phi_amplitude = 1
        self.lambda_amplitude = 1

        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(T)
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(T)

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

    def initialize_params(self, psi_config=None, true_psi=None, true_gamma_slope=None, **kwargs):
        """Initialize parameters including gamma_slope"""
        Y_avg = torch.mean(self.Y, dim=2)
        epsilon = 1e-6
        Y_avg = torch.clamp(Y_avg, epsilon, 1.0 - epsilon)
        Y_avg = torch.log(Y_avg / (1 - Y_avg))

        # Initialize psi (same as before)
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

        # Initialize gamma_level and gamma_slope (NEW!)
        gamma_level_init = torch.zeros((self.P, self.K_total))
        gamma_slope_init = torch.zeros((self.P, self.K_total))  # NEW!

        lambda_init = torch.zeros((self.N, self.K_total, self.T))
        phi_init = torch.zeros((self.K_total, self.D, self.T))

        # Initialize phi
        for k in range(self.K):
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps

        # Initialize lambda and gamma for disease clusters
        for k in range(self.K):
            print(f"\nCalculating gamma_level and gamma_slope for k={k}:")
            if true_psi is None:
                cluster_diseases = (self.clusters == k)
                base_value = Y_avg[:, cluster_diseases].mean(dim=1)
                base_value_centered = base_value - base_value.mean()
                print(f"Number of diseases in cluster: {cluster_diseases.sum()}")
            else:
                strong_diseases = (true_psi[k] > 0).float()
                base_value = Y_avg[:, strong_diseases > 0].mean(dim=1)
                base_value_centered = base_value - base_value.mean()
                print(f"Number of diseases in cluster: {strong_diseases.sum()}")

            # Initialize gamma_level (same as before)
            gamma_level_init[:, k] = torch.linalg.lstsq(self.G, base_value_centered.unsqueeze(1)).solution.squeeze()
            print(f"Gamma_level init for k={k} (first 5): {gamma_level_init[:5, k]}")

            # Initialize gamma_slope (NEW!)
            if true_gamma_slope is not None:
                # Use true values if provided (for simulation)
                gamma_slope_init[:, k] = torch.tensor(true_gamma_slope[:, k], dtype=torch.float32)
                print(f"Using true gamma_slope for k={k} (first 5): {gamma_slope_init[:5, k]}")
            else:
                # Initialize small random values
                gamma_slope_init[:, k] = 0.001 * torch.randn(self.P)
                print(f"Gamma_slope init for k={k} (small random): {gamma_slope_init[:5, k]}")

            # Initialize lambda with both level and slope effects (NEW!)
            level_effect = self.genetic_scale * (self.G @ gamma_level_init[:, k])  # (N,)
            slope_effect = self.genetic_scale * (self.G @ gamma_slope_init[:, k])  # (N,)

            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                # Mean = r_k + level_effect + t * slope_effect
                mean_trajectory = (self.signature_refs[k] + level_effect[i] +
                                 self.t_centered * slope_effect[i])
                lambda_init[i, k, :] = mean_trajectory + eps

        # Healthy state
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
            gamma_level_init[:, self.K] = 0.0
            gamma_slope_init[:, self.K] = 0.0

        # Register parameters
        self.gamma_level = nn.Parameter(gamma_level_init)  # Renamed from gamma
        if self.learn_slope:
            self.gamma_slope = nn.Parameter(gamma_slope_init)  # NEW!
        else:
            self.register_buffer('gamma_slope', gamma_slope_init)  # Fixed, not learned

        self.lambda_ = nn.Parameter(lambda_init)
        self.phi = nn.Parameter(phi_init)
        self.psi = nn.Parameter(psi_init)

        print(f"\nInitialization complete!")
        print(f"  Disease states: {self.K}")
        if self.healthy_ref is not None:
            print(f"  + 1 healthy state")
        print(f"  Learning genetic slope: {self.learn_slope}")

    def forward(self):
        """Forward pass - same as before"""
        theta = torch.softmax(self.lambda_, dim=1)
        epsilon = 1e-6
        phi_prob = torch.sigmoid(self.phi)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        """Modified loss function - same as before"""
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)

        N, D, T = self.Y.shape
        event_times_tensor = torch.tensor(event_times, dtype=torch.long)
        event_times_expanded = event_times_tensor.unsqueeze(-1)
        time_grid = torch.arange(T, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        mask_before_event = (time_grid < event_times_expanded).float()
        mask_at_event = (time_grid == event_times_expanded).float()

        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))
        total_data_loss = (loss_censored + loss_event + loss_no_event) / self.N

        # GP prior loss
        if self.gpweight > 0:
            gp_loss = self.compute_gp_prior_loss()
        else:
            gp_loss = 0.0

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

        total_loss = total_data_loss + self.gpweight * gp_loss + self.lrtpen * signature_update_loss / (self.N * self.T)
        return total_loss

    def compute_gp_prior_loss(self):
        """
        Vectorized GP prior loss with TIME-VARYING genetic effects (NEW!)
        """
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        L_phi = torch.linalg.cholesky(self.K_phi)

        # Lambda GP prior - MODIFIED to include genetic slope
        # Mean = r_k + g_i^T * gamma_level + t * g_i^T * gamma_slope
        mean_lambda = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)

        for k in range(self.K):
            # Level effect: constant over time
            level_effect = self.genetic_scale * (self.G @ self.gamma_level[:, k])  # (N,)
            # Slope effect: varies with time (NEW!)
            slope_effect = self.genetic_scale * (self.G @ self.gamma_slope[:, k])  # (N,)

            # Broadcast to (N, T)
            # mean = r_k + level_effect + t_centered * slope_effect
            mean_lambda[:, k, :] = (
                self.signature_refs[k] +
                level_effect.unsqueeze(1) +
                self.t_centered.unsqueeze(0) * slope_effect.unsqueeze(1)
            )

        # Healthy state
        if self.healthy_ref is not None:
            mean_lambda[:, self.K, :] = self.healthy_ref

        # Compute deviations and loss (vectorized, same as before)
        deviations_lambda = self.lambda_ - mean_lambda
        deviations_flat = deviations_lambda.reshape(-1, self.T)
        deviations_flat_T = deviations_flat.T
        v_flat_T = torch.cholesky_solve(deviations_flat_T, L_lambda)
        v_flat = v_flat_T.T
        gp_loss_lambda = 0.5 * torch.sum(deviations_flat * v_flat)

        # Phi GP prior (unchanged)
        mean_phi = self.logit_prev_t.unsqueeze(0) + self.psi.unsqueeze(2)
        deviations_phi = self.phi - mean_phi
        deviations_phi_flat = deviations_phi.reshape(-1, self.T)
        deviations_phi_flat_T = deviations_phi_flat.T
        v_phi_flat_T = torch.cholesky_solve(deviations_phi_flat_T, L_phi)
        v_phi_flat = v_phi_flat_T.T
        gp_loss_phi = 0.5 * torch.sum(deviations_phi_flat * v_phi_flat)

        return gp_loss_lambda / self.N + gp_loss_phi / self.D

    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01):
        """Modified fit with gamma_slope in optimizer (NEW!)"""

        param_groups = [
            {'params': [self.lambda_], 'lr': learning_rate},
            {'params': [self.phi], 'lr': learning_rate * 0.1},
            {'params': [self.psi], 'lr': learning_rate * 0.1},
            {'params': [self.gamma_level], 'weight_decay': lambda_reg, 'lr': learning_rate},
        ]

        # Add gamma_slope if learning it (NEW!)
        if self.learn_slope:
            param_groups.append({
                'params': [self.gamma_slope],
                'weight_decay': lambda_reg,  # Same regularization as gamma_level
                'lr': learning_rate * 0.5  # Slightly slower LR for stability
            })

        if isinstance(self.kappa, nn.Parameter):
            param_groups.append({'params': [self.kappa], 'lr': learning_rate})

        optimizer = optim.Adam(param_groups)

        gradient_history = {
            'lambda_grad': [],
            'phi_grad': [],
            'gamma_level_grad': [],
            'gamma_slope_grad': []
        }

        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()

            gradient_history['lambda_grad'].append(self.lambda_.grad.clone().detach())
            gradient_history['phi_grad'].append(self.phi.grad.clone().detach())
            gradient_history['gamma_level_grad'].append(self.gamma_level.grad.clone().detach())
            if self.learn_slope:
                gradient_history['gamma_slope_grad'].append(self.gamma_slope.grad.clone().detach())

            optimizer.step()
            losses.append(loss.item())

            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}, Loss: {loss.item():.4f}")
                if self.learn_slope:
                    print(f"  Gamma_slope range: [{self.gamma_slope.data.min():.4f}, {self.gamma_slope.data.max():.4f}]")
                    print(f"  Gamma_slope grad norm: {self.gamma_slope.grad.norm().item():.4f}")

        return losses, gradient_history

    def analyze_genetic_slopes(self):
        """NEW! Analyze learned genetic slope parameters"""
        print("\n" + "=" * 70)
        print("GENETIC SLOPE ANALYSIS")
        print("=" * 70)

        with torch.no_grad():
            for k in range(self.K):
                print(f"\nSignature {k}:")
                print(f"  Gamma_level range: [{self.gamma_level[:, k].min():.4f}, {self.gamma_level[:, k].max():.4f}]")
                print(f"  Gamma_slope range: [{self.gamma_slope[:, k].min():.4f}, {self.gamma_slope[:, k].max():.4f}]")

                # Find individuals with high/low first genetic component
                first_pc = self.G[:, 0]
                high_idx = first_pc > first_pc.quantile(0.75)
                low_idx = first_pc < first_pc.quantile(0.25)

                # Compute mean slopes for high/low groups
                level_high = (self.G[high_idx] @ self.gamma_level[:, k]).mean()
                level_low = (self.G[low_idx] @ self.gamma_level[:, k]).mean()
                slope_high = (self.G[high_idx] @ self.gamma_slope[:, k]).mean()
                slope_low = (self.G[low_idx] @ self.gamma_slope[:, k]).mean()

                print(f"  High PRS group (top 25%):")
                print(f"    Level effect: {level_high:.4f}")
                print(f"    Slope effect: {slope_high:.4f} per year")
                print(f"  Low PRS group (bottom 25%):")
                print(f"    Level effect: {level_low:.4f}")
                print(f"    Slope effect: {slope_low:.4f} per year")
                print(f"  Difference (high - low):")
                print(f"    Level: {level_high - level_low:.4f}")
                print(f"    Slope: {slope_high - slope_low:.4f} per year")

    def visualize_slopes(self, n_individuals=10):
        """NEW! Visualize lambda trajectories showing genetic slope effects"""
        with torch.no_grad():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Select individuals with high/low first genetic component
            first_pc = self.G[:, 0]
            high_idx = torch.where(first_pc > first_pc.quantile(0.9))[0][:n_individuals]
            low_idx = torch.where(first_pc < first_pc.quantile(0.1))[0][:n_individuals]

            ages = np.arange(30, 30 + self.T)

            # Plot lambda trajectories for first signature
            ax = axes[0, 0]
            for idx in high_idx:
                ax.plot(ages, self.lambda_[idx, 0, :].numpy(), 'r-', alpha=0.3)
            for idx in low_idx:
                ax.plot(ages, self.lambda_[idx, 0, :].numpy(), 'b-', alpha=0.3)

            high_mean = self.lambda_[high_idx, 0, :].mean(dim=0).numpy()
            low_mean = self.lambda_[low_idx, 0, :].mean(dim=0).numpy()
            ax.plot(ages, high_mean, 'r-', linewidth=3, label='High PRS')
            ax.plot(ages, low_mean, 'b-', linewidth=3, label='Low PRS')
            ax.set_xlabel('Age')
            ax.set_ylabel('λ (Signature 0)')
            ax.set_title('Lambda Trajectories: High vs. Low PRS')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot theta trajectories
            theta = torch.softmax(self.lambda_, dim=1)
            ax = axes[0, 1]
            for idx in high_idx:
                ax.plot(ages, theta[idx, 0, :].numpy(), 'r-', alpha=0.3)
            for idx in low_idx:
                ax.plot(ages, theta[idx, 0, :].numpy(), 'b-', alpha=0.3)

            ax.plot(ages, theta[high_idx, 0, :].mean(dim=0).numpy(), 'r-', linewidth=3, label='High PRS')
            ax.plot(ages, theta[low_idx, 0, :].mean(dim=0).numpy(), 'b-', linewidth=3, label='Low PRS')
            ax.set_xlabel('Age')
            ax.set_ylabel('θ (Signature 0)')
            ax.set_title('Theta Trajectories (after softmax)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot gamma_level vs gamma_slope
            ax = axes[1, 0]
            for k in range(min(self.K, 3)):
                ax.scatter(self.gamma_level[0, k].item(), self.gamma_slope[0, k].item(),
                          s=100, label=f'Signature {k}')
            ax.set_xlabel('Gamma_level (first genetic feature)')
            ax.set_ylabel('Gamma_slope (first genetic feature)')
            ax.set_title('Level vs. Slope Effects')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

            # Plot heatmap of gamma_slope
            ax = axes[1, 1]
            im = ax.imshow(self.gamma_slope.numpy(), aspect='auto', cmap='RdBu_r',
                          vmin=-0.02, vmax=0.02)
            ax.set_xlabel('Signature')
            ax.set_ylabel('Genetic Feature')
            ax.set_title('Gamma_slope (Progression Speed Effects)')
            plt.colorbar(im, ax=ax)

            plt.tight_layout()
            return fig

    # Keep all other methods from original (visualize_clusters, etc.)
    def visualize_clusters(self, disease_names):
        """Same as original"""
        if not hasattr(self, 'clusters'):
            raise ValueError("Model must be initialized with clusters")

        Y_avg = torch.mean(self.Y, dim=2)
        print("\nCluster Assignments:")
        for k in range(self.K):
            print(f"\nCluster {k}:")
            cluster_diseases = [disease_names[i] for i in range(len(self.clusters))
                              if self.clusters[i] == k]
            cluster_mask = (self.clusters == k)
            prevalences = Y_avg[:, cluster_mask].mean(dim=0)
            for disease, prev in zip(cluster_diseases, prevalences):
                print(f"  - {disease} (prevalence: {prev:.4f})")


def subset_data(Y, E, G, start_index, end_index):
    """Utility function for subsetting data"""
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices
