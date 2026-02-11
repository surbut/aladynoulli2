"""
Reparameterized version of clust_huge_amp_vectorized: gamma and psi flow through the NLL.

Key difference from original:
- Original: lambda_ and phi are free parameters; gamma and psi only in GP prior (indirect learning).
- Reparam: lambda = mean_lambda(gamma) + delta, phi = mean_phi(psi) + epsilon.
  Delta and epsilon have GP priors. Gamma and psi are in the forward pass -> get NLL gradients.

Use: Same interface as clust_huge_amp_vectorized. Swap the import in run_aladyn_batch_vector_e_censor_nolor.py
     to compare: from clust_huge_amp_vectorized_reparam import *
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
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns


class AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(nn.Module):
    """
    Reparameterized: lambda = mean(gamma) + delta, phi = mean(psi) + epsilon.
    Gamma and psi get NLL gradients (full chain rule).
    """
    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 signature_references=None, healthy_reference=None, disease_names=None, flat_lambda=False, learn_kappa=True):
        super().__init__()
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.gpweight = W
        self.jitter = 1e-6
        if learn_kappa:
            self.kappa = nn.Parameter(torch.ones(1))
        else:
            self.kappa = torch.ones(1)
        self.lrtpen = R
        self.lambda_length_scale = T / 4
        self.phi_length_scale = T / 3
        self.init_amplitude = 1.0
        self.lambda_amplitude_init = init_sd_scaler
        self.phi_amplitude_init = init_sd_scaler

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

        self.psi = None
        self.disease_names = disease_names

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

        self.G = torch.tensor(G, dtype=torch.float32)
        G_centered = G - G.mean(axis=0, keepdims=True)
        G_scaled = G_centered / G_centered.std(axis=0, keepdims=True)
        self.G = torch.tensor(G_scaled, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )
        self.initialize_params()

    def get_mean_lambda(self):
        """Mean of lambda GP: signature_refs + genetic_scale * (G @ gamma)"""
        mean_lambda = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        for k in range(self.K):
            mean_lambda[:, k, :] = self.signature_refs[k].unsqueeze(0).unsqueeze(1) + \
                self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
        if self.healthy_ref is not None:
            mean_lambda[:, self.K, :] = self.healthy_ref.unsqueeze(0).unsqueeze(1)
        return mean_lambda

    def get_mean_phi(self):
        """Mean of phi GP: logit_prev_t + psi"""
        return self.logit_prev_t.unsqueeze(0) + self.psi.unsqueeze(2)

    def get_lambda(self):
        """lambda = mean(gamma) + delta -- gamma flows through NLL"""
        return self.get_mean_lambda() + self.delta

    def get_phi(self):
        """phi = mean(psi) + epsilon -- psi flows through NLL"""
        return self.get_mean_phi() + self.epsilon

    @property
    def lambda_(self):
        """For backward compatibility with code that expects self.lambda_"""
        return self.get_lambda()

    @property
    def phi(self):
        """For backward compatibility (save, visualize, etc.)"""
        return self.get_phi()

    def initialize_params(self, psi_config=None, true_psi=None, **kwargs):
        """Initialize gamma, psi, delta, epsilon. Delta and epsilon are the GP residuals."""
        Y_avg = torch.mean(self.Y, dim=2)
        eps_num = 1e-6
        Y_avg = torch.clamp(Y_avg, eps_num, 1.0 - eps_num)
        Y_avg = torch.log(Y_avg / (1 - Y_avg))

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

        for k in range(self.K):
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps

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

        # Reparameterization: delta = lambda - mean_lambda, epsilon = phi - mean_phi
        mean_lambda_init = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        for k in range(self.K):
            mean_lambda_init[:, k, :] = self.signature_refs[k].unsqueeze(0).unsqueeze(1) + \
                self.genetic_scale * (self.G @ gamma_init[:, k]).unsqueeze(1)
        if self.healthy_ref is not None:
            mean_lambda_init[:, self.K, :] = self.healthy_ref.unsqueeze(0).unsqueeze(1)
        mean_phi_init = self.logit_prev_t.unsqueeze(0) + psi_init.unsqueeze(2)
        delta_init = lambda_init - mean_lambda_init
        epsilon_init = phi_init - mean_phi_init

        self.gamma = nn.Parameter(gamma_init)
        self.psi = nn.Parameter(psi_init)
        self.delta = nn.Parameter(delta_init)
        self.epsilon = nn.Parameter(epsilon_init)

        if self.healthy_ref is not None:
            print(f"Initializing with {self.K} disease states + 1 healthy state (REPARAM)")
        else:
            print(f"Initializing with {self.K} disease states only (REPARAM)")
        print("Reparameterized init complete: gamma, psi in NLL path; delta, epsilon have GP prior.")

    def forward(self):
        lam = self.get_lambda()
        theta = torch.softmax(lam, dim=1)
        phi_val = self.get_phi()
        phi_prob = torch.sigmoid(phi_val)
        eps_clamp = 1e-6
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, eps_clamp, 1 - eps_clamp)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
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
        total_data_loss = (loss_censored + loss_event + loss_no_event) / (self.N)

        if self.gpweight > 0:
            gp_loss = self.compute_gp_prior_loss()
        else:
            gp_loss = 0.0
        signature_update_loss = 0.0

        if self.lrtpen > 0:
            diagnoses = self.Y
            phi_avg = phi_prob.mean(dim=2)
            lam = self.get_lambda()
            for d in range(self.D):
                if torch.any(diagnoses[:, d, :]):
                    spec_d = phi_avg[:, d]
                    max_sig = torch.argmax(spec_d)
                    other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (self.K_total - 1)
                    lr = spec_d[max_sig] / (other_mean + epsilon)
                    if lr > 2:
                        diagnosis_mask = diagnoses[:, d, :].bool()
                        patient_idx, time_idx = torch.where(diagnosis_mask)
                        lambda_at_diagnosis = lam[patient_idx, max_sig, time_idx]
                        target_value = 2.0
                        disease_prevalence = diagnoses[:, d, :].float().mean() + epsilon
                        prevalence_scaling = min(0.1 / disease_prevalence, 10.0)
                        signature_update_loss += torch.sum(
                            torch.log(lr) * prevalence_scaling * (target_value - lambda_at_diagnosis)
                        )

        total_loss = total_data_loss + self.gpweight * gp_loss + self.lrtpen * signature_update_loss / (self.N * self.T)
        return total_loss

    def compute_gp_prior_loss(self):
        """GP prior on delta and epsilon only (residuals). Gamma and psi are in the mean."""
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        L_phi = torch.linalg.cholesky(self.K_phi)
        deviations_flat = self.delta.reshape(-1, self.T)
        deviations_flat_T = deviations_flat.T
        v_flat_T = torch.cholesky_solve(deviations_flat_T, L_lambda)
        v_flat = v_flat_T.T
        gp_loss_lambda = 0.5 * torch.sum(deviations_flat * v_flat)
        deviations_phi_flat = self.epsilon.reshape(-1, self.T)
        deviations_phi_flat_T = deviations_phi_flat.T
        v_phi_flat_T = torch.cholesky_solve(deviations_phi_flat_T, L_phi)
        v_phi_flat = v_phi_flat_T.T
        gp_loss_phi = 0.5 * torch.sum(deviations_phi_flat * v_phi_flat)
        return gp_loss_lambda / self.N + gp_loss_phi / self.D

    def fit(self, event_times, num_epochs=100, learning_rate=0.01):
        param_groups = [
            {'params': [self.delta], 'lr': learning_rate},
            {'params': [self.epsilon], 'lr': learning_rate * 0.1},
            {'params': [self.psi], 'lr': learning_rate * 0.1},
            {'params': [self.gamma], 'lr': learning_rate},
        ]
        if isinstance(self.kappa, nn.Parameter):
            param_groups.append({'params': [self.kappa], 'lr': learning_rate})
        optimizer = optim.Adam(param_groups)
        gradient_history = {'delta_grad': [], 'epsilon_grad': []}
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            gradient_history['delta_grad'].append(self.delta.grad.clone().detach())
            gradient_history['epsilon_grad'].append(self.epsilon.grad.clone().detach())
            optimizer.step()
            losses.append(loss.item())
            if epoch % 1 == 0:
                print(f"\nEpoch {epoch} (REPARAM)")
                print(f"Loss: {loss.item():.4f}")
                self.analyze_signature_responses()
        return losses, gradient_history

    def visualize_clusters(self, disease_names):
        if not hasattr(self, 'clusters'):
            raise ValueError("Model must be initialized with clusters before visualization.")
        Y_avg = torch.mean(self.Y, dim=2)
        print("\nCluster Assignments:")
        for k in range(self.K):
            print(f"\nCluster {k}:")
            cluster_diseases = [disease_names[i] for i in range(len(self.clusters)) if self.clusters[i] == k]
            cluster_mask = (self.clusters == k)
            prevalences = Y_avg[:, cluster_mask].mean(dim=0)
            for disease, prev in zip(cluster_diseases, prevalences):
                print(f"  - {disease} (prevalence: {prev:.4f})")
        if self.healthy_ref is not None:
            print(f"\nHealthy State (Topic {self.K}):")
            print(f"Mean psi value: {self.psi[self.K].mean().item():.4f}")

    def analyze_signature_responses(self, top_n=5):
        with torch.no_grad():
            pi, theta, phi_prob = self.forward()
            phi_avg = phi_prob.mean(dim=2)
            print(f"\nMonitoring signature responses (REPARAM):")
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

    def plot_initial_params(self, n_samples=5):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        time_points = np.arange(self.T)
        disease_idx = np.random.choice(self.D, n_samples, replace=False)
        indiv_idx = np.random.choice(self.N, n_samples, replace=False)
        for k in range(2):
            for d in range(n_samples):
                axes[0, k].plot([0, self.T], self.psi[k, disease_idx[d]].detach().numpy(), label=f'Disease {disease_idx[d]}')
            axes[0, k].set_title(f'Psi values (K={k})')
            axes[0, k].set_xlabel('Time')
            axes[0, k].legend()
        phi_val = self.get_phi()
        for k in range(2):
            for d in range(n_samples):
                axes[1, k].plot(time_points, phi_val[k, disease_idx[d], :].detach().numpy(), label=f'Disease {disease_idx[d]}')
            axes[1, k].set_title(f'Phi values (K={k})')
            axes[1, k].set_xlabel('Time')
            axes[1, k].legend()
        lam = self.get_lambda()
        for k in range(2):
            for i in range(n_samples):
                axes[2, k].plot(time_points, lam[indiv_idx[i], k, :].detach().numpy(), label=f'Individual {indiv_idx[i]}')
            axes[2, k].set_title(f'Lambda values (K={k})')
            axes[2, k].set_xlabel('Time')
            axes[2, k].legend()
        plt.tight_layout()
        plt.show()
        print("\nCluster membership for sampled diseases:")
        for d in disease_idx:
            print(f"Disease {d}: Cluster {self.clusters[d]}")

    def visualize_initialization(self):
        fig = plt.figure(figsize=(20, 15))
        ax1 = plt.subplot(3, 2, 1)
        cluster_matrix = np.zeros((self.K, self.D))
        for k in range(self.K):
            cluster_matrix[k, self.clusters == k] = 1
        im1 = ax1.imshow(cluster_matrix, aspect='auto', cmap='binary')
        ax1.set_title('Cluster Assignments')
        ax1.set_xlabel('Disease')
        ax1.set_ylabel('State')
        plt.colorbar(im1, ax=ax1)
        ax2 = plt.subplot(3, 2, 2)
        im2 = ax2.imshow(self.psi.data.numpy(), aspect='auto', cmap='RdBu_r')
        ax2.set_title('ψ (Cluster Deviations)')
        ax2.set_xlabel('Disease')
        ax2.set_ylabel('State')
        plt.colorbar(im2, ax=ax2)
        lam = self.get_lambda()
        ax3 = plt.subplot(3, 2, 3)
        for k in range(self.K):
            for i in range(min(3, self.N)):
                ax3.plot(lam[i, k, :].detach().numpy(), alpha=0.7, label=f'Individual {i}, State {k}')
        ax3.set_title('λ Trajectories (Sample Individuals)')
        ax3.set_xlabel('Time')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        phi_val = self.get_phi()
        ax4 = plt.subplot(3, 2, 4)
        for k in range(self.K):
            for d in range(min(2, self.D)):
                ax4.plot(phi_val[k, d, :].detach().numpy(), alpha=0.7, label=f'State {k}, Disease {d}')
        ax4.set_title('φ Trajectories (Sample Diseases)')
        ax4.set_xlabel('Time')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5 = plt.subplot(3, 2, 5)
        im5 = ax5.imshow(self.gamma.data.numpy(), aspect='auto', cmap='RdBu_r')
        ax5.set_title('γ (Genetic Effects)')
        ax5.set_xlabel('State')
        ax5.set_ylabel('Genetic Component')
        plt.colorbar(im5, ax=ax5)
        plt.subplot(3, 2, 6)
        plt.axis('off')
        lam = self.get_lambda()
        phi_val = self.get_phi()
        stats_text = (
            f"Parameter Ranges:\n"
            f"ψ: [{self.psi.data.min():.3f}, {self.psi.data.max():.3f}]\n"
            f"λ: [{lam.min().item():.3f}, {lam.max().item():.3f}]\n"
            f"φ: [{phi_val.min().item():.3f}, {phi_val.max().item():.3f}]\n"
            f"γ: [{self.gamma.data.min():.3f}, {self.gamma.data.max():.3f}]\n\n"
            f"Cluster Sizes:\n"
        )
        for k in range(self.K):
            stats_text += f"Cluster {k}: {(self.clusters == k).sum()} diseases\n"
        plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top')
        plt.tight_layout()
        plt.show()


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices
