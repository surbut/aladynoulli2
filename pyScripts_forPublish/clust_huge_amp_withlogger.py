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

class AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, R,W,prevalence_t, init_sd_scaler, genetic_scale,
                 signature_references=None, healthy_reference=None, disease_names=None, flat_lambda=False, learn_kappa=True):
        super().__init__()
        # Basic dimensions and settings
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.gpweight=W
        self.jitter = 1e-6
        # Make kappa configurable
        if learn_kappa:
            self.kappa = nn.Parameter(torch.ones(1))  # Learnable kappa
        else:
            self.kappa = torch.ones(1)  # Fixed kappa
        self.lrtpen = R# Stronger LRT penalty
        # Fixed kernel parameters
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
        self.init_amplitude = 1.0  # Fixed initial amplitude for both kernels
        # Fixed amplitude as hyperparameter
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
        self.phi_amplitude=1
        self.lambda_amplitude=1

        # Add jitter and store
        jitter_matrix = self.jitter * torch.eye(T)

        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(self.T) # Phi kernel stays fixed
          # Only for initialization
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(self.T)
        
        # Remove learnable amplitude
        #self.log_lambda_amplitude = nn.Parameter(torch.tensor(1.0))  # DELETE
        
         # or whatever value works well
        
        # Store other needed values (prevalence, etc.)
        self.psi = None 
        self.disease_names = disease_names
          
    # Handle signature references
        if flat_lambda:
            self.signature_refs = torch.zeros(K)
            self.genetic_scale=genetic_scale   # 
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
        
        # Initialize parameters (lambda, phi, gamma, psi)
        self.initialize_params()

    def initialize_params(self, psi_config=None, true_psi=None, **kwargs):
        """Initialize parameters with K disease clusters plus one healthy cluster"""
        Y_avg = torch.mean(self.Y, dim=2)
        epsilon = 1e-6  # Increased epsilon for better numerical stability
        # Clamp Y_avg to valid probability range before logit transform
        Y_avg = torch.clamp(Y_avg, epsilon, 1.0 - epsilon)
        Y_avg = torch.log(Y_avg/(1-Y_avg))  # Now safe to do logit transform

        # Initialize psi for disease clusters
        if true_psi is not None:
            # Use provided psi and add healthy cluster
            psi_init = torch.zeros((self.K_total, self.D))
            psi_init[:self.K, :] = true_psi  # Copy disease clusters
            if self.healthy_ref is not None:
                psi_init[self.K, :] = -5.0 + 0.01 * torch.randn(self.D)  # Add healthy cluster
      
        elif psi_config is not None:
            # Initialize with config and add healthy cluster
            psi_init = torch.zeros((self.K_total, self.D))
            for k in range(self.K):
                cluster_mask = (self.clusters == k)
                psi_init[k, cluster_mask] = psi_config['in_cluster'] + psi_config['noise_in'] * torch.randn(cluster_mask.sum())
                psi_init[k, ~cluster_mask] = psi_config['out_cluster'] + psi_config['noise_out'] * torch.randn((~cluster_mask).sum())
            if self.healthy_ref is not None:
                psi_init[self.K, :] = -5.0 + 0.01 * torch.randn(self.D)

        else:
            # Original clustering code for disease clusters
            Y_corr = torch.corrcoef(Y_avg.T)
            # Handle NaN values in correlation matrix
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
            
            # Initialize psi with cluster deviations
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

        # Initialize phi for disease clusters
        for k in range(self.K):
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps
   

        # Initialize lambda and gamma for disease clusters
        for k in range(self.K):
            print(f"\nCalculating gamma for k={k}:")
            if true_psi is None:
                cluster_diseases = (self.clusters == k)
                base_value = Y_avg[:, cluster_diseases].mean(dim=1)
                base_value_centered = base_value - base_value.mean()
                print(f"Number of diseases in cluster: {cluster_diseases.sum()}")
            else:
                # Use diseases with strong psi values for this signature
                strong_diseases = (true_psi[k] > 0).float()
                base_value = Y_avg[:, strong_diseases > 0].mean(dim=1)
                base_value_centered = base_value - base_value.mean()
                print(f"Number of diseases in cluster: {strong_diseases.sum()}")
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
        epsilon=1e-6
        phi_prob = torch.sigmoid(self.phi)
        # Apply fixed kappa
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, epsilon, 1-epsilon)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        """Modified loss function with vectorized LRT updates"""
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

        # GP prior loss
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
        # compute with amplitude above 
        # Fixed K_lambda using amplitude of 2.0
        
        
        # Initialize losses
        gp_loss_lambda = 0.0
        gp_loss_phi = 0.0
        
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
            
        # Phi GP prior (unchanged, uses fixed K_phi)
        L_phi = torch.linalg.cholesky(self.K_phi)
        for k in range(self.K_total):
            phi_k = self.phi[k]  # D x T
            for d in range(self.D):
                mean_phi_d = self.logit_prev_t[d, :] + self.psi[k, d]
                dev_d = (phi_k[d:d+1, :] - mean_phi_d).T
                v_d = torch.cholesky_solve(dev_d, L_phi)
                gp_loss_phi += 0.5 * torch.sum(v_d.T @ dev_d)
        
        # Return combined loss with appropriate scaling
        return gp_loss_lambda / self.N + gp_loss_phi / self.D

        
    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01):
        """Modified fit method with separate learning rates"""
        
        # Create parameter groups based on whether kappa is learnable
        param_groups = [
            {'params': [self.lambda_], 'lr': learning_rate},      # e.g. 1e-2
            {'params': [self.phi], 'lr': learning_rate * 0.1},
            {'params': [self.psi], 'lr': learning_rate*0.1},          
            {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': learning_rate}
     
        ]
        
        # Add kappa to optimizer only if it's learnable
        if isinstance(self.kappa, nn.Parameter):
            param_groups.append({'params': [self.kappa], 'lr': learning_rate})
        
        optimizer = optim.Adam(param_groups)

        #Initialize gradient history
        gradient_history = {
            'lambda_grad': [],
            'phi_grad': []
        }
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            # Remove kappa gradient printing
            
            gradient_history['lambda_grad'].append(self.lambda_.grad.clone().detach())
            gradient_history['phi_grad'].append(self.phi.grad.clone().detach())
        
            optimizer.step()
            losses.append(loss.item())

            with torch.no_grad():
                gp = self.compute_gp_prior_loss().item()
                total = loss.item()
                scaled_gp = float(self.gpweight) * gp
                frac = scaled_gp / max(total, 1e-12)
                print({"data_plus_other": total, "gp_loss": gp, "scaled_gp": scaled_gp, "gp_frac": frac})
                
            if epoch % 1 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss.item():.4f}")
                self.analyze_signature_responses()  # Call as method

        
        return losses, gradient_history
    
### now  do some viausliaziont ###

    def visualize_clusters(self, disease_names):
        """
        Visualize cluster assignments and disease names
        
        Parameters:
        disease_names: list of disease names corresponding to columns in Y
        """
        if not hasattr(self, 'clusters'):
            raise ValueError("Model must be initialized with clusters before visualization. Call initialize_params() first.")
            
        Y_avg = torch.mean(self.Y, dim=2)
        
        print("\nCluster Assignments:")
        for k in range(self.K):
            print(f"\nCluster {k}:")
            cluster_diseases = [disease_names[i] for i in range(len(self.clusters)) 
                            if self.clusters[i] == k]
            # Get prevalence for each disease
            cluster_mask = (self.clusters == k)
            prevalences = Y_avg[:, cluster_mask].mean(dim=0)
            
            for disease, prev in zip(cluster_diseases, prevalences):
                print(f"  - {disease} (prevalence: {prev:.4f})")
        
        if self.healthy_ref is not None:
            print(f"\nHealthy State (Topic {self.K}):")
            print(f"Mean psi value: {self.psi[self.K].mean().item():.4f}")


    def analyze_signature_responses(self, top_n=5):
        ## look at tehta around a window 
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
                    theta_vals = theta[diagnosed, max_sig, :] ## time around disease window
                    theta_others = theta[~diagnosed, max_sig, :]
                    
                    print(f"\nDisease {d} (signature {max_sig}, LR={lr:.2f}):")
                    print(f"  Theta for diagnosed: {theta_vals.mean():.3f} ± {theta_vals.std():.3f}")
                    print(f"  Theta for others: {theta_others.mean():.3f}")
                    print(f"  Proportion difference: {(theta_vals.mean() - theta_others.mean()):.3f}")

    def plot_initial_params(self, n_samples=5):
        """
        Visualize initial parameters for sample diseases and individuals
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        time_points = np.arange(self.T)
        
        # Sample indices
        disease_idx = np.random.choice(self.D, n_samples, replace=False)
        indiv_idx = np.random.choice(self.N, n_samples, replace=False)
        
        # Plot psi (static over time, but we'll repeat it)
        for k in range(2):  # Plot for two different K values
            for d in range(n_samples):
                axes[0,k].plot([0, self.T], 
                            self.psi[k,disease_idx[d]].detach().numpy(), 
                            label=f'Disease {disease_idx[d]}')
            axes[0,k].set_title(f'Psi values (K={k})')
            axes[0,k].set_xlabel('Time')
            axes[0,k].legend()
        
        # Plot phi
        for k in range(2):
            for d in range(n_samples):
                axes[1,k].plot(time_points, 
                            self.phi[k,disease_idx[d],:].detach().numpy(), 
                            label=f'Disease {disease_idx[d]}')
            axes[1,k].set_title(f'Phi values (K={k})')
            axes[1,k].set_xlabel('Time')
            axes[1,k].legend()
        
        # Plot lambda
        for k in range(2):
            for i in range(n_samples):
                axes[2,k].plot(time_points, 
                            self.lambda_[indiv_idx[i],k,:].detach().numpy(), 
                            label=f'Individual {indiv_idx[i]}')
            axes[2,k].set_title(f'Lambda values (K={k})')
            axes[2,k].set_xlabel('Time')
            axes[2,k].legend()
        
        plt.tight_layout()
        plt.show()

        # Print cluster membership for sampled diseases
        print("\nCluster membership for sampled diseases:")
        for d in disease_idx:
            print(f"Disease {d}: Cluster {self.clusters[d]}")
    def visualize_initialization(self):
        """Visualize all initial parameters and cluster structure"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cluster assignments and psi (2 plots)
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
        
        # 2. Lambda trajectories for different states
        ax3 = plt.subplot(3, 2, 3)
        for k in range(self.K):
            # Plot first 3 individuals for each state
            for i in range(min(3, self.N)):
                ax3.plot(self.lambda_[i, k, :].data.numpy(), 
                        alpha=0.7, label=f'Individual {i}, State {k}')
        ax3.set_title('λ Trajectories (Sample Individuals)')
        ax3.set_xlabel('Time')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Phi trajectories
        ax4 = plt.subplot(3, 2, 4)
        for k in range(self.K):
            # Plot first 2 diseases for each state
            for d in range(min(2, self.D)):
                ax4.plot(self.phi[k, d, :].data.numpy(), 
                        alpha=0.7, label=f'State {k}, Disease {d}')
        ax4.set_title('φ Trajectories (Sample Diseases)')
        ax4.set_xlabel('Time')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Gamma (genetic effects)
        ax5 = plt.subplot(3, 2, 5)
        im5 = ax5.imshow(self.gamma.data.numpy(), aspect='auto', cmap='RdBu_r')
        ax5.set_title('γ (Genetic Effects)')
        ax5.set_xlabel('State')
        ax5.set_ylabel('Genetic Component')
        plt.colorbar(im5, ax=ax5)
        
        # 5. Print summary statistics
        plt.subplot(3, 2, 6)
        plt.axis('off')
        stats_text = (
            f"Parameter Ranges:\n"
            f"ψ: [{self.psi.data.min():.3f}, {self.psi.data.max():.3f}]\n"
            f"λ: [{self.lambda_.data.min():.3f}, {self.lambda_.data.max():.3f}]\n"
            f"φ: [{self.phi.data.min():.3f}, {self.phi.data.max():.3f}]\n"
            f"γ: [{self.gamma.data.min():.3f}, {self.gamma.data.max():.3f}]\n\n"
            f"Cluster Sizes:\n"
        )
        for k in range(self.K):
            stats_text += f"Cluster {k}: {(self.clusters == k).sum()} diseases\n"
        plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def check_gp_kernels(self):
        """Check GP kernel initialization"""
        print(f"T = {self.T}")
        print(f"lambda_length_scale = {self.lambda_length_scale}")
        print(f"phi_length_scale = {self.phi_length_scale}")
        
        # Print kernel matrices for first state
        print("\nLambda kernel (first 5x5):")
        print(self.K_lambda[0][:5, :5].detach().numpy())
        print("\nPhi kernel (first 5x5):")
        print(self.K_phi[0][:5, :5].detach().numpy())
        
        # Check condition numbers
        print("\nCondition numbers:")
        print(f"Lambda kernel: {torch.linalg.cond(self.K_lambda[0]).item():.2f}")
        print(f"Phi kernel: {torch.linalg.cond(self.K_phi[0]).item():.2f}")
        
        # Check Cholesky factors
        L_lambda = torch.linalg.cholesky(self.K_lambda[0])
        L_phi = torch.linalg.cholesky(self.K_phi[0])
        
        print("\nCholesky factor norms:")
        print(f"Lambda: {torch.norm(L_lambda).item():.2f}")
        print(f"Phi: {torch.norm(L_phi).item():.2f}")
        
        # Sample and plot trajectories to check smoothness
        n_samples = 5
        times = torch.arange(self.T)
        samples_lambda = torch.zeros((n_samples, self.T))
        samples_phi = torch.zeros((n_samples, self.T))
        
        for i in range(n_samples):
            eps_lambda = L_lambda @ torch.randn(self.T)
            eps_phi = L_phi @ torch.randn(self.T)
            samples_lambda[i] = eps_lambda
            samples_phi[i] = eps_phi
        
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.title("Lambda GP samples")
        plt.plot(times.numpy(), samples_lambda.T.numpy())
        plt.subplot(122)
        plt.title("Phi GP samples")
        plt.plot(times.numpy(), samples_phi.T.numpy())
        plt.show()


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]  # Changed from slice to index list
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices

