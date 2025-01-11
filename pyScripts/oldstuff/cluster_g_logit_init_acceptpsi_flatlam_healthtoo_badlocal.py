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
    def __init__(self, N, D, T, K, P, G, Y, prevalence_t, signature_references=None, healthy_reference=None, disease_names=None,flat_lambda=False):
        super().__init__()
        self.N = N
        self.D = D
        self.T = T
        self.K = K
        # Make K_total conditional on whether healthy_reference is provided
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.psi = None 
        self.disease_names = disease_names
        self.jitter = 1e-6
        # Store whether to use flat lambda
        self.flat_lambda = flat_lambda
        
        # If using flat lambda, modify signature references
        
    # Handle signature references
        if flat_lambda:
            self.signature_refs = torch.zeros(K)
            self.genetic_scale=1    # Zeros instead of ones
        else:
            if signature_references is None:
                raise ValueError("signature_references must be provided when flat_lambda=False")
            self.signature_refs = torch.tensor(signature_references, dtype=torch.float32)
            self.genetic_scale = 1.0 
        if healthy_reference is not None:
            self.healthy_ref = torch.tensor(healthy_reference, dtype=torch.float32)
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
        
        # Fixed kernel parameters
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
         
        self.amplitude = 1
        self.lambda_amp=1

        # Initialize parameters
        self.update_kernels()
        self.initialize_params()
        
    def initialize_params(self, psi_config=None, true_psi=None, **kwargs):
        """Initialize parameters with K disease clusters plus one healthy cluster"""
        Y_avg = torch.mean(self.Y, dim=2)
        epsilon = 1e-8
        Y_avg = torch.log((Y_avg + epsilon)/(1-Y_avg+epsilon))

        
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
            L_phi = torch.linalg.cholesky(self.K_phi)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps

        # Initialize lambda and gamma for disease clusters
        for k in range(self.K):
            cluster_mask = (self.clusters == k)
            ''' 
            base_value = Y_avg[:, cluster_diseases].mean(dim=1)
            gamma_init[:, k] = torch.linalg.lstsq(self.G, base_value.unsqueeze(1)).solution.squeeze()
            lambda_means = self.genetic_scale * (self.G @ gamma_init[:, k])
            L_k = torch.linalg.cholesky(self.K_lambda)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = self.signature_refs[k] + lambda_means[i] + eps*0.01
            '''
            # New proposed approach:
            # Calculate log proportions for all individuals
            indiv_prop_logs = torch.zeros(self.N)
            for i in range(self.N):
                Y_counts_i = self.Y[i].sum(dim=0)  # T vec
                total_counts_i = Y_counts_i.sum() + 1e-8
                # Get proportion of their diagnoses from signature k
                indiv_signature_props = self.Y[i, cluster_mask].sum(dim=0) / total_counts_i
                indiv_prop_avg = indiv_signature_props.mean()
                indiv_prop_logs[i] = torch.log(indiv_prop_avg + epsilon)

            gamma_init[:, k] = torch.linalg.lstsq(self.G, (indiv_prop_logs - self.signature_refs[k].mean()).unsqueeze(1)).solution.squeeze()
            lambda_means = self.genetic_scale * (self.G @ gamma_init[:, k])
            L_k = torch.linalg.cholesky(self.K_lambda)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = self.signature_refs[k] + lambda_means[i] + eps*0.01

        if self.healthy_ref is not None:
            L_phi = torch.linalg.cholesky(self.K_phi)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[self.K, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[self.K, d, :] = mean_phi + eps

            L_k = torch.linalg.cholesky(self.K_lambda)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, self.K, :] = self.healthy_ref + eps*0.01
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
    
    def update_kernels(self):
        """Compute fixed covariance matrices"""
        times = torch.arange(self.T, dtype=torch.float32)
        sq_dists = (times.unsqueeze(0) - times.unsqueeze(1)) ** 2
        
        # Compute single kernel for each type
        K_lambda = self.lambda_amp ** 2 * torch.exp(-0.5 * sq_dists / (self.lambda_length_scale ** 2))
        K_phi = self.amplitude ** 2 * torch.exp(-0.5 * sq_dists / (self.phi_length_scale ** 2))
        
        # Add jitter once
        jitter_matrix = self.jitter * torch.eye(self.T)
        
        # Store single kernel for each type
        self.K_lambda = K_lambda + jitter_matrix
        self.K_phi = K_phi + jitter_matrix

        print(f"Lambda kernel condition number: {torch.linalg.cond(self.K_lambda):.2f}")
        print(f"Phi kernel condition number: {torch.linalg.cond(self.K_phi):.2f}")

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
        Compute the average GP prior loss with time-dependent mean.
        Lambda terms averaged by N, Phi terms averaged by D.
        """
        gp_loss_lambda = 0.0
        gp_loss_phi = 0.0
        
        for k in range(self.K_total):
            L_lambda = torch.linalg.cholesky(self.K_lambda)
            L_phi = torch.linalg.cholesky(self.K_phi)
            
            # Lambda GP prior (unchanged)
            lambda_k = self.lambda_[:, k, :]
       
            if k == self.K and self.healthy_ref is not None:  # Healthy state
                mean_lambda_k = self.healthy_ref.unsqueeze(0) 
            else:  # Disease signatures
                mean_lambda_k = self.signature_refs[k].unsqueeze(0) + self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
            
            deviations_lambda = lambda_k - mean_lambda_k
            for i in range(self.N):
                dev_i = deviations_lambda[i:i+1].T
                v_i = torch.cholesky_solve(dev_i, L_lambda)
                gp_loss_lambda += 0.5 * torch.sum(v_i.T @ dev_i)
            
            # Phi GP prior (updated to include psi)
            phi_k = self.phi[k]  # D x T
            for d in range(self.D):
                mean_phi_d = self.logit_prev_t[d, :] + self.psi[k, d]  # Include psi deviation
                dev_d = (phi_k[d:d+1, :] - mean_phi_d).T
                v_d = torch.cholesky_solve(dev_d, L_phi)
                gp_loss_phi += 0.5 * torch.sum(v_d.T @ dev_d)
        
        return gp_loss_lambda / (self.N ) + gp_loss_phi / (self.D)
        
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


    def fit(self, event_times, num_epochs=1000, learning_rate=1e-4, lambda_reg=1e-2,
        convergence_threshold=1e-3, patience=20):
        """
        Fit model with early stopping and parameter monitoring
        
        optimizer = optim.Adam([
            {'params': [self.lambda_, self.phi,self.psi]},
            {'params': [self.gamma], 'weight_decay': lambda_reg}
        ], lr=learning_rate)
        """

            # Create optimizer with consistent learning rates
        optimizer = optim.Adam([
            {'params': [self.lambda_, self.phi], 'lr': learning_rate},
            {'params': [self.psi], 'lr': learning_rate},  # Same base learning rate
            {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': learning_rate}
        ])

        
        history = {
            'loss': [],
            'max_grad_lambda': [],
            'max_grad_phi': [],
            'max_grad_gamma': [],
            'max_grad_psi': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        prev_loss = float('inf')
        
        print("Starting training...")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Compute loss and backprop
            loss = self.compute_loss(event_times)
            loss_val = loss.item()
            history['loss'].append(loss_val)
            loss.backward()

        
            # Get and track gradients
            grad_lambda = self.lambda_.grad.abs().max().item() if self.lambda_.grad is not None else 0
            grad_phi = self.phi.grad.abs().max().item() if self.phi.grad is not None else 0
            grad_gamma = self.gamma.grad.abs().max().item() if self.gamma.grad is not None else 0
            grad_psi = self.psi.grad.abs().max().item() if self.psi.grad is not None else 0
            
            history['max_grad_lambda'].append(grad_lambda)
            history['max_grad_phi'].append(grad_phi)
            history['max_grad_gamma'].append(grad_gamma)
            history['max_grad_psi'].append(grad_psi)

                # Monitor psi gradients
            print(f"\nEpoch {epoch}")
            print(f"Loss: {loss.item():.4f}")
            print("Psi gradient stats:")
            print(f"Mean: {self.psi.grad.mean().item():.4e}")
            print(f"Std:  {self.psi.grad.std().item():.4e}")
            print(f"Max:  {self.psi.grad.max().item():.4e}")
            print(f"Min:  {self.psi.grad.min().item():.4e}")
            
            # Check if psi is actually changing
            ## old_psi = self.psi.detach().clone()
            ## optimizer.step()
            ## psi_change = (self.psi - old_psi).abs().mean().item()
            ### print(f"Average psi change: {psi_change:.4e}")


            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val:.4f}, "
                    f"Gradients - Lambda: {grad_lambda:.3e}, Phi: {grad_phi:.3e}, "
                    f"Gamma: {grad_gamma:.3e}, Psi: {grad_psi:.3e}")

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

         
        
            optimizer.step()
            prev_loss = loss_val
            
            # Time estimate on first epoch
            if epoch == 0:
                import time
                start_time = time.time()
            elif epoch == 1:
                time_per_epoch = time.time() - start_time
                estimated_total_time = time_per_epoch * num_epochs
                print(f"\nEstimated total training time: {estimated_total_time/60:.1f} minutes")
    
        return history
    
    def visualize_healthy_state(self):
        """
        Visualize characteristics of the healthy state
        """
        # Get initial theta (proportions) from softmax of lambda
        theta = torch.softmax(self.lambda_, dim=1)
        healthy_props = theta[:, self.K, :].mean(dim=0)  # Average across individuals
        
        # Get disease probabilities for healthy state
        phi_probs = torch.sigmoid(self.phi[self.K])  # D x T
        
        print("\nHealthy State Statistics:")
        print(f"Average proportion in healthy state: {healthy_props.mean().item():.3f}")
        print(f"Range of proportions: [{healthy_props.min().item():.3f}, {healthy_props.max().item():.3f}]")
        print(f"\nDisease probabilities in healthy state:")
        print(f"Mean: {phi_probs.mean().item():.3f}")
        print(f"Range: [{phi_probs.min().item():.3f}, {phi_probs.max().item():.3f}]")
        
        # Optional: Plot distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot proportion distribution
        ax1.hist(healthy_props.detach().numpy(), bins=30)
        ax1.set_title('Distribution of Healthy State Proportions')
        ax1.set_xlabel('Proportion')
        ax1.set_ylabel('Count')
        
        # Plot disease probability distribution
        ax2.hist(phi_probs.flatten().detach().numpy(), bins=30)
        ax2.set_title('Distribution of Disease Probabilities\nin Healthy State')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def plot_genetic_scores(self, original_G=None):
        """
        Create box plots of genetic scores to compare original and transformed versions
        
        Parameters:
        original_G: numpy array or torch tensor, the original G matrix before transformation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot original G if provided
        if original_G is not None:
            if torch.is_tensor(original_G):
                original_G = original_G.numpy()
            
            ax1.boxplot([original_G[:, p] for p in range(self.P)],
                    labels=[f'Component {p+1}' for p in range(self.P)])
            ax1.set_title('Original Genetic Components')
            ax1.set_ylabel('Score')
            ax1.grid(True, alpha=0.3)
            
            # Add mean line for original plot
            orig_mean = original_G.mean()
            ax1.axhline(y=orig_mean, color='r', linestyle='--', alpha=0.5, label=f'Mean ({orig_mean:.3f})')
        
        # Plot transformed G
        G_np = self.G.numpy()
        ax2.boxplot([G_np[:, p] for p in range(self.P)],
                    labels=[f'Component {p+1}' for p in range(self.P)])
        
        # Add exact reference lines for transformed plot
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Mean (0)')
        ax2.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='±1 std')
        ax2.axhline(y=-1, color='g', linestyle='--', alpha=0.5)
        
        ax2.set_title('Transformed Genetic Components\n(Centered and Scaled)')
        ax2.set_ylabel('Standardized Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics for both
        print("\nGenetic Components Summary Statistics:")
        print(f"{'Component':<10} {'Original Mean':>13} {'Original Std':>12} {'Trans. Mean':>12} {'Trans. Std':>11}")
        print("-" * 60)
        for p in range(self.P):
            orig_mean = original_G[:, p].mean() if original_G is not None else float('nan')
            orig_std = original_G[:, p].std() if original_G is not None else float('nan')
            trans_mean = G_np[:, p].mean()
            trans_std = G_np[:, p].std()
            print(f"{p+1:<10} {orig_mean:>13.3f} {orig_std:>12.3f} {trans_mean:>12.3f} {trans_std:>11.3f}")
                  
    def plot_qq_genetic_scores(self, original_G=None, n_components=4):
        """
        Create Q-Q plots comparing original and transformed genetic components
        
        Parameters:
        original_G: original genetic components tensor/array
        n_components: number of components to plot (default=4)
        """
        import numpy as np
        from scipy import stats
        
        # Convert tensors to numpy if needed
        G_np = self.G.numpy()
        if original_G is not None:
            if torch.is_tensor(original_G):
                original_G = original_G.numpy()
        
        # Randomly select components
        selected_components = np.random.choice(self.P, size=n_components, replace=False)
        
        # Create subplot grid
        fig, axes = plt.subplots(n_components, 2, figsize=(12, 4*n_components))
        
        for idx, comp in enumerate(selected_components):
            # Original data Q-Q plot
            if original_G is not None:
                stats.probplot(original_G[:, comp], dist="norm", plot=axes[idx, 0])
                axes[idx, 0].set_title(f'Original Component {comp+1}')
            
            # Transformed data Q-Q plot
            stats.probplot(G_np[:, comp], dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'Transformed Component {comp+1}')
        
        plt.tight_layout()
        plt.show()
        # Print summary statistics
        print("\nHealthy State Disease Probabilities:")
        print(f"Mean: {phi_probs[:, :, :].mean(dim=2)[self.K].mean().item():.3f}")
        print(f"Max:  {phi_probs[:, :, :].mean(dim=2)[self.K].max().item():.3f}")
                
    def plot_disease_logits(self, n_diseases=5):
        """
        Plot logit values (phi) across states for randomly selected diseases,
        highlighting that the healthy state (K) has lower values
        """
        # Select random diseases
        random_diseases = np.random.choice(self.D, size=n_diseases, replace=False)
        
        # Create plot
        fig, axes = plt.subplots(n_diseases, 1, figsize=(10, 3*n_diseases))
        if n_diseases == 1:
            axes = [axes]
        
        for idx, d in enumerate(random_diseases):
            # Get logit values for each state
            state_logits = self.phi[:, d, :].mean(dim=1).detach().numpy()  # Average over time
            
            # Create bar plot
            bars = axes[idx].bar(range(self.K_total), state_logits)
            
            # Highlight healthy state
            bars[self.K].set_color('green')
            
            # Add disease name if available
            disease_label = f"Disease {d}" if self.disease_names is None else self.disease_names[d]
            axes[idx].set_title(disease_label)
            axes[idx].set_xlabel('State')
            axes[idx].set_ylabel('Logit Value')
            
            # Annotate healthy state
            axes[idx].text(self.K, state_logits[self.K], 
                        f'Healthy\n{state_logits[self.K]:.3f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nHealthy State Logit Values:")
        print(f"Mean: {self.phi[self.K, :, :].mean().item():.3f}")
        print(f"Max:  {self.phi[self.K, :, :].max().item():.3f}")
        
    def fit_efficient(self, event_times, num_epochs=1000, learning_rate=1e-4, lambda_reg=1e-2,
                 param_change_threshold=1e-5, consecutive_threshold=3):
        """
        Efficient fitting with parameter change monitoring
        Args:
            param_change_threshold: threshold for parameter changes
            consecutive_threshold: number of consecutive small changes before stopping
        """
        optimizer = optim.Adam([
            {'params': [self.lambda_, self.phi], 'lr': learning_rate},
            {'params': [self.psi], 'lr': learning_rate},
            {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': learning_rate}
        ])
        
        history = {
            'loss': [],
            'param_changes': [],
            'gradients': []
        }
        
        consecutive_small_changes = 0
        print("Starting efficient training...")
        
        # Store initial parameter values
        old_params = {
            'lambda': self.lambda_.detach().clone(),
            'phi': self.phi.detach().clone(),
            'psi': self.psi.detach().clone(),
            'gamma': self.gamma.detach().clone()
        }
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Compute loss and backprop
            loss = self.compute_loss(event_times)
            loss_val = loss.item()
            history['loss'].append(loss_val)
            loss.backward()
            
            # Track parameter changes every 5 epochs
            if epoch % 5 == 0:
                param_changes = {
                    'lambda': (self.lambda_ - old_params['lambda']).abs().mean().item(),
                    'phi': (self.phi - old_params['phi']).abs().mean().item(),
                    'psi': (self.psi - old_params['psi']).abs().mean().item(),
                    'gamma': (self.gamma - old_params['gamma']).abs().mean().item()
                }
                
                max_change = max(param_changes.values())
                history['param_changes'].append(param_changes)
                
                # Print progress every 10 epochs
                if epoch % 10 == 0:
                    print(f"\nEpoch {epoch}")
                    print(f"Loss: {loss_val:.4f}")
                    print(f"Max parameter change: {max_change:.2e}")
                    
                    # Store gradients for monitoring
                    if self.lambda_.grad is not None:
                        grads = {
                            'lambda': self.lambda_.grad.abs().max().item(),
                            'phi': self.phi.grad.abs().max().item(),
                            'psi': self.psi.grad.abs().max().item(),
                            'gamma': self.gamma.grad.abs().max().item()
                        }
                        history['gradients'].append(grads)
                        print(f"Max gradients: {grads}")
                
                # Check for convergence based on parameter changes
                if max_change < param_change_threshold:
                    consecutive_small_changes += 1
                    if consecutive_small_changes >= consecutive_threshold:
                        print(f"\nParameters stabilized at epoch {epoch}")
                        print(f"Final loss: {loss_val:.4f}")
                        break
                else:
                    consecutive_small_changes = 0
                
                # Update old parameters
                old_params = {
                    'lambda': self.lambda_.detach().clone(),
                    'phi': self.phi.detach().clone(),
                    'psi': self.psi.detach().clone(),
                    'gamma': self.gamma.detach().clone()
                }
            # Only update parameters if we haven't converged/stopped
            optimizer.step()
            
            # Time estimate on first epoch
            if epoch == 0:
                import time
                start_time = time.time()
            elif epoch == 1:
                time_per_epoch = time.time() - start_time
                estimated_total_time = time_per_epoch * num_epochs
                print(f"\nEstimated total training time: {estimated_total_time/60:.1f} minutes")
        
        return history
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
                            [self.psi[k,disease_idx[d]].detach().numpy(), 
                            self.psi[k,disease_idx[d]].detach().numpy()], 
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





   

## plotting code from here down
def plot_training_diagnostics(history):
    """Plot training diagnostics for fixed kernel model"""
    plt.figure(figsize=(15, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'])
    plt.yscale('log')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot gradients
    plt.subplot(2, 2, 2)
    plt.plot(history['max_grad_lambda'], label='λ')
    plt.plot(history['max_grad_phi'], label='φ')
    plt.plot(history['max_grad_gamma'], label='γ')
    plt.yscale('log')
    plt.title('Maximum Gradients')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Plot condition numbers
    plt.subplot(2, 2, 3)
    plt.plot(history['condition_number_lambda'], label='λ kernels')
    plt.plot(history['condition_number_phi'], label='φ kernels')
    plt.yscale('log')
    plt.title('Kernel Condition Numbers')
    plt.xlabel('Epoch')
    plt.ylabel('Condition Number')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage:
"""
history = model.fit(event_times, num_epochs=1000)
plot_training_diagnostics(history)
"""


# In a separate script/notebook:
def calculate_population_references(Y_full, initial_clusters, K):
    """
    Pre-compute population reference trajectories
    Returns:
        - healthy_reference: T-length vector of healthy proportions
        - signature_references: K x T matrix of signature trajectories
    """
    # Calculate and smooth trajectories as before
    ...
    return healthy_reference, signature_references


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
    length_scales = np.random.uniform(T / 4, T / 3, K)
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





def generate_clustered_survival_data(N=1000, D=20, T=50, K=5, P=5):
    """
    Generate synthetic data matching our fitted model structure
    """
    # Fixed kernel parameters as in the fit
    lambda_length = T/4
    phi_length = T/3
    amplitude = 1.0
    
    # Setup time grid
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]
    K_lambda = amplitude**2 * np.exp(-0.5 * (time_diff**2) / lambda_length**2)
    K_phi = amplitude**2 * np.exp(-0.5 * (time_diff**2) / phi_length**2)
    
        # 1. Generate more realistic baseline trajectories
    logit_prev_t = np.zeros((D, T))
    for d in range(D):
        # More diverse base rates
        base_rate = np.random.choice([
            #np.random.uniform(-18, -16),  # Very rare
            #np.random.uniform(-16, -14),  # Rare
            np.random.uniform(-14, -12),  # Uncommon
            np.random.uniform(-12, -10),  # Moderate
            np.random.uniform(-10, -8),   # Common
            np.random.uniform(-8, -6)     # Very common
        ], p=[0.40, 0.40, 0.15, 0.05])
        
        # More diverse trajectory shapes
        peak_age = np.random.uniform(20, 40)  # Wider range for peak
        # Reduce slope range
        slope = np.random.uniform(0.10, 0.4)  # More modest increase

# Increase decay to ensure prevalence plateaus
        decay = np.random.uniform(0.002, 0.01)
        
        # Add possibility of early vs late onset patterns
        onset_shift = np.random.uniform(-10, 10)
        time_points_shifted = time_points - onset_shift
        
        # Generate trajectory with more complex patterns
        logit_prev_t[d, :] = base_rate + \
                            slope * time_points_shifted - \
                            decay * (time_points_shifted - peak_age)**2 #+ \
                           # np.random.normal(0, 0.5, T)  # Add some noise
        
    # 2. Generate cluster assignments
    clusters = np.zeros(D)
    diseases_per_cluster = D // K
    for k in range(K):
        clusters[k*diseases_per_cluster:(k+1)*diseases_per_cluster] = k
    
    # 3. Generate lambda (individual trajectories)
    G = np.random.randn(N, P)  # Genetic covariates
    Gamma_k = np.random.randn(P, K)  # Genetic effects
    lambda_ik = np.zeros((N, K, T))
    
    for i in range(N):
        mean_lambda = G[i] @ Gamma_k  # Individual-specific means
        for k in range(K):
            lambda_ik[i,k,:] = multivariate_normal.rvs(
                mean=mean_lambda[k] * np.ones(T), 
                cov=K_lambda
            )
    
    # 4. Generate phi with cluster structure
    phi_kd = np.zeros((K, D, T))
    psi = np.zeros((K, D))
    
    for k in range(K):
        for d in range(D):
            # Set cluster-specific offsets
            if clusters[d] == k:
                psi[k,d] = 1.0  # In-cluster
            else:
                psi[k,d] = -3.0  # Out-cluster
                
            # Generate phi around mu_d + psi
            mean_phi = logit_prev_t[d,:] + psi[k,d]
            phi_kd[k,d,:] = multivariate_normal.rvs(mean=mean_phi, cov=K_phi)
    
    # 5. Compute probabilities
    theta = softmax(lambda_ik, axis=1)
    eta = expit(phi_kd)
    pi = np.einsum('nkt,kdt->ndt', theta, eta)
    
    # 6. Generate events
    Y = np.zeros((N, D, T))
    event_times = np.full((N, D), T)
    
    for n in range(N):
        for d in range(D):
            for t in range(T):
                if Y[n,d,:t].sum() == 0:  # Still at risk
                    if np.random.rand() < pi[n,d,t]:
                        Y[n,d,t] = 1
                        event_times[n,d] = t
                        break
    
    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'clusters': clusters,
        'logit_prev_t': logit_prev_t,
        'theta': theta,
        'phi': phi_kd,
        'lambda': lambda_ik,
        'psi': psi,
        'pi': pi
    }


def plot_synthetic_components(data, num_samples=5):
    plt.figure(figsize=(20, 12))
    
    # 1. Plot sample phi trajectories for each cluster
    plt.subplot(231)
    for k in range(data['phi'].shape[0]):  # For each cluster
        for d in range(min(num_samples, data['phi'].shape[1])):  # Sample diseases
            plt.plot(data['phi'][k,d,:], alpha=0.5, label=f'Cluster {k}' if d==0 else '')
    plt.title('Sample φ Trajectories by Cluster')
    plt.xlabel('Time')
    plt.ylabel('φ Value')
    plt.legend()
    
    # 2. Plot sample lambda trajectories
    plt.subplot(232)
    for i in range(min(num_samples, data['lambda'].shape[0])):  # Sample individuals
        for k in range(data['lambda'].shape[1]):  # For each cluster
            plt.plot(data['lambda'][i,k,:], alpha=0.5, label=f'Cluster {k}' if i==0 else '')
    plt.title('Sample λ Trajectories')
    plt.xlabel('Time')
    plt.ylabel('λ Value')
    plt.legend()
    
    # 3. Plot psi heatmap
    plt.subplot(233)
    sns.heatmap(data['psi'], cmap='RdBu_r', center=0)
    plt.title('ψ Values (Cluster-Disease Assignment)')
    plt.xlabel('Disease')
    plt.ylabel('Cluster')
    
     # 4. Plot sample theta (signature weights) as bars instead of lines
    plt.subplot(234)
    width = 0.15  # Width of bars
    x = np.arange(data['theta'].shape[1])  # Cluster indices
    
    for i in range(min(num_samples, data['theta'].shape[0])):
        plt.bar(x + i*width, data['theta'][i,:,0], 
               width, alpha=0.5, label=f'Individual {i}')
    
    plt.title('Sample θ Values (t=0)')
    plt.xlabel('Cluster')
    plt.ylabel('Weight')
    plt.legend()
    plt.xticks(x + width*2, [f'Cluster {i}' for i in x])
    
    # 5. Plot sample pi trajectories
    plt.subplot(235)
    for i in range(min(num_samples, data['pi'].shape[0])):
        for d in range(min(3, data['pi'].shape[1])):
            plt.plot(data['pi'][i,d,:], alpha=0.5, label=f'Ind {i}, Disease {d}' if i==0 else '')
    plt.title('Sample π Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def calculate_calibration_stats(model, Y):
    """Calculate calibration stats for a model"""
    with torch.no_grad():
        predicted = model.forward()
        pi_pred = predicted[0] if isinstance(predicted, tuple) else predicted
        pi_pred = pi_pred.cpu().detach().numpy()
        Y_np = Y.cpu().detach().numpy() if torch.is_tensor(Y) else Y
        
        # Convert to numpy and calculate means
        observed_risk = Y_np.mean(axis=0).flatten()
        predicted_risk = pi_pred.mean(axis=0).flatten()
        
        scale_factor = np.mean(observed_risk) / np.mean(predicted_risk)
        calibrated_risk = predicted_risk * scale_factor
        
        ss_res = np.sum((observed_risk - calibrated_risk) ** 2)
        ss_tot = np.sum((observed_risk - np.mean(observed_risk)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2, scale_factor, observed_risk, predicted_risk, calibrated_risk
