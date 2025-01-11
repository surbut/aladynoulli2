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
from torch_lr_finder import LRFinder
from torch_lr_finder import LRFinder
from torch.utils.data import TensorDataset, DataLoader

class AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, prevalence_t, disease_names=None):
        super().__init__()
        self.N = N
        self.D = D
        self.T = T
        self.K = K
        self.P = P
        self.psi = None 
        self.disease_names = disease_names
        self.jitter = 1e-6

        # Convert inputs to tensors
        self.G = torch.tensor(G, dtype=torch.float32)
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

        # Initialize parameters
        self.update_kernels()
        self.initialize_params()
        
    def initialize_params(self, psi_config=None, true_psi=None, **kwargs):
        """Initialize parameters with either true psi from simulation or clustering"""
        Y_avg = torch.mean(self.Y, dim=2)
        epsilon=1e-8
        Y_avg = torch.log((Y_avg + epsilon)/(1-Y_avg+epsilon))

        if true_psi is not None:
            # Use true psi from simulation
            self.psi = nn.Parameter(true_psi)
            print("\nUsing true psi from simulation")
            
        elif psi_config is not None:
            # Use provided psi configuration
            psi_init = torch.zeros((self.K, self.D))
            for k in range(self.K):
                cluster_mask = (self.clusters == k)
                psi_init[k, cluster_mask] = psi_config['in_cluster'] + psi_config['noise_in'] * torch.randn(cluster_mask.sum())
                psi_init[k, ~cluster_mask] = psi_config['out_cluster'] + psi_config['noise_out'] * torch.randn((~cluster_mask).sum())
            self.psi = nn.Parameter(psi_init)
            print("\nUsing psi configuration")
            
        else:
            # Original clustering code
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
            psi_init = torch.zeros((self.K, self.D))
            for k in range(self.K):
                cluster_mask = (self.clusters == k)
                psi_init[k, cluster_mask] = 1.0 + 0.1 * torch.randn(cluster_mask.sum())
                psi_init[k, ~cluster_mask] = -3.0 + 0.01 * torch.randn((~cluster_mask).sum())
            
            self.psi = nn.Parameter(psi_init)
            
            print("\nCluster Sizes:")
            unique, counts = np.unique(self.clusters, return_counts=True)
            for k, count in zip(unique, counts):
                print(f"Cluster {k}: {count} diseases")
        
        # Initialize other parameters using the psi (true or clustered)
        gamma_init = torch.zeros((self.P, self.K))
        lambda_init = torch.zeros((self.N, self.K, self.T))
        phi_init = torch.zeros((self.K, self.D, self.T))
        
        # Rest of initialization remains the same
        for k in range(self.K):
            if true_psi is None:
                cluster_diseases = (self.clusters == k)
                base_value = Y_avg[:, cluster_diseases].mean(dim=1)
            else:
                # Use diseases with strong psi values for this signature 
                
                strong_diseases = (true_psi[k] > 0).float()
                base_value = Y_avg[:, strong_diseases > 0].mean(dim=1)
                # check these values in R
            gamma_init[:, k] = torch.linalg.lstsq(self.G, base_value.unsqueeze(1)).solution.squeeze()
            lambda_means = self.G @ gamma_init[:, k]
            L_k = torch.linalg.cholesky(self.K_lambda[k])
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = lambda_means[i] + eps
            
            L_phi = torch.linalg.cholesky(self.K_phi[k])
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + self.psi[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps
        
        self.gamma = nn.Parameter(gamma_init)
        self.lambda_ = nn.Parameter(lambda_init)
        self.phi = nn.Parameter(phi_init)
        
        print("Initialization complete!")
        
    def find_best_lr(self, event_times):
        """Use torch-lr-finder to automatically find best learning rate"""
        from torch_lr_finder import LRFinder
        from torch.utils.data import TensorDataset, DataLoader

        # Create dataset with both inputs and dummy labels
        dummy_labels = torch.zeros_like(event_times)  # dummy labels required by LRFinder
        dataset = TensorDataset(event_times, dummy_labels)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters())
        
        # Create LR finder with custom training step
        def training_step(batch, batch_idx):
            inputs, _ = batch  # Ignore the dummy labels
            loss = self.compute_loss(inputs)
            return loss
        
        lr_finder = LRFinder(self, optimizer, training_step, device="cpu")
        
        # Run range test with just a few iterations
        lr_finder.range_test(train_loader, end_lr=1, num_iter=20, start_lr=1e-7)
        
        # Plot results
        lr_finder.plot()
        
        # Get suggestion
        best_lr = lr_finder.suggestion()
        print(f"\nSuggested learning rate: {best_lr:.2e}")
        
        # Reset the model
        lr_finder.reset()
        
        return best_lr

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

    def forward(self, x=None):
        """Forward pass of the model. x parameter added for LRFinder compatibility"""
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
        
        for k in range(self.K):
            L_lambda = torch.linalg.cholesky(self.K_lambda[k])
            L_phi = torch.linalg.cholesky(self.K_phi[k])
            
            # Lambda GP prior (unchanged)
            lambda_k = self.lambda_[:, k, :]
            mean_lambda_k = (self.G @ self.gamma[:, k]).unsqueeze(1)
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
        
        #print("\nHealthy State (Topic {self.K-1})")


    def fit(self, event_times, num_epochs=1000, optimizer=None, lambda_reg=1e-2,
        convergence_threshold=1e-3, patience=10):
        """
        Fit model with early stopping and parameter monitoring
        """
        if optimizer is None:
            # Default optimizer if none provided
            optimizer = optim.Adam([
                {'params': [self.lambda_, self.phi], 'lr': 1e-3},
                {'params': [self.psi], 'lr': 1e-3},
                {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': 1e-3}
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
            
            if epoch % 10 == 0:
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






