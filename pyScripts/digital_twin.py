import numpy as np
import torch
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DigitalTwin:
    def __init__(self, model, disease_names=None):
        """
        Initialize digital twin with a trained model
        
        Parameters:
        model: Trained AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest model
        disease_names: List of disease names
        """
        self.model = model
        self.disease_names = disease_names
        self.N = model.N
        self.D = model.D
        self.T = model.T
        self.K = model.K
        
        # Get model parameters
        with torch.no_grad():
            self.lambda_ = model.lambda_.detach().numpy()  # Shape (N, K, T)
            self.phi = model.phi.detach().numpy()  # Shape (K, D, T)
            self.psi = model.psi.detach().numpy()  # Shape (K, D)
            self.gamma = model.gamma.detach().numpy()  # Shape (P, K)
            self.G = model.G.detach().numpy()  # Shape (N, P)
            
        # Pre-compute theta
        self.theta = np.exp(self.lambda_) / np.sum(np.exp(self.lambda_), axis=1, keepdims=True)
        
    def create_twin(self, genetic_profile, age=None):
        """
        Create a digital twin for a new individual based on their genetic profile
        
        Parameters:
        genetic_profile: Array of genetic features (P,)
        age: Current age (optional)
        
        Returns:
        Dictionary containing twin's predicted trajectories
        """
        # Ensure genetic_profile is 1D
        genetic_profile = np.asarray(genetic_profile).squeeze()
        #print(f"genetic_profile shape: {genetic_profile.shape}")
        #print(f"G shape: {self.G.shape}")
        #print(f"gamma shape: {self.gamma.shape}")
        
        # Standardize genetic profile using the same scaling as training data
        G_mean = self.G.mean(axis=0, keepdims=True)
        G_std = self.G.std(axis=0, keepdims=True)
        genetic_profile_scaled = (genetic_profile - G_mean.squeeze()) / G_std.squeeze()
        #print(f"genetic_profile_scaled shape: {genetic_profile_scaled.shape}")
        
        # Compute genetic effects (constant over time)
        genetic_effects = np.dot(genetic_profile_scaled, self.gamma)  # Shape (K,)
        #print(f"genetic_effects shape: {genetic_effects.shape}")
        
        # Get the kernel matrix
        K_lambda = self.model.K_lambda_init.detach().numpy()
        #print(f"K_lambda shape: {K_lambda.shape}")
        
        # Initialize twin's lambda (K+1 signatures including health)
        K_total = self.K + (1 if self.model.healthy_ref is not None else 0)
        twin_lambda = np.zeros((K_total, self.T))
        
        # Create disease signatures
        for k in range(self.K):
            # Get base signature trajectory
            base_trajectory = self.model.signature_refs[k].detach().numpy()  # Shape (T,)
            #print(f"base_trajectory shape: {base_trajectory.shape}")
            #print(f"genetic_effects[{k}] shape: {genetic_effects[k].shape}")
            
            # Add constant genetic effect to each time point
            # genetic_effects[k] is a scalar, so it will be broadcast to all T time points
            mean_lambda = base_trajectory + genetic_effects[k]
            
            # Add GP noise
            L = np.linalg.cholesky(K_lambda)
            eps = L @ np.random.randn(self.T)
            twin_lambda[k, :] = mean_lambda + eps
            
        # Add health signature if it exists
        if self.model.healthy_ref is not None:
            base_trajectory = self.model.healthy_ref.detach().numpy()
            L = np.linalg.cholesky(K_lambda)
            eps = L @ np.random.randn(self.T)
            twin_lambda[self.K, :] = base_trajectory + eps
            
        # Compute twin's theta
        twin_theta = np.exp(twin_lambda) / np.sum(np.exp(twin_lambda), axis=0, keepdims=True)
        
        # Compute disease probabilities
        twin_phi_prob = expit(self.phi)
        twin_pi = np.einsum('kt,kdt->dt', twin_theta, twin_phi_prob)
        
        return {
            'theta': twin_theta,  # Signature proportions over time
            'pi': twin_pi,  # Disease probabilities over time
            'lambda': twin_lambda,  # Raw signature values
            'genetic_effects': genetic_effects  # Genetic contributions to each signature
        }
    
    def simulate_intervention(self, twin_data, intervention_type, target_signature=None, 
                            effect_size=0.5, start_time=None, end_time=None):
        """
        Simulate the effect of an intervention on the digital twin
        
        Parameters:
        twin_data: Dictionary from create_twin()
        intervention_type: 'reduce' or 'increase'
        target_signature: Which signature to modify (if None, affects all)
        effect_size: Magnitude of intervention effect
        start_time: When to start intervention (if None, starts immediately)
        end_time: When to end intervention (if None, continues to end)
        
        Returns:
        Modified twin data with intervention effects
        """
        modified_data = twin_data.copy()
        
        # Set time window
        start_idx = 0 if start_time is None else start_time
        end_idx = self.T if end_time is None else end_time
        
        # Modify lambda values
        if target_signature is not None:
            signatures = [target_signature]
        else:
            signatures = range(self.K)
            
        for k in signatures:
            if intervention_type == 'reduce':
                modified_data['lambda'][k, start_idx:end_idx] *= (1 - effect_size)
            else:  # increase
                modified_data['lambda'][k, start_idx:end_idx] *= (1 + effect_size)
                
        # Recompute theta and pi with modified lambda
        modified_data['theta'] = np.exp(modified_data['lambda']) / \
                                np.sum(np.exp(modified_data['lambda']), axis=0, keepdims=True)
        modified_data['pi'] = np.einsum('kt,kdt->dt', modified_data['theta'], expit(self.phi))
        
        return modified_data
    
    def plot_twin_trajectories(self, twin_data, modified_data=None, 
                             selected_diseases=None, figsize=(15, 15)):
        """
        Plot disease trajectories for the digital twin
        
        Parameters:
        twin_data: Dictionary from create_twin()
        modified_data: Optional modified twin data from simulate_intervention()
        selected_diseases: List of disease indices to plot (if None, plots all)
        figsize: Figure size
        """
        if selected_diseases is None:
            selected_diseases = range(self.D)
            
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize)

        time_points = np.arange(self.T)

        # 1. Plot raw lambda (signature trajectories)
        for k in range(self.K):
            ax0.plot(time_points, twin_data['lambda'][k], label=f'Signature {k}', alpha=0.7)
            if modified_data is not None:
                ax0.plot(time_points, modified_data['lambda'][k], '--', alpha=0.7)
        ax0.set_title('Signature Trajectories (lambda) Over Time')
        ax0.set_xlabel('Time')
        ax0.set_ylabel('Lambda Value')
        ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Plot signature proportions (theta)
        for k in range(self.K):
            ax1.plot(time_points, twin_data['theta'][k], label=f'Signature {k}', alpha=0.7)
            if modified_data is not None:
                ax1.plot(time_points, modified_data['theta'][k], '--', alpha=0.7)
        ax1.set_title('Signature Proportions (theta) Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Proportion')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Plot disease probabilities (pi)
        for d in selected_diseases:
            if isinstance(self.disease_names, pd.DataFrame):
                disease_name = self.disease_names.iloc[d, 0] if d < len(self.disease_names) else f'Disease {d}'
            else:
                disease_name = self.disease_names[d] if self.disease_names is not None else f'Disease {d}'
            ax2.plot(time_points, twin_data['pi'][d], label=disease_name, alpha=0.7)
            if modified_data is not None:
                ax2.plot(time_points, modified_data['pi'][d], '--', label=f'{disease_name} (modified)', alpha=0.7)
        ax2.set_title('Disease Probabilities (pi) Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
        
    def compare_twins(self, twin1_data, twin2_data, metric='pi'):
        """
        Compare two digital twins
        
        Parameters:
        twin1_data: First twin's data
        twin2_data: Second twin's data
        metric: What to compare ('pi' for disease probabilities or 'theta' for signatures)
        
        Returns:
        Dictionary of comparison metrics
        """
        if metric == 'pi':
            data1 = twin1_data['pi']
            data2 = twin2_data['pi']
        else:
            data1 = twin1_data['theta']
            data2 = twin2_data['theta']
            
        # Compute differences
        differences = data1 - data2
        
        # Calculate summary statistics
        mean_diff = np.mean(differences, axis=1)
        max_diff = np.max(np.abs(differences), axis=1)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        sns.heatmap(differences, cmap='RdBu_r', center=0)
        plt.title(f'Differences between Twins ({metric})')
        plt.xlabel('Time')
        plt.ylabel('Disease' if metric == 'pi' else 'Signature')
        plt.show()
        
        return {
            'mean_difference': mean_diff,
            'max_difference': max_diff,
            'raw_differences': differences
        }