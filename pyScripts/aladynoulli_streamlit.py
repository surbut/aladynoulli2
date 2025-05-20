import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax, expit
import gdown  # Add this to requirements.txt
import os
from sklearn.metrics import roc_curve, auc

class ModelVisualizer:
    def __init__(self, model_state_dict, G=None, disease_names=None):
        # Get model parameters
        self.lambda_ = model_state_dict['lambda_'].detach().numpy()  # Shape (N, K, T)
        self.phi = model_state_dict['phi'].detach().numpy()  # Shape (K, D, T)
        self.psi = model_state_dict['psi'].detach().numpy()  # Shape (K, D)
        
        # Add G and gamma - handle G differently since it's not a tensor
        self.G = G if G is not None else None  # Remove .numpy() since G is already numpy
        self.gamma = model_state_dict['gamma'].detach().numpy() if 'gamma' in model_state_dict else None
        
        # Store disease names
        if hasattr(disease_names, 'values'):
            self.disease_names = disease_names.values.tolist()
        elif hasattr(disease_names, 'tolist'):
            self.disease_names = disease_names.tolist()
        else:
            self.disease_names = disease_names if disease_names is not None else [f"Disease_{i}" for i in range(self.D)]
        
        # Get dimensions
        self.N, self.K, self.T = self.lambda_.shape
        self.D = self.phi.shape[1]
        
        # Pre-compute theta
        self.theta = softmax(self.lambda_, axis=1)  # Shape (N, K, T)
        
        # Add placeholder for genomic data
        self.genomic_data = None
    
    def set_genomic_data(self, genomic_data):
        """Set genomic data for visualization"""
        self.genomic_data = genomic_data
    
    def compute_person_trajectory(self, person_idx, time_idx):
        # Get theta for specific person and timepoint
        theta_t = self.theta[person_idx, :, time_idx]  # Shape (K,)
        
        # Get phi for specific timepoint
        phi_t = self.phi[:, :, time_idx]  # Shape (K, D)
        
        # Compute disease probabilities
        eta_t = expit(phi_t)  # Shape (K, D)
        pi = np.dot(theta_t, eta_t)  # Shape (D,)
        
        return theta_t, pi
    
    def plot_visualization(self, person_idx, time_idx):
        theta, pi = self.compute_person_trajectory(person_idx, time_idx)
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Plot signature proportions
        plt.subplot(221)
        plt.bar(range(self.K), theta)
        plt.title(f'Signature Proportions (Person {person_idx}, Time {time_idx})')
        plt.xlabel('Signature')
        plt.ylabel('Proportion')
        
        # 2. Plot disease probabilities
        plt.subplot(222)
        plt.hist(pi, bins=30)
        plt.title('Distribution of Disease Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        
        # 3. Plot psi heatmap
        plt.subplot(223)
        sns.heatmap(self.psi, cmap='RdBu_r', center=0)
        plt.title('ψ Values (Signature-Disease Associations)')
        plt.xlabel('Disease')
        plt.ylabel('Signature')
        
        # 4. Plot lambda trajectories for this person
        plt.subplot(224)
        for k in range(self.K):
            plt.plot(self.lambda_[person_idx, k, :], 
                    label=f'Signature {k}')
        plt.axvline(x=time_idx, color='r', linestyle='--')
        plt.title(f'λ Trajectories for Person {person_idx}')
        plt.xlabel('Time')
        plt.ylabel('λ Value')
        plt.legend()
        
        plt.tight_layout()
        return fig

    def plot_theta_comparison(self, person_indices, time_idx):
        """Plot theta comparison for multiple people at a given time"""
        n_people = len(person_indices)
        fig, axes = plt.subplots(n_people, 1, figsize=(12, 3*n_people))
        if n_people == 1:
            axes = [axes]
            
        for idx, person_idx in enumerate(person_indices):
            theta_t = self.theta[person_idx, :, time_idx]
            axes[idx].bar(range(self.K), theta_t)
            axes[idx].set_title(f'Person {person_idx} Signature Proportions (Time {time_idx})')
            axes[idx].set_xlabel('Signature')
            axes[idx].set_ylabel('Proportion')
            
        plt.tight_layout()
        return fig
    
    def plot_genomic_heatmap(self, person_idx):
        """Plot genomic data heatmap for a person"""
        if self.genomic_data is None:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 4))
        person_genomics = self.genomic_data[person_idx]
        sns.heatmap(person_genomics.reshape(1, -1), 
                   cmap='RdBu_r',
                   center=0,
                   ax=ax)
        ax.set_title(f'Genomic Profile - Person {person_idx}')
        ax.set_xlabel('Genetic Features')
        ax.set_yticklabels([])
        plt.tight_layout()
        return fig

    def plot_reference_comparison(self, person_idx, time_range=None):
        """Plot lambda and theta values comparing patient-specific vs reference trajectories"""
        if time_range is None:
            time_range = range(self.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot Lambda Values
        ax1.set_title('Lambda Values\n(dashed=reference, solid=patient-specific)')
        for k in range(self.K):
            # Plot reference trajectory (dashed)
            ref_lambda = np.mean(self.lambda_[:, k, :], axis=0)
            ax1.plot(time_range, ref_lambda[time_range], '--', label=f'Ref {k}', alpha=0.5)
            
            # Plot patient-specific trajectory (solid)
            ax1.plot(time_range, self.lambda_[person_idx, k, time_range], 
                    '-', label=f'Patient {person_idx} - State {k}')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Lambda (logit scale)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot Proportions (Theta)
        ax2.set_title('Final Proportions\n(dashed=reference, solid=patient-specific)')
        for k in range(self.K):
            # Plot reference trajectory (dashed)
            ref_theta = np.mean(self.theta[:, k, :], axis=0)
            ax2.plot(time_range, ref_theta[time_range], '--', label=f'Ref {k}', alpha=0.5)
            
            # Plot patient-specific trajectory (solid)
            ax2.plot(time_range, self.theta[person_idx, k, time_range], 
                    '-', label=f'Patient {person_idx} - State {k}')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Proportion')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig

    def plot_genetic_scores(self, person_idx):
        """Plot genetic scores for a person"""
        if self.G is None or self.gamma is None:
            return None
            
        # Calculate genetic effects
        # G[person_idx] is (36,) and gamma is (36, 21)
        # Need to transpose gamma to get (21, 36) for correct multiplication
        genetic_effects = np.dot(self.G[person_idx], self.gamma)  # Shape (21,)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot raw genetic scores
        sns.heatmap(self.G[person_idx].reshape(1, -1), 
                   cmap='RdBu_r',
                   center=0,
                   ax=ax1)
        ax1.set_title(f'Raw Genetic Scores - Person {person_idx}')
        ax1.set_xlabel('Genetic Features (n=36)')
        ax1.set_yticklabels([])
        
        # Plot genetic effects on signatures
        ax2.bar(range(self.K), genetic_effects)
        ax2.set_title('Genetic Effects on Each Signature')
        ax2.set_xlabel('Signature')
        ax2.set_ylabel('Effect Size')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0 for reference
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        plt.tight_layout()
        return fig

    def compute_disease_probabilities(self, person_idx):
        """Compute disease probabilities across all timepoints"""
        # Shape: (T, D)
        pi_t = np.zeros((self.T, self.D))
        for t in range(self.T):
            theta_t = self.theta[person_idx, :, t]  # (K,)
            eta_t = expit(self.phi[:, :, t])  # (K, D)
            pi_t[t] = np.dot(theta_t, eta_t)  # (D,)
        return pi_t
    
    def plot_prediction_explanation(self, person_idx, time_idx):
        """Visualize how predictions are generated"""
        theta_t = self.theta[person_idx, :, time_idx]  # (K,)
        eta_t = expit(self.phi[:, :, time_idx])  # (K, D)
        pi_t = np.dot(theta_t, eta_t)  # (D,)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Show theta (signature proportions)
        axes[0,0].bar(range(self.K), theta_t)
        axes[0,0].set_title('Step 1: Signature Proportions (θ)')
        axes[0,0].set_xlabel('Signature')
        axes[0,0].set_ylabel('Proportion')
        
        # 2. Show eta (signature-disease probabilities)
        sns.heatmap(eta_t, ax=axes[0,1], cmap='RdBu_r',
                   xticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto',
                   yticklabels=[f'Sig {k}' for k in range(self.K)])
        axes[0,1].set_title('Step 2: Signature-Disease Probabilities (η)')
        
        # 3. Show individual contributions from each signature
        contributions = np.zeros((self.K, self.D))
        for k in range(self.K):
            contributions[k] = theta_t[k] * eta_t[k]  # Individual contribution from signature k
        
        sns.heatmap(contributions, ax=axes[1,0], cmap='RdBu_r',
                   xticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto',
                   yticklabels=[f'Sig {k}' for k in range(self.K)])
        axes[1,0].set_title('Step 3: Individual Signature Contributions (θₖ × ηₖ)')
        
        # 4. Show final probabilities (sum of contributions)
        axes[1,1].bar(range(self.D), pi_t)
        axes[1,1].set_title('Step 4: Final Disease Probabilities (π = Σₖ θₖηₖ)')
        axes[1,1].set_xlabel('Disease')
        axes[1,1].set_ylabel('Probability')
        if hasattr(self, 'disease_names'):
            plt.xticks(range(self.D), self.disease_names, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_disease_covariance(self, person_idx):
        """Plot disease probability covariance over time"""
        # Get all disease probabilities over time
        pi_t = self.compute_disease_probabilities(person_idx)  # (T, D)
        
        # Compute covariance
        cov_matrix = np.cov(pi_t.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot covariance heatmap
        sns.heatmap(cov_matrix, ax=ax1, cmap='RdBu_r', center=0,
                   xticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto',
                   yticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto')
        ax1.set_title('Disease Probability Covariance')
        
        # Plot disease probability trajectories
        for d in range(self.D):
            ax2.plot(range(self.T), pi_t[:, d], 
                    label=self.disease_names[d] if hasattr(self, 'disease_names') else f'Disease {d}')
        ax2.set_title('Disease Probability Trajectories')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig

    def plot_disease_correlation(self, disease1_idx, disease2_idx, Y=None, n_patients=10000):
        """Plot correlation between two diseases for disease-free patients"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subset the first n_patients
        subset_idx = slice(0, min(n_patients, self.N))
        
        # Get probabilities for subset of patients
        all_probs = np.zeros((n_patients, self.T, 2))
        for person_idx in range(n_patients):
            pi_t = self.compute_disease_probabilities(person_idx)
            all_probs[person_idx, :, 0] = pi_t[:, disease1_idx]
            all_probs[person_idx, :, 1] = pi_t[:, disease2_idx]
        
        # Left panel: Scatter plot with time progression
        correlations = []
        for t in range(self.T):
            if Y is not None:
                healthy_mask = (Y[subset_idx, disease1_idx, t] == 0) & (Y[subset_idx, disease2_idx, t] == 0)
                probs1 = all_probs[healthy_mask, t, 0]
                probs2 = all_probs[healthy_mask, t, 1]
            else:
                probs1 = all_probs[:, t, 0]
                probs2 = all_probs[:, t, 1]
            
            # Plot scatter
            ax1.scatter(probs1, probs2, 
                       alpha=0.1, 
                       color=plt.cm.viridis(t/self.T), 
                       s=5)
            
            # Calculate correlation for this timepoint
            corr = np.corrcoef(probs1, probs2)[0,1]
            print(f"Time {t}: correlation = {corr}")
            correlations.append(corr)
        
        # Set axis limits using 99.9th percentile
        max1 = np.percentile(all_probs[:,:,0], 99.9)
        max2 = np.percentile(all_probs[:,:,1], 99.9)
        ax1.set_xlim(0, max1 * 1.1)
        ax1.set_ylim(0, max2 * 1.1)
        
        ax1.set_xlabel(f'{self.disease_names[disease1_idx]} probability')
        ax1.set_ylabel(f'{self.disease_names[disease2_idx]} probability')
        ax1.set_title('Disease Probability Correlations\n(Disease-free patients)')
        
        # Calculate overall correlation (all timepoints combined)
        all_disease1 = all_probs[:,:,0].flatten()
        all_disease2 = all_probs[:,:,1].flatten()
        overall_corr = np.corrcoef(all_disease1, all_disease2)[0,1]
        ax1.text(0.05, 0.95, f'Overall r = {overall_corr:.3f}', 
                 transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Calculate per-timepoint correlations for right panel
        correlations = [np.corrcoef(all_probs[:,t,0], all_probs[:,t,1])[0,1] 
                       for t in range(self.T)]
        
        # Right panel: Correlation over time
        times = range(self.T)
        ax2.plot(times, correlations, '-o')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.set_title('Evolution of Risk Correlation Over Time')
        ax2.grid(True)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=0, vmax=self.T-1))
        plt.colorbar(sm, ax=ax1, label='Timepoint')
        
        plt.tight_layout()
        return fig

    def plot_signatures_by_onset(self, disease_idx, Y, focus_signature=None):
        """Analyze signature distributions stratified by age of onset with genetic enrichment"""
        
        # Convert Y to numpy if it's a tensor
        if torch.is_tensor(Y):
            Y = Y.detach().numpy()
        
        # Get first occurrence of disease for each patient
        disease_times = np.argmax(Y[:, disease_idx, :] == 1, axis=1)
        has_disease = disease_times > 0
        
        # Define onset groups
        early = disease_times < 20
        mid = (disease_times >= 20) & (disease_times < 35)
        late = disease_times >= 35
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot signature distributions for each onset group
        for k in range(self.K):
            for onset, label, alpha in zip([early, mid, late], 
                                         ['Early', 'Mid', 'Late'],
                                         [0.3, 0.5, 0.7]):
                if np.sum(onset) > 0:  # Changed from np.any() to np.sum() > 0
                    ax1.hist(self.theta[onset, k, 0], alpha=alpha, 
                            label=f'{label} - Sig {k}', bins=20)
        
        ax1.set_title('Signature Distributions by Age of Onset')
        ax1.set_xlabel('Signature Weight')
        ax1.set_ylabel('Count')
        ax1.legend()
        
        # Plot temporal evolution of signatures for each onset group
        times = range(self.T)
        for k in range(self.K):
            for onset, label, style in zip([early, mid, late],
                                         ['Early', 'Mid', 'Late'],
                                         ['-', '--', ':']):
                if np.sum(onset) > 0:
                    mean_trajectory = np.mean(self.theta[onset, k, :], axis=0)
                    ax2.plot(times, mean_trajectory, style, 
                            label=f'{label} - Sig {k}')
        
        ax2.set_title('Signature Evolution by Age of Onset')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mean Signature Weight')
        ax2.legend()
        
        # New: Genetic enrichment analysis
        if focus_signature is not None:
            # Get population average trajectory
            pop_mean = np.mean(self.theta[:, focus_signature, :], axis=0)  # Shape (T,)
            
            # Calculate means for cases and non-cases at each timepoint
            non_cases = ~(Y[:, disease_idx, :].any(axis=1))  # Never get disease
            
            baseline = self.theta[non_cases, focus_signature, :]
            baseline_mean = np.mean(baseline, axis=0)  # Shape (T,)
            
            # Calculate enrichment for each onset group
            enrichments = []
            errors = []
            for group, label in zip([early, mid, late], ['Early', 'Mid', 'Late']):
                if np.sum(group) > 0:
                    # Get values for this group
                    group_theta = self.theta[group, focus_signature, :]  # Shape (n_group, T)
                    
                    # Calculate enrichment at each timepoint
                    enrichment = np.mean(group_theta - baseline_mean[None, :], axis=(0,1))
                    enrichments.append(enrichment)
                    
                    # Calculate standard error
                    se = np.std(group_theta - baseline_mean[None, :]) / np.sqrt(np.sum(group))
                    errors.append(se)
                else:
                    enrichments.append(0)
                    errors.append(0)
            
            # Plot enrichment with error bars
            groups = ['Early', 'Mid', 'Late']
            ax3.bar(groups, enrichments, yerr=errors)
            ax3.set_title(f'Signature {focus_signature} Enrichment vs Age-Matched Controls')
            ax3.set_ylabel('Enrichment vs Controls')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Add statistical annotation
            from scipy import stats
            early_theta = self.theta[early, focus_signature, :]
            late_theta = self.theta[late, focus_signature, :]
            
            # Compare mean enrichments
            early_enrichment = np.mean(early_theta - baseline_mean[None, :], axis=1)
            late_enrichment = np.mean(late_theta - baseline_mean[None, :], axis=1)
            t_stat, p_val = stats.ttest_ind(early_enrichment, late_enrichment)
            
            ax3.text(0.05, 0.95, f'Early vs Late\np={p_val:.2e}', 
                    transform=ax3.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def plot_signature_enrichments(self, disease_idx, Y):
        """Plot enrichment of all signatures for early vs late onset"""
        
        # Convert Y to numpy if it's a tensor
        if torch.is_tensor(Y):
            Y = Y.detach().numpy()
        
        # Get onset groups
        disease_times = np.argmax(Y[:, disease_idx, :] == 1, axis=1)
        early = disease_times < 20
        late = disease_times >= 35
        non_cases = ~(Y[:, disease_idx, :].any(axis=1))
        
        # Calculate enrichments for all signatures
        enrichments = []
        errors = []
        pvals = []
        
        for k in range(self.K):
            # Get baseline
            baseline = self.theta[non_cases, k, :]
            baseline_mean = np.mean(baseline, axis=0)  # Shape (T,)
            
            # Calculate enrichment for each group relative to baseline
            early_enrichment = np.mean(self.theta[early, k, :] - baseline_mean[None, :])
            late_enrichment = np.mean(self.theta[late, k, :] - baseline_mean[None, :])
            
            # Store difference in enrichment
            enrichments.append(early_enrichment - late_enrichment)
            
            # Calculate standard error of the difference
            early_vals = self.theta[early, k, :] - baseline_mean[None, :]
            late_vals = self.theta[late, k, :] - baseline_mean[None, :]
            
            # Pooled standard error
            n1 = np.sum(early)
            n2 = np.sum(late)
            se = np.sqrt(np.var(early_vals.flatten())/n1 + np.var(late_vals.flatten())/n2)
            errors.append(se)
            
            # Statistical test
            from scipy import stats
            _, p_val = stats.ttest_ind(early_vals.flatten(), late_vals.flatten())
            pvals.append(p_val)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(self.K)
        
        # Plot bars
        bars = ax.bar(x, enrichments, yerr=errors)
        
        # Color significant bars
        for i, (bar, pval) in enumerate(zip(bars, pvals)):
            if pval < 0.05:
                bar.set_color('red')
                ax.text(i, enrichments[i], f'p={pval:.1e}', 
                       ha='center', va='bottom')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title(f'Early vs Late Onset Enrichment by Signature\nDisease: {self.disease_names[disease_idx]}')
        ax.set_xlabel('Signature')
        ax.set_ylabel('Early - Late Enrichment')
        
        plt.tight_layout()
        return fig

    def plot_diagnosis_impact(self, person_idx, disease_idx):
        """Analyze how trajectories change around disease diagnoses"""
        if not hasattr(self, 'Y'):
            return None
            
        # Get diagnosis times for this person and disease
        diagnoses = self.Y[person_idx, disease_idx, :]
        diagnosis_times = np.where(diagnoses)[0]
        
        if len(diagnosis_times) == 0:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Signature trajectories around diagnosis
        window = 10  # Time points before/after diagnosis
        for k in range(self.K):
            # Get trajectory for this signature
            traj = self.theta[person_idx, k, :]
            
            # Plot full trajectory
            ax1.plot(range(self.T), traj, alpha=0.3, color=f'C{k}')
            
            # Highlight window around each diagnosis
            for t in diagnosis_times:
                start = max(0, t - window)
                end = min(self.T, t + window)
                ax1.plot(range(start, end), traj[start:end], 
                        color=f'C{k}', linewidth=2)
                ax1.axvline(x=t, color='red', alpha=0.3, linestyle='--')
        
        ax1.set_title(f'Signature Trajectories Around Diagnosis\nDisease: {self.disease_names[disease_idx]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Signature Proportion')
        ax1.legend([f'Signature {k}' for k in range(self.K)])
        
        # Plot 2: Disease probability changes
        pi_t = self.compute_disease_probabilities(person_idx)
        
        # Get related diseases (same cluster)
        disease_cluster = self.clusters[disease_idx]
        related_diseases = np.where(self.clusters == disease_cluster)[0]
        
        for d in related_diseases:
            ax2.plot(range(self.T), pi_t[:, d], 
                    label=self.disease_names[d], alpha=0.5)
            
            # Highlight window around diagnosis
            for t in diagnosis_times:
                start = max(0, t - window)
                end = min(self.T, t + window)
                ax2.plot(range(start, end), pi_t[start:end, d], 
                        linewidth=2)
                ax2.axvline(x=t, color='red', alpha=0.3, linestyle='--')
        
        ax2.set_title('Disease Probability Changes Around Diagnosis')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig

    def plot_genetic_trajectory_impact(self, person_idx):
        """Analyze how genetic factors influence trajectory changes"""
        if self.G is None or self.gamma is None:
            return None
            
        # Calculate genetic effects
        genetic_effects = np.dot(self.G[person_idx], self.gamma)  # Shape (K,)
        
        # Get top and bottom affected signatures
        top_sigs = np.argsort(genetic_effects)[-3:]  # Top 3
        bottom_sigs = np.argsort(genetic_effects)[:3]  # Bottom 3
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Trajectories for most affected signatures
        for k in top_sigs:
            ax1.plot(range(self.T), self.theta[person_idx, k, :],
                    label=f'Signature {k} (Effect: {genetic_effects[k]:.2f})')
        ax1.set_title('Most Positively Affected Signatures')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Proportion')
        ax1.legend()
        
        # Plot 2: Trajectories for least affected signatures
        for k in bottom_sigs:
            ax2.plot(range(self.T), self.theta[person_idx, k, :],
                    label=f'Signature {k} (Effect: {genetic_effects[k]:.2f})')
        ax2.set_title('Most Negatively Affected Signatures')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Proportion')
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def analyze_prediction_accuracy(self, person_idx, disease_idx):
        """Analyze how well the model predicts future diagnoses"""
        if not hasattr(self, 'Y'):
            return None
            
        # Get actual diagnosis time
        diagnoses = self.Y[person_idx, disease_idx, :]
        diagnosis_time = np.where(diagnoses)[0]
        
        if len(diagnosis_time) == 0:
            return None
            
        diagnosis_time = diagnosis_time[0]
        
        # Get predicted probabilities
        pi_t = self.compute_disease_probabilities(person_idx)
        pred_probs = pi_t[:, disease_idx]
        
        # Calculate ROC curve for different prediction windows
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Prediction trajectory
        ax1.plot(range(self.T), pred_probs, label='Predicted Probability')
        ax1.axvline(x=diagnosis_time, color='red', linestyle='--', 
                   label='Actual Diagnosis')
        
        # Add prediction windows
        windows = [5, 10, 15]
        for window in windows:
            if diagnosis_time - window >= 0:
                ax1.axvspan(diagnosis_time - window, diagnosis_time, 
                          alpha=0.2, color='gray',
                          label=f'{window}-timepoint window')
        
        ax1.set_title(f'Prediction Trajectory for {self.disease_names[disease_idx]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Predicted Probability')
        ax1.legend()
        
        # Plot 2: ROC curves for different windows
        for window in windows:
            if diagnosis_time - window >= 0:
                # Get predictions in window
                pred_window = pred_probs[diagnosis_time - window:diagnosis_time]
                # Create binary labels (1 for diagnosis time, 0 for others)
                labels = np.zeros_like(pred_window)
                labels[-1] = 1
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(labels, pred_window)
                roc_auc = auc(fpr, tpr)
                
                ax2.plot(fpr, tpr, 
                        label=f'{window}-timepoint window (AUC = {roc_auc:.2f})')
        
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_title('ROC Curves for Different Prediction Windows')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        
        plt.tight_layout()
        return fig

def plot_phi_evolution(phi, clusters=None, disease_names=None):
    """Plot the evolution of phi values over time for each signature."""
    K, D, T = phi.shape
    
    # Convert disease_names to list if it's a DataFrame or Series
    if hasattr(disease_names, 'values'):
        disease_names = disease_names.values.tolist()
    elif hasattr(disease_names, 'tolist'):
        disease_names = disease_names.tolist()
    
    if disease_names is None:
        disease_names = [f"Disease_{i}" for i in range(D)]
    
    if clusters is None or isinstance(clusters, float):
        disease_indices = {k: np.arange(D) for k in range(K)}
    else:
        # Ensure clusters is a numpy array
        clusters = np.array(clusters)
        disease_indices = {k: np.where(clusters == k)[0] for k in range(K)}
    
    fig, axes = plt.subplots(K, 1, figsize=(15, 4*K))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        cluster_diseases = disease_indices[k]
        if len(cluster_diseases) > 0:
            phi_k = expit(phi[k, cluster_diseases, :])
            
            # Get disease names for this cluster
            cluster_disease_names = [disease_names[i] for i in cluster_diseases]
            
            sns.heatmap(phi_k, ax=axes[k], cmap='RdBu_r',
                       xticklabels=range(T) if k == K-1 else False,
                       yticklabels=cluster_disease_names,
                       cbar_kws={'label': 'Probability'})
            axes[k].set_title(f'Signature {k} Disease Probabilities Over Time')
            axes[k].set_ylabel(f'Cluster {k} Diseases (n={len(cluster_diseases)})')
            
            # Rotate y-axis labels for better readability
            axes[k].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    return fig

def download_model():
    """Download model from Dropbox if not present"""
    model_path = 'models/model.pt'
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(model_path):
        # Replace with your Dropbox direct download link
        url = "https://www.dropbox.com/scl/fi/YOUR_FILE_ID/model.pt?rlkey=YOUR_RLKEY&dl=1"
        st.info("Downloading model file... This may take a few minutes.")
        try:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            raise
    return model_path

def main():
    st.title("Disease Trajectory Model Visualization")
    
   
    first_model = torch.load('/Users/sarahurbut/Library/Cloudstorage/Dropbox/resultshighamp/results/output_0_10000/model.pt')
    # Load model state dict and additional data
    model_state_dict = first_model['model_state_dict']
    
    # Load and convert data types as needed
    clusters = np.array(first_model['clusters'])
    G = first_model['G']
    disease_names = first_model['disease_names']
    
    visualizer = ModelVisualizer(model_state_dict, G=G, disease_names=disease_names)
    
    # TODO: Load and set genomic data
    # genomic_data = np.load('path_to_genomic_data.npy')
    # visualizer.set_genomic_data(genomic_data)
    
    # Sidebar controls
    st.sidebar.header("Controls")
    person_idx = st.sidebar.slider("Select Person", 0, visualizer.N-1, 0)
    time_idx = st.sidebar.slider("Select Time Point", 0, visualizer.T-1, 0)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Individual Trajectories", 
        "Phi Evolution", 
        "Trajectory Comparison",
        "Genetic Effects",
        "Predictions Analysis",
        "Disease Correlations",
        "Onset Analysis",
        "Predictive Analytics"  # New tab
    ])
    
    with tab1:
        # Main visualization
        fig = visualizer.plot_visualization(person_idx, time_idx)
        st.pyplot(fig)
        
        # Original sidebar information
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### How to interpret:
        - **Top Left**: Shows the proportion of each signature at the selected time point
        - **Top Right**: Shows the distribution of disease probabilities
        - **Bottom Left**: Shows the signature-disease associations (ψ values)
        - **Bottom Right**: Shows how λ values change over time for this person
        
        The vertical red line in the bottom right plot shows the current time point.
        """)
    
    with tab2:
        st.markdown("### Evolution of Disease Probabilities by Signature")
        phi_fig = plot_phi_evolution(visualizer.phi, clusters, visualizer.disease_names)
        st.pyplot(phi_fig)
        
        st.markdown("""
        ### How to interpret Phi Evolution:
        - Each row shows cluster-specific diseases for each signature
        - The x-axis shows time progression
        - The y-axis shows different diseases
        - The color intensity indicates the probability of each disease at each time point
        - Darker red indicates higher probability, darker blue indicates lower probability
        """)
    
    with tab3:
        st.markdown("### Compare Patient Trajectories with Reference")
        
        # Time range selector
        time_start = st.slider("Start Time", 0, visualizer.T-1, 0)
        time_end = st.slider("End Time", time_start, visualizer.T-1, visualizer.T-1)
        time_range = range(time_start, time_end + 1)
        
        # Plot comparison
        comparison_fig = visualizer.plot_reference_comparison(person_idx, time_range)
        st.pyplot(comparison_fig)
        
        st.markdown("""
        ### How to interpret:
        - Dashed lines show reference (population average) trajectories
        - Solid lines show patient-specific trajectories
        - Left plot shows lambda values (logit scale)
        - Right plot shows proportions (theta values)
        - Different colors represent different disease states/signatures
        """)

    with tab4:
        st.markdown("### Genetic Scores and Effects")
        genetic_fig = visualizer.plot_genetic_scores(person_idx)
        if genetic_fig is not None:
            st.pyplot(genetic_fig)
            
            st.markdown("""
            ### How to interpret Genetic Effects:
            - Top plot shows raw genetic scores for the selected person
            - Bottom plot shows how these genetic scores influence each signature
            - Positive values indicate genetic predisposition towards that signature
            - Negative values indicate genetic protection against that signature
            """)
        else:
            st.info("Genetic data not available")

    with tab5:
        st.markdown("### Understanding Disease Predictions")
        
        st.markdown("""
        Disease probabilities (π) are generated through a matrix multiplication of:
        1. Individual signature proportions (θ)
        2. Signature-specific disease probabilities (η)
        
        This creates personalized disease probability predictions that account for 
        both individual trajectory patterns and signature-disease associations.
        """)
        
        # Show prediction generation
        pred_fig = visualizer.plot_prediction_explanation(person_idx, time_idx)
        st.pyplot(pred_fig)
        
        st.markdown("### Disease Probability Relationships")
        st.markdown("""
        The covariance structure shows how disease probabilities move together over time:
        - Positive covariance (red) indicates diseases that tend to increase/decrease together
        - Negative covariance (blue) indicates diseases that tend to move oppositely
        - The trajectory plot shows how individual disease probabilities evolve
        """)
        
        # Show covariance structure
        cov_fig = visualizer.plot_disease_covariance(person_idx)
        st.pyplot(cov_fig)

    with tab6:
        st.markdown("### Disease Risk Correlations")
        
        # Create two dropdown menus for disease selection
        col1, col2 = st.columns(2)
        with col1:
            disease1 = st.selectbox(
                "Select First Disease",
                options=range(len(visualizer.disease_names)),
                format_func=lambda x: visualizer.disease_names[x]
            )
        with col2:
            disease2 = st.selectbox(
                "Select Second Disease",
                options=range(len(visualizer.disease_names)),
                format_func=lambda x: visualizer.disease_names[x]
            )
        
        if disease1 != disease2:
            corr_fig = visualizer.plot_disease_correlation(disease1, disease2)
            st.pyplot(corr_fig)
            
            st.markdown("""
            ### How to interpret:
            - Left plot shows the correlation between risks for the two selected diseases
            - Each point represents one person's risk values
            - Right plot shows how the average risks for both diseases change over time
            - Correlation coefficient shows the strength of the relationship
            """)
        else:
            st.warning("Please select two different diseases to compare")

    with tab7:
        st.markdown("### Signature Analysis by Age of Onset")
        
        col1, col2 = st.columns(2)
        with col1:
            disease_idx = st.selectbox(
                "Select Disease for Onset Analysis",
                options=range(len(visualizer.disease_names)),
                format_func=lambda x: visualizer.disease_names[x]
            )
        with col2:
            focus_sig = st.selectbox(
                "Focus on Signature",
                options=range(visualizer.K),
                format_func=lambda x: f"Signature {x}"
            )
        
        # Plot onset analysis if Y data is available
        if 'Y' in first_model:
            onset_fig = visualizer.plot_signatures_by_onset(
                disease_idx, first_model['Y'], focus_signature=focus_sig)
            st.pyplot(onset_fig)
            
            st.markdown("""
            ### How to interpret:
            - Left: Distribution of signature weights by onset group
            - Middle: How signatures evolve over time for each onset group
            - Right: Focused analysis of selected signature
            - Statistical test compares early vs late onset groups
            """)
        else:
            st.info("Outcome data (Y) not available for onset analysis")

        st.markdown("### Signature Enrichment Analysis")
        enrichment_fig = visualizer.plot_signature_enrichments(disease_idx, first_model['Y'])
        st.pyplot(enrichment_fig)

        st.markdown("### Diagnosis Impact Analysis")
        diagnosis_fig = visualizer.plot_diagnosis_impact(person_idx, disease_idx)
        st.pyplot(diagnosis_fig)

        st.markdown("### Genetic Trajectory Impact Analysis")
        genetic_impact_fig = visualizer.plot_genetic_trajectory_impact(person_idx)
        st.pyplot(genetic_impact_fig)

    with tab8:
        st.markdown("### Predictive Analytics")
        
        # Disease selection
        disease_idx = st.selectbox(
            "Select Disease for Prediction Analysis",
            options=range(len(visualizer.disease_names)),
            format_func=lambda x: visualizer.disease_names[x]
        )
        
        # Show prediction accuracy analysis
        pred_accuracy_fig = visualizer.analyze_prediction_accuracy(person_idx, disease_idx)
        if pred_accuracy_fig is not None:
            st.pyplot(pred_accuracy_fig)
            
            st.markdown("""
            ### How to interpret:
            - Left plot shows the predicted probability trajectory over time
            - Vertical red line indicates actual diagnosis time
            - Gray shaded areas show different prediction windows
            - Right plot shows ROC curves for different prediction windows
            - Higher AUC indicates better prediction accuracy
            """)
        else:
            st.info("No diagnosis data available for this person/disease combination")

if __name__ == "__main__":
    main()