import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import scipy.stats
from sklearn.metrics import roc_auc_score
import pandas as pd

def generate_validation_data(N=1000, D=20, T=50, K=5, P=5):
    """
    Generate synthetic data that validates clust_huge_amp's approach
    """
    # 1. Create reference signatures (population averages)
    signature_references = np.zeros((K, T))
    for k in range(K):
        if k == 0:  # Early onset
            signature_references[k] = scipy.stats.norm.pdf(np.linspace(0, T, T), loc=T/4, scale=T/8)
        elif k == 1:  # Late onset
            signature_references[k] = scipy.stats.norm.pdf(np.linspace(0, T, T), loc=3*T/4, scale=T/8)
        elif k == 2:  # Gradual increase
            signature_references[k] = np.linspace(0, 1, T)
        elif k == 3:  # Middle age peak
            signature_references[k] = scipy.stats.norm.pdf(np.linspace(0, T, T), loc=T/2, scale=T/6)
        else:  # Constant risk
            signature_references[k] = np.ones(T)
        # Normalize
        signature_references[k] = signature_references[k] / signature_references[k].max()

    # 2. Generate genetic effects (G) and true gamma
    G = np.random.randn(N, P)
    gamma_true = np.random.randn(P, K) * 0.5  # Moderate genetic effects

    # 3. Generate lambda with small initial GP noise
    k_init_lambda = T/20  # Small initial length scale
    times = np.arange(T)
    K_init = np.exp(-0.5 * (times[:, None] - times[None, :])**2 / k_init_lambda**2)
    
    lambda_true = np.zeros((N, K, T))
    for i in range(N):
        genetic_mean = G[i] @ gamma_true  # Individual's genetic predisposition
        for k in range(K):
            # Small GP noise around genetic mean
            lambda_true[i,k] = genetic_mean[k] + np.random.multivariate_normal(
                np.zeros(T), K_init * 0.1)  # Small variance

    # 4. Generate disease clusters with clear signature associations
    psi_true = np.zeros((K, D))
    true_clusters = np.zeros(D)
    diseases_per_cluster = D // K
    
    # Handle uneven division
    extra_diseases = D % K
    cluster_sizes = [diseases_per_cluster + 1 if k < extra_diseases else diseases_per_cluster 
                    for k in range(K)]
    
    start_idx = 0
    for k in range(K):
        # Assign diseases to clusters
        end_idx = start_idx + cluster_sizes[k]
        true_clusters[start_idx:end_idx] = k
        
        # Create mask for this cluster
        cluster_mask = (true_clusters == k)
        n_in_cluster = np.sum(cluster_mask)
        n_out_cluster = D - n_in_cluster
        
        # Set positive associations for in-cluster diseases
        psi_true[k, cluster_mask] = 2.0 + 0.1 * np.random.randn(n_in_cluster)
        # Set negative associations for out-of-cluster diseases
        psi_true[k, ~cluster_mask] = -4.0 + 0.1 * np.random.randn(n_out_cluster)
        
        start_idx = end_idx

    # 5. Generate realistic baseline rates
    mu_d = np.zeros((D, T))
    for d in range(D):
        base_rate = np.random.choice([-4, -3, -2])  # Different baseline risks
        age_effect = np.linspace(0, 1, T)  # Common age trend
        mu_d[d] = base_rate + age_effect

    # 6. Generate events
    theta = softmax(lambda_true, axis=1)
    Y = np.zeros((N, D, T))
    event_times = np.full((N, D), T)

    for n in range(N):
        for d in range(D):
            for t in range(T):
                if Y[n,d,:t].sum() == 0:  # Still at risk
                    # Combine baseline, signatures, and kappa
                    logit = mu_d[d,t] + np.sum(theta[n,:,t] * psi_true[:,d])
                    prob = 1 / (1 + np.exp(-logit))
                    if np.random.rand() < prob:
                        Y[n,d,t] = 1
                        event_times[n,d] = t
                        break

    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'true_clusters': true_clusters,
        'signature_references': signature_references,
        'true_psi': psi_true,
        'true_gamma': gamma_true,
        'true_mu': mu_d,
        'true_lambda': lambda_true,
        'theta': theta
    }

def plot_signature_recovery(true_sigs, recovered_sigs, save_path=None):
    """Plot true vs recovered reference signatures"""
    K, T = true_sigs.shape
    fig, axes = plt.subplots(K, 1, figsize=(12, 3*K))
    times = np.arange(T)
    
    for k in range(K):
        ax = axes[k] if K > 1 else axes
        ax.plot(times, true_sigs[k], 'b-', label='True', linewidth=2)
        ax.plot(times, recovered_sigs[k], 'r--', label='Recovered', linewidth=2)
        ax.set_title(f'Signature {k+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_cluster_recovery(true_psi, recovered_psi, save_path=None):
    """Plot true vs recovered cluster structure through psi"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot true psi
    sns.heatmap(true_psi, ax=ax1, cmap='RdBu_r', center=0)
    ax1.set_title('True ψ')
    ax1.set_xlabel('Disease')
    ax1.set_ylabel('Signature')
    
    # Plot recovered psi
    sns.heatmap(recovered_psi, ax=ax2, cmap='RdBu_r', center=0)
    ax2.set_title('Recovered ψ')
    ax2.set_xlabel('Disease')
    ax2.set_ylabel('Signature')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_genetic_effects(true_gamma, recovered_gamma, save_path=None):
    """Plot true vs recovered genetic effects"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot true gamma
    sns.heatmap(true_gamma, ax=ax1, cmap='RdBu_r', center=0)
    ax1.set_title('True γ')
    ax1.set_xlabel('Signature')
    ax1.set_ylabel('Genetic Component')
    
    # Plot recovered gamma
    sns.heatmap(recovered_gamma, ax=ax2, cmap='RdBu_r', center=0)
    ax2.set_title('Recovered γ')
    ax2.set_xlabel('Signature')
    ax2.set_ylabel('Genetic Component')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def evaluate_predictions(true_Y, pred_probs):
    """Evaluate prediction accuracy"""
    aucs = []
    for d in range(true_Y.shape[1]):
        # Flatten predictions and true values for each disease
        y_true = true_Y[:,d,:].flatten()
        y_pred = pred_probs[:,d,:].flatten()
        # Calculate AUC
        if np.sum(y_true) > 0:  # Only if there are positive cases
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
    
    return np.mean(aucs), np.std(aucs)

def plot_prediction_accuracy(true_Y, pred_probs, save_path=None):
    """Plot prediction accuracy metrics"""
    # Calculate AUC for each disease
    D = true_Y.shape[1]
    aucs = []
    for d in range(D):
        y_true = true_Y[:,d,:].flatten()
        y_pred = pred_probs[:,d,:].flatten()
        if np.sum(y_true) > 0:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
    
    plt.figure(figsize=(10, 5))
    plt.hist(aucs, bins=20)
    plt.xlabel('AUC')
    plt.ylabel('Count')
    plt.title('Distribution of Disease-Specific AUCs')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

def plot_simulated_signatures(signature_references, save_path=None):
    """Plot the simulated reference signatures"""
    K, T = signature_references.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    times = np.arange(T)
    
    for k in range(K):
        label = {
            0: 'Early onset',
            1: 'Late onset',
            2: 'Gradual increase',
            3: 'Middle age peak',
            4: 'Constant risk'
        }.get(k, f'Signature {k+1}')
        
        ax.plot(times, signature_references[k], linewidth=2, label=label)
    
    ax.set_title('Simulated Reference Signatures')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signature Strength')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_simulated_structure(data, save_path=None):
    """Plot the key components of the simulated data"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Reference Signatures
    ax1 = plt.subplot(3, 2, 1)
    times = np.arange(data['signature_references'].shape[1])
    for k in range(data['signature_references'].shape[0]):
        ax1.plot(times, data['signature_references'][k], linewidth=2, 
                label=f'Signature {k+1}')
    ax1.set_title('Reference Signatures')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. True Psi (Cluster Structure)
    ax2 = plt.subplot(3, 2, 2)
    sns.heatmap(data['true_psi'], ax=ax2, cmap='RdBu_r', center=0)
    ax2.set_title('True ψ (Disease-Signature Associations)')
    ax2.set_xlabel('Disease')
    ax2.set_ylabel('Signature')
    
    # 3. Sample Individual Trajectories (lambda)
    ax3 = plt.subplot(3, 2, 3)
    for k in range(data['true_lambda'].shape[1]):
        ax3.plot(times, data['true_lambda'][0,k,:], label=f'Signature {k+1}')
    ax3.set_title('Sample Individual Trajectories (λ)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Genetic Effects
    ax4 = plt.subplot(3, 2, 4)
    sns.heatmap(data['true_gamma'], ax=ax4, cmap='RdBu_r', center=0)
    ax4.set_title('True Genetic Effects (γ)')
    ax4.set_xlabel('Signature')
    ax4.set_ylabel('Genetic Component')
    
    # 5. Disease Baselines
    ax5 = plt.subplot(3, 2, 5)
    for d in range(min(5, data['true_mu'].shape[0])):  # Plot first 5 diseases
        ax5.plot(times, data['true_mu'][d], label=f'Disease {d+1}')
    ax5.set_title('Sample Disease Baselines (μ)')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Logit Rate')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Event Distribution
    ax6 = plt.subplot(3, 2, 6)
    event_counts = np.sum(data['Y'], axis=(0,2))  # Sum over individuals and time
    ax6.bar(range(len(event_counts)), event_counts)
    ax6.set_title('Disease Event Counts')
    ax6.set_xlabel('Disease')
    ax6.set_ylabel('Number of Events')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_individual_examples(data, n_individuals=3, save_path=None):
    """Plot example individual trajectories and their disease events"""
    fig = plt.figure(figsize=(15, 5*n_individuals))
    
    for i in range(n_individuals):
        # Signature weights (theta)
        ax1 = plt.subplot(n_individuals, 2, 2*i+1)
        times = np.arange(data['theta'].shape[2])
        for k in range(data['theta'].shape[1]):
            ax1.plot(times, data['theta'][i,k,:], label=f'Signature {k+1}')
        ax1.set_title(f'Individual {i+1} Signature Weights (θ)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Weight')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Disease events
        ax2 = plt.subplot(n_individuals, 2, 2*i+2)
        events = data['Y'][i].T  # Time x Disease
        sns.heatmap(events, ax=ax2, cmap='Reds', cbar_kws={'label': 'Event'})
        ax2.set_title(f'Individual {i+1} Disease Events')
        ax2.set_xlabel('Disease')
        ax2.set_ylabel('Time')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    data = generate_validation_data()
    
    # Plot simulated data structure
    plot_simulated_structure(data, save_path='simulated_structure.pdf')
    
    # Plot individual examples
    plot_individual_examples(data, n_individuals=3, save_path='individual_examples.pdf')
    
    # Plot just the signatures
    plot_simulated_signatures(data['signature_references'], save_path='reference_signatures.pdf')
    
    # Example usage for after model fitting remains the same
    """
    # Plot signature recovery
    plot_signature_recovery(data['signature_references'], 
                          model.get_signatures(),
                          save_path='signature_recovery.pdf')
    
    # Plot cluster recovery
    plot_cluster_recovery(data['true_psi'],
                         model.psi.detach().numpy(),
                         save_path='cluster_recovery.pdf')
    
    # Plot genetic effects
    plot_genetic_effects(data['true_gamma'],
                        model.gamma.detach().numpy(),
                        save_path='genetic_effects.pdf')
    
    # Evaluate predictions
    pred_probs = model.forward()[0].detach().numpy()
    plot_prediction_accuracy(data['Y'], pred_probs,
                           save_path='prediction_accuracy.pdf')
    """ 