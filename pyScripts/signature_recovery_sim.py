import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import scipy.stats
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from simulation_study import generate_validation_data
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

def generate_ground_truth_signatures(T=50, K=5):
    """
    Generate clear, distinct signature patterns that we want to recover.
    """
    signatures = np.zeros((K, T))
    time_points = np.linspace(0, 1, T)
    
    # Early onset signature
    signatures[0] = scipy.stats.norm.pdf(time_points, loc=0.2, scale=0.1)
    
    # Late onset signature
    signatures[1] = scipy.stats.norm.pdf(time_points, loc=0.8, scale=0.1)
    
    # Gradual increase signature
    signatures[2] = time_points
    
    # Middle age peak signature
    signatures[3] = scipy.stats.norm.pdf(time_points, loc=0.5, scale=0.15)
    
    # Bimodal signature
    signatures[4] = 0.5 * (scipy.stats.norm.pdf(time_points, loc=0.3, scale=0.1) +
                          scipy.stats.norm.pdf(time_points, loc=0.7, scale=0.1))
    
    # Normalize each signature
    signatures = signatures / signatures.max(axis=1, keepdims=True)
    return signatures

def evaluate_signature_recovery(true_signatures, recovered_signatures):
    """
    Evaluate how well the recovered signatures match the true ones.
    Returns correlation matrix between true and recovered signatures.
    """
    K = true_signatures.shape[0]
    correlations = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            correlations[i,j] = np.corrcoef(true_signatures[i], recovered_signatures[j])[0,1]
    
    return correlations

def plot_signature_comparison(true_signatures, recovered_signatures, gp_weight, save_path=None):
    """
    Plot true vs recovered signatures.
    """
    K = true_signatures.shape[0]
    fig, axes = plt.subplots(K, 2, figsize=(12, 3*K))
    
    for k in range(K):
        # Plot true signature
        axes[k,0].plot(true_signatures[k])
        axes[k,0].set_title(f'True Signature {k+1}')
        axes[k,0].set_ylim(0, 1)
        
        # Plot recovered signature
        axes[k,1].plot(recovered_signatures[k])
        axes[k,1].set_title(f'Recovered Signature {k+1} (GP weight={gp_weight})')
        axes[k,1].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def run_signature_recovery_simulation(N=1000, D=20, T=50, K=5, P=5, gp_weights=[0.0001, 0.001, 0.01, 0.1, 1.0]):
    """
    Run simulation study comparing signature recovery across different GP weights.
    """
    # Generate ground truth signatures
    true_signatures = generate_ground_truth_signatures(T, K)
    
    # Generate synthetic data using these signatures
    data = generate_validation_data(N=N, D=D, T=T, K=K, P=P)
    
    # Store results
    results = {
        'gp_weight': [],
        'signature_correlation': [],
        'cluster_ari': []
    }
    
    # Run model with different GP weights
    for gp_weight in gp_weights:
        print(f"Running simulation with GP weight = {gp_weight}")
        
        # Initialize model with the same parameters as in your notebook
        model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
            N=N, 
            D=D, 
            T=T, 
            K=K,
            P=P,
            G=data['G'],
            Y=data['Y'],
            R=0,  # LRT penalty
            W=gp_weight,  # GP weight
            prevalence_t=data['prevalence_t'],
            init_sd_scaler=1e-1,  # Same as your notebook
            genetic_scale=1,  # Same as your notebook
            signature_references=true_signatures,  # Use true signatures as references
            healthy_reference=True,  # Include healthy reference
            disease_names=None
        )
        
        # Let the model do its own clustering and psi estimation
        # Don't provide true_psi or clusters
        model.initialize_params()
        
        # Fit model
        history = model.fit(data['event_times'], num_epochs=1000)
        
        # Get recovered signatures
        recovered_signatures = model.get_signatures()
        
        # Evaluate signature recovery
        correlations = evaluate_signature_recovery(true_signatures, recovered_signatures)
        max_correlations = np.max(correlations, axis=1)
        avg_correlation = np.mean(max_correlations)
        
        # Evaluate cluster recovery
        recovered_clusters = np.argmax(model.psi.detach().numpy(), axis=0)
        ari = adjusted_rand_score(data['true_clusters'], recovered_clusters)
        
        # Store results
        results['gp_weight'].append(gp_weight)
        results['signature_correlation'].append(avg_correlation)
        results['cluster_ari'].append(ari)
        
        # Plot comparison for this GP weight
        plot_signature_comparison(
            true_signatures,
            recovered_signatures,
            gp_weight,
            save_path=f'signature_comparison_gp{gp_weight}.pdf'
        )
        
        # Print cluster sizes for this run
        print("\nRecovered Cluster Sizes:")
        unique, counts = np.unique(recovered_clusters, return_counts=True)
        for k, count in zip(unique, counts):
            print(f"Cluster {k}: {count} diseases")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot overall results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['gp_weight'], results_df['signature_correlation'], 'o-')
    plt.xscale('log')
    plt.xlabel('GP Weight')
    plt.ylabel('Average Signature Correlation')
    plt.title('Signature Recovery')
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['gp_weight'], results_df['cluster_ari'], 'o-')
    plt.xscale('log')
    plt.xlabel('GP Weight')
    plt.ylabel('Adjusted Rand Index')
    plt.title('Cluster Recovery')
    
    plt.tight_layout()
    plt.savefig('recovery_results.pdf')
    plt.close()
    
    return results_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run simulation study
    results = run_signature_recovery_simulation()
    
    # Print results
    print("\nSimulation Results:")
    print(results) 