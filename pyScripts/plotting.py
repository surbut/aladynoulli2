import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_individual_trajectories(data, individual_ids=None, n_individuals=5):
    """
    Plot individual lambda trajectories and state proportions side by side.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing model components including 'lambda' and 'theta'
    individual_ids : list
        Specific individual IDs to plot. If None, randomly samples n_individuals
    n_individuals : int
        Number of individuals to plot if individual_ids not provided
    """
    if individual_ids is None:
        individual_ids = np.random.choice(data['lambda'].shape[0], 
                                        size=n_individuals, replace=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get number of states/clusters
    n_states = data['lambda'].shape[1]
    time_points = np.arange(data['lambda'].shape[2])
    
    # Color map for states
    colors = plt.cm.rainbow(np.linspace(0, 1, n_states))
    
    # Plot 1: Individual Lambda Trajectories
    for ind_idx, ind_id in enumerate(individual_ids):
        for state in range(n_states):
            label = f'Individual {ind_id} - State {state}'
            ax1.plot(time_points, data['lambda'][ind_id, state, :],
                    color=colors[state], alpha=0.7, label=label)
    
    ax1.set_title('Individual Lambda Trajectories')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lambda (logit scale)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual State Proportions
    for ind_idx, ind_id in enumerate(individual_ids):
        for state in range(n_states):
            label = f'Individual {ind_id} - State {state}'
            ax2.plot(time_points, data['theta'][ind_id, state, :],
                    color=colors[state], alpha=0.7, label=label)
    
    ax2.set_title('Individual State Proportions')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Proportion')
    ax2.grid(True, alpha=0.3)
    
    # Add legends with smaller font size
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    plt.show()

def plot_synthetic_components(data, num_samples=5, save_path=None):
    """
    Visualize the synthetic data components including individual lambda and state trajectories.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the synthetic data components
    num_samples : int
        Number of samples to plot for each component
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed.
    """
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
    
    # 4. Plot sample theta (signature weights) as bars
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
    plt.legend()
    
    # 6. Plot individual state trajectories if available
    if 'state' in data:
        plt.subplot(236)
        for i in range(min(num_samples, data['state'].shape[0])):
            plt.plot(data['state'][i,:], alpha=0.5, label=f'Individual {i}')
        plt.title('Individual State Trajectories')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def analyze_disease_clusters(initial_clusters, disease_names):
    """
    Analyze and display which diseases are clustering together.
    
    Parameters:
    -----------
    initial_clusters : array-like
        Array of cluster assignments for each disease
    disease_names : array-like
        Array of disease names corresponding to the clusters
    """
    unique_clusters = np.unique(initial_clusters)
    
    print("Disease Cluster Analysis:")
    print("------------------------")
    
    for cluster in unique_clusters:
        cluster_mask = (initial_clusters == cluster)
        cluster_diseases = np.array(disease_names)[cluster_mask]
        
        print(f"\nCluster {cluster} (Size: {len(cluster_diseases)}):")
        for disease in sorted(cluster_diseases):
            print(f"  - {disease}")
            
    # Print some statistics
    print("\nClustering Statistics:")
    print("--------------------")
    values, counts = np.unique(initial_clusters, return_counts=True)
    print("Cluster sizes:", dict(zip(values, counts)))
    print(f"Number of clusters: {len(unique_clusters)}")
    print(f"Average cluster size: {np.mean(counts):.1f}")
    print(f"Std dev of cluster size: {np.std(counts):.1f}") 