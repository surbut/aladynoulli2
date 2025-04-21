import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def group_diseases_by_cluster(clusters, disease_names_df):
    """
    Group diseases by their cluster assignments
    
    Parameters:
    -----------
    clusters : array-like
        Cluster assignment for each disease
    disease_names_df : pandas DataFrame
        DataFrame containing disease names
        
    Returns:
    --------
    dict : Dictionary mapping cluster number to list of diseases in that cluster
    dict : Dictionary mapping cluster number to list of disease indices in that cluster
    """
    # Convert disease names to list if it's a DataFrame
    if isinstance(disease_names_df, pd.DataFrame):
        disease_names = disease_names_df.iloc[:, 0].tolist()
    else:
        disease_names = disease_names_df
    
    # Initialize defaultdict to collect diseases for each cluster
    cluster_diseases = defaultdict(list)
    cluster_indices = defaultdict(list)
    
    # Group diseases by cluster
    for idx, (disease, cluster) in enumerate(zip(disease_names, clusters)):
        cluster_diseases[int(cluster)].append(disease)
        cluster_indices[int(cluster)].append(idx)
    
    # Convert defaultdict to regular dict and sort by cluster number
    cluster_diseases = dict(sorted(cluster_diseases.items()))
    cluster_indices = dict(sorted(cluster_indices.items()))
    
    # Print summary
    print("\nDisease Clusters Summary:")
    print("-" * 80)
    
    # Calculate some statistics
    n_clusters = len(cluster_diseases)
    total_diseases = sum(len(diseases) for diseases in cluster_diseases.values())
    avg_size = total_diseases / n_clusters
    
    print(f"Total clusters: {n_clusters}")
    print(f"Total diseases: {total_diseases}")
    print(f"Average cluster size: {avg_size:.1f}")
    print("\nDetailed breakdown:")
    
    for cluster in cluster_diseases:
        print(f"\nCluster {cluster} ({len(cluster_diseases[cluster])} diseases):")
        for disease in cluster_diseases[cluster]:
            print(f"  - {disease}")
    
    return cluster_diseases, cluster_indices

def plot_confusion_matrix(true_clusters, pred_clusters, K):
    """
    Plot confusion matrix and compute clustering metrics.
    Reorders clusters to maximize diagonal alignment.
    """
    # Compute confusion matrix
    conf_mat = confusion_matrix(true_clusters, pred_clusters)
    
    # Normalize by true cluster sizes
    conf_mat_norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)
    
    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(-conf_mat_norm)
    
    # Reorder the confusion matrix
    conf_mat_reordered = conf_mat_norm[:, col_ind]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat_reordered, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(K), yticklabels=range(K))
    plt.title('Confusion Matrix (Normalized & Reordered)')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Cluster')
    
    # Compute metrics
    ari = adjusted_rand_score(true_clusters, pred_clusters)
    nmi = normalized_mutual_info_score(true_clusters, pred_clusters)
    
    print(f"\nClustering Metrics:")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")
    
    plt.show()
    
    return col_ind  # Return the optimal ordering

if __name__ == "__main__":
    # Example usage
    import torch
    
    # Load checkpoint
    checkpoint = torch.load("/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt")
    
    # Get clusters and disease names
    clusters = checkpoint['clusters']
    disease_names = checkpoint['disease_names']
    
    # Group diseases
    cluster_diseases, cluster_indices = group_diseases_by_cluster(clusters, disease_names) 

    # Load simulation data
    print("Loading simulation data...")
    sim_data = np.load('state_driven_sim.npz')
    true_clusters = sim_data['clusters']

    # Load model results (assuming you've saved them after fitting)
    # You'll need to replace this with actual model results
    print("Loading model results...")
    # pred_clusters = ... # This will come from your model fit

    # For now, let's just look at the true cluster structure
    unique_clusters, counts = np.unique(true_clusters, return_counts=True)
    print("\nTrue Cluster Sizes:")
    for k, count in zip(unique_clusters, counts):
        print(f"Cluster {k}: {count} diseases")

    # Once you have model results, uncomment this:
    # plot_confusion_matrix(true_clusters, pred_clusters, K=20) 