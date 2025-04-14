import numpy as np
import pandas as pd
from collections import defaultdict

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