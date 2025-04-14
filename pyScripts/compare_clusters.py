import torch
import numpy as np
import pandas as pd
from collections import defaultdict

def group_diseases_by_cluster(clusters, disease_names_df, model_name=""):
    """
    Group diseases by their cluster assignments
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
    print(f"\nDisease Clusters Summary for {model_name}:")
    print("-" * 80)
    
    # Calculate some statistics
    n_clusters = len(cluster_diseases)
    total_diseases = sum(len(diseases) for diseases in cluster_diseases.values())
    avg_size = total_diseases / n_clusters
    
    print(f"Total clusters: {n_clusters}")
    print(f"Total diseases: {total_diseases}")
    print(f"Average cluster size: {avg_size:.1f}")
    
    # Find cancer-related clusters
    cancer_clusters = []
    for cluster, diseases in cluster_diseases.items():
        cancer_diseases = [d for d in diseases if 'cancer' in d.lower() or 'malignant' in d.lower() or 'neoplasm' in d.lower()]
        if cancer_diseases:
            cancer_clusters.append(cluster)
            print(f"\nCluster {cluster} - Cancer-related ({len(diseases)} diseases):")
            for disease in diseases:
                print(f"  - {disease}")
    
    return cluster_diseases, cluster_indices, cancer_clusters

def compare_cancer_clusters(models_info):
    """
    Compare how cancers are clustered across different models
    """
    print("\nCancer Cluster Comparison:")
    print("-" * 80)
    
    # Collect all cancer-related diseases
    all_cancers = set()
    for model_info in models_info:
        clusters = model_info['clusters']
        disease_names = model_info['disease_names']
        if isinstance(disease_names, pd.DataFrame):
            disease_names = disease_names.iloc[:, 0].tolist()
        
        for disease in disease_names:
            if 'cancer' in disease.lower() or 'malignant' in disease.lower() or 'neoplasm' in disease.lower():
                all_cancers.add(disease)
    
    # Create comparison table
    print("\nCancer Clustering Across Models:")
    print("-" * 80)
    print(f"{'Cancer Type':<50} | {'UKB':>5} {'MGB':>5} {'AOU':>5}")
    print("-" * 80)
    
    for cancer in sorted(all_cancers):
        clusters_by_model = []
        for model_info in models_info:
            clusters = model_info['clusters']
            disease_names = model_info['disease_names']
            if isinstance(disease_names, pd.DataFrame):
                disease_names = disease_names.iloc[:, 0].tolist()
            
            try:
                idx = disease_names.index(cancer)
                cluster = clusters[idx]
                clusters_by_model.append(f"{cluster:>5}")
            except ValueError:
                clusters_by_model.append("    -")
        
        print(f"{cancer:<50} | {' '.join(clusters_by_model)}")

if __name__ == "__main__":
    # Load models
    models = {
        "UKB": "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt",
        "MGB": "/Users/sarahurbut/Dropbox/model_with_kappa_bigam_MGB.pt",
        "AOU": "/Users/sarahurbut/Dropbox/model_with_kappa_bigam_aou.pt"
    }
    
    models_info = []
    for name, path in models.items():
        try:
            checkpoint = torch.load(path)
            cluster_diseases, cluster_indices, cancer_clusters = group_diseases_by_cluster(
                checkpoint['clusters'], 
                checkpoint['disease_names'],
                name
            )
            models_info.append({
                'name': name,
                'clusters': checkpoint['clusters'],
                'disease_names': checkpoint['disease_names'],
                'cancer_clusters': cancer_clusters
            })
        except Exception as e:
            print(f"\nError loading {name} model: {str(e)}")
    
    # Compare cancer clusters across models
    compare_cancer_clusters(models_info) 