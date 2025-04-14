import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_all_psi(model_path, model_name=""):
    """
    Analyze and visualize psi values for all diseases
    """
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    # Get disease names
    disease_names = checkpoint['disease_names']
    if isinstance(disease_names, pd.DataFrame):
        disease_names = disease_names.iloc[:, 0].tolist()
    print(f"\nNumber of diseases: {len(disease_names)}")
    
    # Get psi values directly from checkpoint
    if 'psi' in checkpoint:
        psi = checkpoint['psi']
    else:
        psi = checkpoint['model_state_dict']['psi']
    print(f"Psi shape before softmax: {psi.shape}")
    
    # Apply softmax and convert to numpy
    if psi.requires_grad:
        psi = psi.detach()
    psi = torch.nn.functional.softmax(psi, dim=0).numpy()  # softmax over clusters
    psi = psi.T  # transpose to get (diseases × clusters)
    print(f"Psi shape after softmax and transpose: {psi.shape}")
    
    # Create heatmap
    plt.figure(figsize=(20, 30))  # Larger figure for all diseases
    
    # Sort diseases by their maximum psi value's cluster
    max_cluster_per_disease = np.argmax(psi, axis=1)
    sorted_indices = np.argsort(max_cluster_per_disease)
    
    # Create heatmap with sorted diseases
    sns.heatmap(psi[sorted_indices], 
                xticklabels=[f"Cluster {i}" for i in range(psi.shape[1])],
                yticklabels=[disease_names[i] for i in sorted_indices],
                cmap='YlOrRd',
                vmin=0, vmax=1)
    plt.title(f"Disease-Cluster Mixed Membership ({model_name})")
    plt.xlabel("Clusters")
    plt.ylabel("Diseases")
    plt.tight_layout()
    plt.savefig(f'all_diseases_psi_{model_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nCluster size summary:")
    print("-" * 80)
    primary_assignments = np.argmax(psi, axis=1)
    for cluster in range(psi.shape[1]):
        diseases_in_cluster = np.where(primary_assignments == cluster)[0]
        print(f"\nCluster {cluster} ({len(diseases_in_cluster)} diseases primarily assigned):")
        # Print top 5 diseases by psi value
        cluster_psi = psi[:, cluster]
        top_diseases = np.argsort(cluster_psi)[-5:][::-1]
        for idx in top_diseases:
            print(f"  - {disease_names[idx]}: {cluster_psi[idx]:.3f}")

if __name__ == "__main__":
    # Load models
    models = {
        "UKB": "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt",
        "MGB": "/Users/sarahurbut/Dropbox/model_with_kappa_bigam_MGB.pt",
        "AOU": "/Users/sarahurbut/Dropbox/model_with_kappa_bigam_aou.pt"
    }
    
    # Analyze each model
    for name, path in models.items():
        try:
            print(f"\nAnalyzing {name} model...")
            analyze_all_psi(path, name)
        except Exception as e:
            print(f"\nError analyzing {name} model: {str(e)}")
            import traceback
            traceback.print_exc() 