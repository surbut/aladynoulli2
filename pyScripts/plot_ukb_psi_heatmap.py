import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ukb_psi_heatmap():
    """
    Create focused heatmaps of UKB's psi values, both raw and normalized
    """
    # Load checkpoint
    model_path = "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt"
    checkpoint = torch.load(model_path)
    
    # Get disease names
    disease_names = checkpoint['disease_names']
    if isinstance(disease_names, pd.DataFrame):
        disease_names = disease_names.iloc[:, 0].tolist()
    
    # Get psi values
    psi = checkpoint['model_state_dict']['psi']
    if psi.requires_grad:
        psi = psi.detach()
    
    # Get raw values before softmax
    raw_psi = psi.numpy().T  # transpose to get (diseases × clusters)
    
    # Apply softmax
    norm_psi = torch.nn.functional.softmax(psi, dim=0).numpy().T
    
    # Sort diseases by their maximum normalized psi value's cluster
    max_values = norm_psi.max(axis=1)
    max_clusters = norm_psi.argmax(axis=1)
    sorted_indices = np.lexsort((max_values, max_clusters))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 48))
    
    # Plot raw psi values
    sns.heatmap(raw_psi[sorted_indices], 
                xticklabels=[f"Cluster {i}" for i in range(raw_psi.shape[1])],
                yticklabels=[disease_names[i] for i in sorted_indices],
                cmap='RdBu_r',
                center=0,
                ax=ax1,
                cbar_kws={'label': 'Raw Psi Value'})
    
    ax1.set_title("UKB Raw Psi Values", fontsize=20, pad=20)
    ax1.set_xlabel("Clusters", fontsize=16)
    ax1.set_ylabel("Diseases", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot normalized psi values
    sns.heatmap(norm_psi[sorted_indices], 
                xticklabels=[f"Cluster {i}" for i in range(norm_psi.shape[1])],
                yticklabels=[disease_names[i] for i in sorted_indices],
                cmap='YlOrRd',
                vmin=0, vmax=1,
                ax=ax2,
                cbar_kws={'label': 'Normalized Membership Strength'})
    
    ax2.set_title("UKB Normalized Disease-Cluster Mixed Membership", fontsize=20, pad=20)
    ax2.set_xlabel("Clusters", fontsize=16)
    ax2.set_ylabel("Diseases", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add gridlines to separate clusters
    for ax in [ax1, ax2]:
        for cluster in range(21):
            cluster_diseases = np.where(max_clusters[sorted_indices] == cluster)[0]
            if len(cluster_diseases) > 0 and cluster > 0:
                ax.axhline(y=cluster_diseases[0], color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('ukb_psi_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary of mixed membership patterns
    print("\nMixed Membership Summary:")
    print("-" * 80)
    print("\nRaw psi range:", raw_psi.min(), "to", raw_psi.max())
    print("Raw psi std:", raw_psi.std())

    # Find and print the location of the maximum raw psi value
    max_raw_psi = raw_psi.max()
    max_indices = np.unravel_index(np.argmax(raw_psi, axis=None), raw_psi.shape)
    max_disease_idx = max_indices[0]
    max_cluster_idx = max_indices[1]
    print(f"\nMaximum raw psi value ({max_raw_psi:.3f}) found for:")
    print(f"  Disease: {disease_names[max_disease_idx]} (index {max_disease_idx})")
    print(f"  Cluster: {max_cluster_idx}")

    # For each disease, show clusters with normalized membership > 0.1
    for i, disease_idx in enumerate(sorted_indices):
        memberships = [(j, norm_psi[disease_idx, j]) for j in range(norm_psi.shape[1]) if norm_psi[disease_idx, j] > 0.1]
        if len(memberships) > 1:  # Only show diseases with multiple significant memberships
            print(f"\n{disease_names[disease_idx]}:")
            for cluster, strength in sorted(memberships, key=lambda x: x[1], reverse=True):
                print(f"  Cluster {cluster}: {strength:.3f} (raw: {raw_psi[disease_idx, cluster]:.3f})")

if __name__ == "__main__":
    plot_ukb_psi_heatmap() 