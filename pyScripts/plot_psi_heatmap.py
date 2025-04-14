import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_psi_heatmap(model_path, model_name=""):
    """
    Create a heatmap visualization of psi values (disease-cluster memberships)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    # Get disease names
    disease_names = checkpoint['disease_names']
    if isinstance(disease_names, pd.DataFrame):
        disease_names = disease_names.iloc[:, 0].tolist()
    
    # Get psi values
    if 'psi' in checkpoint:
        psi = checkpoint['psi']
    else:
        psi = checkpoint['model_state_dict']['psi']
    
    # Apply softmax and convert to numpy
    if psi.requires_grad:
        psi = psi.detach()
    psi = torch.nn.functional.softmax(psi, dim=0).numpy()  # softmax over clusters
    psi = psi.T  # transpose to get (diseases × clusters)
    
    # Sort diseases by their maximum psi value's cluster
    max_cluster_per_disease = np.argmax(psi, axis=1)
    sorted_indices = np.argsort(max_cluster_per_disease)
    
    # Create figure
    plt.figure(figsize=(20, 30))
    
    # Create heatmap
    sns.heatmap(psi[sorted_indices], 
                xticklabels=[f"Cluster {i}" for i in range(psi.shape[1])],
                yticklabels=[disease_names[i] for i in sorted_indices],
                cmap='YlOrRd',
                vmin=0, vmax=1)
    
    plt.title(f"Disease-Cluster Mixed Membership ({model_name})", fontsize=16, pad=20)
    plt.xlabel("Clusters", fontsize=14)
    plt.ylabel("Diseases", fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'psi_heatmap_{model_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Load models
    models = {
        "UKB": "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt",
        "MGB": "/Users/sarahurbut/Dropbox/model_with_kappa_bigam_MGB.pt",
        "AOU": "/Users/sarahurbut/Dropbox/model_with_kappa_bigam_aou.pt"
    }
    
    # Create heatmap for each model
    for name, path in models.items():
        try:
            plot_psi_heatmap(path, name)
        except Exception as e:
            print(f"\nError plotting {name} model: {str(e)}")
            import traceback
            traceback.print_exc() 