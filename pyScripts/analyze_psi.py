import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_cancer_psi(model_path, model_name=""):
    """
    Analyze posterior psi values (mixed membership) for cancer-related diseases
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
    
    # Find cancer-related diseases
    cancer_indices = []
    cancer_names = []
    for idx, disease in enumerate(disease_names):
        if 'cancer' in disease.lower() or 'malignant' in disease.lower() or 'neoplasm' in disease.lower():
            cancer_indices.append(idx)
            cancer_names.append(disease)
    print(f"Number of cancer-related diseases found: {len(cancer_indices)}")
    
    # Extract psi values for cancer diseases
    cancer_psi = psi[cancer_indices]
    print(f"Cancer psi shape: {cancer_psi.shape}")
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(cancer_psi, 
                xticklabels=[f"Cluster {i}" for i in range(psi.shape[1])],
                yticklabels=cancer_names,
                cmap='YlOrRd',
                vmin=0, vmax=1)
    plt.title(f"Disease-Cluster Mixed Membership ({model_name})")
    plt.xlabel("Clusters")
    plt.ylabel("Cancer Types")
    plt.tight_layout()
    plt.savefig(f'cancer_psi_{model_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print strongest associations
    print(f"\nMixed Membership Analysis for {model_name}:")
    print("-" * 80)
    for i, disease in enumerate(cancer_names):
        # Get top 3 clusters
        top_clusters = np.argsort(cancer_psi[i])[-3:][::-1]
        print(f"\n{disease}:")
        for cluster in top_clusters:
            print(f"  Cluster {cluster}: {cancer_psi[i][cluster]:.3f}")

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
            analyze_cancer_psi(path, name)
        except Exception as e:
            print(f"\nError analyzing {name} model: {str(e)}")
            import traceback
            traceback.print_exc() 