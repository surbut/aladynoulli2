# Check MGB Signature 17 Diseases

Run this in your notebook to see which diseases are in MGB signature 17:

```python
import torch
import numpy as np
import pandas as pd

# Load MGB model
mgb_model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'
mgb_data = torch.load(mgb_model_path, map_location=torch.device('cpu'))

print(f"Model keys: {list(mgb_data.keys())}")

# Get disease names
disease_names_mgb = mgb_data['disease_names']
if hasattr(disease_names_mgb, 'values'):
    disease_names_mgb = disease_names_mgb.values.tolist()
elif isinstance(disease_names_mgb, (list, tuple)):
    disease_names_mgb = list(disease_names_mgb)
elif isinstance(disease_names_mgb, np.ndarray):
    disease_names_mgb = disease_names_mgb.tolist()

print(f"\nTotal diseases: {len(disease_names_mgb)}")

# Check for clusters
if 'clusters' in mgb_data:
    clusters = mgb_data['clusters']
    print(f"\nFound 'clusters' in model data")
    print(f"Type: {type(clusters)}")
    
    # If clusters is a DataFrame with disease assignments
    if isinstance(clusters, pd.DataFrame):
        print(f"Columns: {list(clusters.columns)}")
        print(f"Shape: {clusters.shape}")
        # Try to find signature 17
        if 17 in clusters.columns:
            sig17_diseases_idx = clusters[clusters[17] > 0].index
            print(f"\nDiseases in signature 17 (from clusters DataFrame):")
            for idx in sig17_diseases_idx[:20]:
                print(f"  {idx}: {disease_names_mgb[idx]}")
        elif 'clusters' in clusters.columns:
            sig17_mask = clusters['clusters'] == 17
            sig17_diseases_idx = clusters[sig17_mask].index
            print(f"\nDiseases in signature 17:")
            for idx in sig17_diseases_idx[:20]:
                print(f"  {idx}: {disease_names_mgb[idx]}")
    elif isinstance(clusters, dict):
        print(f"Clusters dict keys: {list(clusters.keys())[:10]}")
        if 17 in clusters:
            print(f"\nSignature 17 data:")
            print(clusters[17])

# Check for phi (signature-disease association matrix)
if 'model_state_dict' in mgb_data:
    model_state = mgb_data['model_state_dict']
    print(f"\nModel state dict keys: {list(model_state.keys())}")
    
    # Look for phi
    if 'phi' in model_state:
        phi = model_state['phi']
        if hasattr(phi, 'detach'):
            phi = phi.detach().numpy()
        print(f"\nFound 'phi' in model_state_dict")
        print(f"Phi shape: {phi.shape}")
        
        # Get signature 17's disease associations
        if phi.shape[0] > 17:  # Signatures x Diseases
            sig17_phi = phi[17, :]
        elif phi.shape[1] > 17:  # Diseases x Signatures
            sig17_phi = phi[:, 17]
        else:
            print(f"Phi shape {phi.shape} doesn't match expected format")
            sig17_phi = None
        
        if sig17_phi is not None:
            # Get top diseases
            top_indices = np.argsort(sig17_phi)[::-1][:20]
            print(f"\nTop 20 diseases in MGB Signature 17 (by phi values):")
            for idx in top_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (phi={sig17_phi[idx]:.4f})")
```

This will show you which diseases are associated with MGB signature 17.

