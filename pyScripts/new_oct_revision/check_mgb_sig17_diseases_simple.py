#!/usr/bin/env python3
"""
Simple script to check which diseases are in MGB signature 17
Using mgb_data['clusters']==17 as the user suggested
"""

import torch
import numpy as np
import pandas as pd

# Load MGB model
mgb_model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'
mgb_data = torch.load(mgb_model_path, map_location=torch.device('cpu'))

# Get disease names
disease_names_mgb = mgb_data['disease_names']
if hasattr(disease_names_mgb, 'values'):
    disease_names_mgb = disease_names_mgb.values.tolist()
elif isinstance(disease_names_mgb, (list, tuple)):
    disease_names_mgb = list(disease_names_mgb)
elif isinstance(disease_names_mgb, np.ndarray):
    disease_names_mgb = disease_names_mgb.tolist()

print("="*80)
print("DISEASES IN MGB SIGNATURE 17")
print("="*80)

# Check clusters structure
clusters = mgb_data['clusters']
print(f"\nClusters type: {type(clusters)}")

if isinstance(clusters, pd.DataFrame):
    print(f"Clusters shape: {clusters.shape}")
    print(f"Clusters columns: {list(clusters.columns)[:10]}...")  # Show first 10
    
    # Check if there's a column for signature 17
    if 17 in clusters.columns:
        print(f"\nFound column 17 in clusters DataFrame")
        sig17_mask = clusters[17] > 0  # Or clusters[17] == 1, depending on format
        sig17_diseases = clusters[sig17_mask]
        print(f"\nDiseases in signature 17 (column 17 > 0):")
        print(f"Number of diseases: {sig17_diseases.shape[0]}")
        for idx in sig17_diseases.index[:30]:  # Show first 30
            print(f"  {idx}: {disease_names_mgb[idx]}")
    elif 'clusters' in clusters.columns:
        # If clusters column contains signature assignments
        sig17_mask = clusters['clusters'] == 17
        sig17_diseases = clusters[sig17_mask]
        print(f"\nDiseases where clusters == 17:")
        print(f"Number of diseases: {sig17_diseases.shape[0]}")
        for idx in sig17_diseases.index[:30]:
            print(f"  {idx}: {disease_names_mgb[idx]}")
    else:
        # Try boolean indexing
        print(f"\nTrying to find signature 17...")
        # Check if any column matches
        for col in clusters.columns:
            if isinstance(col, (int, np.integer)) and col == 17:
                sig17_mask = clusters[col] > 0
                sig17_indices = clusters[sig17_mask].index
                print(f"\nFound diseases in signature 17 (column {col} > 0):")
                for idx in sig17_indices[:30]:
                    print(f"  {idx}: {disease_names_mgb[idx]}")
                break

elif isinstance(clusters, np.ndarray):
    print(f"Clusters shape: {clusters.shape}")
    # If clusters is a disease x signature matrix
    if clusters.shape[1] > 17:
        sig17_diseases_idx = np.where(clusters[:, 17] > 0)[0]
        print(f"\nDiseases in signature 17 (column 17 > 0):")
        print(f"Number of diseases: {len(sig17_diseases_idx)}")
        for idx in sig17_diseases_idx[:30]:
            print(f"  {idx}: {disease_names_mgb[idx]}")
    elif clusters.shape[0] > 17:
        sig17_diseases_idx = np.where(clusters[17, :] > 0)[0]
        print(f"\nDiseases in signature 17 (row 17 > 0):")
        print(f"Number of diseases: {len(sig17_diseases_idx)}")
        for idx in sig17_diseases_idx[:30]:
            print(f"  {idx}: {disease_names_mgb[idx]}")
    else:
        # Try boolean indexing
        sig17_mask = clusters == 17
        if sig17_mask.ndim == 1:
            sig17_diseases_idx = np.where(sig17_mask)[0]
            print(f"\nDiseases where clusters == 17:")
            print(f"Number of diseases: {len(sig17_diseases_idx)}")
            for idx in sig17_diseases_idx[:30]:
                print(f"  {idx}: {disease_names_mgb[idx]}")

elif isinstance(clusters, dict):
    print(f"Clusters keys: {list(clusters.keys())[:10]}")
    if 17 in clusters:
        sig17_data = clusters[17]
        print(f"\nSignature 17 data type: {type(sig17_data)}")
        if isinstance(sig17_data, (list, np.ndarray, pd.Series)):
            sig17_diseases_idx = np.where(np.array(sig17_data) > 0)[0] if hasattr(sig17_data, '__len__') else []
            print(f"\nDiseases in signature 17:")
            for idx in sig17_diseases_idx[:30]:
                print(f"  {idx}: {disease_names_mgb[idx]}")

# Also check phi if available (signature-disease association matrix)
if 'model_state_dict' in mgb_data:
    model_state = mgb_data['model_state_dict']
    if 'phi' in model_state:
        phi = model_state['phi']
        if hasattr(phi, 'detach'):
            phi = phi.detach().numpy()
        
        print(f"\n{'='*80}")
        print("ALTERNATIVE: Using phi (signature-disease association matrix)")
        print(f"{'='*80}")
        print(f"Phi shape: {phi.shape}")
        
        # Get signature 17's disease associations
        if phi.shape[0] > 17:  # Signatures x Diseases
            sig17_phi = phi[17, :]
            top_indices = np.argsort(sig17_phi)[::-1][:30]
            print(f"\nTop 30 diseases in MGB Signature 17 (by phi values):")
            for idx in top_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (phi={sig17_phi[idx]:.4f})")
        elif phi.shape[1] > 17:  # Diseases x Signatures
            sig17_phi = phi[:, 17]
            top_indices = np.argsort(sig17_phi)[::-1][:30]
            print(f"\nTop 30 diseases in MGB Signature 17 (by phi values):")
            for idx in top_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (phi={sig17_phi[idx]:.4f})")

