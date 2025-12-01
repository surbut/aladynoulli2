#!/usr/bin/env python3
"""
Check which diseases are in MGB signature 17
"""

import torch
import numpy as np
import pandas as pd

# Load MGB model
mgb_model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'
print(f"Loading MGB model from: {mgb_model_path}")

mgb_data = torch.load(mgb_model_path, map_location=torch.device('cpu'))

print(f"\nModel keys: {list(mgb_data.keys())}")

# Get disease names
disease_names_mgb = mgb_data['disease_names']
if hasattr(disease_names_mgb, 'values'):
    disease_names_mgb = disease_names_mgb.values.tolist()
elif isinstance(disease_names_mgb, (list, tuple)):
    disease_names_mgb = list(disease_names_mgb)
elif isinstance(disease_names_mgb, np.ndarray):
    disease_names_mgb = disease_names_mgb.tolist()

print(f"\nTotal diseases: {len(disease_names_mgb)}")

# Check for clusters or phi
if 'clusters' in mgb_data:
    print(f"\nFound 'clusters' in model data")
    clusters = mgb_data['clusters']
    print(f"Clusters shape/type: {type(clusters)}")
    if isinstance(clusters, (dict, pd.DataFrame)):
        print(f"Clusters keys/columns: {list(clusters.keys()) if isinstance(clusters, dict) else list(clusters.columns)}")
    
    # Try to get signature 17 diseases
    if isinstance(clusters, pd.DataFrame):
        if 'clusters' in clusters.columns or 17 in clusters.columns:
            sig17_mask = clusters['clusters'] == 17 if 'clusters' in clusters.columns else clusters[17] > 0
            sig17_diseases = clusters[sig17_mask]
            print(f"\nDiseases in signature 17 (from clusters DataFrame):")
            print(sig17_diseases)
    elif isinstance(clusters, dict):
        if 17 in clusters:
            print(f"\nSignature 17 in clusters dict:")
            print(clusters[17])
        elif 'clusters' in clusters:
            sig17_data = clusters['clusters']
            if isinstance(sig17_data, pd.DataFrame) or isinstance(sig17_data, np.ndarray):
                print(f"\nSignature 17 data shape: {sig17_data.shape if hasattr(sig17_data, 'shape') else 'N/A'}")
                if isinstance(sig17_data, pd.DataFrame):
                    print(sig17_data.head(20))
                elif isinstance(sig17_data, np.ndarray):
                    print(f"First 20 values: {sig17_data[:20] if sig17_data.ndim == 1 else sig17_data[:20, :]}")

# Check for phi (signature-disease association matrix)
if 'phi' in mgb_data:
    phi = mgb_data['phi']
    print(f"\nFound 'phi' in model data")
    print(f"Phi shape: {phi.shape if hasattr(phi, 'shape') else type(phi)}")
    
    if hasattr(phi, 'shape'):
        if phi.shape[0] > 17:  # Check if we have signature 17
            sig17_phi = phi[17, :]  # Get signature 17's association with all diseases
            # Get top diseases
            top_indices = np.argsort(sig17_phi)[::-1][:20]
            print(f"\nTop 20 diseases in MGB Signature 17 (by phi values):")
            for idx in top_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (phi={sig17_phi[idx]:.4f})")

# Check model_state_dict for phi
if 'model_state_dict' in mgb_data:
    model_state = mgb_data['model_state_dict']
    print(f"\nModel state dict keys: {list(model_state.keys())}")
    
    if 'phi' in model_state:
        phi = model_state['phi']
        if hasattr(phi, 'detach'):
            phi = phi.detach().numpy()
        print(f"\nFound 'phi' in model_state_dict")
        print(f"Phi shape: {phi.shape}")
        
        if phi.shape[0] > 17:
            sig17_phi = phi[17, :]
            top_indices = np.argsort(sig17_phi)[::-1][:20]
            print(f"\nTop 20 diseases in MGB Signature 17 (by phi values):")
            for idx in top_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (phi={sig17_phi[idx]:.4f})")

# Also check if there's a direct way to access signature 17 diseases
print(f"\n{'='*80}")
print("Attempting to find diseases in signature 17...")
print(f"{'='*80}")

# If clusters is a patient-level assignment, we need to look at phi or disease associations
# Let's try to find phi or similar structure
if 'model_state_dict' in mgb_data:
    state_dict = mgb_data['model_state_dict']
    for key in state_dict.keys():
        if 'phi' in key.lower() or 'disease' in key.lower():
            print(f"\nFound potential disease association key: {key}")
            data = state_dict[key]
            if hasattr(data, 'detach'):
                data = data.detach().numpy()
            print(f"  Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            if hasattr(data, 'shape') and len(data.shape) == 2:
                if data.shape[0] > 17:  # Signatures x Diseases
                    sig17_assoc = data[17, :]
                    top_indices = np.argsort(sig17_assoc)[::-1][:20]
                    print(f"\n  Top 20 diseases in MGB Signature 17:")
                    for idx in top_indices:
                        print(f"    {idx}: {disease_names_mgb[idx]} (value={sig17_assoc[idx]:.4f})")
                elif data.shape[1] > 17:  # Diseases x Signatures
                    sig17_assoc = data[:, 17]
                    top_indices = np.argsort(sig17_assoc)[::-1][:20]
                    print(f"\n  Top 20 diseases in MGB Signature 17:")
                    for idx in top_indices:
                        print(f"    {idx}: {disease_names_mgb[idx]} (value={sig17_assoc[idx]:.4f})")

