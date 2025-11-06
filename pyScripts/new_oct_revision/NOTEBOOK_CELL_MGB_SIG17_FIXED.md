# Check MGB Signature 17 Diseases (Fixed)

Run this cell in your notebook to see which diseases are in MGB signature 17:

```python
# Check diseases in MGB signature 17
clusters = mgb_data['clusters']

print("="*80)
print("DISEASES IN MGB SIGNATURE 17")
print("="*80)
print(f"\nClusters type: {type(clusters)}")

# If clusters is a DataFrame
if isinstance(clusters, pd.DataFrame):
    print(f"Clusters shape: {clusters.shape}")
    print(f"Columns: {list(clusters.columns)[:10] if len(clusters.columns) > 10 else list(clusters.columns)}")
    
    # Check if column 17 exists (signature 17)
    if 17 in clusters.columns:
        sig17_mask = clusters[17] > 0
        sig17_indices = clusters[sig17_mask].index.tolist()
        print(f"\n✅ Found {len(sig17_indices)} diseases in signature 17 (column 17 > 0):")
        for idx in sig17_indices:
            print(f"  {idx}: {disease_names_mgb[idx]}")
    else:
        print(f"\n⚠️  Column 17 not found. Available columns: {list(clusters.columns)[:20]}")

# If clusters is a numpy array
elif isinstance(clusters, np.ndarray):
    print(f"Clusters shape: {clusters.shape}")
    print(f"Clusters ndim: {clusters.ndim}")
    
    if clusters.ndim == 1:
        # 1D array - might be patient-level assignments
        print(f"\n⚠️  Clusters is 1D array (length: {len(clusters)})")
        print(f"   This is likely patient-level signature assignments, not disease-signature associations.")
        print(f"   Use phi (signature-disease association matrix) instead - see below.")
    elif clusters.ndim == 2:
        # 2D array - check dimensions
        if clusters.shape[1] > 17:  # Diseases x Signatures
            sig17_diseases = clusters[:, 17]
            sig17_indices = np.where(sig17_diseases > 0)[0]
            print(f"\n✅ Found {len(sig17_indices)} diseases in signature 17 (column 17 > 0):")
            for idx in sig17_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (value={sig17_diseases[idx]:.4f})")
        elif clusters.shape[0] > 17:  # Signatures x Diseases
            sig17_diseases = clusters[17, :]
            sig17_indices = np.where(sig17_diseases > 0)[0]
            print(f"\n✅ Found {len(sig17_indices)} diseases in signature 17 (row 17 > 0):")
            for idx in sig17_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (value={sig17_diseases[idx]:.4f})")
        else:
            print(f"\n⚠️  Clusters shape {clusters.shape} doesn't have dimension > 17")
    else:
        print(f"\n⚠️  Clusters has {clusters.ndim} dimensions - unexpected shape")

# Use phi (signature-disease association matrix) - this is the correct way
print(f"\n{'='*80}")
print("USING PHI (SIGNATURE-DISEASE ASSOCIATION MATRIX)")
print(f"{'='*80}")

if 'model_state_dict' in mgb_data:
    model_state = mgb_data['model_state_dict']
    if 'phi' in model_state:
        phi = model_state['phi']
        if hasattr(phi, 'detach'):
            phi = phi.detach().numpy()
        
        print(f"Phi shape: {phi.shape}")
        print(f"Phi ndim: {phi.ndim}")
        
        # Get signature 17's disease associations
        if phi.ndim == 2:
            if phi.shape[0] > 17:  # Signatures x Diseases
                sig17_phi = phi[17, :]
                print(f"\n✅ Using phi[17, :] (signature 17's associations with all diseases)")
            elif phi.shape[1] > 17:  # Diseases x Signatures
                sig17_phi = phi[:, 17]
                print(f"\n✅ Using phi[:, 17] (all diseases' associations with signature 17)")
            else:
                print(f"\n⚠️  Phi shape {phi.shape} doesn't have dimension > 17")
                sig17_phi = None
        else:
            print(f"\n⚠️  Phi has {phi.ndim} dimensions - unexpected")
            sig17_phi = None
        
        if sig17_phi is not None:
            # Get top diseases (highest phi values)
            top_indices = np.argsort(sig17_phi)[::-1][:50]
            print(f"\nTop 50 diseases in MGB Signature 17 (by phi values):")
            print(f"{'Index':<8} {'Disease Name':<60} {'Phi Value':<12}")
            print("-" * 80)
            for idx in top_indices:
                print(f"{idx:<8} {disease_names_mgb[idx]:<60} {sig17_phi[idx]:<12.4f}")
    else:
        print("\n⚠️  'phi' not found in model_state_dict")
        print(f"   Available keys: {list(model_state.keys())}")
else:
    print("\n⚠️  'model_state_dict' not found in mgb_data")
```

This fixed version:
1. Checks the number of dimensions before accessing shape[1]
2. Handles 1D arrays properly
3. Uses `phi` (signature-disease association matrix) which is the correct way to find diseases in a signature

