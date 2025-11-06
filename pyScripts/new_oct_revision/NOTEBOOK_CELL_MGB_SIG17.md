# Check MGB Signature 17 Diseases

Run this cell in your notebook to see which diseases are in MGB signature 17:

```python
# Check diseases in MGB signature 17
clusters = mgb_data['clusters']

print("="*80)
print("DISEASES IN MGB SIGNATURE 17")
print("="*80)
print(f"\nClusters type: {type(clusters)}")
print(f"Clusters shape: {clusters.shape if hasattr(clusters, 'shape') else 'N/A'}")

# If clusters is a DataFrame
if isinstance(clusters, pd.DataFrame):
    print(f"Columns: {list(clusters.columns)[:10] if len(clusters.columns) > 10 else list(clusters.columns)}")
    
    # Check if column 17 exists (signature 17)
    if 17 in clusters.columns:
        sig17_mask = clusters[17] > 0
        sig17_indices = clusters[sig17_mask].index.tolist()
        print(f"\n✅ Found {len(sig17_indices)} diseases in signature 17 (column 17 > 0):")
        for idx in sig17_indices[:50]:  # Show first 50
            print(f"  {idx}: {disease_names_mgb[idx]}")
    else:
        print(f"\n⚠️  Column 17 not found. Available columns: {list(clusters.columns)[:20]}")

# If clusters is a numpy array (likely disease x signature matrix)
elif isinstance(clusters, np.ndarray):
    print(f"Clusters shape: {clusters.shape}")
    
    # Try different interpretations
    if clusters.shape[1] > 17:  # Diseases x Signatures
        sig17_diseases = clusters[:, 17]
        sig17_indices = np.where(sig17_diseases > 0)[0]
        print(f"\n✅ Found {len(sig17_indices)} diseases in signature 17 (column 17 > 0):")
        for idx in sig17_indices[:50]:
            print(f"  {idx}: {disease_names_mgb[idx]} (value={sig17_diseases[idx]:.4f})")
    elif clusters.shape[0] > 17:  # Signatures x Diseases
        sig17_diseases = clusters[17, :]
        sig17_indices = np.where(sig17_diseases > 0)[0]
        print(f"\n✅ Found {len(sig17_indices)} diseases in signature 17 (row 17 > 0):")
        for idx in sig17_indices[:50]:
            print(f"  {idx}: {disease_names_mgb[idx]} (value={sig17_diseases[idx]:.4f})")

# Also check phi (signature-disease association) if available
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
        elif phi.shape[1] > 17:  # Diseases x Signatures
            sig17_phi = phi[:, 17]
        else:
            sig17_phi = None
        
        if sig17_phi is not None:
            top_indices = np.argsort(sig17_phi)[::-1][:30]
            print(f"\nTop 30 diseases in MGB Signature 17 (by phi values):")
            for idx in top_indices:
                print(f"  {idx}: {disease_names_mgb[idx]} (phi={sig17_phi[idx]:.4f})")
```

This will show you which diseases are associated with MGB signature 17, using both the `clusters` structure and the `phi` matrix if available.

