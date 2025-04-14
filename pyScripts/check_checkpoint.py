import torch

# Load checkpoint
checkpoint = torch.load("/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt")

# Print checkpoint keys
print("\nCheckpoint keys:")
print("-" * 40)
for key in checkpoint.keys():
    print(f"- {key}")

# If clusters exist, print their shape/size
if 'clusters' in checkpoint:
    clusters = checkpoint['clusters']
    print(f"\nClusters type: {type(clusters)}")
    if hasattr(clusters, 'shape'):
        print(f"Clusters shape: {clusters.shape}")
    print(f"Number of unique clusters: {len(set(clusters))}")

# If disease_names exist, print some examples
if 'disease_names' in checkpoint:
    disease_names = checkpoint['disease_names']
    print(f"\nDisease names type: {type(disease_names)}")
    print(f"Number of diseases: {len(disease_names)}")
    print("\nFirst few disease names:")
    for i, name in enumerate(disease_names[:5]):
        print(f"{i}: {name}") 