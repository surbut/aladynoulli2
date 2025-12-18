#!/usr/bin/env python3
"""
Verify that for each disease, its top signature (highest PSI) stays consistent across batches.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
BATCH_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized')
OLD_CHECKPOINT_PATH = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam.pt')
N_BATCHES_TO_CHECK = 40

print("="*80)
print("VERIFYING TOP SIGNATURE PER DISEASE CONSISTENCY ACROSS BATCHES")
print("="*80)

# Load disease names
print("\nLoading disease names...")
old_checkpoint = torch.load(OLD_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
disease_names = old_checkpoint['disease_names']
if isinstance(disease_names, (list, tuple)):
    disease_names = list(disease_names)
elif hasattr(disease_names, 'values'):
    disease_names = disease_names.values.tolist()
elif torch.is_tensor(disease_names):
    disease_names = disease_names.tolist()
    
print(f"  Loaded {len(disease_names)} disease names")

# Find batch files
batch_files = sorted(BATCH_DIR.glob('enrollment_model_W0.0001_batch_*_*.pt'))[:N_BATCHES_TO_CHECK]
print(f"\nFound {len(batch_files)} batch files to check")

# For each disease, track which signature has the highest PSI in each batch
disease_top_sigs = defaultdict(list)  # {disease_idx: [sig0, sig1, sig2, ...] across batches}

for batch_idx, batch_file in enumerate(batch_files):
    print(f"Processing batch {batch_idx + 1}/{len(batch_files)}: {batch_file.name}")
    
    checkpoint = torch.load(batch_file, map_location='cpu', weights_only=False)
    
    # Extract PSI
    if 'model_state_dict' in checkpoint:
        psi = checkpoint['model_state_dict']['psi']
    else:
        psi = checkpoint['psi']
    
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    K, D = psi.shape
    
    # For each disease, find the signature with highest PSI
    for d in range(D):
        # Get PSI values for this disease across all signatures
        psi_values = psi[:, d]  # Shape: (K,)
        top_sig = np.argmax(psi_values)
        disease_top_sigs[d].append(top_sig)

print("\n" + "="*80)
print("TOP SIGNATURE CONSISTENCY PER DISEASE")
print("="*80)

# Analyze consistency
consistency_stats = {
    'fully_consistent': 0,  # Same top sig in all batches
    'mostly_consistent': 0,  # Same top sig in >= 80% of batches
    'inconsistent': 0  # Different top sigs across batches
}

# Store results
results = []

for d in range(len(disease_names)):
    if d not in disease_top_sigs:
        continue
    
    top_sigs = disease_top_sigs[d]
    most_common_sig = max(set(top_sigs), key=top_sigs.count)
    consistency_ratio = top_sigs.count(most_common_sig) / len(top_sigs)
    
    # Check if all the same
    if len(set(top_sigs)) == 1:
        consistency_stats['fully_consistent'] += 1
        consistency = 'Fully consistent'
    elif consistency_ratio >= 0.8:
        consistency_stats['mostly_consistent'] += 1
        consistency = 'Mostly consistent'
    else:
        consistency_stats['inconsistent'] += 1
        consistency = 'Inconsistent'
    
    # Get PSI values for the most common signature (from first batch for reference)
    first_batch = torch.load(batch_files[0], map_location='cpu', weights_only=False)
    if 'model_state_dict' in first_batch:
        psi_ref = first_batch['model_state_dict']['psi']
    else:
        psi_ref = first_batch['psi']
    if torch.is_tensor(psi_ref):
        psi_ref = psi_ref.detach().cpu().numpy()
    psi_value = psi_ref[most_common_sig, d]
    
    results.append({
        'Disease_Idx': d,
        'Disease_Name': disease_names[d] if d < len(disease_names) else f'Disease_{d}',
        'Top_Signature': most_common_sig,
        'Consistency_Ratio': consistency_ratio,
        'Consistency': consistency,
        'PSI_Value': psi_value,
        'Batch_Counts': {sig: top_sigs.count(sig) for sig in set(top_sigs)}
    })

# Sort by consistency ratio (most consistent first), then by PSI value
results.sort(key=lambda x: (-x['Consistency_Ratio'], -x['PSI_Value']))

# Create DataFrame and save
results_df = pd.DataFrame(results)
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'top_signature_per_disease_consistency.csv'

# Clean up the Batch_Counts column for CSV (convert dict to string)
results_df_clean = results_df.copy()
results_df_clean['Batch_Counts'] = results_df_clean['Batch_Counts'].astype(str)
results_df_clean.to_csv(output_file, index=False)

print(f"\n{'Disease':<50} {'Top Sig':<10} {'Consistency':<20} {'PSI':<10}")
print("-"*90)

for result in results[:50]:  # Show top 50 most consistent
    name = result['Disease_Name'][:48]
    print(f"{name:<50} {result['Top_Signature']:<10} {result['Consistency']:<20} {result['PSI_Value']:>8.3f}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total diseases: {len(results)}")
print(f"Fully consistent (same top sig in all batches): {consistency_stats['fully_consistent']} ({100*consistency_stats['fully_consistent']/len(results):.1f}%)")
print(f"Mostly consistent (>=80%): {consistency_stats['mostly_consistent']} ({100*consistency_stats['mostly_consistent']/len(results):.1f}%)")
print(f"Inconsistent (<80%): {consistency_stats['inconsistent']} ({100*consistency_stats['inconsistent']/len(results):.1f}%)")

print(f"\n✓ Results saved to: {output_file}")

# Also create summary by signature
print("\n" + "="*80)
print("DISEASES PER SIGNATURE (based on most common top signature)")
print("="*80)

sig_to_diseases = defaultdict(list)
for result in results:
    sig_to_diseases[result['Top_Signature']].append(result)

for sig in sorted(sig_to_diseases.keys()):
    diseases = sig_to_diseases[sig]
    diseases.sort(key=lambda x: -x['PSI_Value'])
    print(f"\nSignature {sig}: {len(diseases)} diseases")
    print(f"  Top 10 diseases by PSI:")
    for i, d in enumerate(diseases[:10], 1):
        consistency_marker = "✓" if d['Consistency_Ratio'] == 1.0 else "~" if d['Consistency_Ratio'] >= 0.8 else "⚠"
        print(f"    {i:2d}. {d['Disease_Name'][:45]:<45} PSI={d['PSI_Value']:>6.3f} {consistency_marker}")

print("="*80)

