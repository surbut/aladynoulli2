#!/usr/bin/env python3
"""
Verify top diseases per signature based on psi values.
Compare across batches to ensure consistency.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
BATCH_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized')
OLD_CHECKPOINT_PATH = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam.pt')
N_BATCHES_TO_CHECK = 10  # Check first 10 batches
TOP_N_DISEASES = 10  # Show top N diseases per signature

print("="*80)
print("VERIFYING TOP DISEASES PER SIGNATURE (based on PSI values)")
print("="*80)
print(f"Batch directory: {BATCH_DIR}")
print(f"Number of batches to check: {N_BATCHES_TO_CHECK}")
print(f"Top N diseases per signature: {TOP_N_DISEASES}")
print("="*80)

# Load disease names from old checkpoint
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
batch_pattern = str(BATCH_DIR / 'enrollment_model_W0.0001_batch_*_*.pt')
batch_files = sorted(BATCH_DIR.glob('enrollment_model_W0.0001_batch_*_*.pt'))[:N_BATCHES_TO_CHECK]

if len(batch_files) == 0:
    print(f"\n⚠️  No batch files found matching pattern: {batch_pattern}")
    exit(1)

print(f"\nFound {len(batch_files)} batch files to check")

# Load PSI from each batch and store top diseases per signature
all_top_diseases = defaultdict(lambda: defaultdict(list))  # {sig: {disease_idx: [counts across batches]}}

for batch_idx, batch_file in enumerate(batch_files):
    print(f"\nProcessing batch {batch_idx + 1}/{len(batch_files)}: {batch_file.name}")
    
    checkpoint = torch.load(batch_file, map_location='cpu', weights_only=False)
    
    # Extract PSI
    if 'model_state_dict' in checkpoint:
        psi = checkpoint['model_state_dict']['psi']
    else:
        psi = checkpoint['psi']
    
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    # Extract clusters
    if 'clusters' in checkpoint:
        clusters = checkpoint['clusters']
        if torch.is_tensor(clusters):
            clusters = clusters.numpy()
    else:
        print(f"  ⚠️  No clusters found in batch {batch_idx}")
        continue
    
    K, D = psi.shape
    print(f"  PSI shape: {psi.shape} (K={K} signatures, D={D} diseases)")
    
    # For each signature, get top diseases by psi value
    for sig in range(K):
        # Get diseases in this signature
        diseases_in_sig = np.where(clusters == sig)[0]
        
        if len(diseases_in_sig) == 0:
            continue
        
        # Get psi values for diseases in this signature
        psi_values = [(psi[sig, d], d) for d in diseases_in_sig]
        psi_values.sort(reverse=True)  # Sort descending (most positive first)
        
        # Store top N diseases
        top_diseases = psi_values[:TOP_N_DISEASES]
        for psi_val, d_idx in top_diseases:
            all_top_diseases[sig][d_idx].append(psi_val)

print("\n" + "="*80)
print("TOP DISEASES PER SIGNATURE (based on PSI values)")
print("="*80)

# For each signature, show top diseases and their consistency
n_sigs = len(all_top_diseases)
for sig in range(n_sigs):
    if sig not in all_top_diseases:
        continue
    
    sig_diseases = all_top_diseases[sig]
    
    # Calculate average psi and frequency across batches
    disease_stats = []
    for d_idx, psi_values in sig_diseases.items():
        avg_psi = np.mean(psi_values)
        frequency = len(psi_values)  # How many batches this disease appeared in top N
        max_psi = np.max(psi_values)
        min_psi = np.min(psi_values)
        
        disease_stats.append({
            'disease_idx': d_idx,
            'disease_name': disease_names[d_idx] if d_idx < len(disease_names) else f'Disease_{d_idx}',
            'avg_psi': avg_psi,
            'max_psi': max_psi,
            'min_psi': min_psi,
            'frequency': frequency,
            'std_psi': np.std(psi_values) if len(psi_values) > 1 else 0.0
        })
    
    # Sort by average psi (descending)
    disease_stats.sort(key=lambda x: x['avg_psi'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"SIGNATURE {sig}")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Disease Name':<40} {'Avg PSI':<12} {'Frequency':<12} {'Std PSI':<12}")
    print("-"*80)
    
    for rank, stats in enumerate(disease_stats[:TOP_N_DISEASES], 1):
        name = stats['disease_name'][:38]  # Truncate if too long
        freq_str = f"{stats['frequency']}/{len(batch_files)}"
        consistency = "✓" if stats['frequency'] == len(batch_files) else "⚠"
        
        print(f"{rank:<6} {name:<40} {stats['avg_psi']:>10.3f}   {freq_str:<12} {stats['std_psi']:>10.3f} {consistency}")
    
    # Show consistency summary
    consistent_count = sum(1 for s in disease_stats[:TOP_N_DISEASES] if s['frequency'] == len(batch_files))
    print(f"\n  Consistency: {consistent_count}/{min(TOP_N_DISEASES, len(disease_stats))} top diseases appear in all batches")

# Create summary DataFrame
summary_data = []
for sig in range(n_sigs):
    if sig not in all_top_diseases:
        continue
    
    sig_diseases = all_top_diseases[sig]
    disease_stats = []
    for d_idx, psi_values in sig_diseases.items():
        disease_stats.append({
            'Signature': sig,
            'Disease_Idx': d_idx,
            'Disease_Name': disease_names[d_idx] if d_idx < len(disease_names) else f'Disease_{d_idx}',
            'Avg_PSI': np.mean(psi_values),
            'Max_PSI': np.max(psi_values),
            'Min_PSI': np.min(psi_values),
            'Frequency': len(psi_values),
            'Std_PSI': np.std(psi_values) if len(psi_values) > 1 else 0.0
        })
    
    disease_stats.sort(key=lambda x: x['Avg_PSI'], reverse=True)
    summary_data.extend(disease_stats[:TOP_N_DISEASES])

summary_df = pd.DataFrame(summary_data)

# Save to CSV
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'top_diseases_per_signature.csv'
summary_df.to_csv(output_file, index=False)
print(f"\n✓ Summary saved to: {output_file}")

print("\n" + "="*80)
print("SUMMARY COMPLETE")
print("="*80)

















