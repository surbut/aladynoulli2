#!/usr/bin/env python3
"""
Recreate case_control_sig{k}.tsv phenotype files for GWAS analysis.

Logic:
- For each signature k, find all diseases assigned to that signature (from initial_clusters)
- Each file has columns: FID, plus one column per disease in that signature
- Column names are the disease names from disease_names.csv (with spaces replaced by underscores)
- Values: 1 = case (ever had disease), 0 = control (never had disease)

Example: If signature 5 contains diseases like "Unstable angina", "Myocardial infarction", etc.
The output file case_control_sig5.tsv will have columns:
FID, Hypercholesterolemia, Unstable_angina_(intermediate_coronary_syndrome), ...
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running')
PIDS_PATH = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/processed_ids.csv')
DISEASE_NAMES_PATH = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/disease_names.csv')

# Output directory - create if doesn't exist
OUTPUT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/case_control_phenotypes')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_disease_name(name):
    """Convert disease name to column-safe format (replace spaces with underscores)."""
    return name.replace(' ', '_').replace(',', ',').replace(';', ';')

def main():
    print("Loading data...")
    
    # Load Y tensor (N, D, T) - patients x diseases x time
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)
    print(f"Y_tensor shape: {Y.shape}")  # Should be (N, D, T)
    N, D, T = Y.shape
    
    # Load initial clusters (disease -> signature mapping)
    clusters = torch.load(DATA_DIR / 'initial_clusters_400k.pt', weights_only=False)
    if isinstance(clusters, torch.Tensor):
        clusters = clusters.numpy()
    print(f"Clusters shape: {clusters.shape}, unique values: {np.unique(clusters)}")
    K = int(clusters.max() + 1)
    print(f"Number of signatures: {K}")
    
    # Load patient IDs
    pids_df = pd.read_csv(PIDS_PATH)
    pids = pids_df['eid'].values
    print(f"Number of patient IDs: {len(pids)}")
    
    # Use only the first 400K patients (Y may have more, but we only have IDs for 400K)
    N_use = len(pids)
    if N > N_use:
        print(f"Note: Y has {N} patients but only {N_use} patient IDs - using first {N_use}")
        Y = Y[:N_use, :, :]
        N = N_use
    
    # Load disease names
    disease_df = pd.read_csv(DISEASE_NAMES_PATH)
    disease_names = disease_df['x'].values  # Column is named 'x' based on the file
    print(f"Number of disease names: {len(disease_names)}")
    
    # Convert Y to numpy and check if any time slice is non-zero per disease
    # Y_ever[n, d] = True if patient n ever had disease d
    Y_np = Y.numpy() if isinstance(Y, torch.Tensor) else Y
    Y_ever = (Y_np != 0).any(axis=2)  # Shape: (N, D)
    print(f"Y_ever shape: {Y_ever.shape}")
    
    # Create case-control for each signature
    for sig in range(K):
        print(f"\n--- Signature {sig} ---")
        
        # Find diseases in this signature
        disease_indices = np.where(clusters == sig)[0]
        print(f"Number of diseases in signature {sig}: {len(disease_indices)}")
        
        # Get disease names for this signature (cleaned for column names)
        sig_disease_names = []
        for d in disease_indices:
            if d < len(disease_names):
                name = clean_disease_name(disease_names[d])
            else:
                name = f"Disease_{d}"
            sig_disease_names.append(name)
        
        print(f"Diseases: {sig_disease_names[:5]}{'...' if len(sig_disease_names) > 5 else ''}")
        
        # Create dataframe with FID only (matching original format)
        pheno_df = pd.DataFrame({
            'FID': pids
        })
        
        # Add column for each disease in this signature
        # Value: 1 = case (ever had), 0 = control (never had)
        for i, d in enumerate(disease_indices):
            col_name = sig_disease_names[i]
            # Y_ever[:, d] is boolean - convert to 1/0
            pheno_df[col_name] = Y_ever[:, d].astype(int)
        
        # Print some stats
        for i, d in enumerate(disease_indices[:3]):
            col_name = sig_disease_names[i]
            n_cases = (pheno_df[col_name] == 1).sum()
            print(f"  {col_name}: {n_cases} cases")
        
        # Save
        output_path = OUTPUT_DIR / f'case_control_sig{sig}.tsv'
        pheno_df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved: {output_path} (shape: {pheno_df.shape})")
    
    print("\n=== Done! ===")
    print(f"Created {K} phenotype files in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
