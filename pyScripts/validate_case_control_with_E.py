#!/usr/bin/env python3
"""
Validate case_control by exploring E_matrix structure.
Figure out how the original case_control files were created.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running')
PIDS_PATH = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/processed_ids.csv')
DISEASE_NAMES_PATH = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/disease_names.csv')
BACKUP_DIR = Path('/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/for_regenie/case_control_phenotypes')

def clean_disease_name(name):
    return name.replace(' ', '_')

def main():
    print("Loading data...")
    
    # Load Y tensor (N, D, T)
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)
    print(f"Y_tensor shape: {Y.shape}")
    N, D, T = Y.shape
    
    # Load E_matrix
    print("Loading E_matrix regular...")
    E = torch.load(DATA_DIR / 'E_matrix.pt', weights_only=False)
    print(f"E_matrix shape: {E.shape}")
    
    # Load initial clusters
    clusters = torch.load(DATA_DIR / 'initial_clusters_400k.pt', weights_only=False)
    if isinstance(clusters, torch.Tensor):
        clusters = clusters.numpy()
    
    # Load patient IDs
    pids_df = pd.read_csv(PIDS_PATH)
    pids = pids_df['eid'].values
    N_use = len(pids)
    
    if N > N_use:
        print(f"Using first {N_use} patients")
        Y = Y[:N_use, :, :]
        E = E[:N_use, :] if len(E) > N_use else E
    
    # Load disease names
    disease_df = pd.read_csv(DISEASE_NAMES_PATH)
    disease_names = disease_df['x'].values
    
    # Convert to numpy
    Y_np = Y.numpy() if isinstance(Y, torch.Tensor) else Y
    E_np = E.numpy() if isinstance(E, torch.Tensor) else E
    
    # Explore E_matrix structure
    print(f"\n=== E_matrix exploration ===")
    print(f"Shape: {E_np.shape}")
    print(f"Dtype: {E_np.dtype}")
    print(f"Min: {E_np.min()}, Max: {E_np.max()}")
    
    # Check unique values
    unique_vals = np.unique(E_np.flatten())
    print(f"Number of unique values: {len(unique_vals)}")
    print(f"Unique values: {unique_vals[:30]}...")
    
    # For a specific disease column, look at the distribution
    sig = 5
    disease_indices = np.where(clusters == sig)[0]
    
    print(f"\n=== Signature {sig} diseases ===")
    for d in disease_indices:
        name = disease_names[d] if d < len(disease_names) else f"Disease_{d}"
        e_col = E_np[:, d]
        
        print(f"\n{name} (index {d}):")
        print(f"  E unique values: {np.unique(e_col)[:10]}")
        print(f"  E == 51 (max/never): {(e_col == 51).sum()}")
        print(f"  E < 51 (occurred): {(e_col < 51).sum()}")
        print(f"  E == 0: {(e_col == 0).sum()}")
        
        # Check Y for comparison
        y_ever = (Y_np[:, d, :] != 0).any(axis=1)
        print(f"  Y ever != 0: {y_ever.sum()}")
    
    # Load backup and compare
    print(f"\n=== Comparing with backup ===")
    backup_df = pd.read_csv(BACKUP_DIR / f'case_control_sig{sig}.tsv', sep='\t')
    
    print(f"\n{'Disease':<50} {'Y!=0':>10} {'E<51':>10} {'Backup':>10}")
    print("-" * 85)
    
    for d in disease_indices:
        name = disease_names[d] if d < len(disease_names) else f"Disease_{d}"
        clean_name = clean_disease_name(name)
        
        y_count = (Y_np[:, d, :] != 0).any(axis=1).sum()
        e_count = (E_np[:, d] < 51).sum()  # E < 51 means disease occurred
        
        # Find backup column
        backup_col = None
        for col in backup_df.columns:
            if col.startswith(clean_name[:15]):
                backup_col = col
                break
        
        backup_count = backup_df[backup_col].sum() if backup_col else "N/A"
        
        match_y = "✓" if y_count == backup_count else ""
        match_e = "✓" if e_count == backup_count else ""
        
        print(f"{name[:50]:<50} {y_count:>10}{match_y} {e_count:>10}{match_e} {backup_count:>10}")

if __name__ == '__main__':
    main()
