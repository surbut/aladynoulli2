#!/usr/bin/env python3
"""
Extract sample patients from censor_e_batchrun_vectorized_nolr batches
and save in compact format for the patient timeline app.
"""

import torch
import numpy as np
from pathlib import Path
from scipy.special import softmax
import glob
from typing import List, Tuple, Optional

def load_batch_file(batch_path: Path, start_idx: int, end_idx: int) -> Optional[dict]:
    """Load a single batch file."""
    # Try different filename patterns
    patterns = [
        f"enrollment_model_VECTORIZED_W0.0001_nolr_batch_{start_idx}_{end_idx}.pt",
        f"enrollment_model_W0.0001_batch_{start_idx}_{end_idx}.pt",
        f"enrollment_model_VECTORIZED_W0.0001_batch_{start_idx}_{end_idx}.pt",
    ]
    
    for pattern in patterns:
        filepath = batch_path / pattern
        if filepath.exists():
            print(f"  Loading {filepath.name}...")
            try:
                data = torch.load(filepath, map_location='cpu', weights_only=False)
                return data
            except Exception as e:
                print(f"    Error loading {filepath.name}: {e}")
                continue
    
    return None

def find_interesting_patients(Y: np.ndarray, psi: np.ndarray, disease_names: List[str],
                               min_diseases: int = 3, min_sigs: int = 2, 
                               min_time_spread: int = 10, top_n: int = 100) -> List[int]:
    """Find interesting patients with multiple diagnoses across multiple signatures."""
    N, D, T = Y.shape
    patient_scores = []
    
    # Find MI index if available
    mi_idx = None
    for i, name in enumerate(disease_names):
        name_lower = str(name).lower()
        if "myocardial infarction" in name_lower or "heart attack" in name_lower:
            mi_idx = i
            break
    
    for n in range(N):
        diag_events = []
        sig_set = set()
        times = []
        
        for d in range(D):
            diag_times = np.where(Y[n, d, :] > 0.5)[0]
            if len(diag_times) > 0:
                t_diag = diag_times[0]
                times.append(t_diag)
                sig_idx = np.argmax(psi[:, d])
                diag_events.append((d, t_diag, sig_idx))
                sig_set.add(sig_idx)
        
        # Prefer MI patients if available
        has_mi = (mi_idx is not None and np.any(Y[n, mi_idx, :] > 0.5))
        mi_bonus = 5.0 if has_mi else 0.0
        
        if len(diag_events) >= min_diseases and len(sig_set) >= min_sigs:
            if len(times) > 1 and (max(times) - min(times)) >= min_time_spread:
                # Score: more diseases, more sigs, more spread, bonus for MI
                score = len(diag_events) + len(sig_set) + (max(times) - min(times)) / 10.0 + mi_bonus
                patient_scores.append((score, n, has_mi))
    
    # Sort by score, descending
    patient_scores.sort(reverse=True)
    
    # Prefer MI patients in top selection
    selected = []
    mi_selected = []
    regular_selected = []
    
    for score, n, has_mi in patient_scores:
        if has_mi and len(mi_selected) < top_n // 2:
            mi_selected.append(n)
        elif not has_mi and len(regular_selected) < top_n // 2:
            regular_selected.append(n)
        
        if len(mi_selected) + len(regular_selected) >= top_n:
            break
    
    selected = mi_selected + regular_selected
    print(f"  Selected {len(mi_selected)} MI patients and {len(regular_selected)} regular patients")
    return selected

def create_compact_app_data(
    batch_dir: str,
    master_checkpoint_path: Optional[str] = None,
    output_path: str = "app_patients_compact.pt",
    max_batches: Optional[int] = None,
    n_patients: int = 100,
    batch_size: int = 10000,
    total_patients: int = 400000
):
    """
    Create compact app data from batch files.
    
    Args:
        batch_dir: Directory containing batch files
        master_checkpoint_path: Path to master checkpoint for phi, psi, gamma
        output_path: Where to save the compact file
        max_batches: Maximum number of batches to load (None = all)
        n_patients: Number of sample patients to select
        batch_size: Size of each batch
        total_patients: Total number of patients in dataset
    """
    batch_dir = Path(batch_dir)
    
    print("="*80)
    print("CREATING COMPACT APP DATA")
    print("="*80)
    
    # 1. Load phi, psi, gamma from master checkpoint or batch files
    print("\n1. Loading model parameters (phi, psi, gamma)...")
    
    if master_checkpoint_path and Path(master_checkpoint_path).exists():
        print(f"  Loading from master checkpoint: {master_checkpoint_path}")
        master = torch.load(master_checkpoint_path, map_location='cpu', weights_only=False)
        phi = master['model_state_dict']['phi']
        psi = master['model_state_dict']['psi']
        gamma = master['model_state_dict'].get('gamma', None)
        if 'disease_names' in master:
            disease_names = master['disease_names']
        elif 'model_state_dict' in master and 'disease_names' in master['model_state_dict']:
            disease_names = master['model_state_dict']['disease_names']
        else:
            # Load from data directory
            disease_names_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv')
            if disease_names_path.exists():
                import pandas as pd
                disease_names = pd.read_csv(disease_names_path, header=None).iloc[:, 0].tolist()
            else:
                raise FileNotFoundError("Could not find disease_names")
    else:
        print("  Master checkpoint not found, will need to pool from batches")
        phi = None
        psi = None
        gamma = None
        disease_names = None
    
    # Convert to numpy
    if phi is not None and torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    if psi is not None and torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    if gamma is not None and torch.is_tensor(gamma):
        gamma = gamma.detach().cpu().numpy()
    
    if disease_names is None:
        # Try to load from first batch or data directory
        disease_names_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv')
        if disease_names_path.exists():
            import pandas as pd
            disease_names = pd.read_csv(disease_names_path, header=None).iloc[:, 0].tolist()
            print(f"  Loaded {len(disease_names)} disease names from CSV")
        else:
            raise FileNotFoundError("Could not find disease_names")
    
    K = phi.shape[0] if phi is not None else 21
    D = len(disease_names)
    T = phi.shape[2] if phi is not None else 52
    
    print(f"  ✓ Model dimensions: K={K}, D={D}, T={T}")
    
    # 2. Load batches and collect lambda, G, Y, pids
    print(f"\n2. Loading batches from {batch_dir}...")
    
    all_lambdas = []
    all_G = []
    all_Y = []
    all_pids = []
    batch_start_indices = []
    
    num_batches_to_load = (total_patients // batch_size) if max_batches is None else max_batches
    
    for batch_num in range(num_batches_to_load):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_patients)
        
        batch_data = load_batch_file(batch_dir, start_idx, end_idx)
        if batch_data is None:
            print(f"  ⚠️  Skipping batch {start_idx}-{end_idx}")
            continue
        
        # Extract data
        if 'model_state_dict' in batch_data:
            lambda_batch = batch_data['model_state_dict']['lambda_']
            if torch.is_tensor(lambda_batch):
                lambda_batch = lambda_batch.detach().cpu().numpy()
        else:
            lambda_batch = batch_data.get('lambda_', None)
            if lambda_batch is not None and torch.is_tensor(lambda_batch):
                lambda_batch = lambda_batch.detach().cpu().numpy()
        
        if lambda_batch is None:
            print(f"  ⚠️  No lambda_ found in batch {start_idx}-{end_idx}")
            continue
        
        # Get G, Y, pids if available
        G_batch = batch_data.get('G', None)
        if G_batch is not None and torch.is_tensor(G_batch):
            G_batch = G_batch.detach().cpu().numpy()
        
        Y_batch = batch_data.get('Y', None)
        if Y_batch is not None and torch.is_tensor(Y_batch):
            Y_batch = Y_batch.detach().cpu().numpy()
        
        pids_batch = batch_data.get('pids', None)
        if pids_batch is None and 'processed_ids' in batch_data:
            pids_batch = batch_data['processed_ids']
        
        all_lambdas.append(lambda_batch)
        if G_batch is not None:
            all_G.append(G_batch)
        if Y_batch is not None:
            all_Y.append(Y_batch)
        if pids_batch is not None:
            all_pids.append(pids_batch)
        
        batch_start_indices.append(start_idx)
        
        if (batch_num + 1) % 10 == 0:
            print(f"  Loaded {batch_num + 1}/{num_batches_to_load} batches...")
    
    print(f"  ✓ Loaded {len(all_lambdas)} batches")
    
    # Concatenate all batches
    print(f"\n3. Concatenating data...")
    lambda_all = np.concatenate(all_lambdas, axis=0)
    print(f"  ✓ Lambda shape: {lambda_all.shape}")
    
    if all_G:
        G_all = np.concatenate(all_G, axis=0)
        print(f"  ✓ G shape: {G_all.shape}")
    else:
        print("  ⚠️  No G data found in batches")
        G_all = None
    
    if all_Y:
        Y_all = np.concatenate(all_Y, axis=0)
        print(f"  ✓ Y shape: {Y_all.shape}")
    else:
        print("  ⚠️  Loading Y from data directory...")
        Y_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
        if Y_path.exists():
            Y_all = torch.load(Y_path, map_location='cpu', weights_only=False)
            if torch.is_tensor(Y_all):
                Y_all = Y_all.detach().cpu().numpy()
            # Subset to loaded patients
            Y_all = Y_all[:len(lambda_all)]
            print(f"  ✓ Y shape: {Y_all.shape}")
        else:
            raise FileNotFoundError("Could not find Y_tensor.pt")
    
    if all_pids:
        pids_all = np.concatenate(all_pids, axis=0)
        print(f"  ✓ PIDs shape: {pids_all.shape}")
    else:
        pids_all = None
    
    N_total = lambda_all.shape[0]
    print(f"  ✓ Total patients: {N_total}")
    
    # 4. Select interesting patients
    print(f"\n4. Selecting {n_patients} interesting patients...")
    
    if psi is None:
        # Use max phi as proxy for psi
        psi = np.max(phi, axis=2) if phi is not None else None
    
    if psi is not None:
        selected_indices = find_interesting_patients(
            Y_all, psi, disease_names, 
            min_diseases=2, min_sigs=2, min_time_spread=5, top_n=n_patients
        )
    else:
        # Fallback: random selection
        selected_indices = np.random.choice(N_total, size=min(n_patients, N_total), replace=False).tolist()
        print("  ⚠️  No psi available, using random selection")
    
    print(f"  ✓ Selected {len(selected_indices)} patients")
    
    # 5. Extract selected patients' data
    print(f"\n5. Extracting selected patients' data...")
    
    lambda_selected = lambda_all[selected_indices]
    print(f"  ✓ Lambda selected: {lambda_selected.shape}")
    
    if G_all is not None:
        G_selected = G_all[selected_indices]
        print(f"  ✓ G selected: {G_selected.shape}")
    else:
        G_selected = None
        print("  ⚠️  No G data available")
    
    Y_selected = Y_all[selected_indices]
    print(f"  ✓ Y selected: {Y_selected.shape}")
    
    if pids_all is not None:
        pids_selected = pids_all[selected_indices]
        print(f"  ✓ PIDs selected: {pids_selected.shape}")
    else:
        pids_selected = None
    
    # Convert lambda to theta
    theta_selected = softmax(lambda_selected, axis=1)
    
    # 6. Get cluster assignments (signature assignments for diseases)
    print(f"\n6. Computing cluster assignments...")
    if psi is not None:
        clusters = np.argmax(psi, axis=0)  # [D] - dominant signature for each disease
        print(f"  ✓ Cluster assignments: {clusters.shape}")
    else:
        clusters = None
        print("  ⚠️  No cluster assignments available")
    
    # 7. Prepare final data structure
    print(f"\n7. Preparing final data structure...")
    
    model_state_dict = {
        'lambda_': torch.from_numpy(lambda_selected),
        'phi': torch.from_numpy(phi) if phi is not None else None,
        'psi': torch.from_numpy(psi) if psi is not None else None,
    }
    
    if gamma is not None:
        model_state_dict['gamma'] = torch.from_numpy(gamma)
    
    output_data = {
        'model_state_dict': model_state_dict,
        'disease_names': disease_names,
        'clusters': clusters if clusters is not None else np.zeros(D, dtype=int),
        'G': torch.from_numpy(G_selected) if G_selected is not None else None,
        'Y': torch.from_numpy(Y_selected),
        'meta': {
            'original_N': N_total,
            'kept_indices': selected_indices,
            'K': K,
            'D': D,
            'T': T,
            'selection_criteria': f'App-selected patients with multi-morbidity (from {len(all_lambdas)} batches)',
            'batch_dir': str(batch_dir),
            'master_checkpoint': master_checkpoint_path if master_checkpoint_path else None,
        }
    }
    
    # 8. Save
    print(f"\n8. Saving to {output_path}...")
    torch.save(output_data, output_path)
    print(f"  ✓ Saved successfully!")
    
    print("\n" + "="*80)
    print("COMPACT APP DATA CREATION COMPLETE")
    print("="*80)
    print(f"Output file: {output_path}")
    print(f"Selected patients: {len(selected_indices)}")
    print(f"Model dimensions: K={K}, D={D}, T={T}")
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create compact app data from batch files")
    parser.add_argument("--batch-dir", type=str, 
                       default="/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr",
                       help="Directory containing batch files (e.g., censor_e_batchrun_vectorized_nolr)")
    parser.add_argument("--master-checkpoint", type=str,
                       default="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE_nolr.pt",
                       help="Path to master checkpoint for phi, psi, gamma")
    parser.add_argument("--output", type=str, default="app_patients_compact_nolr.pt",
                       help="Output file path")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Maximum number of batches to load (None = all)")
    parser.add_argument("--n-patients", type=int, default=100,
                       help="Number of sample patients to select")
    
    args = parser.parse_args()
    
    create_compact_app_data(
        batch_dir=args.batch_dir,
        master_checkpoint_path=args.master_checkpoint,
        output_path=args.output,
        max_batches=args.max_batches,
        n_patients=args.n_patients,
    )

