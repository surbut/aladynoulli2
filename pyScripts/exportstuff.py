#!/usr/bin/env python3
import torch
import numpy as np
from scipy.special import softmax

def find_good_mi_overlap_patients(Y,lambda_full, psi, mi_idx, sig_idx=5, window=3, min_diseases=2):
    """Same logic as the app to find good patients"""
    # Convert Y to numpy if it's a torch tensor
   # fallback: all MI patients - convert to numpy first

        
    for n in range(lambda_full.shape[0]):
        if np.any(Y_np[n, mi_idx, :] > 0.5):
            good_patients.append(n)
    
    N, D, T = Y.shape
    selected = []
    for n in range(N):
        # Must have MI
        mi_times = np.where(Y[n, mi_idx, :] > 0.5)[0]
        if len(mi_times) == 0:
            continue
        # Multi-morbid: at least min_diseases diagnosed
        n_diagnoses = sum(1 for d in range(D) if np.any(Y[n, d, :] > 0.5))
        if n_diagnoses < min_diseases:
            continue
        # Check if MI diagnosis is near signature 5 peak
        sig5_traj = softmax(psi, axis=0)[sig_idx, :]
        peak_time = int(np.argmax(sig5_traj))
        # Check if any MI diagnosis is within window of peak
        if any(abs(t - peak_time) <= window for t in mi_times):
            selected.append(n)
    return selected

def export_good_patients_subset(model_path, out_path):
    """Export subset of good patients using app logic"""
    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location='cpu')
    
    # Extract components
    msd = model['model_state_dict']
    lambda_full = msd['lambda_']              # [N, K, T]
    phi = msd['phi']                          # [K, D, T] 
    psi = msd['psi']                          # [K, D, T]
    gamma = msd.get('gamma', None)            # [P, K]
    kappa = msd.get('kappa', None)
    
    Y_full = model['Y']            # [N, D, T]
    G_full = model['G']            # [N, P]
    clusters_full = model['clusters']
    disease_names = model['disease_names']
    Y_full = Y_full.detach().cpu().numpy() 
    print(type(Y_full))
    print(f"Model loaded:")
    print(f"- lambda_: {lambda_full.detach().cpu().numpy().shape}")
    print(f"- phi: {phi.detach().cpu().numpy().shape}")
    print(f"- psi: {psi.detach().cpu().numpy().shape}")
    print(f"- Y: {Y_full.shape if Y_full is not None else 'None'}")
    print(f"- G: {G_full.shape if G_full is not None else 'None'}")
    
    # Use known MI index instead of searching
    mi_idx = 112
    print(f"Using MI index: {mi_idx}")
    
    # Find good patients using same logic as app
    good_patients = []
    if Y_full is not None:
        good_patients = find_good_mi_overlap_patients(Y_full, psi, mi_idx, sig_idx=5, window=3, min_diseases=2)
    
    if not good_patients:
        # fallback: all MI patients
        for n in range(lambda_full.shape[0]):
            if np.any(Y_full[n, mi_idx, :] > 0.5):
                good_patients.append(n)
    
    if not good_patients:
        good_patients = list(range(lambda_full.shape[0]))  # fallback to all patients
    
    print(f"\nSelected {len(good_patients)} patients:")
    print(f"Indices: {good_patients[:20]}{'...' if len(good_patients) > 20 else ''}")
    
    # Slice to just good patients
    indices = np.array(sorted(good_patients))
    lambda_subset = lambda_full[indices, :, :]
    Y_subset = Y_full[indices, :, :] if Y_full is not None else None
    G_subset = G_full[indices, :] if G_full is not None else None
    clusters_subset = clusters_full[indices] if clusters_full.size > 0 else None
    
    # Build compact model
    compact_msd = {
        'lambda_': lambda_subset,
        'phi': phi,
        'psi': psi
    }
    if gamma is not None:
        compact_msd['gamma'] = gamma
    if kappa is not None:
        compact_msd['kappa'] = kappa
    
    compact = {
        'model_state_dict': compact_msd,
        'disease_names': disease_names,
        'clusters': clusters_subset,
        'G': G_subset,
        'Y': Y_subset,
        'meta': {
            'original_N': lambda_full.shape[0],
            'kept_indices': indices.tolist(),
            'K': lambda_full.shape[1],
            'T': lambda_full.shape[2],
            'selection_criteria': 'MI patients with multi-morbidity near sig5 peak'
        }
    }
    
    # Save
    torch.save(compact, out_path)
    print(f"\nSaved compact subset to: {out_path}")
    print(f"- Patients kept: {len(indices)}")
    print(f"- lambda_ shape: {lambda_subset.shape}")
    if Y_subset is not None:
        print(f"- Y shape: {Y_subset.shape}")
    if G_subset is not None:
        print(f"- G shape: {G_subset.shape}")
    
    # Save just the indices too
    indices_path = out_path.replace('.pt', '_indices.txt')
    with open(indices_path, 'w') as f:
        f.write(f"# Good patient indices from {model_path}\n")
        f.write(f"# Selection criteria: MI patients with multi-morbidity near sig5 peak\n")
        f.write(f"# Total patients: {len(indices)}\n")
        f.write(f"# Original model size: {lambda_full.shape[0]}\n\n")
        for i, idx in enumerate(indices):
            f.write(f"{i}\t{idx}\n")
    print(f"Saved indices mapping to: {indices_path}")

if __name__ == "__main__":
    model_path = '/Users/sarahurbut/Library/Cloudstorage/Dropbox-Personal/resultshighamp/results/output_0_10000/model.pt'
    out_path = '/Users/sarahurbut/Library/Cloudstorage/Dropbox-Personal/resultshighamp/results/output_0_10000/model_good_patients.pt'
    
    export_good_patients_subset(model_path, out_path)