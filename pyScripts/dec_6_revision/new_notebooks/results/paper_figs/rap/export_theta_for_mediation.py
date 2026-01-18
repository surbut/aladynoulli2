#!/usr/bin/env python3
"""
Export patient-level signature loadings (theta) to CSV for R mediation analysis.

Creates a CSV file with:
- Patient ID (eid)
- Mean signature loadings WITH genetics (actual theta) - Sig0, Sig1, ...
- Mean signature loadings WITHOUT genetics (counterfactual theta) - Sig0_noG, Sig1_noG, ...

This allows R to:
1. Match patients by ID
2. Correlate gene burden with signature loadings (with and without G)
3. Test Gene → Signature → Disease mediation
   - Compare gene-sig correlation with vs without G effects in sig
   - Test if gene-disease effect is mediated through signature
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from scipy.special import softmax

def load_theta_from_file(theta_path):
    """Load theta from a precomputed file."""
    print(f"Loading theta from: {theta_path}")
    theta_full = torch.load(theta_path, map_location='cpu', weights_only=False)
    
    # Handle different file structures
    if isinstance(theta_full, dict):
        if 'theta' in theta_full:
            theta = theta_full['theta']
        elif 'thetas' in theta_full:
            theta = theta_full['thetas']
        elif 'lambda_' in theta_full:
            lambda_ = theta_full['lambda_']
            # Convert lambda to theta via softmax
            if torch.is_tensor(lambda_):
                theta = torch.softmax(lambda_, dim=1)
            else:
                theta = softmax(lambda_, axis=1)
        else:
            # Try first tensor value
            theta = list(theta_full.values())[0]
            if torch.is_tensor(theta):
                if theta.dim() == 3:
                    theta = torch.softmax(theta, dim=1)
    else:
        theta = theta_full
    
    # Convert to numpy
    if torch.is_tensor(theta):
        theta = theta.numpy()
    
    print(f"  Theta shape: {theta.shape}")
    return theta

def sort_by_index(dir_name):
    """Extract start index from directory/file name for sorting.
    
    Example: 'output_10000_20000' -> 10000
             'enrollment_model_W0.0001_batch_10000_20000.pt' -> 10000
    """
    import re
    # Try pattern: output_START_END or batch_START_END
    match = re.search(r'(?:output_|batch_)(\d+)_\d+', dir_name)
    if match:
        return int(match.group(1))
    return 0

def load_theta_from_batches(batch_dir, pattern="enrollment_model_W0.0001_batch_*.pt", 
                            include_noG=False, noG_batch_dir=None):
    """Load and pool theta from multiple batch files.
    
    If include_noG=True, also computes counterfactual theta without genetic effects.
    Requires G and gamma to be in batch files.
    """
    batch_files = sorted(glob.glob(str(Path(batch_dir) / pattern)))
    print(f"Found {len(batch_files)} batch files in {batch_dir}")
    
    all_lambda = []
    all_G = []
    all_gamma = None
    all_genetic_scale = None
    all_pids = []
    
    for batch_file in batch_files:
        print(f"  Loading {Path(batch_file).name}...")
        try:
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)
            
            # Extract lambda
            if 'model_state_dict' in batch_data:
                lambda_batch = batch_data['model_state_dict']['lambda_']
                gamma_batch = batch_data['model_state_dict'].get('gamma', None)
            elif 'lambda_' in batch_data:
                lambda_batch = batch_data['lambda_']
                gamma_batch = batch_data.get('gamma', None)
            else:
                print(f"    ⚠️  No lambda_ found in {batch_file}")
                continue
            
            # Convert to numpy if needed
            if torch.is_tensor(lambda_batch):
                lambda_batch = lambda_batch.numpy()
            if gamma_batch is not None and torch.is_tensor(gamma_batch):
                gamma_batch = gamma_batch.numpy()
            
            all_lambda.append(lambda_batch)
            
            # Extract G if needed for noG computation
            if include_noG:
                if 'G' in batch_data:
                    G_batch = batch_data['G']
                elif 'model_state_dict' in batch_data and 'G' in batch_data['model_state_dict']:
                    G_batch = batch_data['model_state_dict']['G']
                else:
                    print(f"    ⚠️  No G found in {batch_file}, skipping noG computation")
                    include_noG = False
                    continue
                
                if torch.is_tensor(G_batch):
                    G_batch = G_batch.numpy()
                all_G.append(G_batch)
                
                # Get gamma and genetic_scale (should be same across batches)
                if all_gamma is None and gamma_batch is not None:
                    all_gamma = gamma_batch
                if all_genetic_scale is None:
                    # Try to get genetic_scale from model or use default
                    if 'genetic_scale' in batch_data:
                        all_genetic_scale = batch_data['genetic_scale']
                    elif 'model_state_dict' in batch_data and 'genetic_scale' in batch_data['model_state_dict']:
                        all_genetic_scale = batch_data['model_state_dict']['genetic_scale']
                    else:
                        all_genetic_scale = 1.0  # Default
                        print(f"    Using default genetic_scale=1.0")
            
            # Try to extract patient IDs if available
            if 'pids' in batch_data:
                pids_batch = batch_data['pids']
                if torch.is_tensor(pids_batch):
                    pids_batch = pids_batch.numpy()
                all_pids.extend(pids_batch.tolist() if isinstance(pids_batch, np.ndarray) else pids_batch)
            
        except Exception as e:
            print(f"    ⚠️  Error loading {batch_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_lambda) == 0:
        raise ValueError("No lambda data loaded from batch files!")
    
    # Concatenate lambda
    lambda_all = np.concatenate(all_lambda, axis=0)
    print(f"  Total lambda shape: {lambda_all.shape}")
    
    # Convert to theta via softmax (WITH genetic effects)
    theta = softmax(lambda_all, axis=1)  # Softmax over signatures (axis=1)
    print(f"  Theta (with G) shape: {theta.shape}")
    
    # Compute counterfactual theta WITHOUT genetic effects
    theta_noG = None
    if include_noG and all_G and all_gamma is not None:
        print(f"  Computing counterfactual theta WITHOUT genetic effects...")
        G_all = np.concatenate(all_G, axis=0)
        print(f"  G shape: {G_all.shape}, gamma shape: {all_gamma.shape}")
        
        # Compute genetic effects: genetic_scale * (G @ gamma)
        # G: [N, P], gamma: [P, K], result: [N, K]
        genetic_effects = all_genetic_scale * (G_all @ all_gamma)  # [N, K]
        
        # Expand genetic_effects to match lambda time dimension: [N, K] -> [N, K, T]
        N, K = genetic_effects.shape
        _, _, T = lambda_all.shape
        genetic_effects_expanded = genetic_effects[:, :, np.newaxis]  # [N, K, 1]
        genetic_effects_expanded = np.repeat(genetic_effects_expanded, T, axis=2)  # [N, K, T]
        
        # Remove genetic effects from lambda
        lambda_noG = lambda_all - genetic_effects_expanded
        
        # Convert to theta via softmax
        theta_noG = softmax(lambda_noG, axis=1)  # [N, K, T]
        print(f"  Theta (no G) shape: {theta_noG.shape}")
    
    return theta, theta_noG, all_pids if all_pids else None

def load_patient_ids(processed_ids_path):
    """Load patient IDs from processed_ids.csv."""
    print(f"Loading patient IDs from: {processed_ids_path}")
    pid_df = pd.read_csv(processed_ids_path)
    if 'eid' in pid_df.columns:
        pids = pid_df['eid'].values
    else:
        pids = pid_df.iloc[:, 0].values
    print(f"  Loaded {len(pids)} patient IDs")
    return pids

def export_theta_to_csv(theta, output_path, pids=None, average_over_time=True, 
                       theta_path=None, batch_dir=None, processed_ids_path=None,
                       theta_noG=None):
    """
    Export theta to CSV for R.
    
    Args:
        theta: numpy array of shape (N, K, T) or (N, K) - theta WITH genetic effects
        output_path: Path to save CSV
        pids: Patient IDs (optional, will try to load if not provided)
        average_over_time: If True, average theta over time dimension
        theta_noG: Optional numpy array - theta WITHOUT genetic effects (counterfactual)
    """
    # Handle time dimension for theta (with G)
    if theta.ndim == 3:
        N, K, T = theta.shape
        if average_over_time:
            print(f"Averaging theta (with G) over time (T={T} timepoints)...")
            theta_mean = theta.mean(axis=2)  # (N, K)
        else:
            # Keep all timepoints - will create columns like Sig0_t0, Sig0_t1, etc.
            theta_mean = theta.reshape(N, K * T)  # (N, K*T)
            T_flat = T
    elif theta.ndim == 2:
        theta_mean = theta
        N, K = theta.shape
        T_flat = 1
    else:
        raise ValueError(f"Unexpected theta shape: {theta.shape}")
    
    # Handle theta_noG if provided
    theta_noG_mean = None
    if theta_noG is not None:
        if theta_noG.ndim == 3:
            N_noG, K_noG, T_noG = theta_noG.shape
            if N_noG != N or K_noG != K:
                raise ValueError(f"Theta_noG shape {theta_noG.shape} doesn't match theta {theta.shape}")
            if average_over_time:
                print(f"Averaging theta (no G) over time (T={T_noG} timepoints)...")
                theta_noG_mean = theta_noG.mean(axis=2)  # (N, K)
            else:
                theta_noG_mean = theta_noG.reshape(N, K * T_noG)  # (N, K*T)
        elif theta_noG.ndim == 2:
            theta_noG_mean = theta_noG
        else:
            raise ValueError(f"Unexpected theta_noG shape: {theta_noG.shape}")
    
    # Load or create patient IDs
    if pids is None:
        # Try to load from file
        if processed_ids_path and Path(processed_ids_path).exists():
            pids = load_patient_ids(processed_ids_path)
        elif theta_path:
            # Try common locations
            possible_paths = [
                Path(theta_path).parent / 'processed_ids.csv',
                Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/processed_ids.csv'),
            ]
            for path in possible_paths:
                if path.exists():
                    pids = load_patient_ids(path)
                    break
        
        # If still no PIDs, create sequential IDs
        if pids is None:
            print("  ⚠️  No patient IDs found, using sequential indices")
            pids = np.arange(N)
    
    # Ensure pids match theta dimensions
    if len(pids) != N:
        print(f"  ⚠️  PID length ({len(pids)}) doesn't match theta N ({N})")
        print(f"  Using first {N} PIDs")
        pids = pids[:N]
    
    # Create DataFrame columns
    if average_over_time:
        # Columns: eid, Sig0, Sig1, ..., Sig0_noG, Sig1_noG, ...
        columns = ['eid'] + [f'Sig{k}' for k in range(K)]
        if theta_noG_mean is not None:
            columns.extend([f'Sig{k}_noG' for k in range(K)])
    else:
        # Columns: eid, Sig0_t0, Sig0_t1, ..., Sig0_noG_t0, ...
        columns = ['eid'] + [f'Sig{k}_t{t}' for k in range(K) for t in range(T_flat)]
        if theta_noG_mean is not None:
            columns.extend([f'Sig{k}_noG_t{t}' for k in range(K) for t in range(T_flat)])
    
    # Combine data
    data_list = [pids, theta_mean]
    if theta_noG_mean is not None:
        data_list.append(theta_noG_mean)
    data = np.column_stack(data_list)
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved theta to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:10])}... (showing first 10)")
    if theta_noG_mean is not None:
        print(f"  ✓ Includes counterfactual theta (no G) columns: Sig*_noG")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export patient-level theta for R mediation analysis')
    parser.add_argument('--theta-file', type=str, 
                       help='Path to precomputed theta file (.pt)')
    parser.add_argument('--batch-dir', type=str,
                       help='Directory containing batch files (alternative to --theta-file)')
    parser.add_argument('--batch-pattern', type=str, default='enrollment_model_W0.0001_batch_*.pt',
                       help='Pattern for batch files (default: enrollment_model_W0.0001_batch_*.pt)')
    parser.add_argument('--processed-ids', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/processed_ids.csv',
                       help='Path to processed_ids.csv with patient IDs')
    parser.add_argument('--output', type=str, default='patient_signature_loadings.csv',
                       help='Output CSV path')
    parser.add_argument('--average-time', action='store_true', default=True,
                       help='Average theta over time (default: True)')
    parser.add_argument('--keep-timepoints', action='store_false', dest='average_time',
                       help='Keep all timepoints (alternative to --average-time)')
    parser.add_argument('--include-noG', action='store_true',
                       help='Also compute and export counterfactual theta WITHOUT genetic effects (requires G and gamma in batch files)')
    
    args = parser.parse_args()
    
    # Load theta
    theta_noG = None
    if args.theta_file:
        theta = load_theta_from_file(args.theta_file)
        pids = None
        if args.include_noG:
            print("  ⚠️  --include-noG requires --batch-dir (need G and gamma data)")
            print("  Proceeding without noG computation")
    elif args.batch_dir:
        theta, theta_noG, pids = load_theta_from_batches(
            args.batch_dir, 
            args.batch_pattern, 
            include_noG=args.include_noG
        )
    else:
        # Try default locations
        default_theta = '/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective_correctE.pt'
        if Path(default_theta).exists():
            print(f"Using default theta file: {default_theta}")
            theta = load_theta_from_file(default_theta)
            pids = None
            if args.include_noG:
                print("  ⚠️  --include-noG requires --batch-dir (need G and gamma data)")
                print("  Proceeding without noG computation")
        else:
            raise ValueError("Must specify --theta-file or --batch-dir, or have default theta file available")
    
    # Export to CSV
    export_theta_to_csv(
        theta=theta,
        output_path=args.output,
        pids=pids,
        average_over_time=args.average_time,
        processed_ids_path=args.processed_ids,
        theta_noG=theta_noG
    )
    
    print("\n✓ Done! You can now use this CSV in R for mediation analysis.")
    print("\nUsage example:")
    print("  # Export with genetic effects only:")
    print("  python export_theta_for_mediation.py --batch-dir /path/to/batches --output theta_withG.csv")
    print("\n  # Export with BOTH with-G and without-G (for mediation analysis):")
    print("  python export_theta_for_mediation.py --batch-dir /path/to/batches --include-noG --output theta_for_mediation.csv")

