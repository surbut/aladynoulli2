import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from utils import *
from clust_huge_amp_fixedPhi_vectorized import *
import sys
import os
import gc
import argparse
import pandas as pd


def parse_args():
    """Parse command-line arguments for AWS compatibility"""
    parser = argparse.ArgumentParser(description='Run Aladyn predictions for age 70, filtered to patients with max_censor > 70')
    parser.add_argument('--data_dir', type=str, 
                       default=os.getenv('DATA_DIR', './data_for_running/'),
                       help='Directory containing data files (default: from DATA_DIR env var or ./data_for_running/)')
    parser.add_argument('--output_dir', type=str,
                       default=os.getenv('OUTPUT_DIR', './output/'),
                       help='Directory to save output files (default: from OUTPUT_DIR env var or ./output/)')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for data subset (default: 0)')
    parser.add_argument('--end_index', type=int, default=25000,
                       help='End index for data subset (default: 25000)')
    parser.add_argument('--min_censor_age', type=float, default=70.0,
                       help='Minimum max_censor age to include (default: 70.0)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization lambda (default: 0.01)')
    parser.add_argument('--censor_info_path', type=str, default=None,
                       help='Path to censor_info.csv (default: data_dir/censor_info.csv or data_dir/../censor_info.csv)')
    return parser.parse_args()


def load_model_essentials(base_path):
    """
    Load all essential components
    """
    print(f"Loading components from {base_path}...")
    
    # Ensure path ends with /
    if not base_path.endswith('/'):
        base_path += '/'
    
    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt', map_location='cpu', weights_only=False)
    E = torch.load(base_path + 'E_matrix.pt', map_location='cpu', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', map_location='cpu', weights_only=False)
    
    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt', map_location='cpu', weights_only=False)
    
    print("Loaded all components successfully!")
    
    return Y, E, G, essentials


def filter_patients_by_censor_age(censor_df, min_censor_age=70.0):
    """
    Filter patients to only those with max_censor > min_censor_age
    
    Returns:
        mask: boolean array indicating which patients to keep
        n_filtered: number of patients after filtering
    """
    if 'max_censor' not in censor_df.columns:
        raise ValueError("censor_info.csv must have 'max_censor' column")
    
    mask = censor_df['max_censor'].values > min_censor_age
    n_filtered = mask.sum()
    
    print(f"Filtering patients with max_censor > {min_censor_age}")
    print(f"  Total patients: {len(censor_df)}")
    print(f"  Patients with max_censor > {min_censor_age}: {n_filtered} ({100*n_filtered/len(censor_df):.1f}%)")
    print(f"  Max censor range: {censor_df['max_censor'].min():.1f} - {censor_df['max_censor'].max():.1f}")
    
    return mask, n_filtered


def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_index = args.start_index
    end_index = args.end_index
    
    # Load and initialize model:
    Y, E, G, essentials = load_model_essentials(args.data_dir)
    
    # Load censor_info.csv to filter patients
    if args.censor_info_path is None:
        # Try common locations
        censor_paths = [
            os.path.join(args.data_dir, 'censor_info.csv'),
            os.path.join(args.data_dir, '../censor_info.csv'),
            os.path.join(args.data_dir, '../../censor_info.csv'),
        ]
        censor_path = None
        for path in censor_paths:
            if os.path.exists(path):
                censor_path = path
                break
        if censor_path is None:
            raise FileNotFoundError(f"Could not find censor_info.csv. Tried: {censor_paths}")
    else:
        censor_path = args.censor_info_path
    
    print(f"\nLoading censor info from: {censor_path}")
    censor_df = pd.read_csv(censor_path)
    print(f"Loaded censor info with {len(censor_df)} patients")
    
    # Filter to patients with sufficient follow-up
    censor_mask, n_filtered = filter_patients_by_censor_age(censor_df, args.min_censor_age)
    
    # Apply the filter to the data
    # Note: We need to filter before subsetting, so we need to work with the full dataset first
    print(f"\nFiltering Y, E, G matrices to patients with max_censor > {args.min_censor_age}...")
    Y_filtered = Y[censor_mask]
    E_filtered = E[censor_mask]
    G_filtered = G[censor_mask]
    censor_df_filtered = censor_df[censor_mask].reset_index(drop=True)
    
    print(f"Filtered data shapes: Y={Y_filtered.shape}, E={E_filtered.shape}, G={G_filtered.shape}")
    
    # Get original indices (from full dataset) that passed the filter
    # This maps filtered dataset indices back to original dataset indices
    original_indices_full = np.where(censor_mask)[0]  # All filtered patients' original indices
    
    # Now subset the filtered data
    actual_end = min(end_index, Y_filtered.shape[0])
    Y_subset, E_subset, G_subset, indices = subset_data(
        Y_filtered, E_filtered, G_filtered, 
        start_index=start_index, 
        end_index=actual_end
    )
    
    # Get original indices for this specific batch
    original_indices_batch = original_indices_full[start_index:actual_end]  # This batch's original indices
    
    # Also subset the censor dataframe
    censor_df_subset = censor_df_filtered.iloc[start_index:actual_end].reset_index(drop=True)
    
    print(f"\nAfter subsetting [{start_index}:{actual_end}]:")
    print(f"  Y_subset shape: {Y_subset.shape}")
    print(f"  E_subset shape: {E_subset.shape}")
    print(f"  G_subset shape: {G_subset.shape}")
    print(f"  Original indices range: {original_indices_batch[0]} to {original_indices_batch[-1]}")
    
    del Y, E, G, Y_filtered, E_filtered, G_filtered
    
    # Load references (signatures only, no healthy)
    refs = torch.load(os.path.join(args.data_dir, 'reference_trajectories.pt'), map_location='cpu', weights_only=False)
    signature_refs = refs['signature_refs']
    
    # Load the CSV file for sex and PCs
    csv_path = os.path.join(args.data_dir, 'baselinagefamh_withpcs.csv')
    if not os.path.exists(csv_path):
        # Try alternative path
        csv_path = os.path.join(args.data_dir, '../baselinagefamh_withpcs.csv')
    
    fh_processed = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(fh_processed)} rows")
    
    # Filter the CSV to match our filtered patients
    # We need to align by index since censor_df was filtered
    fh_filtered = fh_processed.iloc[original_indices_full].reset_index(drop=True)
    fh_subset = fh_filtered.iloc[start_index:actual_end].reset_index(drop=True)
    
    sex = fh_subset['sex'].values
    G_with_sex = np.column_stack([G_subset, sex])
    
    # Add PCs
    pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
                  'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
    pcs = fh_subset[pc_columns].values
    G_with_sex = np.column_stack([G_with_sex, pcs])
    print(f"G_with_sex shape: {G_with_sex.shape} (should be [N, 36 PRS + 1 sex + 10 PCs = 47])")
    
    import cProfile
    import pstats
    from pstats import SortKey
    
    # Path to your total fit model (corrected E version)
    total_fit_path = os.path.join(args.data_dir, 'master_for_fitting_pooled_correctedE.pt')
    total_checkpoint = torch.load(total_fit_path, map_location='cpu', weights_only=False)
    phi_total = total_checkpoint['model_state_dict']['phi'].cpu().numpy()  # shape: (K, D, T)
    psi_total = total_checkpoint['model_state_dict']['psi'].cpu().numpy()  # shape: (K, D, T)
    
    # Fixed starting age is 40 (time 10), age offset 0 = predict at age 40
    # We make predictions using information up to age 40 (time 10)
    # Then evaluate AUC for events occurring from age 40-70 (time 10 to time 40)
    fixed_starting_age = 40
    age_offset = 0  # Predict at age 40 (time 10), evaluate 30-year window to age 70
    current_age = fixed_starting_age + age_offset  # Should be 40
    
    print(f"\n=== Predicting for age offset {age_offset} years (age {current_age}) ===")
    print(f"Using information up to age {current_age} (time {current_age - 30})")
    print(f"Evaluating 30-year predictions (age {current_age} to {current_age + 30})")
    print(f"Using only patients with max_censor > {args.min_censor_age}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    model = AladynSurvivalFixedPhi(
        N=Y_subset.shape[0],
        D=Y_subset.shape[1],
        T=Y_subset.shape[2],
        K=20,
        P=G_with_sex.shape[1],
        G=G_with_sex,
        Y=Y_subset,
        R=0,
        W=0.0001,
        prevalence_t=essentials['prevalence_t'],
        init_sd_scaler=1e-1,
        genetic_scale=1,
        pretrained_phi=phi_total,
        pretrained_psi=psi_total,
        signature_references=signature_refs,
        healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    
    if np.allclose(model.phi.cpu().numpy(), phi_total):
        print("phi matches phi_total!")
    else:
        print("phi does NOT match phi_total!")
    
    if np.allclose(model.psi.cpu().numpy(), psi_total):
        print("psi matches psi_total!")
    else:
        print("psi does NOT match psi_total!")
    
    # Reinitialize with psi_total for consistent gamma initialization
    print("Reinitializing gamma with psi_total for consistency...")
    model.initialize_params(init_psi=torch.tensor(psi_total, dtype=torch.float32))
    print("✓ Gamma reinitialized with psi_total")
    
    # Create age-specific event times
    E_age_specific = E_subset.clone()
    
    # Initialize tracking variables
    total_times_changed = 0
    max_cap_applied = 0
    min_cap_applied = float('inf')
    
    for patient_idx, row in enumerate(censor_df_subset.itertuples()):
        if patient_idx >= E_age_specific.shape[0]:
            break
            
        # Current age = fixed starting age (40) + age_offset = 40
        current_age_check = fixed_starting_age + age_offset
        
        # Time since age 30 for this current age
        time_since_30 = max(0, current_age_check - 30)  # Should be 10 (age 40 - age 30)
        
        max_cap_applied = max(max_cap_applied, time_since_30)
        min_cap_applied = min(min_cap_applied, time_since_30)
        
        # Store original times for this patient
        original_times = E_age_specific[patient_idx, :].clone()
        
        # Cap event times to current age
        E_age_specific[patient_idx, :] = torch.minimum(
            E_age_specific[patient_idx, :],
            torch.full_like(E_age_specific[patient_idx, :], time_since_30)
        )
        
        times_changed = torch.sum(E_age_specific[patient_idx, :] != original_times).item()
        total_times_changed += times_changed
    
    # Print censoring verification
    print(f"Censoring verification for age offset {age_offset} (age {current_age}):")
    print(f"  Total event times changed: {total_times_changed}")
    print(f"  Max cap applied: {max_cap_applied:.1f}")
    print(f"  Min cap applied: {min_cap_applied:.1f}")
    
    # Check a few specific patients
    test_patients = [0, 1, min(100, len(censor_df_subset)-1)]
    for test_idx in test_patients:
        if test_idx < len(censor_df_subset):
            row = censor_df_subset.iloc[test_idx]
            max_censor_age = row.max_censor
            current_age_check = fixed_starting_age + age_offset
            expected_cap = max(0, current_age_check - 30)
            
            # Check max value in this patient's event times
            max_time = torch.max(E_age_specific[test_idx, :]).item()
            
            print(f"  Patient {test_idx}: max_censor={max_censor_age:.1f}, fixed_age={current_age_check:.0f}, "
                  f"cap={expected_cap:.1f}, max_event_time={max_time:.1f}")
            
            # Verify cap was applied correctly
            if max_time > expected_cap + 0.01:  # Small tolerance
                print(f"    WARNING: Max time {max_time:.1f} exceeds cap {expected_cap:.1f}!")
    
    # Train model for this specific age
    print(f"Training model for age offset {age_offset} (age {current_age})...")
    profiler = cProfile.Profile()
    profiler.enable()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    history_new = model.fit(
        E_age_specific,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg
    )
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    
    # Get predictions for this age
    with torch.no_grad():
        pi, _, _ = model.forward()
        
        # Save age-specific predictions
        pi_filename = os.path.join(args.output_dir, 
            f'pi_fixedphi_age_{current_age}_offset_{age_offset}_filtered_censor{args.min_censor_age}_batch_{start_index}_{actual_end}.pt')
        torch.save(pi, pi_filename)
        print(f"Saved predictions to {pi_filename}")
    
    # Save model checkpoint
    model_filename = os.path.join(args.output_dir,
        f'model_fixedphi_age_{current_age}_offset_{age_offset}_filtered_censor{args.min_censor_age}_batch_{start_index}_{actual_end}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'E': E_age_specific,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'age_offset': age_offset,
        'current_age': current_age,
        'fixed_starting_age': fixed_starting_age,
        'start_index': start_index,
        'end_index': actual_end,
        'min_censor_age': args.min_censor_age,
        'n_patients_filtered': n_filtered,
        'n_patients_in_batch': Y_subset.shape[0],
        'original_indices': original_indices_batch,  # Original patient indices from full dataset
        'filtered_indices': np.arange(start_index, actual_end)  # Indices within filtered dataset
    }, model_filename)
    print(f"Saved model to {model_filename}")
    
    print("\n=== Age offset 0 (age 40) prediction completed! ===")
    print(f"  Predictions made using data up to age 40 (time 10)")
    print(f"  Ready for 30-year evaluation (age 40-70, time 10-40)")
    print(f"  Total patients in batch: {Y_subset.shape[0]}")
    print(f"  All patients have max_censor > {args.min_censor_age}")


if __name__ == '__main__':
    main()

