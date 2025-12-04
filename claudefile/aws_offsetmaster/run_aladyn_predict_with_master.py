#!/usr/bin/env python
"""
Prediction script for Aladyn model using enrollment data with fixed phi/psi
USES MASTER CHECKPOINT FILES (pooled phi + initial_psi)

This script:
- Uses FIXED phi from pooled batches (master checkpoint) and initial_psi
- Only estimates lambda (genetic effects) per batch
- Processes data in batches (default 10k samples)
- Saves predictions (pi) for each batch
- Concatenates all batches at the end

Master checkpoints should be created using create_master_checkpoints.py:
- master_for_fitting_pooled_all_data.pt (retrospective pooled phi)
- master_for_fitting_pooled_enrollment_data.pt (enrollment pooled phi)

Usage:
    AWS:  python scripts/run_aladyn_predict_with_master.py --trained_model_path data_for_running/master.pt
    
    Example AWS commands:
    # Fixed phi from pooled enrollment data
    nohup python scripts/run_aladyn_predict_with_master.py \
        --trained_model_path data_for_running/master_for_fitting_pooled_enrollment_data.pt \
        --data_dir data_for_running/ \
        --output_dir output/enrollment_pooled/ \
        --covariates_path data_for_running/baselinagefamh_withpcs.csv \
        --max_batches 10 \
        > predict_enrollment_pooled.log 2>&1 &
    
    # Fixed phi from pooled retrospective (all data)
    nohup python scripts/run_aladyn_predict_with_master.py \
        --trained_model_path data_for_running/master_for_fitting_pooled_all_data.pt \
        --data_dir data_for_running/ \
        --output_dir output/retrospective_pooled/ \
        --covariates_path data_for_running/baselinagefamh_withpcs.csv \
        --max_batches 10 \
        > predict_retrospective_pooled.log 2>&1 &
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
import gc
import cProfile
import pstats
from pstats import SortKey
from pathlib import Path

# Import from local directory (AWS setup)
# On AWS, clust_huge_amp_fixedPhi.py and utils.py are in the scripts directory
# Add scripts directory to path so imports work
script_dir = Path(__file__).parent
if 'scripts' in str(script_dir):
    # If script is in scripts/, add scripts/ to path
    sys.path.insert(0, str(script_dir))
else:
    # If script is run from parent directory, add scripts/ to path
    scripts_dir = Path(__file__).parent.parent / 'scripts'
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

from clust_huge_amp_fixedPhi_vectorized import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def subset_data(Y, E, G, start_index, end_index):
    """Subset data based on indices."""
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices


def load_model_essentials(base_path):
    """Load all essential components"""
    print(f"Loading components from {base_path}...")
    
    # Ensure path ends with / for os.path.join compatibility
    if not base_path.endswith('/') and not base_path.endswith(os.sep):
        base_path = base_path + os.sep

    # Load large matrices - NOTE: Using E_enrollment_full.pt for enrollment data!
    Y = torch.load(os.path.join(base_path, 'Y_tensor.pt'), map_location='cpu', weights_only=False)
    E = torch.load(os.path.join(base_path, 'E_enrollment_full.pt'), map_location='cpu', weights_only=False)  # Enrollment data
    G = torch.load(os.path.join(base_path, 'G_matrix.pt'), map_location='cpu', weights_only=False)

    # Load other components
    essentials = torch.load(os.path.join(base_path, 'model_essentials.pt'), map_location='cpu', weights_only=False)

    print("Loaded all components successfully!")

    return Y, E, G, essentials


def load_covariates_data(csv_path):
    """Load and process covariates from CSV file"""
    print("Loading covariates data...")
    fh_processed = pd.read_csv(csv_path)
    print(f"Loaded {len(fh_processed)} samples from covariates file")
    return fh_processed


def generate_batches(total_size, batch_size=25000):
    """Generate batch indices"""
    batches = []
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batches.append((start, end))
    return batches


def main():
    parser = argparse.ArgumentParser(description='Run Aladyn predictions with fixed phi/psi')
    parser.add_argument('--trained_model_path', type=str, required=True,
                       help='Path to trained model with phi and psi')
    parser.add_argument('--batch_size', type=int, default=25000,
                       help='Batch size for processing')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs per batch')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization parameter')

    # Path configuration - AWS compatible with environment variable defaults
    parser.add_argument('--data_dir', type=str,
                       default=os.getenv('DATA_DIR', './data_for_running/'),
                       help='Directory containing input data (default: from DATA_DIR env var or ./data_for_running/)')
    parser.add_argument('--output_dir', type=str,
                       default=os.getenv('OUTPUT_DIR', './output/'),
                       help='Output directory for predictions (default: from OUTPUT_DIR env var or ./output/)')
    parser.add_argument('--covariates_path', type=str,
                       default=os.getenv('COVARIATES_PATH', './data_for_running/baselinagefamh_withpcs.csv'),
                       help='Path to covariates CSV file (default: from COVARIATES_PATH env var or ./data_for_running/baselinagefamh_withpcs.csv)')
    parser.add_argument('--include_pcs', type=bool, default=True,
                       help='Include principal components in the model')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to process (None = all batches)')
    parser.add_argument('--start_batch', type=int, default=0,
                       help='Batch index to start from (0 = start from beginning, 10 = skip first 10 batches)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Aladyn Prediction Script - Fixed Phi/Psi Mode")
    print(f"{'='*80}")
    print(f"Trained model: {args.trained_model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")

    # Load data
    Y, E, G, essentials = load_model_essentials(args.data_dir)
    fh_processed = load_covariates_data(args.covariates_path)

    # Load references
    print("Loading reference trajectories...")
    refs = torch.load(os.path.join(args.data_dir, 'reference_trajectories.pt'), map_location='cpu', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs  # Free memory - we only need signature_refs
    gc.collect()

    # Load initial_psi for consistent gamma initialization across batches
    print("Loading initial_psi for consistent initialization...")
    initial_psi = torch.load(os.path.join(args.data_dir, 'initial_psi_400k.pt'), map_location='cpu', weights_only=False)
    if torch.is_tensor(initial_psi):
        initial_psi = initial_psi.cpu().numpy()
    print(f"Loaded initial_psi shape: {initial_psi.shape}")

    # Load master checkpoint to get pooled phi and psi
    print(f"Loading master checkpoint from {args.trained_model_path}...")
    master_checkpoint = torch.load(args.trained_model_path, map_location='cpu', weights_only=False)
    
    # Extract phi and psi from master checkpoint
    if 'model_state_dict' in master_checkpoint:
        phi_total = master_checkpoint['model_state_dict']['phi']
        psi_total = master_checkpoint['model_state_dict']['psi']
    else:
        # Fallback: check root level
        phi_total = master_checkpoint['phi']
        psi_total = master_checkpoint['psi']
    
    # Print checkpoint info if available (before deleting)
    if 'description' in master_checkpoint:
        print(f"Master checkpoint description: {master_checkpoint['description']}")
    
    # Convert to numpy if tensor
    if torch.is_tensor(phi_total):
        phi_total = phi_total.cpu().numpy()
    if torch.is_tensor(psi_total):
        psi_total = psi_total.cpu().numpy()
    
    # Free memory - we only need phi_total and psi_total
    del master_checkpoint
    gc.collect()
    
    print(f"Loaded pooled phi shape: {phi_total.shape}")
    print(f"Loaded psi_total shape: {psi_total.shape}")
    
        # Generate batches
    total_samples = Y.shape[0]
    batches = generate_batches(total_samples, args.batch_size)
    
    # Skip first N batches if start_batch is specified
    if args.start_batch > 0:
        if args.start_batch >= len(batches):
            print(f"✗ ERROR: start_batch ({args.start_batch}) >= total batches ({len(batches)})")
            return
        batches = batches[args.start_batch:]
        print(f"\n⚠️  Skipping first {args.start_batch} batches, starting from batch {args.start_batch}")
    
    # Limit to max_batches if specified
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
        print(f"⚠️  Limiting to {args.max_batches} batches")

    print(f"\nWill process {len(batches)} batches of {args.batch_size} samples")
    print(f"Total samples: {total_samples}")
    if batches:
        print(f"Processing samples: {batches[0][0]} to {batches[-1][1]}")
    print()

    # Process each batch
    successful_batches = []

    for batch_idx, (start, stop) in enumerate(batches):
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx+1}/{len(batches)}: Processing samples {start} to {stop}")
        print(f"{'='*80}")

        try:
            # Set random seeds
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.backends.cudnn.deterministic = True

            # Subset the data
            print(f"Subsetting data...")
            Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G,
                                                              start_index=start,
                                                              end_index=stop)

            # Get demographics and add sex
            pce_df_subset = fh_processed.iloc[start:stop].reset_index(drop=True)
            sex = pce_df_subset['sex'].values
            G_with_sex = np.column_stack([G_batch, sex])
            if args.include_pcs:
                pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
                'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
                pcs = pce_df_subset[pc_columns].values
                G_with_sex = np.column_stack([G_batch, sex, pcs])

            print(f"Data shapes: Y={Y_batch.shape}, E={E_batch.shape}, G={G_with_sex.shape}")

            # Initialize model with FIXED phi and psi
            print(f"Initializing model with fixed phi/psi...")
            model = AladynSurvivalFixedPhi(
                N=Y_batch.shape[0],
                D=Y_batch.shape[1],
                T=Y_batch.shape[2],
                K=20,
                P=G_with_sex.shape[1],
                G=G_with_sex,
                Y=Y_batch,
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

            # Verify phi and psi are fixed
            if np.allclose(model.phi.cpu().numpy(), phi_total):
                print("✓ phi matches phi_total!")
            else:
                print("✗ WARNING: phi does NOT match phi_total!")

            if np.allclose(model.psi.cpu().numpy(), psi_total):
                print("✓ psi matches psi_total!")
            else:
                print("✗ WARNING: psi does NOT match psi_total!")
            
            # Reinitialize with psi_total for consistent gamma initialization across batches
            # Use psi_total (from master checkpoint) to determine which diseases belong to which signature
            # This is more accurate than initial_psi since psi_total was learned from enrollment data
            print("Reinitializing gamma with psi_total for consistency...")
            model.initialize_params(init_psi=torch.tensor(psi_total, dtype=torch.float32))
            print("✓ Gamma reinitialized with psi_total")

            # Train model (only lambda is being estimated)
            print(f"Training model (estimating lambda only)...")
            profiler = cProfile.Profile()
            profiler.enable()

            history = model.fit(
                E_batch,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                lambda_reg=args.lambda_reg
            )

            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(10)

            # Generate and save predictions
            print(f"Generating predictions...")
            with torch.no_grad():
                pi, _, _ = model.forward()

                # Save predictions
                pi_filename = os.path.join(args.output_dir,
                                          f"pi_enroll_fixedphi_sex_{start}_{stop}.pt")
                torch.save(pi, pi_filename)
                print(f"✓ Saved predictions to {pi_filename}")

                # Save model state
                model_filename = os.path.join(args.output_dir,
                                            f"model_enroll_fixedphi_sex_{start}_{stop}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'E': E_batch,
                    'prevalence_t': model.prevalence_t,
                    'logit_prevalence_t': model.logit_prev_t,
                    'start_index': start,
                    'end_index': stop,
                }, model_filename)
                print(f"✓ Saved model to {model_filename}")

            successful_batches.append((start, stop))

            # Clean up memory
            print(f"Cleaning up memory...")
            del pi, model, Y_batch, E_batch, G_batch, G_with_sex, pce_df_subset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"✓ Batch {batch_idx+1}/{len(batches)} complete!")

        except Exception as e:
            print(f"✗ ERROR in batch {start}-{stop}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"ALL BATCHES COMPLETE!")
    print(f"{'='*80}")

    # Concatenate all predictions into one file
    print(f"\nConcatenating all predictions into single file...")
    pi_batches = []

    for start, stop in successful_batches:
        pi_filename = os.path.join(args.output_dir, f"pi_enroll_fixedphi_sex_{start}_{stop}.pt")
        try:
            pi_batch = torch.load(pi_filename, weights_only=False)
            pi_batches.append(pi_batch)
            print(f"✓ Loaded {pi_filename}, shape: {pi_batch.shape}")
        except Exception as e:
            print(f"✗ Could not load {pi_filename}: {e}")

    if pi_batches:
        print(f"\nConcatenating {len(pi_batches)} batches...")
        pi_full = torch.cat(pi_batches, dim=0)
        print(f"Final shape: {pi_full.shape}")

        # Save combined file
        full_filename = os.path.join(args.output_dir, "pi_enroll_fixedphi_sex_FULL.pt")
        torch.save(pi_full, full_filename)
        print(f"✓ Saved combined predictions to {full_filename}")

        # Save batch info
        batch_info = {
            'batches': successful_batches,
            'total_patients': pi_full.shape[0],
            'n_diseases': pi_full.shape[1],
            'n_timepoints': pi_full.shape[2],
            'args': vars(args),
        }
        info_filename = os.path.join(args.output_dir, "batch_info.pt")
        torch.save(batch_info, info_filename)
        print(f"✓ Saved batch info to {info_filename}")
    else:
        print("✗ No successful batches to concatenate!")

    print(f"\n{'='*80}")
    print(f"DONE! Ready for washout analysis.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
