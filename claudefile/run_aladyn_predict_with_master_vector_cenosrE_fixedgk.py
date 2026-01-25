#!/usr/bin/env python
"""
Prediction script for Aladyn model using enrollment data with fixed phi/psi, fixed gamma, and fixed kappa
USES MASTER CHECKPOINT FILES (pooled phi + initial_psi) + POOLED GAMMA + POOLED KAPPA

This script:
- Uses FIXED phi from pooled batches (master checkpoint) and initial_psi
- Uses FIXED gamma from pooled training batches
- Uses FIXED kappa from pooled training batches
- Only estimates lambda (genetic effects) per batch
- Processes data in batches (default 10k samples)
- Saves predictions (pi) for each batch
- Concatenates all batches at the end

Master checkpoints should be created using create_master_checkpoints.py:
- master_for_fitting_pooled_correctedE.pt (pooled phi)

Pooled gamma and kappa should be created using pool_kappa_and_gamma_from_batches.py:
- pooled_kappa_gamma.pt (pooled kappa and gamma)

Usage:
    python run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py \
        --trained_model_path /path/to/master.pt \
        --pooled_gamma_kappa_path /path/to/pooled_kappa_gamma.pt
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

# Add AWS directory to path to use the fixed gamma/kappa model
sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))

from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa import *
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
    print("Loading components...")

    # Load large matrices - NOTE: Using E_enrollment_full.pt for enrollment data!
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_enrollment_full.pt', weights_only=False)  # Enrollment data for prediction
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)

    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)

    print("Loaded all components successfully!")

    return Y, E, G, essentials


def load_covariates_data(csv_path):
    """Load and process covariates from CSV file"""
    print("Loading covariates data...")
    fh_processed = pd.read_csv(csv_path)
    print(f"Loaded {len(fh_processed)} samples from covariates file")
    return fh_processed


def generate_batches(total_size, batch_size=10000):
    """Generate batch indices"""
    batches = []
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batches.append((start, end))
    return batches


def main():
    parser = argparse.ArgumentParser(description='Run Aladyn predictions with fixed phi/psi, fixed gamma, and fixed kappa')
    parser.add_argument('--trained_model_path', type=str, required=True,
                       help='Path to trained model with phi and psi')
    parser.add_argument('--pooled_gamma_kappa_path', type=str, required=True,
                       help='Path to pooled gamma and kappa file (from pool_kappa_and_gamma_from_batches.py)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for processing')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs per batch')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization parameter')

    # Path configuration
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing input data')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_vectorized/',
                       help='Output directory for predictions')
    parser.add_argument('--covariates_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv',
                       help='Path to covariates CSV file')
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
    print(f"Aladyn Prediction Script - Fixed Phi/Psi/Gamma/Kappa Mode")
    print(f"{'='*80}")
    print(f"Trained model: {args.trained_model_path}")
    print(f"Pooled gamma/kappa: {args.pooled_gamma_kappa_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")

    # Load data
    Y, E, G, essentials = load_model_essentials(args.data_dir)
    fh_processed = load_covariates_data(args.covariates_path)

    # Load references
    print("Loading reference trajectories...")
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs  # Free memory - we only need signature_refs
    gc.collect()
    prevalence_t=torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)

    # Load initial_psi for consistent initialization (though gamma is fixed)
    print("Loading initial_psi...")
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
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
    
    # Load pooled gamma and kappa
    print(f"Loading pooled gamma and kappa from {args.pooled_gamma_kappa_path}...")
    pooled_gk = torch.load(args.pooled_gamma_kappa_path, map_location='cpu', weights_only=False)
    
    if 'gamma' in pooled_gk:
        gamma_pooled = pooled_gk['gamma']
    else:
        raise ValueError(f"gamma not found in {args.pooled_gamma_kappa_path}")
    
    if 'kappa' in pooled_gk:
        kappa_pooled = pooled_gk['kappa']
    else:
        raise ValueError(f"kappa not found in {args.pooled_gamma_kappa_path}")
    
    # Convert to numpy if tensor
    if torch.is_tensor(gamma_pooled):
        gamma_pooled = gamma_pooled.cpu().numpy()
    if torch.is_tensor(kappa_pooled):
        kappa_pooled = kappa_pooled.item() if kappa_pooled.numel() == 1 else float(kappa_pooled)
    else:
        kappa_pooled = float(kappa_pooled)
    
    print(f"Loaded pooled gamma shape: {gamma_pooled.shape}")
    print(f"Loaded pooled kappa: {kappa_pooled:.6f}")
    
    # Free memory
    del pooled_gk
    gc.collect()
    
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

            # Check that gamma shape matches G_with_sex
            if gamma_pooled.shape[0] != G_with_sex.shape[1]:
                print(f"⚠️  WARNING: gamma shape {gamma_pooled.shape[0]} doesn't match G shape {G_with_sex.shape[1]}")
                print(f"   This might be okay if gamma was pooled from a different G configuration")
                print(f"   Will proceed, but predictions may be incorrect if shapes don't match")

            # Initialize model with FIXED phi, psi, gamma, and kappa
            print(f"Initializing model with fixed phi/psi/gamma/kappa...")
            model = AladynSurvivalFixedPhiFixedGammaFixedKappa(
                N=Y_batch.shape[0],
                D=Y_batch.shape[1],
                T=Y_batch.shape[2],
                K=20,
                P=G_with_sex.shape[1],
                G=G_with_sex,
                Y=Y_batch,
                R=0,
                W=0.0001,
                prevalence_t=prevalence_t,
                init_sd_scaler=1e-1,
                genetic_scale=1,
                pretrained_phi=phi_total,
                pretrained_psi=psi_total,
                pretrained_gamma=gamma_pooled,
                pretrained_kappa=kappa_pooled,
                signature_references=signature_refs,
                healthy_reference=True,
                disease_names=essentials['disease_names']
            )

            # Verify phi, psi, gamma, and kappa are fixed
            if np.allclose(model.phi.cpu().numpy(), phi_total):
                print("✓ phi matches phi_total!")
            else:
                print("✗ WARNING: phi does NOT match phi_total!")

            if np.allclose(model.psi.cpu().numpy(), psi_total):
                print("✓ psi matches psi_total!")
            else:
                print("✗ WARNING: psi does NOT match psi_total!")
            
            if np.allclose(model.gamma.cpu().numpy(), gamma_pooled):
                print("✓ gamma matches gamma_pooled!")
            else:
                print("✗ WARNING: gamma does NOT match gamma_pooled!")
            
            if abs(model.kappa.item() - kappa_pooled) < 1e-6:
                print(f"✓ kappa matches kappa_pooled! ({model.kappa.item():.6f})")
            else:
                print(f"✗ WARNING: kappa does NOT match kappa_pooled! (model: {model.kappa.item():.6f}, pooled: {kappa_pooled:.6f})")

            # Train model (only lambda is being estimated)
            print(f"Training model (estimating lambda only, gamma and kappa are fixed)...")
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
            'gamma_shape': gamma_pooled.shape,
            'kappa_value': kappa_pooled,
        }
        info_filename = os.path.join(args.output_dir, "batch_info.pt")
        torch.save(batch_info, info_filename)
        print(f"✓ Saved batch info to {info_filename}")
    else:
        print("✗ No successful batches to concatenate!")

    print(f"\n{'='*80}")
    print(f"DONE! Predictions use fixed gamma and kappa (only lambda was learned).")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

