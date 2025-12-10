#!/usr/bin/env python
"""
Run Aladyn predictions for first 10K patients with different washout windows.

This script runs predictions for:
1. No washout (enrollment_full)
2. 1-month washout
3. 3-month washout
4. 6-month washout

Uses master_for_fitting_pooled_correctedE.pt as the trained model.

Usage:
    # With defaults (uses master_for_fitting_pooled_correctedE.pt):
    python run_aladyn_predict_washout_comparison.py
    
    # With nohup (background):
    cd /Users/sarahurbut/aladynoulli2/claudefile
    nohup python run_aladyn_predict_washout_comparison.py \
        --output_base_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/washout_comparison_10k/ \
        > washout_comparison_10k.log 2>&1 &
    
    # Check progress:
    tail -f washout_comparison_10k.log
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

# Add AWS directory to path to use the tested vectorized model
sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))

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


def load_model_essentials(base_path, E_matrix_path):
    """Load all essential components with specified E matrix"""
    print("Loading components...")

    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(E_matrix_path, weights_only=False)  # Use specified E matrix
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)

    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)

    print("Loaded all components successfully!")
    print(f"E matrix shape: {E.shape}")

    return Y, E, G, essentials


def load_covariates_data(csv_path):
    """Load and process covariates from CSV file"""
    print("Loading covariates data...")
    fh_processed = pd.read_csv(csv_path)
    print(f"Loaded {len(fh_processed)} samples from covariates file")
    return fh_processed


def run_predictions_for_washout(args, washout_label, E_matrix_path, output_dir):
    """Run predictions for a specific washout version"""
    print(f"\n{'='*80}")
    print(f"RUNNING PREDICTIONS: {washout_label}")
    print(f"{'='*80}")
    print(f"E matrix: {E_matrix_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    Y, E, G, essentials = load_model_essentials(args.data_dir, E_matrix_path)
    fh_processed = load_covariates_data(args.covariates_path)

    # Load references
    print("Loading reference trajectories...")
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    gc.collect()
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)

    # Load initial_psi
    print("Loading initial_psi...")
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    if torch.is_tensor(initial_psi):
        initial_psi = initial_psi.cpu().numpy()
    print(f"Loaded initial_psi shape: {initial_psi.shape}")

    # Load master checkpoint
    print(f"Loading master checkpoint from {args.trained_model_path}...")
    master_checkpoint = torch.load(args.trained_model_path, map_location='cpu', weights_only=False)
    
    # Extract phi and psi from master checkpoint
    if 'model_state_dict' in master_checkpoint:
        phi_total = master_checkpoint['model_state_dict']['phi']
        psi_total = master_checkpoint['model_state_dict']['psi']
    else:
        phi_total = master_checkpoint['phi']
        psi_total = master_checkpoint['psi']
    
    if 'description' in master_checkpoint:
        print(f"Master checkpoint description: {master_checkpoint['description']}")
    
    # Convert to numpy if tensor
    if torch.is_tensor(phi_total):
        phi_total = phi_total.cpu().numpy()
    if torch.is_tensor(psi_total):
        psi_total = psi_total.cpu().numpy()
    
    del master_checkpoint
    gc.collect()
    
    print(f"Loaded pooled phi shape: {phi_total.shape}")
    print(f"Loaded psi_total shape: {psi_total.shape}")
    
    # Process only first 10K patients
    start = 0
    stop = 10000
    print(f"\nProcessing samples {start} to {stop}...")

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
            prevalence_t=prevalence_t,
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
        
        # Reinitialize with psi_total for consistent gamma initialization
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

            # Save predictions with washout label
            pi_filename = os.path.join(output_dir, f"pi_washout_{washout_label}_0_10000.pt")
            torch.save(pi, pi_filename)
            print(f"✓ Saved predictions to {pi_filename}")

            # Save model state
            model_filename = os.path.join(output_dir, f"model_washout_{washout_label}_0_10000.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'E': E_batch,
                'prevalence_t': model.prevalence_t,
                'logit_prevalence_t': model.logit_prev_t,
                'start_index': start,
                'end_index': stop,
                'washout_label': washout_label,
                'E_matrix_path': E_matrix_path,
            }, model_filename)
            print(f"✓ Saved model to {model_filename}")

        # Clean up memory
        print(f"Cleaning up memory...")
        del pi, model, Y_batch, E_batch, G_batch, G_with_sex, pce_df_subset, Y, E, G
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"✓ {washout_label} complete!")
        return True

    except Exception as e:
        print(f"✗ ERROR in {washout_label}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Aladyn predictions with different washout windows')
    parser.add_argument('--trained_model_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt',
                       help='Path to trained model with phi and psi (default: master_for_fitting_pooled_correctedE.pt)')
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
    parser.add_argument('--output_base_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/washout_comparison_10k/',
                       help='Base output directory for predictions')
    parser.add_argument('--covariates_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv',
                       help='Path to covariates CSV file')
    parser.add_argument('--include_pcs', type=bool, default=True,
                       help='Include principal components in the model')
    
    # Washout matrix paths
    parser.add_argument('--washout_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout/',
                       help='Directory containing washout E matrices')
    
    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Aladyn Washout Comparison Script - First 10K Patients")
    print(f"{'='*80}")
    print(f"Trained model: {args.trained_model_path}")
    print(f"Output base directory: {args.output_base_dir}")
    print(f"Washout matrices directory: {args.washout_dir}")
    print(f"{'='*80}\n")

    # Define washout configurations
    washout_configs = [
        #{
         #   'label': 'no_washout',
        #    'E_path': os.path.join(args.data_dir, 'E_enrollment_full.pt'),
        #    'description': 'No washout (enrollment_full)'
       # },
        {
            'label': '1month',
            'E_path': os.path.join(args.washout_dir, 'E_washout_1month_batch_0_10000.pt'),
            'description': '1-month washout'
        },
        {
            'label': '3month',
            'E_path': os.path.join(args.washout_dir, 'E_washout_3month_batch_0_10000.pt'),
            'description': '3-month washout'
        },
        {
            'label': '6month',
            'E_path': os.path.join(args.washout_dir, 'E_washout_6month_batch_0_10000.pt'),
            'description': '6-month washout'
        }
    ]

    # Run predictions for each washout configuration
    results = {}
    for config in washout_configs:
        # Check if E matrix exists
        if not os.path.exists(config['E_path']):
            print(f"✗ WARNING: E matrix not found: {config['E_path']}")
            print(f"  Skipping {config['label']}")
            results[config['label']] = False
            continue

        # Create output directory for this washout version
        output_dir = os.path.join(args.output_base_dir, config['label'])
        
        # Run predictions
        success = run_predictions_for_washout(
            args,
            washout_label=config['label'],
            E_matrix_path=config['E_path'],
            output_dir=output_dir
        )
        
        results[config['label']] = success

    # Print summary
    print(f"\n{'='*80}")
    print(f"WASHOUT COMPARISON SUMMARY")
    print(f"{'='*80}")
    for config in washout_configs:
        status = "✓ SUCCESS" if results.get(config['label'], False) else "✗ FAILED"
        print(f"{config['label']:15} {config['description']:30} {status}")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {args.output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

