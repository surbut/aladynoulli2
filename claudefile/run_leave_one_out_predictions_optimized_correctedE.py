#!/usr/bin/env python
"""
Optimized version: Load data once, process all batches with different checkpoints

This version loads Y, E, G once, then processes all 40 batches sequentially,
loading a different leave-one-out checkpoint for each batch.

Usage:
    python run_leave_one_out_predictions_optimized_correctedE.py
"""

import sys
import argparse
import torch
import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path

# Add path for model import
sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))
from clust_huge_amp_fixedPhi_vectorized import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model_essentials(base_path):
    """Load all essential components"""
    print("Loading components...")
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_enrollment_full.pt', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)
    print("Loaded all components successfully!")
    return Y, E, G, essentials


def subset_data(Y, E, G, start_index, end_index):
    """Subset data based on indices."""
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices


def main():
    parser = argparse.ArgumentParser(description='Run optimized leave-one-out predictions')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Data directory')
    parser.add_argument('--output_base_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_correctedE/',
                       help='Base output directory for predictions')
    parser.add_argument('--covariates_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv',
                       help='Path to covariates CSV file')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs per batch')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization parameter')
    parser.add_argument('--skip_completed', action='store_true',
                       help='Skip batches that already have output files')
    args = parser.parse_args()
    
    print("="*80)
    print("Optimized Leave-One-Out Predictions (Load Data Once)")
    print("="*80)
    print(f"Total batches: {args.total_batches}")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Load data ONCE (this is the key optimization)
    print("Loading data files (this happens ONCE)...")
    Y, E, G, essentials = load_model_essentials(args.data_dir)
    fh_processed = pd.read_csv(args.covariates_path)
    
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
    
    print(f"\n✓ All data loaded! Processing {args.total_batches} batches...")
    print("="*80)
    
    # Process each batch
    for batch_idx in range(args.total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        output_dir = f"{args.output_base_dir}batch_{batch_idx}/"
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_path = f"{args.data_dir}master_for_fitting_pooled_correctedE_exclude_batch_{batch_idx}.pt"
        
        # Check if already completed
        if args.skip_completed:
            pi_file = os.path.join(output_dir, f"pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt")
            if Path(pi_file).exists():
                print(f"\n⏭️  Batch {batch_idx} already completed (found {Path(pi_file).name}), skipping...")
                continue
        
        if not Path(checkpoint_path).exists():
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping batch {batch_idx}")
            continue
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx+1}/{args.total_batches}: Processing samples {start_idx}-{end_idx}")
        print(f"{'='*80}")
        print(f"Using checkpoint: {Path(checkpoint_path).name}")
        print(f"✓ Verification: Checkpoint excludes batch {batch_idx}, will predict on batch {batch_idx} (samples {start_idx}-{end_idx})")
        
        try:
            # Load leave-one-out checkpoint for this batch
            print(f"Loading leave-one-out checkpoint...")
            master_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in master_checkpoint:
                phi_total = master_checkpoint['model_state_dict']['phi']
                psi_total = master_checkpoint['model_state_dict']['psi']
            else:
                phi_total = master_checkpoint['phi']
                psi_total = master_checkpoint['psi']
            
            if torch.is_tensor(phi_total):
                phi_total = phi_total.cpu().numpy()
            if torch.is_tensor(psi_total):
                psi_total = psi_total.cpu().numpy()
            
            del master_checkpoint
            gc.collect()
            
            # Subset data for this batch
            print(f"Subsetting data...")
            Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G, start_idx, end_idx)
            
            # Get demographics
            pce_df_subset = fh_processed.iloc[start_idx:end_idx].reset_index(drop=True)
            sex = pce_df_subset['sex'].values
            G_with_sex = np.column_stack([G_batch, sex])
            
            pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
                         'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
            pcs = pce_df_subset[pc_columns].values
            G_with_sex = np.column_stack([G_batch, sex, pcs])
            
            print(f"Data shapes: Y={Y_batch.shape}, E={E_batch.shape}, G={G_with_sex.shape}")
            
            # Initialize model with fixed phi and psi
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
            
            # Reinitialize gamma
            print("Reinitializing gamma with psi_total...")
            model.initialize_params(init_psi=torch.tensor(psi_total, dtype=torch.float32))
            
            # Train model (only lambda is being estimated)
            print(f"Training model (estimating lambda only, {args.num_epochs} epochs)...")
            history = model.fit(
                E_batch,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                lambda_reg=args.lambda_reg
            )
            
            # Generate predictions
            print(f"Generating predictions...")
            with torch.no_grad():
                pi, _, _ = model.forward()
                
                # Save predictions
                pi_filename = os.path.join(output_dir, f"pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt")
                torch.save(pi, pi_filename)
                print(f"✓ Saved predictions to {pi_filename}")
                
                # Save model state
                model_filename = os.path.join(output_dir, f"model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'E': E_batch,
                    'prevalence_t': model.prevalence_t,
                    'logit_prevalence_t': model.logit_prev_t,
                    'start_index': start_idx,
                    'end_index': end_idx,
                }, model_filename)
            
            # Clean up
            del pi, model, Y_batch, E_batch, G_batch, G_with_sex, pce_df_subset, phi_total, psi_total
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"✓ Batch {batch_idx+1}/{args.total_batches} complete!")
            
        except Exception as e:
            print(f"✗ ERROR in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("ALL BATCHES COMPLETE!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("1. Calculate 10-year AUC for each batch using calculate_leave_one_out_auc_correctedE.py")
    print("2. Compare to overall pooled AUC")


if __name__ == '__main__':
    main()

