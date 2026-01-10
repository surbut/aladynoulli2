#!/usr/bin/env python
"""
Batch script to run Aladyn model on sample batches WITHOUT sex and PCs
Variant: PRS only (36 PRS), no sex, no principal components, no lambda_reg

This version uses ONLY PRS values (G matrix) - no sex covariate, no PCs.
Useful for comparing results with/without sex and PCs to show robustness.

Converted from aladynoulli_fit_for_understanding_and_discovery.ipynb

Usage:
    Local:  python run_aladyn_batch_vector_e_censor_nolor_nosex_nopc.py --start_index 0 --end_index 10000
    AWS:    python run_aladyn_batch_vector_e_censor_nolor_nosex_nopc.py --start_index 0 --end_index 10000 --data_dir /data --output_dir /results

    This was run on the full dataset (400k samples) and saved in the Dropbox/censor_e_batchrun_vectorized_nolr_nopcs_nosex directory.
    
Differences from _nolr version:
    - No lambda_reg (same as _nolr)
    - No sex covariate (G_batch only, P=36)
    - No principal components (G_batch only, P=36)
    - Output filename includes '_nopcs_nosex' suffix
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to import clust_huge_amp
sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))

from clust_huge_amp_vectorized_nolr import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model_essentials(base_path):
    """Load all essential components"""
    print("Loading components...")

    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_matrix_corrected.pt', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)

    

    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)

    print("Loaded all components successfully!")

    return Y, E, G, essentials


def load_covariates_data(csv_path):
    """Load and process covariates from CSV file"""
    print("Loading covariates data...")
    fh_processed = pd.read_csv(csv_path)

    # Convert sex to numeric: Female=0, Male=1
    if 'Sex' in fh_processed.columns:
        fh_processed['sex_numeric'] = fh_processed['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
        sex = fh_processed['sex_numeric'].values
    elif 'sex' in fh_processed.columns:
        # If already numeric or lowercase
        sex = fh_processed['sex'].values
    else:
        raise ValueError("No 'Sex' or 'sex' column found in covariates CSV")

    return sex, fh_processed


def main():
    parser = argparse.ArgumentParser(description='Run Aladyn model on batch of samples')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for batch')
    parser.add_argument('--end_index', type=int, default=10000,
                       help='End index for batch')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--K', type=int, default=20,
                       help='Number of clusters')
    parser.add_argument('--W', type=float, default=0.0001,
                       help='W parameter')

    # Path configuration - defaults to local Dropbox, can be overridden for AWS
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing input data (use /data for AWS)')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr_nopcs_nosex',
                       help='Output directory for saved models (use /results for AWS)')
    parser.add_argument('--covariates_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv',
                       help='Path to covariates CSV file (not used - PRS only)')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}")
    print(f"Running Aladyn batch (VECTORIZED_nolr_nopcs_nosex): samples {args.start_index} to {args.end_index}")
    print(f"Using PRS only (no sex, no PCs)")
    print(f"{'='*60}\n")

    # Load data
    Y, E, G, essentials = load_model_essentials(args.data_dir)

    # Subset the data
    print(f"Subsetting data from {args.start_index} to {args.end_index}...")
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G,
                                                      start_index=args.start_index,
                                                      end_index=args.end_index)

    # Free up memory
    del Y

    # Use PRS only (no sex, no PCs)
    # G_batch contains only PRS values (36 PRS)
    G_final = G_batch
    print(f"Using PRS only (no sex, no PCs)")
    print(f"G_final shape: {G_final.shape} (should be [N, 36] for 36 PRS)")
    print(f"Expected P dimension: {G_final.shape[1]} (36 PRS only)")

    # Load reference trajectories
    print("Loading reference trajectories...")
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)
 
    # Initialize model
    print(f"\nInitializing model with K={args.K} clusters (VECTORIZED_nolr_nopcs_nosex)...")
    print(f"P dimension: {G_final.shape[1]} (PRS only, no sex, no PCs)")
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_batch.shape[0],
        D=Y_batch.shape[1],
        T=Y_batch.shape[2],
        K=args.K,
        P=G_final.shape[1],  # Should be 36 (PRS only)
        init_sd_scaler=1e-1,
        G=G_final,
        Y=Y_batch,
        genetic_scale=1,
        W=args.W,
        R=0,
        prevalence_t=prevalence_t,
        signature_references=signature_refs,
        healthy_reference=True,
        disease_names=essentials['disease_names']
    )

    # Load and initialize with saved psi and clusters
    print("Loading initial psi and clusters...")
    torch.manual_seed(0)
    np.random.seed(0)

    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(args.data_dir + 'initial_clusters_400k.pt', weights_only=False)

 
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters

    # Verify clusters match
    clusters_match = np.array_equal(initial_clusters, model.clusters)
    print(f"Clusters match exactly: {clusters_match}")

    # Train the model
    print(f"\nTraining model for {args.num_epochs} epochs (VECTORIZED_nolr_nopcs_nosex)...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Model configuration: PRS only (P={G_final.shape[1]}), no lambda_reg, no sex, no PCs")

    history = model.fit(E_batch,
                       num_epochs=args.num_epochs,
                       learning_rate=args.learning_rate
                       )   #no lambda_reg   

    # Save model
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'enrollment_model_VECTORIZED_W{args.W}_nolr_nopcs_nosex_batch_{args.start_index}_{args.end_index}.pt'
    print(f"\nSaving model to {output_path}...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'phi': model.phi,
        'Y': model.Y,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'G': model.G,
        #'history': history,
        'args': vars(args),
        'indices': indices,
        'clusters': initial_clusters,  # Save initial_clusters directly to ensure it's saved
        'version': 'VECTORIZED_nolr_nopcs_nosex',  # Mark this as PRS-only version (no sex, no PCs)
        'P_dimension': G_final.shape[1],  # Save P dimension for reference (should be 36)
    }, output_path)

    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to:")
    print(f"{output_path}")
    print(f"{'='*60}\n")

    # Print final loss
    if history and len(history) > 0 and 'loss' in history[-1]:
        print(f"Final loss: {history[-1]['loss']:.4f}")


if __name__ == '__main__':
    main()
