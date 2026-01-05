#!/usr/bin/env python
"""
Batch script to run Aladyn model on sample batches with JOINT ESTIMATION
Uses E_enrollment_full.pt (enrollment matrix) for training

This script:
- Uses E_enrollment_full.pt (same as prediction scripts)
- Performs JOINT ESTIMATION (trains phi, psi, and lambda together)
- AWS-compatible with environment variable defaults

Requirements on AWS:
- clust_huge_amp_vectorized.py must be available (script will search multiple paths)
- Data files: Y_tensor.pt, E_enrollment_full.pt, G_matrix.pt, model_essentials.pt
- Reference files: reference_trajectories.pt, prevalence_t_corrected.pt
- Initialization files: initial_psi_400k.pt, initial_clusters_400k.pt
- Covariates CSV: baselinagefamh_withpcs.csv

Usage:
    Local:  python run_aladyn_batch_vector_e_enrollment_joint.py --start_index 0 --end_index 10000
    AWS:    python run_aladyn_batch_vector_e_enrollment_joint.py --start_index 0 --end_index 10000 --data_dir /data --output_dir /results
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
from pathlib import Path

# Add scripts directory to path to import clust_huge_amp_vectorized
# Try multiple possible locations (for local vs AWS)
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent / 'pyScripts_forPublish',  # Local: claudefile/../pyScripts_forPublish
    script_dir / 'pyScripts_forPublish',  # If pyScripts_forPublish is in same dir
    Path.home() / 'pyScripts_forPublish',  # Fallback: home directory
    Path('/home/ec2-user/pyScripts_forPublish'),  # AWS EC2 default
    Path('/home/ubuntu/pyScripts_forPublish'),  # AWS Ubuntu default
    script_dir,  # Same directory as script (if clust_huge_amp_vectorized.py is in scripts/)
    script_dir.parent / 'scripts',  # Parent/scripts
]

clust_huge_amp_found = False
for path in possible_paths:
    if (path / 'clust_huge_amp_vectorized.py').exists():
        sys.path.insert(0, str(path))
        clust_huge_amp_found = True
        print(f"Found clust_huge_amp_vectorized.py at: {path}")
        break

if not clust_huge_amp_found:
    # If none found, print error
    print(f"ERROR: Could not find clust_huge_amp_vectorized.py. Checked:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

from clust_huge_amp_vectorized import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model_essentials(base_path):
    """Load all essential components using E_enrollment_full.pt"""
    print("Loading components...")
    
    # Ensure path ends with / for os.path.join compatibility
    if not base_path.endswith('/') and not base_path.endswith(os.sep):
        base_path = base_path + os.sep

    # Load large matrices - NOTE: Using E_enrollment_full.pt for enrollment data!
    Y = torch.load(os.path.join(base_path, 'Y_tensor.pt'), map_location='cpu', weights_only=False)
    E = torch.load(os.path.join(base_path, 'E_enrollment_full.pt'), map_location='cpu', weights_only=False)  # Enrollment data
    G = torch.load(os.path.join(base_path, 'G_matrix.pt'), map_location='cpu', weights_only=False)

    # Load other components
    essentials = torch.load(os.path.join(base_path, 'model_essentials.pt'), map_location='cpu', weights_only=False)

    print(f"Loaded all components successfully!")
    print(f"  Y shape: {Y.shape}")
    print(f"  E shape: {E.shape} (enrollment matrix)")
    print(f"  G shape: {G.shape}")

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
    parser = argparse.ArgumentParser(description='Run Aladyn model on batch of samples with joint estimation (enrollment E)')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for batch')
    parser.add_argument('--end_index', type=int, default=10000,
                       help='End index for batch')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization parameter')
    parser.add_argument('--K', type=int, default=20,
                       help='Number of clusters')
    parser.add_argument('--W', type=float, default=0.0001,
                       help='W parameter')

    # Path configuration - AWS compatible with environment variable defaults
    parser.add_argument('--data_dir', type=str,
                       default=os.getenv('DATA_DIR', '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'),
                       help='Directory containing input data (default: from DATA_DIR env var)')
    parser.add_argument('--output_dir', type=str,
                       default=os.getenv('OUTPUT_DIR', '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_joint_estimation_batchrun'),
                       help='Output directory for saved models (default: from OUTPUT_DIR env var)')
    parser.add_argument('--covariates_path', type=str,
                       default=os.getenv('COVARIATES_PATH', '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'),
                       help='Path to covariates CSV file (default: from COVARIATES_PATH env var)')
    parser.add_argument('--include_pcs', type=bool, default=True,
                       help='Include principal components in the model')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}")
    print(f"Running Aladyn batch (VECTORIZED, JOINT ESTIMATION, ENROLLMENT E):")
    print(f"  Samples {args.start_index} to {args.end_index}")
    print(f"  Using E_enrollment_full.pt (enrollment matrix)")
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

    # Load covariates data and subset it
    sex, fh_processed = load_covariates_data(args.covariates_path)
    sex_batch = sex[args.start_index:args.end_index]

    # Combine G with sex
    G_with_sex = np.column_stack([G_batch, sex_batch])
    if args.include_pcs:
        pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
        'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
        pcs = fh_processed.iloc[args.start_index:args.end_index][pc_columns].values
        G_with_sex = np.column_stack([G_batch, sex_batch, pcs])
    print(f"G_with_sex shape: {G_with_sex.shape}")
    print(f"Covariates loaded: {fh_processed.shape[0]} total samples")

    # Load reference trajectories
    print("Loading reference trajectories...")
    refs = torch.load(os.path.join(args.data_dir, 'reference_trajectories.pt'), map_location='cpu', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(os.path.join(args.data_dir, 'prevalence_t_corrected.pt'), map_location='cpu', weights_only=False)
 
    # Initialize model
    print(f"\nInitializing model with K={args.K} clusters (VECTORIZED, JOINT ESTIMATION)...")
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_batch.shape[0],
        D=Y_batch.shape[1],
        T=Y_batch.shape[2],
        K=args.K,
        P=G_with_sex.shape[1],
        init_sd_scaler=1e-1,
        G=G_with_sex,
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

    initial_psi = torch.load(os.path.join(args.data_dir, 'initial_psi_400k.pt'), map_location='cpu', weights_only=False)
    initial_clusters = torch.load(os.path.join(args.data_dir, 'initial_clusters_400k.pt'), map_location='cpu', weights_only=False)

 
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters

    # Verify clusters match
    clusters_match = np.array_equal(initial_clusters, model.clusters)
    print(f"Clusters match exactly: {clusters_match}")

    # Train the model (JOINT ESTIMATION: phi, psi, and lambda are all trained)
    print(f"\nTraining model for {args.num_epochs} epochs (JOINT ESTIMATION)...")
    print(f"Learning rate: {args.learning_rate}, Lambda: {args.lambda_reg}")
    print(f"Note: All parameters (phi, psi, lambda) will be jointly estimated")

    history = model.fit(E_batch,
                       num_epochs=args.num_epochs,
                       learning_rate=args.learning_rate,
                       lambda_reg=args.lambda_reg)

    # Save model
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'enrollment_joint_model_VECTORIZED_W{args.W}_batch_{args.start_index}_{args.end_index}.pt'
    print(f"\nSaving model to {output_path}...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'phi': model.phi,
        'psi': model.psi,  # Save trained psi (jointly estimated)
        'Y': model.Y,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'G': model.G,
        #'history': history,
        'args': vars(args),
        'indices': indices,
        'clusters': initial_clusters,  # Save initial_clusters directly to ensure it's saved
        'version': 'VECTORIZED_JOINT_ENROLLMENT',  # Mark this as the vectorized joint estimation version with enrollment E
        'E_source': 'E_enrollment_full.pt',  # Mark which E matrix was used
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

