#!/usr/bin/env python
"""
Batch script to run Aladyn model on sample batches with JOINT ESTIMATION (NO lambda_reg)
Uses E_enrollment_full.pt (enrollment matrix) for training

This script:
- Uses E_enrollment_full.pt (same as prediction scripts)
- Performs JOINT ESTIMATION (trains phi, psi, and lambda together)
- Uses clust_huge_amp_vectorized_nolr.py — no lambda_reg / no learning-rate regularization
- AWS-compatible with environment variable defaults

Requirements on AWS:
- clust_huge_amp_vectorized_nolr.py must be available (script will search multiple paths)
- Data files: Y_tensor.pt, E_enrollment_full.pt, G_matrix.pt, model_essentials.pt
- Reference files: reference_trajectories.pt, prevalence_t_corrected.pt
- Initialization files: initial_psi_400k.pt, initial_clusters_400k.pt
- Covariates CSV: baselinagefamh_withpcs.csv

Usage:
    Local:  python run_aladyn_batch_vector_e_enrollment_joint_nolr.py --start_index 0 --end_index 10000
    AWS:    python run_aladyn_batch_vector_e_enrollment_joint_nolr.py --start_index 0 --end_index 10000 --data_dir /data --output_dir /results
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
from pathlib import Path

# Add scripts directory to path to import clust_huge_amp_vectorized_nolr
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent / 'pyScripts_forPublish',
    script_dir / 'pyScripts_forPublish',
    Path.home() / 'pyScripts_forPublish',
    Path('/home/ec2-user/pyScripts_forPublish'),
    Path('/home/ubuntu/pyScripts_forPublish'),
    Path('/home/ubuntu/aladyn_project/scripts'),  # AWS scripts dir
    script_dir,
    script_dir.parent / 'scripts',
]

clust_nolr_found = False
for path in possible_paths:
    if (path / 'clust_huge_amp_vectorized_nolr.py').exists():
        sys.path.insert(0, str(path))
        clust_nolr_found = True
        print(f"Found clust_huge_amp_vectorized_nolr.py at: {path}")
        break

if not clust_nolr_found:
    print(f"ERROR: Could not find clust_huge_amp_vectorized_nolr.py. Checked:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

from clust_huge_amp_vectorized_nolr import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model_essentials(base_path):
    """Load all essential components using E_enrollment_full.pt"""
    print("Loading components...")
    if not base_path.endswith('/') and not base_path.endswith(os.sep):
        base_path = base_path + os.sep

    Y = torch.load(os.path.join(base_path, 'Y_tensor.pt'), map_location='cpu', weights_only=False)
    E = torch.load(os.path.join(base_path, 'E_enrollment_full.pt'), map_location='cpu', weights_only=False)
    G = torch.load(os.path.join(base_path, 'G_matrix.pt'), map_location='cpu', weights_only=False)
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
    if 'Sex' in fh_processed.columns:
        fh_processed['sex_numeric'] = fh_processed['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
        sex = fh_processed['sex_numeric'].values
    elif 'sex' in fh_processed.columns:
        sex = fh_processed['sex'].values
    else:
        raise ValueError("No 'Sex' or 'sex' column found in covariates CSV")
    return sex, fh_processed


def main():
    parser = argparse.ArgumentParser(description='Run Aladyn model on batch with joint estimation (enrollment E, NO lambda_reg)')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for batch')
    parser.add_argument('--end_index', type=int, default=10000, help='End index for batch')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--K', type=int, default=20, help='Number of clusters')
    parser.add_argument('--W', type=float, default=0.0001, help='W parameter')

    parser.add_argument('--data_dir', type=str,
                        default=os.getenv('DATA_DIR', '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'),
                        help='Directory containing input data')
    parser.add_argument('--output_dir', type=str,
                        default=os.getenv('OUTPUT_DIR', '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_joint_estimation_batchrun_nolr'),
                        help='Output directory for saved models')
    parser.add_argument('--covariates_path', type=str,
                        default=os.getenv('COVARIATES_PATH', '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'),
                        help='Path to covariates CSV file')
    parser.add_argument('--include_pcs', type=bool, default=True, help='Include principal components in the model')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}")
    print(f"Running Aladyn batch (VECTORIZED, JOINT ESTIMATION, ENROLLMENT E, NOLR):")
    print(f"  Samples {args.start_index} to {args.end_index}")
    print(f"  Using E_enrollment_full.pt (enrollment matrix)")
    print(f"  No lambda_reg (unregularized)")
    print(f"{'='*60}\n")

    Y, E, G, essentials = load_model_essentials(args.data_dir)
    print(f"Subsetting data from {args.start_index} to {args.end_index}...")
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G,
                                                     start_index=args.start_index,
                                                     end_index=args.end_index)
    del Y

    sex, fh_processed = load_covariates_data(args.covariates_path)
    sex_batch = sex[args.start_index:args.end_index]
    G_with_sex = np.column_stack([G_batch, sex_batch])
    if args.include_pcs:
        pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
                      'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
        pcs = fh_processed.iloc[args.start_index:args.end_index][pc_columns].values
        G_with_sex = np.column_stack([G_batch, sex_batch, pcs])
    print(f"G_with_sex shape: {G_with_sex.shape}")

    print("Loading reference trajectories...")
    refs = torch.load(os.path.join(args.data_dir, 'reference_trajectories.pt'), map_location='cpu', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(os.path.join(args.data_dir, 'prevalence_t_corrected.pt'), map_location='cpu', weights_only=False)

    print(f"\nInitializing model with K={args.K} clusters (VECTORIZED, JOINT, NOLR)...")
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

    print("Loading initial psi and clusters...")
    torch.manual_seed(0)
    np.random.seed(0)
    initial_psi = torch.load(os.path.join(args.data_dir, 'initial_psi_400k.pt'), map_location='cpu', weights_only=False)
    initial_clusters = torch.load(os.path.join(args.data_dir, 'initial_clusters_400k.pt'), map_location='cpu', weights_only=False)
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters
    print(f"Clusters match exactly: {np.array_equal(initial_clusters, model.clusters)}")

    print(f"\nTraining model for {args.num_epochs} epochs (JOINT ESTIMATION, NOLR)...")
    print(f"Learning rate: {args.learning_rate} (no lambda_reg)")
    history = model.fit(E_batch,
                        num_epochs=args.num_epochs,
                        learning_rate=args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'enrollment_joint_model_VECTORIZED_nolr_W{args.W}_batch_{args.start_index}_{args.end_index}.pt'
    print(f"\nSaving model to {output_path}...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'phi': model.phi,
        'psi': model.psi,
        'Y': model.Y,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'G': model.G,
        'args': vars(args),
        'indices': indices,
        'clusters': initial_clusters,
        'version': 'VECTORIZED_JOINT_ENROLLMENT_NOLR',
        'E_source': 'E_enrollment_full.pt',
    }, output_path)

    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {output_path}")
    print(f"{'='*60}\n")
    if history and len(history) > 0 and 'loss' in history[-1]:
        print(f"Final loss: {history[-1]['loss']:.4f}")


if __name__ == '__main__':
    main()
