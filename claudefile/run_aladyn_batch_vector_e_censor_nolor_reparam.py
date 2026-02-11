#!/usr/bin/env python
"""
Same as run_aladyn_batch_vector_e_censor_nolor.py but uses REPARAMETERIZED model.
Gamma and psi flow through NLL (full chain rule). Use to compare with original.

Usage:
    python run_aladyn_batch_vector_e_censor_nolor_reparam.py --start_index 0 --end_index 10000

Output goes to censor_e_batchrun_vectorized_REPARAM by default.
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))

from clust_huge_amp_vectorized_reparam import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model_essentials(base_path):
    print("Loading components...")
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_matrix_corrected.pt', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)
    print("Loaded all components successfully!")
    return Y, E, G, essentials


def load_covariates_data(csv_path):
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
    parser = argparse.ArgumentParser(description='Run Aladyn REPARAM model on batch')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--W', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    parser.add_argument('--covariates_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--include_pcs', type=bool, default=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}")
    print(f"Running Aladyn batch (REPARAM): samples {args.start_index} to {args.end_index}")
    print(f"{'='*60}\n")

    Y, E, G, essentials = load_model_essentials(args.data_dir)
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

    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)

    print(f"\nInitializing REPARAM model with K={args.K} clusters...")
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

    torch.manual_seed(0)
    np.random.seed(0)
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(args.data_dir + 'initial_clusters_400k.pt', weights_only=False)

    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters

    print(f"\nTraining REPARAM model for {args.num_epochs} epochs...")
    print(f"Learning rate: {args.learning_rate}, GP weight: {args.W}")
    history = model.fit(E_batch,
                       num_epochs=args.num_epochs,
                       learning_rate=args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'enrollment_model_REPARAM_W{args.W}_batch_{args.start_index}_{args.end_index}.pt'
    print(f"\nSaving model to {output_path}...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'phi': model.phi,
        'Y': model.Y,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'G': model.G,
        'args': vars(args),
        'indices': indices,
        'clusters': initial_clusters,
        'version': 'VECTORIZED_REPARAM',
    }, output_path)

    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {output_path}")
    print(f"{'='*60}\n")

    if history:
        losses, _ = history
        if losses:
            print(f"Final loss: {losses[-1]:.4f}")


if __name__ == '__main__':
    main()
