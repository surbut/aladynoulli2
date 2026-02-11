#!/usr/bin/env python
"""
Prediction script for REPARAM Aladyn model using enrollment data with fixed phi/psi, fixed gamma, and fixed kappa.
Uses the non-centered parameterization: lambda = mean(gamma) + delta.

This is the reparam counterpart to run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py.
Key difference: uses AladynSurvivalFixedPhiFixedGammaFixedKappaReparam which optimizes delta
(not lambda directly), and gamma enters the forward pass via lambda = mean(gamma) + delta.

Usage:
    python run_aladyn_predict_reparam_fixedgk.py \
        --pooled_reparam_path /path/to/pooled_phi_kappa_gamma_reparam.pt \
        --output_dir /path/to/output/ \
        --max_batches 5
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
import gc
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))

from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=FutureWarning)


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    return Y[indices], E[indices], G[indices], indices


def main():
    parser = argparse.ArgumentParser(description='Run REPARAM Aladyn predictions with fixed phi/psi/gamma/kappa')
    parser.add_argument('--pooled_reparam_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/pooled_phi_kappa_gamma_reparam.pt',
                        help='Path to pooled reparam params (phi, psi, gamma, kappa)')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_vectorized/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--max_batches', type=int, default=5)
    parser.add_argument('--start_batch', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"REPARAM Aladyn Prediction - Fixed Phi/Psi/Gamma/Kappa, optimize delta")
    print(f"{'='*80}")
    print(f"Pooled reparam params: {args.pooled_reparam_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    Y = torch.load(args.data_dir + 'Y_tensor.pt', weights_only=False)
    E = torch.load(args.data_dir + 'E_enrollment_full.pt', weights_only=False)
    G = torch.load(args.data_dir + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(args.data_dir + 'model_essentials.pt', weights_only=False)
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)
    fh_processed = pd.read_csv(args.covariates_path)

    # Load pooled reparam params
    print(f"Loading pooled reparam params...")
    pooled = torch.load(args.pooled_reparam_path, weights_only=False)
    phi = pooled['phi']
    gamma = pooled['gamma']
    kappa = pooled['kappa']
    if torch.is_tensor(phi):
        phi = phi.cpu().numpy()
    if torch.is_tensor(gamma):
        gamma = gamma.cpu().numpy()
    kappa = float(kappa) if not hasattr(kappa, 'item') else kappa.item()

    psi = pooled.get('psi')
    if psi is None:
        ip = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
        psi = ip.numpy() if torch.is_tensor(ip) else np.array(ip)
        if psi.shape[0] == 20:
            psi = np.vstack([psi, np.full((1, psi.shape[1]), -5.0)])
    elif torch.is_tensor(psi):
        psi = psi.cpu().numpy()

    print(f"  phi {phi.shape}, psi {psi.shape}, kappa {kappa:.4f}, gamma {gamma.shape}")
    del pooled
    gc.collect()

    # Generate batches
    total_samples = Y.shape[0]
    batches = []
    for start in range(0, total_samples, args.batch_size):
        end = min(start + args.batch_size, total_samples)
        batches.append((start, end))

    if args.start_batch > 0:
        batches = batches[args.start_batch:]
    if args.max_batches is not None:
        batches = batches[:args.max_batches]

    print(f"Will process {len(batches)} batches of {args.batch_size}")
    if batches:
        print(f"Samples: {batches[0][0]} to {batches[-1][1]}")

    successful_batches = []

    for batch_idx, (start, stop) in enumerate(batches):
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx+1}/{len(batches)}: samples {start}-{stop}")
        print(f"{'='*80}")

        pi_filename = os.path.join(args.output_dir, f"pi_enroll_fixedphi_sex_{start}_{stop}.pt")
        if os.path.exists(pi_filename):
            print(f"Already exists, skipping: {pi_filename}")
            successful_batches.append((start, stop))
            continue

        batch_start_time = time.time()

        try:
            torch.manual_seed(42)
            np.random.seed(42)

            Y_batch, E_batch, G_batch, _ = subset_data(Y, E, G, start, stop)

            # Add sex + PCs
            pce_df_subset = fh_processed.iloc[start:stop].reset_index(drop=True)
            sex = pce_df_subset['sex'].values
            pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
            pcs = pce_df_subset[pc_cols].values
            G_with_sex = np.column_stack([G_batch, sex, pcs])

            print(f"Data: Y={Y_batch.shape}, E={E_batch.shape}, G={G_with_sex.shape}")

            # Initialize REPARAM model
            model = AladynSurvivalFixedPhiFixedGammaFixedKappaReparam(
                N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
                K=20, P=G_with_sex.shape[1],
                G=G_with_sex, Y=Y_batch,
                R=0, W=0.0001, prevalence_t=prevalence_t,
                init_sd_scaler=1e-1, genetic_scale=1,
                pretrained_phi=phi, pretrained_psi=psi,
                pretrained_gamma=gamma, pretrained_kappa=kappa,
                signature_references=signature_refs, healthy_reference=True,
                disease_names=essentials['disease_names'],
            )

            # Fit delta (reparam forward: lambda = mean(gamma) + delta)
            print(f"Training (optimizing delta, {args.num_epochs} epochs, lr={args.learning_rate})...")
            result = model.fit(E_batch, num_epochs=args.num_epochs, learning_rate=args.learning_rate)
            losses = result[0] if isinstance(result, tuple) else result
            print(f"  Final loss: {losses[-1]:.4f}")

            # Extract pi
            with torch.no_grad():
                pi, _, _ = model.forward()

            # Check for NaN
            n_nan = torch.isnan(pi).sum().item()
            if n_nan > 0:
                print(f"  WARNING: {n_nan} NaN in pi -- clamping")
                pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
                pi = torch.clamp(pi, 1e-8, 1 - 1e-8)

            torch.save(pi, pi_filename)
            print(f"Saved: {pi_filename}")

            successful_batches.append((start, stop))

            del pi, model, Y_batch, E_batch, G_batch, G_with_sex
            gc.collect()

            batch_time = (time.time() - batch_start_time) / 60.0
            print(f"Batch {batch_idx+1} done ({batch_time:.1f} min)")

        except Exception as e:
            print(f"ERROR batch {start}-{stop}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"DONE! {len(successful_batches)}/{len(batches)} batches successful")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
