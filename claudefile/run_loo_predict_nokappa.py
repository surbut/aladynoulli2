#!/usr/bin/env python3
"""
Leave-one-out (LOO) prediction pipeline for nokappa reparam.

For each prediction batch i (default: first 5 batches = 50k patients):
  1. Pool phi, psi, gamma from ALL training batches EXCEPT batch i (kappa=1 fixed)
  2. Fit reparam model (optimize delta) with LOO-pooled params
  3. Save pi tensors

Usage:
    python run_loo_predict_nokappa.py --n_pred_batches 5
"""

import argparse
import gc
import glob
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_all_checkpoints(pattern, max_batches=None):
    """Load phi, psi, kappa, gamma from all batch checkpoints."""
    files = sorted(glob.glob(pattern))
    if max_batches is not None:
        files = files[:max_batches]
    print(f"  Found {len(files)} checkpoints matching pattern")

    all_phi, all_psi, all_kappa, all_gamma = [], [], [], []
    loaded_files = []

    def _extract(ckpt, name):
        if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
            return ckpt['model_state_dict'][name]
        if name in ckpt:
            return ckpt[name]
        return None

    for fp in files:
        try:
            ckpt = torch.load(fp, weights_only=False)
            phi = _extract(ckpt, 'phi')
            if phi is None:
                continue
            phi = phi.detach().cpu().numpy() if torch.is_tensor(phi) else np.array(phi)

            psi = _extract(ckpt, 'psi')
            if psi is not None:
                psi = psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.array(psi)

            kappa = _extract(ckpt, 'kappa')
            k = kappa.item() if torch.is_tensor(kappa) else float(kappa) if kappa is not None else 1.0

            gamma = _extract(ckpt, 'gamma')
            if gamma is None:
                continue
            gamma = gamma.detach().cpu().numpy() if torch.is_tensor(gamma) else np.array(gamma)

            all_phi.append(phi)
            all_psi.append(psi)
            all_kappa.append(k)
            all_gamma.append(gamma)
            loaded_files.append(Path(fp).name)
        except Exception as e:
            print(f"  Error loading {Path(fp).name}: {e}")
            continue

    print(f"  Loaded {len(all_phi)} checkpoints")
    return all_phi, all_psi, all_kappa, all_gamma, loaded_files


def loo_pool(all_phi, all_psi, all_kappa, all_gamma, exclude_idx):
    """Compute LOO-pooled params by averaging all batches except exclude_idx."""
    n = len(all_phi)
    indices = [j for j in range(n) if j != exclude_idx]

    phi = np.mean(np.stack([all_phi[j] for j in indices]), axis=0)
    kappa = float(np.mean([all_kappa[j] for j in indices]))
    gamma = np.mean(np.stack([all_gamma[j] for j in indices]), axis=0)

    psi_valid = [all_psi[j] for j in indices if all_psi[j] is not None]
    psi = np.mean(np.stack(psi_valid), axis=0) if psi_valid else None

    return phi, psi, kappa, gamma


def fit_and_extract_pi(Y_batch, E_batch, G_with_sex, phi, psi, kappa, gamma,
                       signature_refs, prevalence_t, disease_names,
                       num_epochs, learning_rate):
    """Fit reparam model with fixed params and extract pi."""
    torch.manual_seed(42)
    np.random.seed(42)

    N, D, T = Y_batch.shape
    K = phi.shape[0] - 1 if phi.shape[0] == 21 else phi.shape[0]
    P = G_with_sex.shape[1]

    model = AladynSurvivalFixedPhiFixedGammaFixedKappaReparam(
        N=N, D=D, T=T, K=K, P=P,
        G=G_with_sex, Y=Y_batch,
        R=0, W=0.0001, prevalence_t=prevalence_t,
        init_sd_scaler=1e-1, genetic_scale=1,
        pretrained_phi=phi, pretrained_psi=psi,
        pretrained_gamma=gamma, pretrained_kappa=kappa,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=disease_names,
    )

    result = model.fit(E_batch, num_epochs=num_epochs, learning_rate=learning_rate)
    losses = result[0] if isinstance(result, tuple) else result

    with torch.no_grad():
        pi, _, _ = model.forward()

    n_nan = torch.isnan(pi).sum().item()
    if n_nan > 0:
        print(f"    WARNING: {n_nan} NaN in pi -- clamping")
        pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
        pi = torch.clamp(pi, 1e-8, 1 - 1e-8)

    final_loss = losses[-1] if len(losses) > 0 else float('nan')
    return pi, final_loss


def main():
    parser = argparse.ArgumentParser(description='LOO prediction for nokappa reparam')
    parser.add_argument('--n_pred_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--n_train_batches', type=int, default=40)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--nokappa_train_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nokappa_loo/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LEAVE-ONE-OUT PREDICTION: NOKAPPA REPARAM")
    print("=" * 80)
    print(f"Prediction batches: {args.n_pred_batches}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    Y = torch.load(args.data_dir + 'Y_tensor.pt', weights_only=False)
    E = torch.load(args.data_dir + 'E_enrollment_full.pt', weights_only=False)
    G = torch.load(args.data_dir + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(args.data_dir + 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)
    fh_processed = pd.read_csv(args.covariates_path)
    print(f"Y: {Y.shape}, E: {E.shape}, G: {G.shape}")

    # Load nokappa training checkpoints
    print("\nLoading NOKAPPA training checkpoints...")
    pattern = str(Path(args.nokappa_train_dir) / 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_*_*.pt')
    nk_phi, nk_psi, nk_kappa, nk_gamma, nk_files = \
        load_all_checkpoints(pattern, args.n_train_batches)

    n_loaded = len(nk_phi)
    print(f"\nLoaded {n_loaded} nokappa checkpoints")
    print(f"Mean kappa (should be ~1.0): {np.mean(nk_kappa):.4f}")
    print(f"Mean |gamma|: {np.abs(np.mean(np.stack(nk_gamma), axis=0)).mean():.4f}")

    if n_loaded < args.n_pred_batches:
        print(f"ERROR: Need at least {args.n_pred_batches} checkpoints, got {n_loaded}")
        return

    # LOO prediction loop
    for batch_idx in range(args.n_pred_batches):
        start = batch_idx * args.batch_size
        stop = (batch_idx + 1) * args.batch_size

        print(f"\n{'=' * 80}")
        print(f"BATCH {batch_idx + 1}/{args.n_pred_batches}: samples {start}-{stop}")
        print(f"  LOO: exclude batch {batch_idx}, pool from {n_loaded - 1} batches")
        print(f"{'=' * 80}")

        pi_path = os.path.join(args.output_dir, f'pi_enroll_fixedphi_sex_{start}_{stop}.pt')
        if os.path.exists(pi_path):
            print(f"  Already exists, skipping: {pi_path}")
            continue

        # Subset data
        Y_batch = Y[start:stop]
        E_batch = E[start:stop]
        G_batch = G[start:stop]

        pce_subset = fh_processed.iloc[start:stop].reset_index(drop=True)
        if 'Sex' in pce_subset.columns:
            sex = pce_subset['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
        else:
            sex = pce_subset['sex'].values
        pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
        pcs = pce_subset[pc_cols].values
        G_with_sex = np.column_stack([G_batch, sex, pcs])

        print(f"  Data: Y={Y_batch.shape}, G_with_sex={G_with_sex.shape}")

        # LOO pool
        phi, psi, kappa, gamma = loo_pool(nk_phi, nk_psi, nk_kappa, nk_gamma, batch_idx)
        print(f"  LOO kappa={kappa:.4f}, mean|gamma|={np.abs(gamma).mean():.4f}")

        # Fit and extract pi
        t0 = time.time()
        pi, loss = fit_and_extract_pi(
            Y_batch, E_batch, G_with_sex, phi, psi, kappa, gamma,
            signature_refs, prevalence_t, disease_names,
            args.num_epochs, args.learning_rate)
        elapsed = (time.time() - t0) / 60
        print(f"  Loss: {loss:.4f}, Time: {elapsed:.1f} min")

        torch.save(pi, pi_path)
        print(f"  Saved: {pi_path}")
        del pi
        gc.collect()

    print(f"\n{'=' * 80}")
    print("NOKAPPA LOO PREDICTION COMPLETE")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
