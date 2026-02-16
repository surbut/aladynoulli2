#!/usr/bin/env python3
"""
Leave-one-out (LOO) prediction pipeline for nolr and reparam.

For each prediction batch i (default: first 5 batches = 50k patients):
  1. Pool phi, psi, kappa, gamma from ALL training batches EXCEPT batch i
  2. Fit nolr model (optimize lambda) with LOO-pooled params
  3. Fit reparam model (optimize delta) with LOO-pooled params
  4. Save pi tensors

This eliminates any data leakage from the prediction batch into pooled params.

Usage:
    python run_loo_predict_both.py --n_pred_batches 5
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
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa import (
    AladynSurvivalFixedPhiFixedGammaFixedKappa,
)
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_all_checkpoints(pattern, max_batches=None):
    """Load phi, psi, kappa, gamma from all batch checkpoints into lists."""
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
                       num_epochs, learning_rate, use_reparam):
    """Fit model with fixed params and extract pi."""
    torch.manual_seed(42)
    np.random.seed(42)

    N, D, T = Y_batch.shape
    K = phi.shape[0] - 1 if phi.shape[0] == 21 else phi.shape[0]
    P = G_with_sex.shape[1]

    if use_reparam:
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
    else:
        model = AladynSurvivalFixedPhiFixedGammaFixedKappa(
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
        print(f"    WARNING: {n_nan} NaN in pi — clamping")
        pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
        pi = torch.clamp(pi, 1e-8, 1 - 1e-8)

    final_loss = losses[-1] if len(losses) > 0 else float('nan')
    return pi, final_loss


def main():
    parser = argparse.ArgumentParser(description='LOO prediction for nolr and reparam')
    parser.add_argument('--n_pred_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--n_train_batches', type=int, default=40,
                        help='Total training batches to load (default 40)')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--nolr_train_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr')
    parser.add_argument('--reparam_train_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--output_base', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix for output dirs (e.g. "_v2" → nolr_loo_v2/)')
    args = parser.parse_args()

    suffix = args.output_suffix
    nolr_out = os.path.join(args.output_base, f'enrollment_predictions_fixedphi_fixedgk_nolr_loo{suffix}/')
    reparam_out = os.path.join(args.output_base, f'enrollment_predictions_fixedphi_fixedgk_reparam_loo{suffix}/')
    os.makedirs(nolr_out, exist_ok=True)
    os.makedirs(reparam_out, exist_ok=True)

    print("=" * 80)
    print("LEAVE-ONE-OUT PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Prediction batches: {args.n_pred_batches} (first {args.n_pred_batches * args.batch_size} patients)")
    print(f"Training batches to load: {args.n_train_batches}")
    print(f"Output: nolr → {nolr_out}")
    print(f"        reparam → {reparam_out}")
    print("=" * 80)

    # ================================================================
    # Load data
    # ================================================================
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

    # ================================================================
    # Load all training checkpoints
    # ================================================================
    print("\nLoading NOLR training checkpoints...")
    nolr_pattern = str(Path(args.nolr_train_dir) / 'enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt')
    nolr_phi, nolr_psi, nolr_kappa, nolr_gamma, nolr_files = \
        load_all_checkpoints(nolr_pattern, args.n_train_batches)

    print("\nLoading REPARAM training checkpoints...")
    reparam_pattern = str(Path(args.reparam_train_dir) / 'enrollment_model_REPARAM_W0.0001_batch_*_*.pt')
    reparam_phi, reparam_psi, reparam_kappa, reparam_gamma, reparam_files = \
        load_all_checkpoints(reparam_pattern, args.n_train_batches)

    # Verify we have enough batches
    n_nolr = len(nolr_phi)
    n_reparam = len(reparam_phi)
    print(f"\nLoaded: {n_nolr} nolr, {n_reparam} reparam checkpoints")

    if n_nolr < args.n_pred_batches or n_reparam < args.n_pred_batches:
        print("ERROR: Not enough training checkpoints for LOO")
        return

    # Show full-pool vs what LOO will produce
    full_kappa_nolr = float(np.mean(nolr_kappa))
    full_kappa_reparam = float(np.mean(reparam_kappa))
    print(f"\nFull-pool kappa: nolr={full_kappa_nolr:.4f}, reparam={full_kappa_reparam:.4f}")
    print(f"Full-pool mean|gamma|: nolr={np.abs(np.mean(np.stack(nolr_gamma), axis=0)).mean():.4f}, "
          f"reparam={np.abs(np.mean(np.stack(reparam_gamma), axis=0)).mean():.4f}")

    # ================================================================
    # LOO prediction loop
    # ================================================================
    for batch_idx in range(args.n_pred_batches):
        start = batch_idx * args.batch_size
        stop = (batch_idx + 1) * args.batch_size

        print(f"\n{'=' * 80}")
        print(f"BATCH {batch_idx + 1}/{args.n_pred_batches}: samples {start}-{stop} (LOO: exclude batch {batch_idx})")
        print(f"{'=' * 80}")

        # Check if both already exist
        nolr_pi_path = os.path.join(nolr_out, f'pi_enroll_fixedphi_sex_{start}_{stop}.pt')
        reparam_pi_path = os.path.join(reparam_out, f'pi_enroll_fixedphi_sex_{start}_{stop}.pt')

        if os.path.exists(nolr_pi_path) and os.path.exists(reparam_pi_path):
            print(f"  Both already exist, skipping")
            continue

        # Subset data
        Y_batch = Y[start:stop]
        E_batch = E[start:stop]
        G_batch = G[start:stop]

        pce_subset = fh_processed.iloc[start:stop].reset_index(drop=True)
        sex = pce_subset['sex'].values
        pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
        pcs = pce_subset[pc_cols].values
        G_with_sex = np.column_stack([G_batch, sex, pcs])

        print(f"  Data: Y={Y_batch.shape}, G_with_sex={G_with_sex.shape}")

        # ---- NOLR LOO ----
        if not os.path.exists(nolr_pi_path):
            print(f"\n  NOLR LOO (exclude batch {batch_idx}, pool from {n_nolr - 1} batches)...")
            phi_n, psi_n, kappa_n, gamma_n = loo_pool(nolr_phi, nolr_psi, nolr_kappa, nolr_gamma, batch_idx)
            print(f"    LOO kappa={kappa_n:.4f}, mean|gamma|={np.abs(gamma_n).mean():.4f}")

            t0 = time.time()
            pi_nolr, loss_nolr = fit_and_extract_pi(
                Y_batch, E_batch, G_with_sex, phi_n, psi_n, kappa_n, gamma_n,
                signature_refs, prevalence_t, disease_names,
                args.num_epochs, args.learning_rate, use_reparam=False)
            elapsed = (time.time() - t0) / 60
            print(f"    Loss: {loss_nolr:.4f}, Time: {elapsed:.1f} min")

            torch.save(pi_nolr, nolr_pi_path)
            print(f"    Saved: {nolr_pi_path}")
            del pi_nolr, phi_n, psi_n, gamma_n
            gc.collect()
        else:
            print(f"  NOLR already exists: {nolr_pi_path}")

        # ---- REPARAM LOO ----
        if not os.path.exists(reparam_pi_path):
            print(f"\n  REPARAM LOO (exclude batch {batch_idx}, pool from {n_reparam - 1} batches)...")
            phi_r, psi_r, kappa_r, gamma_r = loo_pool(reparam_phi, reparam_psi, reparam_kappa, reparam_gamma, batch_idx)
            print(f"    LOO kappa={kappa_r:.4f}, mean|gamma|={np.abs(gamma_r).mean():.4f}")

            t0 = time.time()
            pi_reparam, loss_reparam = fit_and_extract_pi(
                Y_batch, E_batch, G_with_sex, phi_r, psi_r, kappa_r, gamma_r,
                signature_refs, prevalence_t, disease_names,
                args.num_epochs, args.learning_rate, use_reparam=True)
            elapsed = (time.time() - t0) / 60
            print(f"    Loss: {loss_reparam:.4f}, Time: {elapsed:.1f} min")

            torch.save(pi_reparam, reparam_pi_path)
            print(f"    Saved: {reparam_pi_path}")
            del pi_reparam, phi_r, psi_r, gamma_r
            gc.collect()
        else:
            print(f"  REPARAM already exists: {reparam_pi_path}")

    print(f"\n{'=' * 80}")
    print("LOO PREDICTION COMPLETE")
    print(f"  NOLR output:    {nolr_out}")
    print(f"  REPARAM output:  {reparam_out}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
