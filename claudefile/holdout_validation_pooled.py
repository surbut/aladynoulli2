#!/usr/bin/env python
"""
Holdout validation with pooled phi, kappa, gamma.
Train 39 batches, pool phi/kappa/gamma, then evaluate on holdout with fixed params.
Uses fixedgk model: only lambda is fitted on holdout.

Prerequisites:
  1. Run pool_phi_kappa_gamma_from_batches.py for both nolr and reparam
  2. Requires pooled_phi_kappa_gamma_nolr.pt and pooled_phi_kappa_gamma_reparam.pt

Usage:
    python holdout_validation_pooled.py --holdout_start 390000 --holdout_end 400000
"""
import numpy as np
import torch
import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa import (
    AladynSurvivalFixedPhiFixedGammaFixedKappa,
)
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    return Y[indices], E[indices], G[indices], indices


def load_model_essentials(base_path):
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_matrix_corrected.pt', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)
    return Y, E, G, essentials


def load_pooled(pooled_path, data_dir):
    """Load pooled params, with psi fallback to initial_psi."""
    data = torch.load(pooled_path, weights_only=False)
    phi = data['phi']
    kappa = float(data['kappa']) if not hasattr(data['kappa'], 'item') else data['kappa'].item()
    gamma = data['gamma']
    if torch.is_tensor(gamma):
        gamma = gamma.numpy()
    if torch.is_tensor(phi):
        phi = phi.numpy()

    psi = data.get('psi')
    if psi is None:
        ip = torch.load(Path(data_dir) / 'initial_psi_400k.pt', weights_only=False)
        psi = ip.numpy() if torch.is_tensor(ip) else np.array(ip)
        if psi.shape[0] == 20:
            healthy = np.full((1, psi.shape[1]), -5.0)
            psi = np.vstack([psi, healthy])
    elif torch.is_tensor(psi):
        psi = psi.numpy()

    return phi, psi, kappa, gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout_start', type=int, default=390000)
    parser.add_argument('--holdout_end', type=int, default=400000)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--pooled_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pooled_dir = Path(args.pooled_dir)
    nolr_path = pooled_dir / 'pooled_phi_kappa_gamma_nolr.pt'
    reparam_path = pooled_dir / 'pooled_phi_kappa_gamma_reparam.pt'

    if not nolr_path.exists():
        print(f"Run: python pool_phi_kappa_gamma_from_batches.py --model_type nolr --max_batches 39")
        print(f"Expected: {nolr_path}")
        return
    if not reparam_path.exists():
        print(f"Run: python pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39")
        print(f"Expected: {reparam_path}")
        return

    print("Loading data...")
    Y, E, G, essentials = load_model_essentials(str(data_dir) + '/')
    Y_holdout, E_holdout, G_holdout, _ = subset_data(
        Y, E, G, args.holdout_start, args.holdout_end
    )
    del Y, E, G

    fh = pd.read_csv(args.covariates_path)
    if 'Sex' in fh.columns:
        sex = fh['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
    else:
        sex = fh['sex'].values
    sex_h = sex[args.holdout_start:args.holdout_end]
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh.iloc[args.holdout_start:args.holdout_end][pc_cols].values
    G_holdout = np.column_stack([G_holdout, sex_h, pcs])

    refs = torch.load(str(data_dir) + '/reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(str(data_dir) + '/prevalence_t_corrected.pt', weights_only=False)

    K = 20
    P = G_holdout.shape[1]

    def run_holdout(phi, psi, kappa, gamma, name, use_reparam=False):
        torch.manual_seed(42)
        np.random.seed(42)
        ModelClass = AladynSurvivalFixedPhiFixedGammaFixedKappaReparam if use_reparam else AladynSurvivalFixedPhiFixedGammaFixedKappa
        model = ModelClass(
            N=Y_holdout.shape[0], D=Y_holdout.shape[1], T=Y_holdout.shape[2],
            K=K, P=P, G=G_holdout, Y=Y_holdout,
            R=0, W=0.0001, prevalence_t=prevalence_t,
            init_sd_scaler=1e-1, genetic_scale=1,
            pretrained_phi=phi, pretrained_psi=psi,
            pretrained_gamma=gamma, pretrained_kappa=kappa,
            signature_references=signature_refs, healthy_reference=True,
            disease_names=essentials['disease_names']
        )
        result = model.fit(E_holdout, num_epochs=args.num_epochs, learning_rate=args.learning_rate)
        losses = result[0] if isinstance(result, tuple) else result
        return losses[-1] if losses else float('nan')

    print("\n" + "="*60)
    print("HOLDOUT VALIDATION (fixed phi, kappa, gamma from pooled batches)")
    print("="*60)
    print(f"Holdout: samples {args.holdout_start}-{args.holdout_end}")
    print("")

    phi_n, psi_n, kappa_n, gamma_n = load_pooled(nolr_path, data_dir)
    phi_r, psi_r, kappa_r, gamma_r = load_pooled(reparam_path, data_dir)

    print("Evaluating NOLR pooled params (optimize lambda)...")
    loss_nolr = run_holdout(phi_n, psi_n, kappa_n, gamma_n, "nolr", use_reparam=False)
    print(f"  Nolr holdout loss: {loss_nolr:.6f}")

    print("Evaluating REPARAM pooled params (optimize delta, lambda=mean(gamma)+delta)...")
    loss_reparam = run_holdout(phi_r, psi_r, kappa_r, gamma_r, "reparam", use_reparam=True)
    print(f"  Reparam holdout loss: {loss_reparam:.6f}")

    print("")
    print("="*60)
    if loss_reparam < loss_nolr:
        print("-> Reparam generalizes better (lower holdout loss)")
    else:
        print("-> Nolr generalizes better (lower holdout loss)")
    print("="*60)


if __name__ == '__main__':
    main()
