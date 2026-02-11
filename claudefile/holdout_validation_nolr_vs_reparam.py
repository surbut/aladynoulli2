#!/usr/bin/env python
"""
Holdout validation: train on batch 0-10000, evaluate on holdout batch with gamma, phi, kappa fixed.
Tests whether reparam's lower training loss generalizes or is overfitting.
"""
import numpy as np
import torch
import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))

from clust_huge_amp_vectorized_nolr import (
    AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest as ModelNolr,
    subset_data,
)
from clust_huge_amp_vectorized_reparam import (
    AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest as ModelReparam,
)


def load_model_essentials(base_path):
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_matrix_corrected.pt', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)
    return Y, E, G, essentials


def load_covariates_data(csv_path):
    fh_processed = pd.read_csv(csv_path)
    if 'Sex' in fh_processed.columns:
        fh_processed['sex_numeric'] = fh_processed['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
        sex = fh_processed['sex_numeric'].values
    else:
        sex = fh_processed['sex'].values
    return sex, fh_processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=10000)
    parser.add_argument('--holdout_start', type=int, default=10000)
    parser.add_argument('--holdout_end', type=int, default=20000)
    parser.add_argument('--W', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--nolr_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr')
    parser.add_argument('--reparam_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    args = parser.parse_args()

    # Load full data
    print("Loading data...")
    Y, E, G, essentials = load_model_essentials(args.data_dir)

    # Holdout batch
    Y_holdout, E_holdout, G_holdout, _ = subset_data(
        Y, E, G, start_index=args.holdout_start, end_index=args.holdout_end
    )
    del Y, E, G

    sex, fh_processed = load_covariates_data(args.covariates_path)
    sex_holdout = sex[args.holdout_start:args.holdout_end]
    pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
                  'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
    pcs_holdout = fh_processed.iloc[args.holdout_start:args.holdout_end][pc_columns].values
    G_holdout = np.column_stack([G_holdout, sex_holdout, pcs_holdout])

    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(args.data_dir + 'initial_clusters_400k.pt', weights_only=False)

    K = 20
    P = G_holdout.shape[1]

    # Build models with HOLDOUT data
    print(f"Building models with holdout data (samples {args.holdout_start}-{args.holdout_end})...")
    model_nolr = ModelNolr(
        N=Y_holdout.shape[0], D=Y_holdout.shape[1], T=Y_holdout.shape[2],
        K=K, P=P, init_sd_scaler=1e-1, G=G_holdout, Y=Y_holdout,
        genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    model_nolr.initialize_params(true_psi=initial_psi)
    model_nolr.clusters = initial_clusters

    model_reparam = ModelReparam(
        N=Y_holdout.shape[0], D=Y_holdout.shape[1], T=Y_holdout.shape[2],
        K=K, P=P, init_sd_scaler=1e-1, G=G_holdout, Y=Y_holdout,
        genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    model_reparam.initialize_params(true_psi=initial_psi)
    model_reparam.clusters = initial_clusters

    # Load trained params (gamma, phi, kappa, etc.) - trained on 0-10000
    nolr_path = Path(args.nolr_dir) / f'enrollment_model_VECTORIZED_W{args.W}_nolr_batch_{args.train_start}_{args.train_end}.pt'
    reparam_path = Path(args.reparam_dir) / f'enrollment_model_REPARAM_W{args.W}_batch_{args.train_start}_{args.train_end}.pt'

    if not nolr_path.exists() or not reparam_path.exists():
        print(f"Checkpoints not found: {nolr_path} / {reparam_path}")
        return

    ckpt_nolr = torch.load(nolr_path, weights_only=False)
    ckpt_reparam = torch.load(reparam_path, weights_only=False)

    # Load state_dict - this sets gamma, phi, psi, lambda_/delta/epsilon, kappa
    model_nolr.load_state_dict(ckpt_nolr['model_state_dict'], strict=False)
    model_reparam.load_state_dict(ckpt_reparam['model_state_dict'], strict=False)

    model_nolr.eval()
    model_reparam.eval()

    # Evaluate on holdout (gamma, phi, kappa fixed from training)
    with torch.no_grad():
        loss_nolr = model_nolr.compute_loss(E_holdout).item()
        loss_reparam = model_reparam.compute_loss(E_holdout).item()

    # Also get training loss for reference (from checkpoint metadata or we'd need to reload train data)
    # We'll load train batch just for train loss comparison
    print("\nLoading training batch for train-loss comparison...")
    Y_full = torch.load(args.data_dir + 'Y_tensor.pt', weights_only=False)
    E_full = torch.load(args.data_dir + 'E_matrix_corrected.pt', weights_only=False)
    G_full = torch.load(args.data_dir + 'G_matrix.pt', weights_only=False)
    Y_train, E_train, G_train, _ = subset_data(Y_full, E_full, G_full,
                                               start_index=args.train_start, end_index=args.train_end)
    sex_train = sex[args.train_start:args.train_end]
    pcs_train = fh_processed.iloc[args.train_start:args.train_end][pc_columns].values
    G_train = np.column_stack([G_train, sex_train, pcs_train])

    model_nolr_train = ModelNolr(
        N=Y_train.shape[0], D=Y_train.shape[1], T=Y_train.shape[2],
        K=K, P=P, init_sd_scaler=1e-1, G=G_train, Y=Y_train,
        genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    model_nolr_train.initialize_params(true_psi=initial_psi)
    model_nolr_train.clusters = initial_clusters
    model_nolr_train.load_state_dict(ckpt_nolr['model_state_dict'], strict=False)
    model_nolr_train.eval()

    model_reparam_train = ModelReparam(
        N=Y_train.shape[0], D=Y_train.shape[1], T=Y_train.shape[2],
        K=K, P=P, init_sd_scaler=1e-1, G=G_train, Y=Y_train,
        genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    model_reparam_train.initialize_params(true_psi=initial_psi)
    model_reparam_train.clusters = initial_clusters
    model_reparam_train.load_state_dict(ckpt_reparam['model_state_dict'], strict=False)
    model_reparam_train.eval()

    with torch.no_grad():
        train_loss_nolr = model_nolr_train.compute_loss(E_train).item()
        train_loss_reparam = model_reparam_train.compute_loss(E_train).item()

    print("\n" + "="*60)
    print("HOLDOUT VALIDATION (gamma, phi, kappa fixed from training)")
    print("="*60)
    print(f"  Train batch: {args.train_start}-{args.train_end}")
    print(f"  Holdout batch: {args.holdout_start}-{args.holdout_end}")
    print("")
    print("  TRAIN loss:")
    print(f"    Nolr:    {train_loss_nolr:.6f}")
    print(f"    Reparam: {train_loss_reparam:.6f}")
    print("")
    print("  HOLDOUT loss:")
    print(f"    Nolr:    {loss_nolr:.6f}")
    print(f"    Reparam: {loss_reparam:.6f}")
    print("")
    print("  Generalization gap (holdout - train):")
    print(f"    Nolr:    {loss_nolr - train_loss_nolr:.6f}")
    print(f"    Reparam: {loss_reparam - train_loss_reparam:.6f}")
    print("")
    if loss_reparam < loss_nolr:
        print("  -> Reparam generalizes better (lower holdout loss)")
    else:
        print("  -> Nolr generalizes better (lower holdout loss)")
    if loss_reparam - train_loss_reparam > loss_nolr - train_loss_nolr:
        print("  -> Reparam has larger generalization gap (more overfitting)")
    print("="*60)


if __name__ == '__main__':
    main()
