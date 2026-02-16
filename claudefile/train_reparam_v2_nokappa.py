#!/usr/bin/env python
"""
Reparam training WITHOUT kappa (kappa fixed at 1.0, not learned).
Same as train_reparam_v2.py but with learn_kappa=False.

pi = einsum(theta, phi_prob) — no kappa scaling, pi naturally in [0,1].

Usage:
    python train_reparam_v2_nokappa.py --start_index 0 --end_index 10000 --num_epochs 500
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import argparse
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))
from clust_huge_amp_vectorized_reparam import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fit_v2(model, event_times, num_epochs=500, learning_rate=0.1,
           grad_clip=5.0, patience=75, min_improvement=1e-4):
    """Improved training loop with cosine annealing LR and gradient clipping."""
    param_groups = [
        {'params': [model.delta], 'lr': learning_rate, 'label': 'delta'},
        {'params': [model.epsilon], 'lr': learning_rate * 0.1, 'label': 'epsilon'},
        {'params': [model.psi], 'lr': learning_rate * 0.1, 'label': 'psi'},
        {'params': [model.gamma], 'lr': learning_rate, 'label': 'gamma'},
    ]
    # No kappa in optimizer — it's fixed at 1.0

    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )

    losses = []
    best_loss = float('inf')
    best_epoch = 0
    best_state = None

    all_params = []
    for pg in param_groups:
        all_params.extend(pg['params'])

    t0 = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model.compute_loss(event_times)

        if torch.isnan(loss):
            print(f"Epoch {epoch}: Loss=nan (stopping, reverting to best)")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        cur_loss = loss.item()
        losses.append(cur_loss)

        if cur_loss < best_loss * (1 - min_improvement):
            best_loss = cur_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = (time.time() - t0) / 60
            gamma_mag = model.gamma.abs().mean().item()
            print(f"Epoch {epoch:4d}: Loss={cur_loss:.4f}, LR={lr_now:.1e}, "
                  f"kappa=1.0 (fixed), |gamma|={gamma_mag:.4f}, "
                  f"time={elapsed:.1f}min")

        if epoch - best_epoch > patience and epoch > 100:
            print(f"Epoch {epoch}: Early stop (no improvement for {patience} epochs, "
                  f"best={best_loss:.4f} at epoch {best_epoch})")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    # Restore best state
    if best_state is not None and len(losses) > 0 and losses[-1] > best_loss * 1.001:
        print(f"Restoring best model from epoch {best_epoch} (loss {best_loss:.4f})")
        model.load_state_dict(best_state)

    return losses


def load_model_essentials(base_path):
    Y = torch.load(base_path + 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path + 'E_matrix_corrected.pt', weights_only=False)
    G = torch.load(base_path + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path + 'model_essentials.pt', weights_only=False)
    return Y, E, G, essentials


def main():
    parser = argparse.ArgumentParser(description='Train Aladyn REPARAM v2 NO KAPPA')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=75)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--W', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n{'='*60}")
    print(f"REPARAM v2 NO KAPPA: samples {args.start_index} to {args.end_index}")
    print(f"Epochs: {args.num_epochs}, LR: {args.learning_rate}, grad_clip: {args.grad_clip}")
    print(f"kappa = 1.0 (FIXED, not learned)")
    print(f"{'='*60}\n")

    Y, E, G, essentials = load_model_essentials(args.data_dir)
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G,
                                                      start_index=args.start_index,
                                                      end_index=args.end_index)
    del Y

    fh_processed = pd.read_csv(args.covariates_path)
    if 'Sex' in fh_processed.columns:
        sex = fh_processed['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
    else:
        sex = fh_processed['sex'].values
    sex_batch = sex[args.start_index:args.end_index]
    pc_columns = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh_processed.iloc[args.start_index:args.end_index][pc_columns].values
    G_with_sex = np.column_stack([G_batch, sex_batch, pcs])
    print(f"G_with_sex shape: {G_with_sex.shape}")

    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)

    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
        K=args.K, P=G_with_sex.shape[1],
        init_sd_scaler=1e-1, G=G_with_sex, Y=Y_batch,
        genetic_scale=1, W=args.W, R=0,
        prevalence_t=prevalence_t, signature_references=signature_refs,
        healthy_reference=True, disease_names=essentials['disease_names'],
        learn_kappa=False,  # <-- THE ONLY DIFFERENCE: kappa fixed at 1.0
    )

    torch.manual_seed(0)
    np.random.seed(0)
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(args.data_dir + 'initial_clusters_400k.pt', weights_only=False)
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters

    print(f"\nTraining with cosine annealing + gradient clipping (NO KAPPA)...")
    t0 = time.time()
    losses = fit_v2(model, E_batch,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    grad_clip=args.grad_clip,
                    patience=args.patience)
    elapsed = (time.time() - t0) / 60
    print(f"\nTraining complete: {len(losses)} epochs in {elapsed:.1f} min")
    print(f"Final loss: {losses[-1]:.4f}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'enrollment_model_REPARAM_NOKAPPA_W{args.W}_batch_{args.start_index}_{args.end_index}.pt'

    gamma_mag = model.gamma.abs().mean().item()
    print(f"\nFinal params: kappa=1.0 (fixed), mean|gamma|={gamma_mag:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'phi': model.phi,
        'psi': model.psi,
        'gamma': model.gamma,
        'kappa': torch.ones(1),
        'Y': model.Y,
        'prevalence_t': model.prevalence_t,
        'logit_prevalence_t': model.logit_prev_t,
        'G': model.G,
        'args': vars(args),
        'indices': indices,
        'clusters': initial_clusters,
        'losses': losses,
        'version': 'VECTORIZED_REPARAM_v2_NOKAPPA',
    }, output_path)

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
