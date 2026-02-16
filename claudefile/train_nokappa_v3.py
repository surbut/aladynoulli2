#!/usr/bin/env python
"""
Nokappa v3 training: constant LR, no cosine, no clipping, 300 epochs.
Grid search winner recipe applied to no-kappa reparam.

Usage:
    python claudefile/train_nokappa_v3.py --start_index 0 --end_index 10000 --W 0.0001
"""

import numpy as np
import torch
import torch.optim as optim
import warnings
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))
from clust_huge_amp_vectorized_reparam import *
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fit_v3(model, event_times, num_epochs=300, learning_rate=0.1):
    """Constant LR training — no cosine, no clipping, no early stopping."""
    param_groups = [
        {'params': [model.delta], 'lr': learning_rate, 'label': 'delta'},
        {'params': [model.epsilon], 'lr': learning_rate * 0.1, 'label': 'epsilon'},
        {'params': [model.psi], 'lr': learning_rate * 0.1, 'label': 'psi'},
        {'params': [model.gamma], 'lr': learning_rate, 'label': 'gamma'},
    ]

    optimizer = optim.Adam(param_groups)
    losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model.compute_loss(event_times)

        if torch.isnan(loss):
            print(f"Epoch {epoch}: Loss=nan (stopping)")
            break

        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        losses.append(cur_loss)

        if epoch % 10 == 0:
            elapsed = (time.time() - t0) / 60
            gamma_mag = model.gamma.abs().mean().item()
            print(f"Epoch {epoch:4d}: Loss={cur_loss:.4f}, |gamma|={gamma_mag:.4f}, "
                  f"time={elapsed:.1f}min")

    return losses


def main():
    parser = argparse.ArgumentParser(description='Train Aladyn nokappa v3 (constant LR)')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--W', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output dir (default: auto based on W)')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--e_matrix', type=str, default='E_matrix_corrected.pt',
                        help='E matrix filename (default: E_matrix_corrected.pt)')
    args = parser.parse_args()

    if args.output_dir is None:
        w_str = f"{args.W:.0e}".replace('+', '').replace('-0', '-')
        args.output_dir = f'/Users/sarahurbut/Library/CloudStorage/Dropbox/nokappa_v3_W{w_str}'

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n{'='*60}")
    print(f"NOKAPPA v3: samples {args.start_index} to {args.end_index}")
    print(f"Epochs: {args.num_epochs}, LR: {args.learning_rate} (constant)")
    print(f"W: {args.W}, kappa=1.0 (fixed)")
    print(f"No cosine, no clipping, no early stopping")
    print(f"E matrix: {args.e_matrix}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load data
    Y = torch.load(args.data_dir + 'Y_tensor.pt', weights_only=False)
    E = torch.load(args.data_dir + args.e_matrix, weights_only=False)
    G = torch.load(args.data_dir + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(args.data_dir + 'model_essentials.pt', weights_only=False)

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

    # Build model
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
        K=args.K, P=G_with_sex.shape[1],
        init_sd_scaler=1e-1, G=G_with_sex, Y=Y_batch,
        genetic_scale=1, W=args.W, R=0,
        prevalence_t=prevalence_t, signature_references=signature_refs,
        healthy_reference=True, disease_names=essentials['disease_names'],
        learn_kappa=False,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(args.data_dir + 'initial_clusters_400k.pt', weights_only=False)
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters

    # Train
    print(f"\nTraining (constant LR={args.learning_rate}, {args.num_epochs} epochs)...")
    t0 = time.time()
    losses = fit_v3(model, E_batch,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate)
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
        'version': 'NOKAPPA_v3',
    }, output_path)

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
