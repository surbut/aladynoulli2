#!/usr/bin/env python
"""
Plot loss vs epoch during training for nolr vs reparam (Loss Convergence).
Simple: train both on same batch, plot loss over epochs.

This is training loss only — for external validation, run holdout_validation_pooled.py.

Usage:
    python plot_loss_convergence_nolr_vs_reparam.py --start_index 0 --end_index 10000 --num_epochs 200
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
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--W', type=float, default=0.0001)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--out', type=str, default='', help='Save plot path')
    args = parser.parse_args()

    print("Loading data...")
    Y, E, G, essentials = load_model_essentials(args.data_dir)
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G,
                                                     start_index=args.start_index,
                                                     end_index=args.end_index)
    del Y

    sex, fh_processed = load_covariates_data(args.covariates_path)
    sex_batch = sex[args.start_index:args.end_index]
    G_with_sex = np.column_stack([G_batch, sex_batch])
    pc_columns = ['f.22009.0.1', 'f.22009.0.2', 'f.22009.0.3', 'f.22009.0.4', 'f.22009.0.5',
                  'f.22009.0.6', 'f.22009.0.7', 'f.22009.0.8', 'f.22009.0.9', 'f.22009.0.10']
    pcs = fh_processed.iloc[args.start_index:args.end_index][pc_columns].values
    G_with_sex = np.column_stack([G_batch, sex_batch, pcs])

    refs = torch.load(args.data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    prevalence_t = torch.load(args.data_dir + 'prevalence_t_corrected.pt', weights_only=False)
    initial_psi = torch.load(args.data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(args.data_dir + 'initial_clusters_400k.pt', weights_only=False)

    K = 20
    P = G_with_sex.shape[1]

    def build_and_train(model_class, name):
        torch.manual_seed(42)
        np.random.seed(42)
        model = model_class(
            N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
            K=K, P=P, init_sd_scaler=1e-1, G=G_with_sex, Y=Y_batch,
            genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
            signature_references=signature_refs, healthy_reference=True,
            disease_names=essentials['disease_names']
        )
        model.initialize_params(true_psi=initial_psi)
        model.clusters = initial_clusters

        result = model.fit(E_batch, num_epochs=args.num_epochs, learning_rate=args.learning_rate)
        losses = result[0] if isinstance(result, tuple) else result
        print(f"  {name}: final loss = {losses[-1]:.4f}")
        return losses

    print("\nTraining nolr...")
    losses_nolr = build_and_train(ModelNolr, "nolr")

    print("Training reparam...")
    losses_reparam = build_and_train(ModelReparam, "reparam")

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = np.arange(len(losses_nolr))
    ax.plot(epochs, losses_nolr, label='nolr', color='C1', linewidth=1.5)
    ax.plot(epochs[:len(losses_reparam)], losses_reparam, label='reparam', color='C2', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence (training loss only — not external validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    out_path = args.out or str(Path(__file__).parent / 'loss_convergence_nolr_vs_reparam.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    print("\nNote: For external validation (holdout AUC), run holdout_validation_pooled.py")


if __name__ == '__main__':
    main()
