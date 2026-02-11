#!/usr/bin/env python
"""
Loss landscape: compare loss at nolr vs reparam and along gamma interpolation.
Also test: does zeroing out gamma on PCs worsen loss? (if we don't want PCs)
Requires data in data_dir (default Dropbox).
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
    parser.add_argument('--W', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--nolr_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr')
    parser.add_argument('--reparam_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    parser.add_argument('--n_interp', type=int, default=11, help='Number of points for gamma interpolation')
    parser.add_argument('--out', type=str, default='', help='Save plot path')
    args = parser.parse_args()

    # Load data
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
    # PC indices: 36 PRS + Sex(36) + PC1..PC10(37-46)
    PC_INDICES = list(range(37, 47))  # 0-based: 36=Sex, 37-46=PC1..PC10

    # Build models
    print("Building models...")
    model_nolr = ModelNolr(
        N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
        K=K, P=P, init_sd_scaler=1e-1, G=G_with_sex, Y=Y_batch,
        genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    model_nolr.initialize_params(true_psi=initial_psi)
    model_nolr.clusters = initial_clusters

    model_reparam = ModelReparam(
        N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
        K=K, P=P, init_sd_scaler=1e-1, G=G_with_sex, Y=Y_batch,
        genetic_scale=1, W=args.W, R=0, prevalence_t=prevalence_t,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    model_reparam.initialize_params(true_psi=initial_psi)
    model_reparam.clusters = initial_clusters

    # Load checkpoints
    nolr_path = Path(args.nolr_dir) / f'enrollment_model_VECTORIZED_W{args.W}_nolr_batch_{args.start_index}_{args.end_index}.pt'
    reparam_path = Path(args.reparam_dir) / f'enrollment_model_REPARAM_W{args.W}_batch_{args.start_index}_{args.end_index}.pt'
    if not nolr_path.exists() or not reparam_path.exists():
        print(f"Checkpoints not found: {nolr_path} / {reparam_path}")
        return

    ckpt_nolr = torch.load(nolr_path, weights_only=False)
    ckpt_reparam = torch.load(reparam_path, weights_only=False)
    model_nolr.load_state_dict(ckpt_nolr['model_state_dict'], strict=False)
    model_reparam.load_state_dict(ckpt_reparam['model_state_dict'], strict=False)

    # Eval mode
    model_nolr.eval()
    model_reparam.eval()

    gamma_nolr = ckpt_nolr['model_state_dict']['gamma'].detach()
    gamma_reparam = ckpt_reparam['model_state_dict']['gamma'].detach()

    # 1) Loss at nolr and reparam
    with torch.no_grad():
        loss_nolr = model_nolr.compute_loss(E_batch).item()
        loss_reparam = model_reparam.compute_loss(E_batch).item()

    print("\n" + "="*60)
    print("LOSS AT SOLUTIONS")
    print("="*60)
    print(f"  Nolr:    {loss_nolr:.6f}")
    print(f"  Reparam: {loss_reparam:.6f}")
    print(f"  Diff:    {loss_reparam - loss_nolr:.6f}")

    # 2) Interpolate gamma: gamma_alpha = (1-alpha)*nolr + alpha*reparam
    alphas = np.linspace(0, 1, args.n_interp)
    losses_interp = []
    for alpha in alphas:
        with torch.no_grad():
            gamma_alpha = (1 - alpha) * gamma_nolr + alpha * gamma_reparam
            model_reparam.gamma.data.copy_(gamma_alpha)
            losses_interp.append(model_reparam.compute_loss(E_batch).item())
    losses_interp = np.array(losses_interp)
    print("\n" + "="*60)
    print("LOSS ALONG GAMMA INTERPOLATION (alpha: 0=nolr, 1=reparam)")
    print("="*60)
    for i, alpha in enumerate(alphas):
        print(f"  alpha={alpha:.2f}: loss={losses_interp[i]:.6f}")
    print(f"  Min at alpha={alphas[np.argmin(losses_interp)]:.2f}")

    # 3) Zero out gamma on PCs: do we want PCs?
    with torch.no_grad():
        gamma_no_pc = gamma_reparam.clone()
        gamma_no_pc[PC_INDICES, :] = 0.0
        model_reparam.gamma.data.copy_(gamma_no_pc)
        loss_reparam_no_pc = model_reparam.compute_loss(E_batch).item()
        model_reparam.gamma.data.copy_(gamma_reparam)  # restore

    print("\n" + "="*60)
    print("PC PENALTY: loss when gamma[PCs] = 0")
    print("="*60)
    print(f"  Reparam (full):     {loss_reparam:.6f}")
    print(f"  Reparam (gamma_PC=0): {loss_reparam_no_pc:.6f}")
    print(f"  Delta:              {loss_reparam_no_pc - loss_reparam:.6f}")
    if loss_reparam_no_pc > loss_reparam:
        print("  -> PCs improve fit (larger loss without them)")
    else:
        print("  -> PCs hurt fit (smaller loss without them)")

    # 4) Reparam loss along phi interpolation (phi_nolr -> phi_reparam)
    phi_nolr = ckpt_nolr['phi'].detach() if 'phi' in ckpt_nolr else ckpt_nolr['model_state_dict']['phi'].detach()
    phi_reparam = model_reparam.phi.detach()  # get_phi()
    mean_phi = model_reparam.get_mean_phi()
    eps_reparam = model_reparam.epsilon.data.clone()
    losses_phi_interp = []
    for alpha in alphas:
        with torch.no_grad():
            phi_alpha = (1 - alpha) * phi_nolr + alpha * phi_reparam
            model_reparam.epsilon.data.copy_(phi_alpha - mean_phi)
            losses_phi_interp.append(model_reparam.compute_loss(E_batch).item())
    model_reparam.epsilon.data.copy_(eps_reparam)  # restore
    losses_phi_interp = np.array(losses_phi_interp)
    print("\n" + "="*60)
    print("LOSS ALONG PHI INTERPOLATION (0=nolr phi, 1=reparam phi)")
    print("="*60)
    for i, alpha in enumerate(alphas):
        print(f"  alpha={alpha:.2f}: loss={losses_phi_interp[i]:.6f}")
    print(f"  Min at alpha={alphas[np.argmin(losses_phi_interp)]:.2f}")

    # 5) Reparam loss along lambda interpolation (delta: 0=nolr equiv, 1=reparam)
    mean_lambda = model_reparam.get_mean_lambda()
    lambda_nolr = ckpt_nolr['model_state_dict']['lambda_'].detach()
    lambda_reparam = model_reparam.get_lambda().detach()
    delta_reparam = model_reparam.delta.data.clone()
    losses_lambda_interp = []
    for alpha in alphas:
        with torch.no_grad():
            lambda_alpha = (1 - alpha) * lambda_nolr + alpha * lambda_reparam
            delta_alpha = lambda_alpha - mean_lambda
            model_reparam.delta.data.copy_(delta_alpha)
            losses_lambda_interp.append(model_reparam.compute_loss(E_batch).item())
    model_reparam.delta.data.copy_(delta_reparam)  # restore
    losses_lambda_interp = np.array(losses_lambda_interp)
    print("\n" + "="*60)
    print("LOSS ALONG LAMBDA INTERPOLATION (0=nolr lambda, 1=reparam lambda)")
    print("="*60)
    for i, alpha in enumerate(alphas):
        print(f"  alpha={alpha:.2f}: loss={losses_lambda_interp[i]:.6f}")
    print(f"  Min at alpha={alphas[np.argmin(losses_lambda_interp)]:.2f}")

    # 6) Plot (2x2: gamma, phi, lambda, summary)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax = axes[0, 0]
        ax.plot(alphas, losses_interp, 'o-', label='gamma interp')
        ax.axhline(loss_nolr, color='C1', ls='--', alpha=0.7, label=f'Nolr={loss_nolr:.4f}')
        ax.axhline(loss_reparam, color='C2', ls='--', alpha=0.7, label=f'Reparam={loss_reparam:.4f}')
        ax.set_xlabel('alpha (0=nolr, 1=reparam)')
        ax.set_ylabel('Loss')
        ax.set_title('Gamma interpolation')
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(alphas, losses_phi_interp, 'o-', color='C3', label='phi interp')
        ax.axhline(loss_nolr, color='C1', ls='--', alpha=0.7)
        ax.axhline(loss_reparam, color='C2', ls='--', alpha=0.7)
        ax.set_xlabel('alpha (0=nolr phi, 1=reparam phi)')
        ax.set_ylabel('Loss')
        ax.set_title('Phi interpolation (reparam model)')
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        ax.plot(alphas, losses_lambda_interp, 'o-', color='C4', label='lambda interp')
        ax.axhline(loss_nolr, color='C1', ls='--', alpha=0.7)
        ax.axhline(loss_reparam, color='C2', ls='--', alpha=0.7)
        ax.set_xlabel('alpha (0=nolr lambda, 1=reparam lambda)')
        ax.set_ylabel('Loss')
        ax.set_title('Lambda interpolation (reparam model)')
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        ax.plot(alphas, losses_interp, 'o-', label='gamma')
        ax.plot(alphas, losses_phi_interp, 's-', label='phi')
        ax.plot(alphas, losses_lambda_interp, '^-', label='lambda')
        ax.axhline(loss_nolr, color='C1', ls='--', alpha=0.7)
        ax.axhline(loss_reparam, color='C2', ls='--', alpha=0.7)
        ax.set_xlabel('alpha')
        ax.set_ylabel('Loss')
        ax.set_title('All (0=nolr, 1=reparam)')
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = args.out or str(Path(__file__).parent / 'loss_landscape.png')
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot to {out_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == '__main__':
    main()
