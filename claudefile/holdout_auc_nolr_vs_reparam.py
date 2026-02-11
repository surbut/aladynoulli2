#!/usr/bin/env python
"""
Holdout AUC comparison: nolr vs reparam.

For each model:
  1. Fix pooled phi, psi, gamma, kappa
  2. Re-estimate lambda (nolr) or delta (reparam) on holdout individuals
  3. Extract pi and compute per-disease AUC with bootstrap CIs

Usage:
    python holdout_auc_nolr_vs_reparam.py --holdout_start 399000 --holdout_end 400000 --num_epochs 50 --n_bootstraps 10
"""
import argparse
import gc
import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')

from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa import (
    AladynSurvivalFixedPhiFixedGammaFixedKappa,
)
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_from_pi,
)


def load_nolr_params(master_path, pooled_gk_path):
    """Load nolr params: phi/psi from master checkpoint, gamma/kappa from pooled nolr.
    Matches production prediction pipeline (run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py).
    """
    # phi/psi from master checkpoint
    master = torch.load(master_path, weights_only=False)
    sd = master['model_state_dict']
    phi = sd['phi'].detach().cpu().numpy() if torch.is_tensor(sd['phi']) else np.array(sd['phi'])
    psi = sd['psi'].detach().cpu().numpy() if torch.is_tensor(sd['psi']) else np.array(sd['psi'])

    # gamma/kappa from pooled nolr batches
    gk = torch.load(pooled_gk_path, weights_only=False)
    kappa = float(gk['kappa']) if not hasattr(gk['kappa'], 'item') else gk['kappa'].item()
    gamma = gk['gamma']
    if torch.is_tensor(gamma):
        gamma = gamma.detach().cpu().numpy()

    print(f"  Nolr phi/psi from: {Path(master_path).name}")
    print(f"  Nolr gamma/kappa from: {Path(pooled_gk_path).name}")
    print(f"  phi {phi.shape}, psi {psi.shape}, kappa {kappa:.4f}, gamma {gamma.shape}")
    return phi, psi, kappa, gamma


def load_reparam_params(pooled_path, data_dir):
    """Load reparam params: all from pooled_phi_kappa_gamma_reparam.pt."""
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

    print(f"  Reparam all from: {Path(pooled_path).name}")
    print(f"  phi {phi.shape}, psi {psi.shape}, kappa {kappa:.4f}, gamma {gamma.shape}")
    return phi, psi, kappa, gamma


def fit_and_extract_pi(Y_holdout, E_holdout, G_holdout, phi, psi, kappa, gamma,
                       signature_refs, prevalence_t, disease_names,
                       num_epochs, learning_rate, use_reparam=False):
    """Construct model, fit lambda/delta, extract pi."""
    torch.manual_seed(42)
    np.random.seed(42)

    K = 20
    P = G_holdout.shape[1]
    ModelClass = AladynSurvivalFixedPhiFixedGammaFixedKappaReparam if use_reparam else AladynSurvivalFixedPhiFixedGammaFixedKappa

    model = ModelClass(
        N=Y_holdout.shape[0], D=Y_holdout.shape[1], T=Y_holdout.shape[2],
        K=K, P=P, G=G_holdout, Y=Y_holdout,
        R=0, W=0.0001, prevalence_t=prevalence_t,
        init_sd_scaler=1e-1, genetic_scale=1,
        pretrained_phi=phi, pretrained_psi=psi,
        pretrained_gamma=gamma, pretrained_kappa=kappa,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=disease_names,
    )

    result = model.fit(E_holdout, num_epochs=num_epochs, learning_rate=learning_rate)
    losses = result[0] if isinstance(result, tuple) else result
    final_loss = losses[-1] if losses else float('nan')

    with torch.no_grad():
        pi, theta, phi_prob = model.forward()

    # Validate pi
    n_nan = torch.isnan(pi).sum().item()
    n_inf = torch.isinf(pi).sum().item()
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: pi has {n_nan} NaN, {n_inf} Inf values -- clamping")
        pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
        pi = torch.clamp(pi, 1e-8, 1 - 1e-8)

    return pi, final_loss, len(losses)


def results_to_df(results_dict, model_name):
    """Convert AUC results dict to DataFrame."""
    rows = []
    for disease, metrics in results_dict.items():
        rows.append({
            'disease': disease,
            f'{model_name}_auc': metrics.get('auc', np.nan),
            f'{model_name}_ci_lower': metrics.get('ci_lower', np.nan),
            f'{model_name}_ci_upper': metrics.get('ci_upper', np.nan),
            f'{model_name}_n_events': metrics.get('n_events', 0),
        })
    return pd.DataFrame(rows)


def print_comparison(merged_df):
    """Print side-by-side AUC comparison table."""
    print(f"\n{'DISEASE':<25} {'NOLR AUC (95% CI)':<30} {'REPARAM AUC (95% CI)':<30} {'DELTA':>8}")
    print("-" * 95)
    for _, row in merged_df.iterrows():
        nolr_auc = row.get('nolr_auc', np.nan)
        reparam_auc = row.get('reparam_auc', np.nan)
        nolr_ci = f"{nolr_auc:.3f} ({row.get('nolr_ci_lower', np.nan):.3f}-{row.get('nolr_ci_upper', np.nan):.3f})"
        reparam_ci = f"{reparam_auc:.3f} ({row.get('reparam_ci_lower', np.nan):.3f}-{row.get('reparam_ci_upper', np.nan):.3f})"
        delta = reparam_auc - nolr_auc if not (np.isnan(nolr_auc) or np.isnan(reparam_auc)) else np.nan
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "N/A"
        print(f"{row['disease']:<25} {nolr_ci:<30} {reparam_ci:<30} {delta_str:>8}")


def main():
    parser = argparse.ArgumentParser(description='Holdout AUC comparison: nolr vs reparam')
    parser.add_argument('--holdout_start', type=int, default=390000)
    parser.add_argument('--holdout_end', type=int, default=400000)
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--pce_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    parser.add_argument('--pooled_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--nolr_master', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt',
                        help='Master checkpoint for nolr phi/psi')
    parser.add_argument('--nolr_gk', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/pooled_kappa_gamma_nolr.pt',
                        help='Pooled gamma/kappa for nolr')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--lr_reparam', type=float, default=1e-1)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pooled_dir = Path(args.pooled_dir)

    # Check pooled files exist
    reparam_path = pooled_dir / 'pooled_phi_kappa_gamma_reparam.pt'
    for p in [Path(args.nolr_master), Path(args.nolr_gk), reparam_path]:
        if not p.exists():
            print(f"Missing: {p}")
            return

    # ================================================================
    # Load data
    # ================================================================
    print("Loading data...")
    Y_full = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
    E_enrollment_full = torch.load(str(data_dir / 'E_enrollment_full.pt'), weights_only=False)
    G_full = torch.load(str(data_dir / 'G_matrix.pt'), weights_only=False)
    essentials = torch.load(str(data_dir / 'model_essentials.pt'), weights_only=False)
    refs = torch.load(str(data_dir / 'reference_trajectories.pt'), weights_only=False)
    prevalence_t = torch.load(str(data_dir / 'prevalence_t_corrected.pt'), weights_only=False)
    disease_names = essentials['disease_names']
    signature_refs = refs['signature_refs']

    # Subset to holdout
    idx = list(range(args.holdout_start, args.holdout_end))
    Y_holdout = Y_full[idx]
    E_enrollment_holdout = E_enrollment_full[idx]
    G_holdout = G_full[idx]
    del Y_full, E_enrollment_full, G_full
    gc.collect()

    # Add sex + PCs to G
    fh = pd.read_csv(args.covariates_path)
    if 'Sex' in fh.columns:
        sex = fh['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
    else:
        sex = fh['sex'].values
    sex_h = sex[args.holdout_start:args.holdout_end]
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh.iloc[args.holdout_start:args.holdout_end][pc_cols].values
    G_holdout = np.column_stack([G_holdout, sex_h, pcs])

    # Load pce_df for AUC evaluation
    pce_df_full = pd.read_csv(args.pce_path)
    pce_df_holdout = pce_df_full.iloc[args.holdout_start:args.holdout_end].reset_index(drop=True)
    del pce_df_full

    # Ensure Sex/age columns
    if 'Sex' not in pce_df_holdout.columns:
        if 'sex' in pce_df_holdout.columns:
            pce_df_holdout['Sex'] = pce_df_holdout['sex'].map({0: 'Female', 1: 'Male'}).fillna('Unknown')
    if 'sex' not in pce_df_holdout.columns:
        if 'Sex' in pce_df_holdout.columns:
            pce_df_holdout['sex'] = pce_df_holdout['Sex'].map({'Female': 0, 'Male': 1}).fillna(-1)
    if 'age' not in pce_df_holdout.columns and 'Age' in pce_df_holdout.columns:
        pce_df_holdout['age'] = pce_df_holdout['Age']

    N_holdout = Y_holdout.shape[0]
    print(f"Holdout: {N_holdout} individuals ({args.holdout_start}-{args.holdout_end})")
    print(f"Y: {Y_holdout.shape}, E_enrollment: {E_enrollment_holdout.shape}")

    # Load pooled params
    print("\nLoading nolr params (master phi/psi + pooled nolr gamma/kappa)...")
    phi_n, psi_n, kappa_n, gamma_n = load_nolr_params(args.nolr_master, args.nolr_gk)
    print("\nLoading reparam params...")
    phi_r, psi_r, kappa_r, gamma_r = load_reparam_params(reparam_path, data_dir)
    print(f"\nNolr kappa: {kappa_n:.4f}, Reparam kappa: {kappa_r:.4f}")

    # ================================================================
    # Fit models and extract pi
    # ================================================================
    print("\n" + "=" * 70)
    print("FITTING NOLR MODEL (optimize lambda)...")
    print("=" * 70)
    pi_nolr, loss_nolr, epochs_nolr = fit_and_extract_pi(
        Y_holdout, E_enrollment_holdout, G_holdout,
        phi_n, psi_n, kappa_n, gamma_n,
        signature_refs, prevalence_t, disease_names,
        num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        use_reparam=False,
    )
    print(f"  Final loss: {loss_nolr:.4f}, epochs: {epochs_nolr}")

    print("\n" + "=" * 70)
    print("FITTING REPARAM MODEL (optimize delta, lambda=mean(gamma)+delta)...")
    print("=" * 70)
    pi_reparam, loss_reparam, epochs_reparam = fit_and_extract_pi(
        Y_holdout, E_enrollment_holdout, G_holdout,
        phi_r, psi_r, kappa_r, gamma_r,
        signature_refs, prevalence_t, disease_names,
        num_epochs=args.num_epochs, learning_rate=args.lr_reparam,
        use_reparam=True,
    )
    print(f"  Final loss: {loss_reparam:.4f}, epochs: {epochs_reparam}")

    # ================================================================
    # Evaluate AUC -- dynamic 10-year
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPUTING DYNAMIC 10-YEAR AUC: NOLR")
    print("=" * 70)
    results_nolr = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
        pi=pi_nolr,
        Y_100k=Y_holdout,
        E_100k=E_enrollment_holdout,
        disease_names=disease_names,
        pce_df=pce_df_holdout,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10,
    )

    print("\n" + "=" * 70)
    print("COMPUTING DYNAMIC 10-YEAR AUC: REPARAM")
    print("=" * 70)
    results_reparam = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
        pi=pi_reparam,
        Y_100k=Y_holdout,
        E_100k=E_enrollment_holdout,
        disease_names=disease_names,
        pce_df=pce_df_holdout,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10,
    )

    # ================================================================
    # Compare
    # ================================================================
    df_nolr = results_to_df(results_nolr, 'nolr')
    df_reparam = results_to_df(results_reparam, 'reparam')
    merged = df_nolr.merge(df_reparam, on='disease', how='outer')

    print("\n" + "=" * 70)
    print("DYNAMIC 10-YEAR AUC COMPARISON")
    print("=" * 70)
    print_comparison(merged)

    # Summary
    valid = merged.dropna(subset=['nolr_auc', 'reparam_auc'])
    if len(valid) > 0:
        nolr_mean = valid['nolr_auc'].mean()
        reparam_mean = valid['reparam_auc'].mean()
        nolr_wins = (valid['nolr_auc'] > valid['reparam_auc']).sum()
        reparam_wins = (valid['reparam_auc'] > valid['nolr_auc']).sum()
        print(f"\nMean AUC -- nolr: {nolr_mean:.4f}, reparam: {reparam_mean:.4f}")
        print(f"Nolr wins: {nolr_wins}, Reparam wins: {reparam_wins}, Tied: {len(valid) - nolr_wins - reparam_wins}")
        print(f"Final holdout loss -- nolr: {loss_nolr:.4f}, reparam: {loss_reparam:.4f}")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'holdout_auc_nolr_vs_reparam.csv')
    merged.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
