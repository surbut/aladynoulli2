#!/usr/bin/env python
"""
Compare phi, psi, and gamma between original (nolr) and reparameterized model runs.
"""
import numpy as np
import torch
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolr_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr',
                        help='Directory with original nolr checkpoints')
    parser.add_argument('--reparam_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM',
                        help='Directory with reparam checkpoints')
    parser.add_argument('--batch_start', type=int, default=0)
    parser.add_argument('--batch_end', type=int, default=10000)
    parser.add_argument('--W', type=float, default=0.0001)
    args = parser.parse_args()

    nolr_path = Path(args.nolr_dir) / f'enrollment_model_VECTORIZED_W{args.W}_nolr_batch_{args.batch_start}_{args.batch_end}.pt'
    reparam_path = Path(args.reparam_dir) / f'enrollment_model_REPARAM_W{args.W}_batch_{args.batch_start}_{args.batch_end}.pt'

    if not nolr_path.exists():
        print(f"NOT FOUND: {nolr_path}")
        return
    if not reparam_path.exists():
        print(f"NOT FOUND: {reparam_path}")
        return

    print(f"Loading nolr: {nolr_path}")
    nolr = torch.load(nolr_path, weights_only=False)
    print(f"Loading reparam: {reparam_path}")
    reparam = torch.load(reparam_path, weights_only=False)

    # Phi: saved directly
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().numpy()
        return np.array(x)
    phi_nolr = to_np(nolr['phi'])
    phi_reparam = to_np(reparam['phi'])

    # Psi and gamma: from state_dict
    sd_nolr = nolr['model_state_dict']
    sd_reparam = reparam['model_state_dict']
    psi_nolr = sd_nolr['psi'].detach().numpy()
    psi_reparam = sd_reparam['psi'].detach().numpy()
    gamma_nolr = sd_nolr['gamma'].detach().numpy()
    gamma_reparam = sd_reparam['gamma'].detach().numpy()

    print("\n" + "="*60)
    print("COMPARISON: Original (nolr) vs Reparameterized")
    print("="*60)

    def compare(name, a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        corr = np.corrcoef(a_flat, b_flat)[0, 1] if a_flat.size > 1 else 1.0
        mae = np.mean(np.abs(a_flat - b_flat))
        max_diff = np.max(np.abs(a_flat - b_flat))
        print(f"\n{name}:")
        print(f"  Correlation: {corr:.6f}")
        print(f"  Mean abs diff: {mae:.6f}")
        print(f"  Max abs diff: {max_diff:.6f}")
        print(f"  Shape: {a.shape}")

    compare("phi", phi_nolr, phi_reparam)
    compare("psi", psi_nolr, psi_reparam)
    compare("gamma", gamma_nolr, gamma_reparam)

    # Sanity check: do top PRS-signature associations still make sense?
    print("\n" + "="*60)
    print("PRS-SIGNATURE SANITY CHECK (do known associations hold?)")
    print("="*60)
    prs_path = Path(__file__).parent.parent / 'prs_names.csv'
    if prs_path.exists():
        prs_names = pd.read_csv(prs_path, header=None).iloc[:, 0].tolist()
        feat_names = prs_names + ["Sex"] + [f"PC{i}" for i in range(1, 11)]
    else:
        feat_names = [f"Feat{i}" for i in range(47)]

    # Show top 3 PRS per signature for all 21 signatures
    sig_labels = {5: "CV", 15: "Metabolic"}
    n_sig = gamma_nolr.shape[1]
    print(f"\nTop 3 PRS by |gamma| per signature (all {n_sig}):")
    print("-" * 80)
    for sig_k in range(n_sig):
        g_nolr = gamma_nolr[:, sig_k]
        g_reparam = gamma_reparam[:, sig_k]
        top_nolr = np.argsort(np.abs(g_nolr))[-3:][::-1]
        top_reparam = np.argsort(np.abs(g_reparam))[-3:][::-1]
        label = sig_labels.get(sig_k, "")
        hdr = f"Sig {sig_k:2d}" + (f" ({label})" if label else "")
        print(f"{hdr}: nolr [{', '.join(f'{feat_names[i]}={g_nolr[i]:.3f}' for i in top_nolr)}]")
        print(f"       rep [{', '.join(f'{feat_names[i]}={g_reparam[i]:.3f}' for i in top_reparam)}]")
    print("-" * 80)
    # Magnitude comparison: reparam absorbing more?
    scale_nolr = np.abs(gamma_nolr).mean()
    scale_reparam = np.abs(gamma_reparam).mean()
    print(f"Mean |gamma|: nolr={scale_nolr:.4f}, reparam={scale_reparam:.4f} (ratio={scale_reparam/scale_nolr:.2f}x)")

    print("\n" + "="*60)
    print("Summary: phi and psi get NLL gradients in reparam; gamma does too.")
    print("If reparam learned differently, we'd see lower correlation or larger diffs.")
    print("Sanity: Reparam should still show CAD/CVD/LDL->Sig5, T2D/BMI->Sig15 if sensible.")
    print("="*60)


if __name__ == '__main__':
    main()
