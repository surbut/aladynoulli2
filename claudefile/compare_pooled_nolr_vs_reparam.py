#!/usr/bin/env python
"""
Compare pooled phi, gamma, kappa, and psi between nolr and reparam.

Prerequisites: Run pool_phi_kappa_gamma_from_batches.py for both model types:
    python pool_phi_kappa_gamma_from_batches.py --model_type nolr --max_batches 39
    python pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39

Usage:
    python compare_pooled_nolr_vs_reparam.py [--plot]
"""
import numpy as np
import torch
import argparse
from pathlib import Path


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return np.array(x)


def compare_arr(name, a, b):
    """Compare two arrays: correlation, MAE, max diff."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    if a_flat.size != b_flat.size:
        print(f"  {name}: shape mismatch {a.shape} vs {b.shape}")
        return
    corr = np.corrcoef(a_flat, b_flat)[0, 1] if a_flat.size > 1 and np.std(a_flat) > 0 and np.std(b_flat) > 0 else np.nan
    mae = np.mean(np.abs(a_flat - b_flat))
    max_diff = np.max(np.abs(a_flat - b_flat))
    print(f"  Correlation: {corr:.6f}")
    print(f"  Mean abs diff: {mae:.6f}")
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Shape: {a.shape}")
    return corr, mae, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pooled_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--plot', action='store_true', help='Save scatter/diff plots')
    args = parser.parse_args()

    pooled_dir = Path(args.pooled_dir)
    nolr_path = pooled_dir / 'pooled_phi_kappa_gamma_nolr.pt'
    reparam_path = pooled_dir / 'pooled_phi_kappa_gamma_reparam.pt'

    if not nolr_path.exists():
        print(f"NOT FOUND: {nolr_path}")
        print("Run: python pool_phi_kappa_gamma_from_batches.py --model_type nolr --max_batches 39")
        return
    if not reparam_path.exists():
        print(f"NOT FOUND: {reparam_path}")
        print("Run: python pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39")
        return

    print(f"Loading {nolr_path.name}...")
    nolr = torch.load(nolr_path, weights_only=False)
    print(f"Loading {reparam_path.name}...")
    reparam = torch.load(reparam_path, weights_only=False)

    phi_nolr = to_np(nolr['phi'])
    phi_reparam = to_np(reparam['phi'])
    gamma_nolr = to_np(nolr['gamma'])
    gamma_reparam = to_np(reparam['gamma'])
    kappa_nolr = float(nolr['kappa']) if 'kappa' in nolr else np.nan
    kappa_reparam = float(reparam['kappa']) if 'kappa' in reparam else np.nan

    psi_nolr = to_np(nolr['psi']) if nolr.get('psi') is not None else None
    psi_reparam = to_np(reparam['psi']) if reparam.get('psi') is not None else None

    print("\n" + "="*70)
    print("POOLED COMPARISON: nolr vs reparam")
    print("="*70)

    # Kappa
    print("\n--- KAPPA ---")
    print(f"  nolr:    {kappa_nolr:.6f}")
    print(f"  reparam: {kappa_reparam:.6f}")
    print(f"  diff:    {kappa_reparam - kappa_nolr:.6f}")

    # Phi
    print("\n--- PHI ---")
    compare_arr("phi", phi_nolr, phi_reparam)

    # Psi
    if psi_nolr is not None and psi_reparam is not None:
        print("\n--- PSI ---")
        compare_arr("psi", psi_nolr, psi_reparam)
    else:
        print("\n--- PSI --- (not available in one or both)")

    # Gamma
    print("\n--- GAMMA ---")
    compare_arr("gamma", gamma_nolr, gamma_reparam)
    print(f"  Mean |gamma| nolr:    {np.abs(gamma_nolr).mean():.6f}")
    print(f"  Mean |gamma| reparam: {np.abs(gamma_reparam).mean():.6f}")

    # Plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # Phi scatter
            ax = axes[0, 0]
            ax.scatter(phi_nolr.flatten(), phi_reparam.flatten(), alpha=0.3, s=5)
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], 'r--', label='y=x')
            ax.set_xlabel('phi (nolr)')
            ax.set_ylabel('phi (reparam)')
            ax.set_title('Phi')
            ax.legend()

            # Gamma scatter
            ax = axes[0, 1]
            ax.scatter(gamma_nolr.flatten(), gamma_reparam.flatten(), alpha=0.5, s=10)
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], 'r--', label='y=x')
            ax.set_xlabel('gamma (nolr)')
            ax.set_ylabel('gamma (reparam)')
            ax.set_title('Gamma')
            ax.legend()

            # Phi diff histogram
            ax = axes[1, 0]
            diff = phi_reparam.flatten() - phi_nolr.flatten()
            ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='r', ls='--')
            ax.set_xlabel('phi_reparam - phi_nolr')
            ax.set_ylabel('Count')
            ax.set_title('Phi difference')

            # Gamma diff histogram
            ax = axes[1, 1]
            diff = gamma_reparam.flatten() - gamma_nolr.flatten()
            ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='r', ls='--')
            ax.set_xlabel('gamma_reparam - gamma_nolr')
            ax.set_ylabel('Count')
            ax.set_title('Gamma difference')

            plt.suptitle('Pooled nolr vs reparam', fontsize=12)
            plt.tight_layout()
            out_path = Path(__file__).parent / 'pooled_nolr_vs_reparam_comparison.png'
            plt.savefig(out_path, dpi=150)
            print(f"\nSaved plot to {out_path}")
        except Exception as e:
            print(f"Plot failed: {e}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
