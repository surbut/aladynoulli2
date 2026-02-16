#!/usr/bin/env python
"""
Pool phi, kappa, and gamma from training batches (nolr or reparam).
Use for holdout validation with fixed phi, kappa, gamma.

Usage:
    # Pool from nolr batches (39 batches)
    python pool_phi_kappa_gamma_from_batches.py --model_type nolr --max_batches 39

    # Pool from reparam batches
    python pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39

Output: pooled_phi_kappa_gamma_{model_type}.pt in output_dir
"""
import torch
import numpy as np
import glob
from pathlib import Path
import argparse


def pool_phi_kappa_gamma(pattern, max_batches=None):
    """
    Load and pool phi, psi, kappa, gamma from batch files.

    Returns:
        phi_pooled: (K, D, T)
        psi_pooled: (K_total, D)
        kappa_pooled: float
        gamma_pooled: (P, K)
        n_batches: int
    """
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching: {pattern}")

    if max_batches is not None:
        files = files[:max_batches]

    all_phi, all_psi, all_kappa, all_gamma = [], [], [], []

    def _extract(ckpt, name):
        """Extract from checkpoint - same pattern as create_master_checkpoint_nolr (avoids bool(tensor))."""
        if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
            return ckpt['model_state_dict'][name]
        if name in ckpt:
            return ckpt[name]
        return None

    for fp in files:
        try:
            ckpt = torch.load(fp, weights_only=False)

            # Phi (same pattern as create_master_checkpoint_nolr)
            phi = _extract(ckpt, 'phi')
            if phi is None:
                print(f"  Skip {Path(fp).name}: no phi")
                continue
            phi = phi.detach().cpu().numpy() if torch.is_tensor(phi) else np.array(phi)
            all_phi.append(phi)

            # Psi (for fixedgk)
            psi = _extract(ckpt, 'psi')
            if psi is not None:
                p = psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.array(psi)
                all_psi.append(p)
            else:
                all_psi.append(None)

            # Kappa
            kappa = _extract(ckpt, 'kappa')
            k = kappa.item() if torch.is_tensor(kappa) else float(kappa) if kappa is not None else 1.0
            all_kappa.append(k)

            # Gamma
            gamma = _extract(ckpt, 'gamma')
            if gamma is None:
                print(f"  Skip {Path(fp).name}: no gamma")
                all_phi.pop()
                all_psi.pop()
                all_kappa.pop()
                continue
            g = gamma.detach().cpu().numpy() if torch.is_tensor(gamma) else np.array(gamma)
            all_gamma.append(g)

            if len(all_phi) <= 3:
                print(f"  Loaded phi={phi.shape}, kappa={k:.4f}, gamma={g.shape} from {Path(fp).name}")

        except Exception as e:
            print(f"  Error {Path(fp).name}: {e}")
            continue

    if not all_phi:
        raise ValueError("No batches loaded")

    phi_pooled = np.mean(np.stack(all_phi), axis=0)
    kappa_pooled = float(np.mean(all_kappa))
    gamma_pooled = np.mean(np.stack(all_gamma), axis=0)

    # Psi: use first valid if any
    psi_valid = [p for p in all_psi if p is not None]
    psi_pooled = np.mean(np.stack(psi_valid), axis=0) if psi_valid else None

    n = len(all_phi)
    print(f"\n{'='*60}")
    print(f"POOLED: phi {phi_pooled.shape}, kappa {kappa_pooled:.6f}, gamma {gamma_pooled.shape}")
    if psi_pooled is not None:
        print(f"  psi {psi_pooled.shape}")
    print(f"  n_batches: {n}")
    print(f"{'='*60}")

    return phi_pooled, psi_pooled, kappa_pooled, gamma_pooled, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['nolr', 'reparam', 'nokappa'], default='nolr')
    parser.add_argument('--max_batches', type=int, default=39)
    parser.add_argument('--nolr_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr')
    parser.add_argument('--reparam_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    parser.add_argument('--nokappa_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--W', type=float, default=0.0001)
    args = parser.parse_args()

    if args.model_type == 'nolr':
        pattern = str(Path(args.nolr_dir) / f'enrollment_model_VECTORIZED_W{args.W}_nolr_batch_*_*.pt')
    elif args.model_type == 'nokappa':
        pattern = str(Path(args.nokappa_dir) / f'enrollment_model_REPARAM_NOKAPPA_W{args.W}_batch_*_*.pt')
    else:
        pattern = str(Path(args.reparam_dir) / f'enrollment_model_REPARAM_W{args.W}_batch_*_*.pt')

    phi, psi, kappa, gamma, n = pool_phi_kappa_gamma(pattern, args.max_batches)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'pooled_phi_kappa_gamma_{args.model_type}.pt'
    save_dict = {
        'phi': phi,
        'kappa': kappa,
        'gamma': gamma,
        'n_batches': n,
        'model_type': args.model_type,
    }
    if psi is not None:
        save_dict['psi'] = psi
    torch.save(save_dict, out_path)
    print(f"\n✓ Saved to {out_path}")


if __name__ == '__main__':
    main()
