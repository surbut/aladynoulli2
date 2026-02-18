#!/usr/bin/env python3
"""
Age-offset predictions for nokappa v3 (reparameterized model).

For each offset k (0..9):
  1. Censor E at enrollment + k years for 10k patients (batch 0)
  2. Fix phi, psi, gamma, kappa from pooled 40-batch training checkpoints
  3. Fit only individual delta (the "testing" step)
  4. Save pi tensor

Population params: pooled from all 40 nokappa v3 training checkpoints (corrected E)
Data: Y, E_matrix_corrected (censored per offset), G, covariates
Batch: first 10k patients (0–10000)

Usage:
    python run_age_offset_nokappa_v3.py
    nohup python run_age_offset_nokappa_v3.py > age_offset_nokappa_v3.log 2>&1 &
"""

import numpy as np
import pandas as pd
import torch
import glob
import os
import sys
import time
import gc
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
CKPT_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/'
OUTPUT_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/age_offset_nokappa_v3/'
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'

START_IDX = 0
END_IDX = 10000
MAX_OFFSET = 9
NUM_EPOCHS = 200
LR = 0.1

# ── Model imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)


def load_all_checkpoints(ckpt_dir):
    """Load phi, psi, kappa, gamma from all 40 batch checkpoints."""
    pattern = os.path.join(ckpt_dir, 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_*_*.pt')
    files = sorted(glob.glob(pattern), key=lambda f: int(Path(f).stem.split('_')[-2]))
    print(f'Found {len(files)} checkpoint files')

    all_phi, all_psi, all_kappa, all_gamma = [], [], [], []
    for fp in files:
        ckpt = torch.load(fp, weights_only=False)
        def _get(name):
            if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
                return ckpt['model_state_dict'][name]
            return ckpt.get(name)

        phi = _get('phi')
        psi = _get('psi')
        gamma = _get('gamma')
        kappa = _get('kappa')
        if phi is None or gamma is None:
            print(f'  SKIP {Path(fp).name}: missing phi or gamma')
            continue

        all_phi.append(phi.detach().cpu().numpy() if torch.is_tensor(phi) else np.array(phi))
        all_psi.append(psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.array(psi) if psi is not None else None)
        all_kappa.append(kappa.item() if torch.is_tensor(kappa) else float(kappa) if kappa is not None else 1.0)
        all_gamma.append(gamma.detach().cpu().numpy() if torch.is_tensor(gamma) else np.array(gamma))

    print(f'Loaded {len(all_phi)} checkpoints')
    return all_phi, all_psi, all_kappa, all_gamma


def pool_all(all_phi, all_psi, all_kappa, all_gamma):
    """Pool parameters from all checkpoints."""
    phi = np.mean(np.stack(all_phi), axis=0)
    kappa = float(np.mean(all_kappa))
    gamma = np.mean(np.stack(all_gamma), axis=0)
    psi_valid = [p for p in all_psi if p is not None]
    psi = np.mean(np.stack(psi_valid), axis=0) if psi_valid else None
    return phi, psi, kappa, gamma


def censor_E_at_offset(E_batch, pce_df_subset, age_offset):
    """Censor E matrix so each patient's data is capped at enrollment + offset years."""
    E_censored = E_batch.clone()
    total_changed = 0

    for i, row in enumerate(pce_df_subset.itertuples()):
        if i >= E_censored.shape[0]:
            break
        current_age = row.age + age_offset
        time_since_30 = max(0, current_age - 30)
        original = E_censored[i, :].clone()
        E_censored[i, :] = torch.minimum(
            E_censored[i, :],
            torch.full_like(E_censored[i, :], time_since_30)
        )
        total_changed += torch.sum(E_censored[i, :] != original).item()

    return E_censored, total_changed


def main():
    print('=' * 70)
    print('Age-Offset Predictions — Nokappa v3 (Reparameterized)')
    print('=' * 70)
    print(f'Batch: {START_IDX}–{END_IDX}')
    print(f'Offsets: 0–{MAX_OFFSET}')
    print(f'Epochs: {NUM_EPOCHS}, LR: {LR}')
    print(f'Output: {OUTPUT_DIR}')
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load & pool population parameters ───────────────────────────────
    print('Loading training checkpoints...')
    all_phi, all_psi, all_kappa, all_gamma = load_all_checkpoints(CKPT_DIR)
    phi, psi, kappa, gamma = pool_all(all_phi, all_psi, all_kappa, all_gamma)
    print(f'Pooled: phi {phi.shape}, psi {psi.shape}, kappa={kappa:.4f}, gamma {gamma.shape}')
    print()

    # ── Load shared data ────────────────────────────────────────────────
    print('Loading data...')
    Y_full = torch.load(DATA_DIR + 'Y_tensor.pt', weights_only=False)
    E_full = torch.load(DATA_DIR + 'E_matrix_corrected.pt', weights_only=False)
    G_full = torch.load(DATA_DIR + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(DATA_DIR + 'model_essentials.pt', weights_only=False)
    refs = torch.load(DATA_DIR + 'reference_trajectories.pt', weights_only=False)
    prevalence_t = torch.load(DATA_DIR + 'prevalence_t_corrected.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    disease_names = essentials['disease_names']

    fh = pd.read_csv(COV_PATH)

    # Subset to batch
    Y_batch = Y_full[START_IDX:END_IDX]
    E_batch = E_full[START_IDX:END_IDX]
    G_batch = G_full[START_IDX:END_IDX]
    pce_df_subset = fh.iloc[START_IDX:END_IDX].reset_index(drop=True)

    # Build G_with_sex (36 PRS + sex + 10 PCs = 47)
    sex = pce_df_subset['sex'].values
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = pce_df_subset[pc_cols].values
    G_with_sex = np.column_stack([G_batch, sex, pcs])
    print(f'G_with_sex: {G_with_sex.shape}')

    del Y_full, E_full, G_full
    gc.collect()

    N, D, T = Y_batch.shape
    K = phi.shape[0] - 1 if phi.shape[0] == 21 else phi.shape[0]
    P = G_with_sex.shape[1]

    print(f'Batch: N={N}, D={D}, T={T}, K={K}, P={P}')
    print()

    # ── Loop over offsets ───────────────────────────────────────────────
    for age_offset in range(MAX_OFFSET + 1):
        pi_path = os.path.join(OUTPUT_DIR,
            f'pi_nokappa_v3_age_offset_{age_offset}_{START_IDX}_{END_IDX}.pt')

        if os.path.exists(pi_path):
            print(f'Offset {age_offset}: already exists, skipping')
            continue

        t0 = time.time()
        print(f'{"=" * 60}')
        print(f'Offset {age_offset} / {MAX_OFFSET}')
        print(f'{"=" * 60}')

        # Censor E
        E_censored, n_changed = censor_E_at_offset(E_batch, pce_df_subset, age_offset)
        print(f'  Censored E: {n_changed} event times changed')

        # Seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Build model with fixed population params, fit only delta
        model = AladynSurvivalFixedPhiFixedGammaFixedKappaReparam(
            N=N, D=D, T=T, K=K, P=P,
            G=G_with_sex, Y=Y_batch,
            R=0, W=0.0001,
            prevalence_t=prevalence_t,
            init_sd_scaler=1e-1, genetic_scale=1,
            pretrained_phi=phi, pretrained_psi=psi,
            pretrained_gamma=gamma, pretrained_kappa=kappa,
            signature_references=signature_refs, healthy_reference=True,
            disease_names=disease_names,
        )

        # Fit individual delta using censored E
        result = model.fit(E_censored, num_epochs=NUM_EPOCHS, learning_rate=LR)
        losses = result[0] if isinstance(result, tuple) else result

        # Extract pi
        with torch.no_grad():
            pi, _, _ = model.forward()

        # Clean NaN/Inf
        n_nan = torch.isnan(pi).sum().item()
        n_inf = torch.isinf(pi).sum().item()
        if n_nan > 0 or n_inf > 0:
            print(f'  WARNING: {n_nan} NaN, {n_inf} Inf in pi — fixing')
            pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
        pi = torch.clamp(pi, 1e-8, 1 - 1e-8)

        # Save
        torch.save(pi, pi_path)
        elapsed = time.time() - t0
        final_loss = losses[-1] if losses else float('nan')
        print(f'  Saved: {pi_path}')
        print(f'  Final loss: {final_loss:.2f}, Time: {elapsed:.0f}s')
        print(f'  pi shape: {pi.shape}, range: [{pi.min():.6f}, {pi.max():.6f}]')

        del model, pi, E_censored, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print('=' * 60)
    print('ALL OFFSETS COMPLETE')
    print(f'Results in: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
