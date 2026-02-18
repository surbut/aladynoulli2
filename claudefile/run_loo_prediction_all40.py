#!/usr/bin/env python3
"""
Full 40-batch LOO prediction for nokappa v3 (reparameterized model).

For each batch i (0..39):
  1. Pool phi, psi, gamma, kappa from all 39 OTHER training checkpoints
  2. Predict on batch i using enrollment E with the reparam fixed-phi model
  3. Save pi tensor

Training checkpoints: corrected E (standard discovery model)
Prediction E matrix:  E_enrollment_full.pt

Usage:
    python run_loo_prediction_all40.py
    nohup python run_loo_prediction_all40.py > loo_prediction_all40.log 2>&1 &
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
OUTPUT_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_nokappa_v3_loo_all40/'
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'

N_BATCHES = 40
BATCH_SIZE = 10000
NUM_PRED_EPOCHS = 200
PRED_LR = 0.1

# ── Model imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'aws_offsetmaster'))
from nokappa_prediction_utils import fit_and_extract_pi


def load_all_checkpoints(ckpt_dir):
    """Load phi, psi, kappa, gamma from all 40 batch checkpoints."""
    pattern = os.path.join(ckpt_dir, 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_*_*.pt')
    files = sorted(glob.glob(pattern), key=lambda f: int(Path(f).stem.split('_')[-2]))
    print(f'Found {len(files)} checkpoint files in {ckpt_dir}')

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

        phi_np = phi.detach().cpu().numpy() if torch.is_tensor(phi) else np.array(phi)
        psi_np = psi.detach().cpu().numpy() if torch.is_tensor(psi) else np.array(psi) if psi is not None else None
        gamma_np = gamma.detach().cpu().numpy() if torch.is_tensor(gamma) else np.array(gamma)
        k = kappa.item() if torch.is_tensor(kappa) else float(kappa) if kappa is not None else 1.0

        all_phi.append(phi_np)
        all_psi.append(psi_np)
        all_kappa.append(k)
        all_gamma.append(gamma_np)

    print(f'Loaded {len(all_phi)} checkpoints successfully')
    return all_phi, all_psi, all_kappa, all_gamma


def loo_pool(all_phi, all_psi, all_kappa, all_gamma, exclude_idx):
    """Pool all batches except exclude_idx."""
    indices = [j for j in range(len(all_phi)) if j != exclude_idx]
    phi = np.mean(np.stack([all_phi[j] for j in indices]), axis=0)
    kappa = float(np.mean([all_kappa[j] for j in indices]))
    gamma = np.mean(np.stack([all_gamma[j] for j in indices]), axis=0)
    psi_valid = [all_psi[j] for j in indices if all_psi[j] is not None]
    psi = np.mean(np.stack(psi_valid), axis=0) if psi_valid else None
    return phi, psi, kappa, gamma


def main():
    print('=' * 70)
    print('NOkappa v3 LOO Prediction — All 40 Batches')
    print('=' * 70)
    print(f'Checkpoint dir: {CKPT_DIR}')
    print(f'Output dir:     {OUTPUT_DIR}')
    print(f'Pred epochs:    {NUM_PRED_EPOCHS}, LR: {PRED_LR}')
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load checkpoints ────────────────────────────────────────────────
    print('Loading training checkpoints...')
    all_phi, all_psi, all_kappa, all_gamma = load_all_checkpoints(CKPT_DIR)
    n_loaded = len(all_phi)
    if n_loaded != N_BATCHES:
        print(f'WARNING: expected {N_BATCHES} checkpoints, got {n_loaded}')
    print()

    # ── Load shared data ────────────────────────────────────────────────
    print('Loading shared data...')
    Y = torch.load(DATA_DIR + 'Y_tensor.pt', weights_only=False)
    E = torch.load(DATA_DIR + 'E_enrollment_full.pt', weights_only=False)
    G = torch.load(DATA_DIR + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(DATA_DIR + 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    refs = torch.load(DATA_DIR + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    prevalence_t = torch.load(DATA_DIR + 'prevalence_t_corrected.pt', weights_only=False)
    fh_processed = pd.read_csv(COV_PATH)
    print(f'Y: {Y.shape}, E: {E.shape}, G: {G.shape}')
    print()

    # ── LOO prediction loop ─────────────────────────────────────────────
    total_t0 = time.time()
    completed = 0
    skipped = 0

    for batch_idx in range(n_loaded):
        start = batch_idx * BATCH_SIZE
        stop = (batch_idx + 1) * BATCH_SIZE

        pi_path = os.path.join(OUTPUT_DIR, f'pi_enroll_fixedphi_sex_{start}_{stop}.pt')

        print(f'\n{"=" * 70}')
        print(f'BATCH {batch_idx + 1}/{n_loaded}: samples {start}-{stop} (LOO: exclude batch {batch_idx})')
        print(f'{"=" * 70}')

        if os.path.exists(pi_path):
            print(f'  Already exists: {pi_path}')
            skipped += 1
            continue

        # Subset data
        Y_batch = Y[start:stop]
        E_batch = E[start:stop]
        G_batch = G[start:stop]

        pce_sub = fh_processed.iloc[start:stop].reset_index(drop=True)
        if 'Sex' in pce_sub.columns:
            sex = pce_sub['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
        else:
            sex = pce_sub['sex'].values
        pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
        pcs = pce_sub[pc_cols].values
        G_with_sex = np.column_stack([G_batch, sex, pcs])

        # LOO pool
        phi, psi, kappa, gamma = loo_pool(all_phi, all_psi, all_kappa, all_gamma, batch_idx)
        print(f'  LOO pool (excl batch {batch_idx}): kappa={kappa:.4f}, mean|gamma|={np.abs(gamma).mean():.4f}')

        # Predict
        t0 = time.time()
        pi, loss = fit_and_extract_pi(
            Y_batch, E_batch, G_with_sex, phi, psi, kappa, gamma,
            signature_refs, prevalence_t, disease_names,
            num_epochs=NUM_PRED_EPOCHS, learning_rate=PRED_LR)
        elapsed = (time.time() - t0) / 60
        print(f'  Loss: {loss:.4f}, Time: {elapsed:.1f} min')

        # Save
        torch.save(pi, pi_path)
        print(f'  Saved: {pi_path}')
        completed += 1

        del pi, Y_batch, E_batch, G_batch, G_with_sex
        gc.collect()

    total_elapsed = (time.time() - total_t0) / 60
    print(f'\n{"=" * 70}')
    print(f'LOO PREDICTION COMPLETE')
    print(f'  Completed: {completed}, Skipped (already exist): {skipped}')
    print(f'  Total time: {total_elapsed:.1f} min ({total_elapsed / 60:.1f} hours)')
    print(f'  Output: {OUTPUT_DIR}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
