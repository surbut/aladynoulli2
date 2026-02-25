#!/usr/bin/env python3
"""
LOO prediction for 1-phase slope — same strategy as no-slope LOO.

Prerequisites:
  Step 1: Train 1-phase slope on all 40 batches (10k each), like no-slope training:
          python train_slopes_single_phase.py --start_batch 0 --end_batch 40
          → produces slope_model_nokappa_v3_single_phase/slope_model_batch_{start}_{stop}.pt

  Step 2: This script. For each batch i (0..39):
          - Pool gamma_level, gamma_slope, psi, epsilon, gamma_health from the other 39 slope checkpoints
          - Fit delta-only on batch i (enrollment E) with fit_slope_delta_and_extract_pi
          - Save pi (same layout as no-slope LOO)

Same as no-slope LOO: pool 39 checkpoints, fit delta on held-out batch, save pi. All CPU.

Usage:
    nohup python run_loo_slope_1phase_all40.py > loo_slope_1phase.log 2>&1 &
    python run_loo_slope_1phase_all40.py --start_batch 0 --end_batch 2   # test 2 folds
"""

import argparse
import gc
import glob
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
SLOPE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3_single_phase/')
BASE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/')
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
OUTPUT_PI_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_slope_1phase_loo_all40/')

BATCH_SIZE = 10000
N_BATCHES = 40
PRED_EPOCHS = 200

sys.path.insert(0, str(Path(__file__).parent))
from slope_holdout_auc import fit_slope_delta_and_extract_pi, load_and_pool_slope_params

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def get_no_slope_delta_for_batch(batch_idx):
    """Load delta for one batch from no-slope checkpoint (for pretrained_delta in slope prediction)."""
    start = batch_idx * BATCH_SIZE
    end = start + BATCH_SIZE
    pattern = str(BASE_CKPT_DIR / f'*_batch_{start}_{end}.pt')
    files = glob.glob(pattern)
    if not files:
        pattern = str(BASE_CKPT_DIR / f'*_{start}_{end}.pt')
        files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No no-slope checkpoint for batch {batch_idx} ({start}-{end})')
    ck = torch.load(files[0], weights_only=False)
    sd = ck.get('model_state_dict', ck)
    delta = sd['delta'].detach()
    return delta


def main():
    # Unbuffer stdout so nohup log shows progress as each batch completes
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description='LOO slope prediction: pool 39 slope checkpoints, predict on held-out. Run after train_slopes_single_phase 0-40.')
    parser.add_argument('--start_batch', type=int, default=0)
    parser.add_argument('--resume_from', type=int, default=None)
    parser.add_argument('--end_batch', type=int, default=N_BATCHES)
    parser.add_argument('--pred_epochs', type=int, default=PRED_EPOCHS)
    args = parser.parse_args()
    if args.resume_from is not None:
        args.start_batch = args.resume_from

    OUTPUT_PI_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('LOO 1-PHASE SLOPE PREDICTION (pool 39, predict held-out)')
    print('=' * 70)
    print(f'Slope checkpoints: {SLOPE_CKPT_DIR}')
    print(f'Pi output:        {OUTPUT_PI_DIR}')
    print(f'Batches: {args.start_batch} to {args.end_batch - 1}')
    print(flush=True)

    # Check we have 40 slope checkpoints
    n_ckpt = len(sorted(glob.glob(str(SLOPE_CKPT_DIR / 'slope_model_batch_*_*.pt'))))
    if n_ckpt < N_BATCHES:
        print(f'WARNING: expected {N_BATCHES} slope checkpoints, found {n_ckpt}. Run first:')
        print(f'  python train_slopes_single_phase.py --start_batch 0 --end_batch 40')
        sys.exit(1)

    # Load shared data
    print('Loading data...')
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)
    E_enroll = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)
    G_prs = torch.load(DATA_DIR / 'G_matrix.pt', weights_only=False)
    refs = torch.load(DATA_DIR / 'reference_trajectories.pt', weights_only=False)
    prevalence_t = torch.load(DATA_DIR / 'prevalence_t_corrected.pt', weights_only=False)
    signature_refs = refs['signature_refs']

    fh = pd.read_csv(COV_PATH)
    sex = fh['sex'].values
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh[pc_cols].values
    G_prs_np = _to_numpy(G_prs) if torch.is_tensor(G_prs) else G_prs
    G_full = np.column_stack([G_prs_np, sex, pcs])
    print(f'Y: {Y.shape}, E_enroll: {E_enroll.shape}, G_full: {G_full.shape}')
    sys.stdout.flush()

    completed = 0
    skipped = 0
    total_t0 = time.time()

    for batch_idx in range(args.start_batch, args.end_batch):
        start = batch_idx * BATCH_SIZE
        stop = (batch_idx + 1) * BATCH_SIZE
        pi_path = OUTPUT_PI_DIR / f'pi_enroll_fixedphi_sex_{start}_{stop}.pt'

        print(f'\n{"=" * 70}')
        print(f'BATCH {batch_idx + 1}/{args.end_batch}: LOO exclude batch {batch_idx} (samples {start}-{stop})')
        print(f'{"=" * 70}')

        if pi_path.exists():
            print(f'  Pi already exists: {pi_path.name}, skipping.', flush=True)
            skipped += 1
            continue

        # LOO: pool slope params from the other 39 batches
        train_indices = [j for j in range(N_BATCHES) if j != batch_idx]
        gamma_level, gamma_slope, psi, epsilon, gamma_health = load_and_pool_slope_params(
            train_indices, slope_ckpt_dir=SLOPE_CKPT_DIR)

        # Held-out batch data
        Y_batch = Y[start:stop]
        E_batch = E_enroll[start:stop]
        G_batch = G_full[start:stop]
        delta_init = get_no_slope_delta_for_batch(batch_idx)

        t0 = time.time()
        pi, nll = fit_slope_delta_and_extract_pi(
            Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
            gamma_level, gamma_slope, psi, epsilon,
            gamma_health=gamma_health,
            pretrained_delta=delta_init,
            n_epochs=args.pred_epochs,
        )
        elapsed = (time.time() - t0) / 60

        torch.save(pi.cpu(), pi_path)
        print(f'  Saved {pi_path.name} (NLL={nll:.4f}, {elapsed:.1f} min). Completed {completed + 1}/{args.end_batch - args.start_batch} LOO folds.', flush=True)
        completed += 1

        del pi, Y_batch, E_batch, G_batch, delta_init, gamma_level, gamma_slope, psi, epsilon
        gc.collect()

    total_elapsed = (time.time() - total_t0) / 60
    print(f'\n{"=" * 70}')
    print('LOO 1-PHASE SLOPE PREDICTION COMPLETE')
    print(f'  Completed: {completed}, Skipped: {skipped}')
    print(f'  Total time: {total_elapsed:.1f} min ({total_elapsed / 60:.1f} h)')
    print(f'  Pi output: {OUTPUT_PI_DIR}')
    print(f'{"=" * 70}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
