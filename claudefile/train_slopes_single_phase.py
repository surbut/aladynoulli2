#!/usr/bin/env python3
"""
Ablation: train slope model with SINGLE PHASE (no delta freeze).
Same model, same total epochs (300), same warm-start — just no freezing.

Compare against train_slopes_all_batches.py (two-phase) to test whether
freezing delta is necessary for gamma_slope recovery.

Usage:
    python train_slopes_single_phase.py                    # all 40 batches (0-39)
    python train_slopes_single_phase.py --start_batch 0 --end_batch 2   # test 2 batches
"""

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from clust_huge_amp_vectorized_reparam_slope import AladynSurvivalReparamWithSlope, subset_data

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
BASE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/')
COV_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
OUTPUT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3_single_phase/')

BATCH_SIZE = 10000
N_TOTAL_BATCHES = 40
W = 1e-4
TOTAL_EPOCHS = 300  # same total as two-phase (100 + 200)
LR = 0.1
LR_SLOPE = 0.1


def pool_gamma_from_all_batches():
    files = sorted(glob.glob(str(BASE_CKPT_DIR / '*.pt')))
    gammas = []
    for f in files:
        ck = torch.load(f, weights_only=False)
        g = ck['gamma']
        gammas.append(g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else np.array(g))
    pooled = np.mean(np.stack(gammas), axis=0)
    print(f'Pooled gamma from {len(gammas)} batches: shape {pooled.shape}, mean|gamma|={np.abs(pooled).mean():.4f}')
    return torch.tensor(pooled, dtype=torch.float32)


def pool_psi_from_all_batches():
    files = sorted(glob.glob(str(BASE_CKPT_DIR / '*.pt')))
    psis = []
    for f in files:
        ck = torch.load(f, weights_only=False)
        p = ck['psi']
        psis.append(p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p))
    pooled = np.mean(np.stack(psis), axis=0)
    print(f'Pooled psi from {len(psis)} batches: shape {pooled.shape}')
    return torch.tensor(pooled, dtype=torch.float32)


def load_data():
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)
    E = torch.load(DATA_DIR / 'E_matrix_corrected.pt', weights_only=False)
    G_prs = torch.load(DATA_DIR / 'G_matrix.pt', weights_only=False)
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    refs = torch.load(DATA_DIR / 'reference_trajectories.pt', weights_only=False)
    disease_names = essentials['disease_names']
    prevalence_t = torch.load(DATA_DIR / 'prevalence_t_corrected.pt', weights_only=False)
    signature_refs = refs['signature_refs']

    fh = pd.read_csv(COV_PATH)
    sex = fh['sex'].values
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh[pc_cols].values
    G_full = np.column_stack([G_prs, sex, pcs])

    print(f'Data loaded: Y {Y.shape}, E {E.shape}, G {G_full.shape}')
    return Y, E, G_full, prevalence_t, signature_refs, disease_names


def get_batch_delta_epsilon(batch_idx):
    start = batch_idx * BATCH_SIZE
    end = start + BATCH_SIZE
    pattern = str(BASE_CKPT_DIR / f'*_batch_{start}_{end}.pt')
    files = glob.glob(pattern)
    if not files:
        pattern = str(BASE_CKPT_DIR / f'*_{start}_{end}.pt')
        files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No checkpoint for batch {batch_idx} (range {start}-{end})')

    ck = torch.load(files[0], weights_only=False)
    sd = ck['model_state_dict']
    delta = sd['delta'].detach()
    epsilon = sd['epsilon'].detach()
    return delta, epsilon


def train_one_batch(batch_idx, Y, E, G_full, prevalence_t, signature_refs,
                    pooled_gamma, pooled_psi):
    start = batch_idx * BATCH_SIZE
    stop = start + BATCH_SIZE
    save_path = OUTPUT_DIR / f'slope_model_batch_{start}_{stop}.pt'

    if save_path.exists():
        print(f'  Batch {batch_idx}: already exists, skipping.')
        return

    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G_full, start, stop)
    N, D, T = Y_batch.shape
    K = len(signature_refs)
    P = G_batch.shape[1]

    delta_pretrained, epsilon_pretrained = get_batch_delta_epsilon(batch_idx)

    model = AladynSurvivalReparamWithSlope(
        N=N, D=D, T=T, K=K, P=P,
        G=G_batch, Y=Y_batch, R=signature_refs, W=W,
        prevalence_t=prevalence_t,
        init_sd_scaler=1.0, genetic_scale=1.0,
        signature_references=signature_refs,
        healthy_reference=-5.0, disease_names=None,
        pretrained_gamma=pooled_gamma,
        pretrained_psi=pooled_psi,
        pretrained_delta=delta_pretrained,
        pretrained_epsilon=epsilon_pretrained,
    )

    t0 = time.time()
    history = model.fit_single_phase(
        event_times=E_batch,
        num_epochs=TOTAL_EPOCHS,
        learning_rate=LR,
        lr_slope=LR_SLOPE,
        verbose_every=50,
    )
    elapsed = time.time() - t0

    ckpt_save = {
        'gamma_level': model.gamma_level.data.clone(),
        'gamma_slope': model.gamma_slope.data.clone(),
        'gamma_health': model.gamma_health.clone(),
        'psi': model.psi.data.clone(),
        'delta': model.delta.data.clone(),
        'epsilon': model.epsilon.data.clone(),
        'history': history,
        'config': {
            'N': N, 'D': D, 'T': T, 'K': K, 'P': P,
            'batch': batch_idx, 'start': start, 'stop': stop,
            'W': W, 'lr': LR, 'lr_slope': LR_SLOPE,
            'total_epochs': TOTAL_EPOCHS,
            'training_mode': 'single_phase',
        },
    }
    torch.save(ckpt_save, save_path)
    final_loss = history['losses'][-1]
    gs_mag = torch.abs(model.gamma_slope.data).mean().item()
    print(f'  Batch {batch_idx}: done in {elapsed:.0f}s ({elapsed/60:.1f}min), '
          f'final_loss={final_loss:.2f}, |gamma_slope|={gs_mag:.4f}', flush=True)

    del model, Y_batch, E_batch, G_batch, delta_pretrained, epsilon_pretrained, history
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_batch', type=int, default=0, help='First batch index (default 0)')
    parser.add_argument('--end_batch', type=int, default=40, help='End batch index exclusive (default 40 = all batches 0-39)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('ABLATION: SINGLE-PHASE SLOPE TRAINING')
    print(f'Batches {args.start_batch} to {args.end_batch - 1}')
    print(f'Total epochs: {TOTAL_EPOCHS} (all params free from start)')
    print(f'Warm-starting gamma_level from POOLED 40-batch gamma')
    print('=' * 60)

    pooled_gamma = pool_gamma_from_all_batches()
    pooled_psi = pool_psi_from_all_batches()

    Y, E, G_full, prevalence_t, signature_refs, disease_names = load_data()

    t_total = time.time()
    failed = []
    for b in range(args.start_batch, args.end_batch):
        print(f'\n--- Batch {b}/{args.end_batch - 1} ---', flush=True)
        try:
            train_one_batch(b, Y, E, G_full, prevalence_t, signature_refs,
                            pooled_gamma, pooled_psi)
            print(f'  Completed batch {b}; {b - args.start_batch + 1}/{args.end_batch - args.start_batch} done.', flush=True)
        except Exception as e:
            failed.append(b)
            print(f'  Batch {b} FAILED: {e}', flush=True)
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - t_total
    if failed:
        print(f'\nFailed batches (re-run with --start_batch): {failed}', flush=True)
    print(f'\nAll done! Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)', flush=True)


if __name__ == '__main__':
    # Unbuffer stdout so nohup log shows progress as each batch completes
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    main()
