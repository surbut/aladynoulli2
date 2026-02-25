#!/usr/bin/env python3
"""
Stability of gamma (gamma_level and gamma_slope) across slope training runs,
and of gamma from the equivalent no-slope (nokappa v3) LOO training.

Loads slope checkpoints from 1-phase (and optionally 2-phase) runs, and
no-slope checkpoints from censor_e_batchrun_vectorized_REPARAM_v3_nokappa
(same batches as used for LOO). Reports:
  - Mean and std across batches (per element, then summarized)
  - Pairwise correlation between batches (flattened gamma)
  - Boxplots over same batch subset: no-slope gamma, slope gamma_level, slope gamma_slope

Usage:
    python gamma_stability_slope_runs.py
    python gamma_stability_slope_runs.py --two_phase   # also load 2-phase slope checkpoints
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SLOPE_1PHASE_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3_single_phase/')
SLOPE_2PHASE_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3/')
NOSLOPE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/')
BATCH_SIZE = 10000
N_BATCHES = 40
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc/')


def load_slope_gammas(ckpt_dir, pattern='slope_model_batch_*_*.pt', max_batches=40):
    """Load gamma_level and gamma_slope from all matching checkpoints. Returns (gl_list, gs_list), paths."""
    files = sorted(glob.glob(str(ckpt_dir / pattern)))
    # sort by batch index (start)
    files = sorted(files, key=lambda f: int(Path(f).stem.split('_')[-2]))
    files = files[:max_batches]
    gl_list, gs_list = [], []
    for fp in files:
        ck = torch.load(fp, weights_only=False)
        gl = ck['gamma_level']
        gs = ck['gamma_slope']
        if isinstance(gl, torch.Tensor):
            gl = gl.detach().cpu().numpy()
        if isinstance(gs, torch.Tensor):
            gs = gs.detach().cpu().numpy()
        gl_list.append(gl)
        gs_list.append(gs)
    return gl_list, gs_list, files


def load_noslope_gammas(ckpt_dir, batch_start=0, batch_end=40):
    """Load gamma from no-slope (nokappa v3) checkpoints for the same batch indices as slope LOO.
    Returns list of (P, K) arrays and list of file paths."""
    gamma_list = []
    paths = []
    for b in range(batch_start, batch_end):
        start = b * BATCH_SIZE
        end = (b + 1) * BATCH_SIZE
        pattern = str(ckpt_dir / f'*_batch_{start}_{end}.pt')
        files = glob.glob(pattern)
        if not files:
            pattern = str(ckpt_dir / f'*_{start}_{end}.pt')
            files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f'No no-slope checkpoint for batch {b} ({start}-{end}) in {ckpt_dir}')
        ck = torch.load(files[0], weights_only=False)
        g = ck['gamma']
        if isinstance(g, torch.Tensor):
            g = g.detach().cpu().numpy()
        gamma_list.append(g)
        paths.append(files[0])
    return gamma_list, paths


def stability_stats(arr_list, name):
    """arr_list: list of (P, K) arrays. Compute mean, std across batches, and batch-batch correlation."""
    stack = np.stack(arr_list, axis=0)  # (n_batches, P, K)
    n, p, k = stack.shape
    mean_ = np.mean(stack, axis=0)
    std_ = np.std(stack, axis=0, ddof=1)
    # Per-element coefficient of variation (avoid div by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(np.abs(mean_) > 1e-10, std_ / np.abs(mean_), np.nan)
    # Flatten each batch -> (n_batches, P*K), then correlation matrix
    flat = stack.reshape(n, -1)
    corr = np.corrcoef(flat)
    return {
        'name': name,
        'shape': (p, k),
        'n_batches': n,
        'mean_over_batches': mean_,
        'std_over_batches': std_,
        'cv_over_batches': cv,
        'mean_abs': np.abs(mean_).mean(),
        'std_abs_mean': np.abs(mean_).std(),
        'mean_std': std_.mean(),
        'min_pairwise_corr': np.min(corr[np.triu_indices(n, k=1)]),
        'mean_pairwise_corr': np.mean(corr[np.triu_indices(n, k=1)]),
        'pairwise_corr': corr,
    }


def main():
    parser = argparse.ArgumentParser(description='Gamma stability across slope runs')
    parser.add_argument('--two_phase', action='store_true', help='Also load 2-phase slope checkpoints')
    args = parser.parse_args()

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []

    # ---- 1-phase ----
    if not SLOPE_1PHASE_DIR.exists():
        print(f'1-phase dir not found: {SLOPE_1PHASE_DIR}')
    else:
        gl_list, gs_list, paths = load_slope_gammas(SLOPE_1PHASE_DIR)
        if not gl_list:
            print(f'No checkpoints in {SLOPE_1PHASE_DIR}')
        else:
            print('=' * 60)
            print('1-PHASE SLOPE (single_phase)')
            print(f'  Checkpoints: {len(gl_list)}')
            print('=' * 60)
            s_gl = stability_stats(gl_list, 'gamma_level_1phase')
            s_gs = stability_stats(gs_list, 'gamma_slope_1phase')
            for s in (s_gl, s_gs):
                print(f"\n  {s['name']} shape {s['shape']}")
                print(f"    mean |value| over (P,K): {s['mean_abs']:.6f}")
                print(f"    std of values (across batches) mean: {s['mean_std']:.6f}")
                print(f"    pairwise batch correlation: min={s['min_pairwise_corr']:.4f}, mean={s['mean_pairwise_corr']:.4f}")
                out_rows.append({
                    'run': '1phase',
                    'param': s['name'],
                    'n_batches': s['n_batches'],
                    'shape_P': s['shape'][0],
                    'shape_K': s['shape'][1],
                    'mean_abs': s['mean_abs'],
                    'mean_std_across_batches': s['mean_std'],
                    'min_pairwise_corr': s['min_pairwise_corr'],
                    'mean_pairwise_corr': s['mean_pairwise_corr'],
                })
            # Save 1-phase pooled mean for reference
            gl_mean = np.mean(np.stack(gl_list), axis=0)
            gs_mean = np.mean(np.stack(gs_list), axis=0)
            np.savez(results_dir / 'gamma_stability_1phase_pooled.npz',
                     gamma_level=gl_mean, gamma_slope=gs_mean)

            # ---- No-slope (same batch subset as slope LOO: 40 batches) ----
            n_batches = len(gl_list)
            if NOSLOPE_CKPT_DIR.exists():
                try:
                    noslope_list, _ = load_noslope_gammas(NOSLOPE_CKPT_DIR, batch_start=0, batch_end=n_batches)
                    s_noslope = stability_stats(noslope_list, 'gamma_noslope')
                    print('\n' + '=' * 60)
                    print('NO-SLOPE (nokappa v3, same batches as slope LOO)')
                    print(f'  Checkpoints: {len(noslope_list)}')
                    print('=' * 60)
                    print(f"  gamma_noslope shape {s_noslope['shape']}")
                    print(f"    mean |value| over (P,K): {s_noslope['mean_abs']:.6f}")
                    print(f"    std of values (across batches) mean: {s_noslope['mean_std']:.6f}")
                    print(f"    pairwise batch correlation: min={s_noslope['min_pairwise_corr']:.4f}, mean={s_noslope['mean_pairwise_corr']:.4f}")
                    out_rows.append({
                        'run': 'noslope',
                        'param': s_noslope['name'],
                        'n_batches': s_noslope['n_batches'],
                        'shape_P': s_noslope['shape'][0],
                        'shape_K': s_noslope['shape'][1],
                        'mean_abs': s_noslope['mean_abs'],
                        'mean_std_across_batches': s_noslope['mean_std'],
                        'min_pairwise_corr': s_noslope['min_pairwise_corr'],
                        'mean_pairwise_corr': s_noslope['mean_pairwise_corr'],
                    })
                    stack_noslope = np.stack(noslope_list, axis=0)
                except Exception as e:
                    print(f'  WARNING: could not load no-slope gammas: {e}')
                    stack_noslope = None
            else:
                print(f'  No-slope dir not found: {NOSLOPE_CKPT_DIR}')
                stack_noslope = None

            # Box plot: no-slope gamma, slope gamma_level, slope gamma_slope (same batches, same PRS×Sig cells)
            n_prs, n_sig = 36, 20  # PRS x disease signatures (first 36 and 20 of 47x21)
            stack_gl = np.stack(gl_list, axis=0)   # (n_batches, 47, 21)
            stack_gs = np.stack(gs_list, axis=0)
            mean_abs_gl = np.abs(stack_gl[:, :n_prs, :n_sig]).mean(axis=0)  # (36, 20)
            mean_abs_gs = np.abs(stack_gs[:, :n_prs, :n_sig]).mean(axis=0)
            combined = mean_abs_gl + mean_abs_gs
            flat_idx = np.argsort(combined.ravel())[::-1]
            n_show = 12
            p_show, k_show = np.unravel_index(flat_idx[:n_show], (n_prs, n_sig))
            labels = [f'PRS{p}_Sig{k}' for p, k in zip(p_show, k_show)]
            data_level = [stack_gl[:, p, k] for p, k in zip(p_show, k_show)]
            data_slope = [stack_gs[:, p, k] for p, k in zip(p_show, k_show)]

            n_plot = 3 if stack_noslope is not None else 2
            fig, axes = plt.subplots(n_plot, 1, figsize=(10, 4 * n_plot), sharex=True)
            if n_plot == 2:
                axes = [axes[0], axes[1]]
            ax_idx = 0

            if stack_noslope is not None:
                data_noslope = [stack_noslope[:, p, k] for p, k in zip(p_show, k_show)]
                bp0 = axes[ax_idx].boxplot(data_noslope, labels=labels, patch_artist=True)
                for b in bp0['boxes']:
                    b.set_facecolor('seagreen')
                    b.set_alpha(0.7)
                axes[ax_idx].set_ylabel(f'gamma no-slope ({n_batches} batches)')
                axes[ax_idx].set_title('No-slope (nokappa v3): value across same runs (sampled PRS×Sig)')
                axes[ax_idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
                axes[ax_idx].tick_params(axis='x', rotation=45)
                ax_idx += 1

            bp1 = axes[ax_idx].boxplot(data_level, labels=labels, patch_artist=True)
            for b in bp1['boxes']:
                b.set_facecolor('steelblue')
                b.set_alpha(0.7)
            axes[ax_idx].set_ylabel(f'gamma_level slope ({n_batches} batches)')
            axes[ax_idx].set_title('1-phase slope: value across same runs (sampled PRS×Sig)')
            axes[ax_idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            axes[ax_idx].tick_params(axis='x', rotation=45)
            ax_idx += 1

            bp2 = axes[ax_idx].boxplot(data_slope, labels=labels, patch_artist=True)
            for b in bp2['boxes']:
                b.set_facecolor('firebrick')
                b.set_alpha(0.7)
            axes[ax_idx].set_ylabel(f'gamma_slope ({n_batches} batches)')
            axes[ax_idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            axes[ax_idx].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plot_path = results_dir / 'gamma_stability_boxplot.pdf'
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f'Saved: {plot_path}')

    # ---- 2-phase (optional) ----
    if args.two_phase and SLOPE_2PHASE_DIR.exists():
        gl_list2, gs_list2, paths2 = load_slope_gammas(SLOPE_2PHASE_DIR)
        if gl_list2:
            print('\n' + '=' * 60)
            print('2-PHASE SLOPE')
            print(f'  Checkpoints: {len(gl_list2)}')
            print('=' * 60)
            s_gl2 = stability_stats(gl_list2, 'gamma_level_2phase')
            s_gs2 = stability_stats(gs_list2, 'gamma_slope_2phase')
            for s in (s_gl2, s_gs2):
                print(f"\n  {s['name']} shape {s['shape']}")
                print(f"    mean |value| over (P,K): {s['mean_abs']:.6f}")
                print(f"    std of values (across batches) mean: {s['mean_std']:.6f}")
                print(f"    pairwise batch correlation: min={s['min_pairwise_corr']:.4f}, mean={s['mean_pairwise_corr']:.4f}")
                out_rows.append({
                    'run': '2phase',
                    'param': s['name'],
                    'n_batches': s['n_batches'],
                    'shape_P': s['shape'][0],
                    'shape_K': s['shape'][1],
                    'mean_abs': s['mean_abs'],
                    'mean_std_across_batches': s['mean_std'],
                    'min_pairwise_corr': s['min_pairwise_corr'],
                    'mean_pairwise_corr': s['mean_pairwise_corr'],
                })

    if out_rows:
        df = pd.DataFrame(out_rows)
        csv_path = results_dir / 'gamma_stability_slope_runs.csv'
        df.to_csv(csv_path, index=False)
        print(f'\nSaved: {csv_path}')
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
