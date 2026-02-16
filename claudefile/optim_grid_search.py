#!/usr/bin/env python
"""
Optimization grid search for NCP (no-kappa reparam) model.

Trains multiple configs on batch 0 (10k patients) and compares metrics.
All configs use kappa=1 (fixed). Searches over LR, scheduler, epochs, clipping.

Usage:
    python claudefile/optim_grid_search.py                                    # Run all configs
    python claudefile/optim_grid_search.py --configs nok_lr01_200 nok_lr01_300  # Run specific
    python claudefile/optim_grid_search.py --compare                           # Compare results
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))
from clust_huge_amp_vectorized_reparam import (
    AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest,
    subset_data,
)
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Config definitions
# ============================================================
CONFIGS = {
    'nok_lr01_200': {
        'lr': 0.1, 'scheduler': 'none', 'epochs': 200,
        'clip': None, 'patience': 75, 'desc': 'v1-style, no kappa',
    },
    'nok_lr01_300': {
        'lr': 0.1, 'scheduler': 'none', 'epochs': 300,
        'clip': None, 'patience': 75, 'desc': 'More epochs',
    },
    'nok_lr01_500': {
        'lr': 0.1, 'scheduler': 'none', 'epochs': 500,
        'clip': None, 'patience': 75, 'desc': 'Even more epochs',
    },
    'nok_lr01_cos300': {
        'lr': 0.1, 'scheduler': 'cosine', 'epochs': 300,
        'clip': None, 'patience': 75, 'desc': 'Cosine schedule',
    },
    'nok_lr01_cos500': {
        'lr': 0.1, 'scheduler': 'cosine', 'epochs': 500,
        'clip': None, 'patience': 75, 'desc': 'Cosine + more epochs',
    },
    'nok_lr005_300': {
        'lr': 0.05, 'scheduler': 'none', 'epochs': 300,
        'clip': None, 'patience': 75, 'desc': 'Lower LR',
    },
    'nok_lr02_200': {
        'lr': 0.2, 'scheduler': 'none', 'epochs': 200,
        'clip': None, 'patience': 75, 'desc': 'Higher LR',
    },
    'nok_lr01_clip300': {
        'lr': 0.1, 'scheduler': 'none', 'epochs': 300,
        'clip': 5.0, 'patience': 75, 'desc': 'With grad clipping',
    },
}


def load_data(data_dir, covariates_path, start_index=0, end_index=10000):
    """Load batch 0 data (same as training scripts)."""
    print("Loading data...")
    Y = torch.load(data_dir + 'Y_tensor.pt', weights_only=False)
    E = torch.load(data_dir + 'E_matrix_corrected.pt', weights_only=False)
    G = torch.load(data_dir + 'G_matrix.pt', weights_only=False)
    essentials = torch.load(data_dir + 'model_essentials.pt', weights_only=False)

    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G,
                                                      start_index=start_index,
                                                      end_index=end_index)
    del Y

    fh_processed = pd.read_csv(covariates_path)
    if 'Sex' in fh_processed.columns:
        sex = fh_processed['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values
    else:
        sex = fh_processed['sex'].values
    sex_batch = sex[start_index:end_index]
    pc_columns = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh_processed.iloc[start_index:end_index][pc_columns].values
    G_with_sex = np.column_stack([G_batch, sex_batch, pcs])

    refs = torch.load(data_dir + 'reference_trajectories.pt', weights_only=False)
    signature_refs = refs['signature_refs']
    del refs
    prevalence_t = torch.load(data_dir + 'prevalence_t_corrected.pt', weights_only=False)
    initial_psi = torch.load(data_dir + 'initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load(data_dir + 'initial_clusters_400k.pt', weights_only=False)

    print(f"Y_batch: {Y_batch.shape}, G_with_sex: {G_with_sex.shape}")
    return {
        'Y_batch': Y_batch, 'E_batch': E_batch, 'G_with_sex': G_with_sex,
        'essentials': essentials, 'signature_refs': signature_refs,
        'prevalence_t': prevalence_t, 'initial_psi': initial_psi,
        'initial_clusters': initial_clusters,
    }


def train_config(config_name, config, data, output_dir, W=0.0001, K=20):
    """Train one config and save metrics."""
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config_name} -- {config['desc']}")
    print(f"  LR={config['lr']}, scheduler={config['scheduler']}, "
          f"epochs={config['epochs']}, clip={config['clip']}")
    print(f"{'=' * 70}")

    # Build model
    torch.manual_seed(42)
    np.random.seed(42)

    Y_batch = data['Y_batch']
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_batch.shape[0], D=Y_batch.shape[1], T=Y_batch.shape[2],
        K=K, P=data['G_with_sex'].shape[1],
        init_sd_scaler=1e-1, G=data['G_with_sex'], Y=Y_batch,
        genetic_scale=1, W=W, R=0,
        prevalence_t=data['prevalence_t'],
        signature_references=data['signature_refs'],
        healthy_reference=True,
        disease_names=data['essentials']['disease_names'],
        learn_kappa=False,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    model.initialize_params(true_psi=data['initial_psi'])
    model.clusters = data['initial_clusters']

    # Set up optimizer
    lr = config['lr']
    param_groups = [
        {'params': [model.delta], 'lr': lr},
        {'params': [model.epsilon], 'lr': lr * 0.1},
        {'params': [model.psi], 'lr': lr * 0.1},
        {'params': [model.gamma], 'lr': lr},
    ]
    optimizer = optim.Adam(param_groups)

    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=lr * 0.01
        )

    all_params = [model.delta, model.epsilon, model.psi, model.gamma]

    # Early stopping setup
    patience = config.get('patience', None)
    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    # Training loop
    metrics = []
    t0 = time.time()
    stopped_early = False

    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        loss = model.compute_loss(data['E_batch'])

        if torch.isnan(loss):
            print(f"  Epoch {epoch}: NaN loss, stopping")
            break

        loss.backward()

        if config['clip'] is not None:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=config['clip'])

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        cur_loss = loss.item()

        # Early stopping check
        if patience is not None:
            if cur_loss < best_loss - 1e-4:
                best_loss = cur_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Epoch {epoch}: early stopping (patience={patience}), "
                          f"best_loss={best_loss:.2f}")
                    stopped_early = True
                    model.load_state_dict(best_state)
                    break

        # Log metrics every 10 epochs
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            with torch.no_grad():
                gamma_mag = model.gamma.abs().mean().item()
                delta_mag = model.delta.abs().mean().item()
                Ggamma = (torch.tensor(data['G_with_sex'], dtype=torch.float32) @ model.gamma)
                Ggamma_mag = Ggamma.abs().mean().item()
                ratio = delta_mag / max(Ggamma_mag, 1e-8)
                psi_vals = model.psi.detach()
                psi_range = (psi_vals.max() - psi_vals.min()).item()

                # Decompose loss: total = NLL + W*gp_loss (avoids E_batch shape mismatch)
                gp_loss = model.compute_gp_prior_loss().item() if W > 0 else 0.0
                nll = cur_loss - W * gp_loss

            lr_now = scheduler.get_last_lr()[0] if scheduler else lr
            elapsed = (time.time() - t0) / 60

            row = {
                'epoch': epoch, 'loss': cur_loss, 'nll': nll,
                'gp_loss': cur_loss - nll,
                'mean_abs_gamma': gamma_mag,
                'mean_abs_delta': delta_mag,
                'mean_abs_Ggamma': Ggamma_mag,
                'delta_over_Ggamma': ratio,
                'psi_range': psi_range,
                'lr': lr_now,
            }
            metrics.append(row)

            if epoch % 50 == 0:
                print(f"  Epoch {epoch:4d}: loss={cur_loss:.2f}, |gamma|={gamma_mag:.4f}, "
                      f"|delta|={delta_mag:.3f}, |Gg|={Ggamma_mag:.3f}, "
                      f"d/Gg={ratio:.2f}, psi_rng={psi_range:.1f}, "
                      f"lr={lr_now:.1e}, t={elapsed:.1f}m")

    elapsed_total = (time.time() - t0) / 60
    stop_info = f" (early stop @ {metrics[-1]['epoch']})" if stopped_early else ""
    print(f"  Done in {elapsed_total:.1f} min, final loss={metrics[-1]['loss']:.2f}{stop_info}")

    # Save results
    config_dir = Path(output_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics CSV
    csv_path = config_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'config_name': config_name,
        'final_metrics': metrics[-1],
        'stopped_early': stopped_early,
        'best_epoch': metrics[-1]['epoch'] if stopped_early else config['epochs'] - 1,
    }, config_dir / 'checkpoint.pt')

    # Save config
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save per-config loss plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        df = pd.DataFrame(metrics)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
        axes[0].plot(df['epoch'], df['loss'], 'b-', linewidth=1.5)
        axes[0].set_title(f'{config_name}: Loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(df['epoch'], df['mean_abs_gamma'], 'r-', linewidth=1.5)
        axes[1].set_title(f'{config_name}: |gamma|')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mean |gamma|')
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(df['epoch'], df['delta_over_Ggamma'], 'g-', linewidth=1.5)
        axes[2].set_title(f'{config_name}: delta/Ggamma')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('ratio')
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config_dir / 'loss_curves.png', dpi=120, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    print(f"  Saved to {config_dir}/")
    return metrics


def compare_results(output_dir):
    """Load all results and print comparison + generate plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    configs_found = {}

    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and (d / 'metrics.csv').exists():
            name = d.name
            metrics = pd.read_csv(d / 'metrics.csv')
            with open(d / 'config.json') as f:
                config = json.load(f)
            configs_found[name] = {'metrics': metrics, 'config': config}

    if not configs_found:
        print("No results found. Run training first.")
        return

    # Print summary table
    print(f"\n{'=' * 120}")
    print(f"GRID SEARCH RESULTS ({len(configs_found)} configs)")
    print(f"{'=' * 120}")
    header = (f"{'CONFIG':<22} {'LR':>5} {'SCHED':>7} {'EP':>4} {'CLIP':>5} "
              f"{'LOSS':>8} {'|gamma|':>8} {'|delta|':>8} {'|Gg|':>8} "
              f"{'d/Gg':>6} {'psi_rng':>8}")
    print(header)
    print("-" * 120)

    for name, data in sorted(configs_found.items()):
        m = data['metrics'].iloc[-1]
        c = data['config']
        clip_str = f"{c['clip']}" if c['clip'] else "none"
        print(f"{name:<22} {c['lr']:>5.3f} {c['scheduler']:>7} {c['epochs']:>4} "
              f"{clip_str:>5} {m['loss']:>8.2f} {m['mean_abs_gamma']:>8.4f} "
              f"{m['mean_abs_delta']:>8.3f} {m['mean_abs_Ggamma']:>8.3f} "
              f"{m['delta_over_Ggamma']:>6.2f} {m['psi_range']:>8.1f}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs_found)))

    for i, (name, data) in enumerate(sorted(configs_found.items())):
        df = data['metrics']
        c = colors[i]
        label = f"{name} (final={df['loss'].iloc[-1]:.1f})"

        axes[0, 0].plot(df['epoch'], df['loss'], color=c, label=label, linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(df['epoch'], df['mean_abs_gamma'], color=c, label=name, linewidth=1.5, alpha=0.8)
        axes[1, 0].plot(df['epoch'], df['delta_over_Ggamma'], color=c, label=name, linewidth=1.5, alpha=0.8)
        axes[1, 1].plot(df['epoch'], df['psi_range'], color=c, label=name, linewidth=1.5, alpha=0.8)

    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('|gamma| (genetic effect magnitude)', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mean |gamma|')
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('|delta| / |G@gamma| ratio (lower = genetics explain more)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ratio')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Psi range (cluster differentiation)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('max(psi) - min(psi)')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimization grid search for NCP model')
    parser.add_argument('--configs', nargs='*', default=None,
                        help='Specific configs to run (default: all)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare existing results instead of training')
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).parent / 'grid_results'))
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override epochs for all configs (for quick tests)')
    args = parser.parse_args()

    if args.compare:
        compare_results(args.output_dir)
        return

    # Determine which configs to run
    if args.configs:
        to_run = {k: CONFIGS[k] for k in args.configs if k in CONFIGS}
        unknown = [k for k in args.configs if k not in CONFIGS]
        if unknown:
            print(f"Unknown configs: {unknown}")
            print(f"Available: {list(CONFIGS.keys())}")
            return
    else:
        to_run = CONFIGS

    if args.num_epochs is not None:
        for cfg in to_run.values():
            cfg['epochs'] = args.num_epochs

    print(f"Running {len(to_run)} configs: {list(to_run.keys())}")
    print(f"Output: {args.output_dir}")

    # Load data once
    data = load_data(args.data_dir, args.covariates_path,
                     args.start_index, args.end_index)

    # Run each config
    t_total = time.time()
    for name, config in to_run.items():
        # Skip if already done
        config_dir = Path(args.output_dir) / name
        if (config_dir / 'metrics.csv').exists():
            print(f"\nSkipping {name} (already exists)")
            continue
        train_config(name, config, data, args.output_dir)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'=' * 70}")
    print(f"All configs complete in {total_min:.0f} min")
    print(f"{'=' * 70}")

    # Auto-compare
    compare_results(args.output_dir)


if __name__ == '__main__':
    main()
