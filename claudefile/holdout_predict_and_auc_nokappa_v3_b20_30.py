#!/usr/bin/env python
"""
Holdout prediction and AUC for nokappa v3 configs (Constant, Cos300, Clip) trained on batches 20-30.

1. Pool phi, psi, gamma from each config's 10 checkpoints
2. Fix params, fit delta on holdout
3. Extract pi, compute holdout loss and AUC (dynamic 10-year)

Holdout: samples 390000-400000 (not in training 200k-300k)

Usage:
    python claudefile/holdout_predict_and_auc_nokappa_v3_b20_30.py
    python claudefile/holdout_predict_and_auc_nokappa_v3_b20_30.py --holdout_start 100000 --holdout_end 200000
"""

import argparse
import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))  # for nokappa_prediction_utils
sys.path.insert(0, str(_script_dir / 'aws_offsetmaster'))
sys.path.append(str(_script_dir.parent / 'pyScripts'))

from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi

from nokappa_prediction_utils import fit_and_extract_pi


DROPBOX = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox')
DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running')
CONFIG_DIRS = {
    'constant': DROPBOX / 'nokappa_v3_W1e-4_b20_30',
    'cos300': DROPBOX / 'nokappa_v3_cos300_b20_30',
    'clip': DROPBOX / 'nokappa_v3_clip_b20_30',
}
# Checkpoint patterns (each dir has different suffix)
CONFIG_PATTERNS = {
    'constant': '*REPARAM_NOKAPPA_W0.0001_batch_*.pt',
    'cos300': '*REPARAM_NOKAPPA_COS300_W0.0001_batch_*.pt',
    'clip': '*REPARAM_NOKAPPA_CLIP_W0.0001_batch_*.pt',
}


def pool_from_checkpoints(ckpt_dir, pattern):
    """Load all checkpoints, average phi, psi, gamma. kappa=1 for all."""
    files = sorted(glob.glob(str(ckpt_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir} matching {pattern}")

    phis, psis, gammas = [], [], []
    for fp in files:
        ckpt = torch.load(fp, weights_only=False)
        phi = ckpt.get('phi') or ckpt.get('model_state_dict', {}).get('phi')
        psi = ckpt.get('psi') or ckpt.get('model_state_dict', {}).get('psi')
        gamma = ckpt.get('gamma') or ckpt.get('model_state_dict', {}).get('gamma')
        if phi is None or gamma is None:
            continue
        phis.append(phi.numpy() if torch.is_tensor(phi) else np.array(phi))
        psis.append(psi.numpy() if torch.is_tensor(psi) else np.array(psi))
        gammas.append(gamma.numpy() if torch.is_tensor(gamma) else np.array(gamma))

    phi_pooled = np.mean(phis, axis=0)
    psi_pooled = np.mean(psis, axis=0)
    gamma_pooled = np.mean(gammas, axis=0)
    kappa = 1.0  # nokappa
    return phi_pooled, psi_pooled, gamma_pooled, kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout_start', type=int, default=390000)
    parser.add_argument('--holdout_end', type=int, default=400000)
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--covariates_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    parser.add_argument('--pce_path', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Check config dirs exist
    for name, d in CONFIG_DIRS.items():
        if not d.exists():
            print(f"Missing: {d} (run training first)")
            return

    # Load holdout data
    print("Loading holdout data...")
    Y_full = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
    E_full = torch.load(str(data_dir / 'E_enrollment_full.pt'), weights_only=False)
    G_full = torch.load(str(data_dir / 'G_matrix.pt'), weights_only=False)
    essentials = torch.load(str(data_dir / 'model_essentials.pt'), weights_only=False)
    refs = torch.load(str(data_dir / 'reference_trajectories.pt'), weights_only=False)
    prevalence_t = torch.load(str(data_dir / 'prevalence_t_corrected.pt'), weights_only=False)
    disease_names = essentials['disease_names']
    signature_refs = refs['signature_refs']

    idx = list(range(args.holdout_start, args.holdout_end))
    Y_holdout = Y_full[idx]
    E_holdout = E_full[idx]
    G_holdout = G_full[idx]
    del Y_full, E_full, G_full
    gc.collect()

    fh = pd.read_csv(args.covariates_path)
    sex = fh['Sex'].map({'Female': 0, 'Male': 1}).astype(int).values if 'Sex' in fh.columns else fh['sex'].values
    sex_h = sex[args.holdout_start:args.holdout_end]
    pc_cols = [f'f.22009.0.{i}' for i in range(1, 11)]
    pcs = fh.iloc[args.holdout_start:args.holdout_end][pc_cols].values
    G_holdout = np.column_stack([G_holdout, sex_h, pcs])

    pce_df = pd.read_csv(args.pce_path).iloc[args.holdout_start:args.holdout_end].reset_index(drop=True)
    if 'Sex' not in pce_df.columns and 'sex' in pce_df.columns:
        pce_df['Sex'] = pce_df['sex'].map({0: 'Female', 1: 'Male'}).fillna('Unknown')
    if 'sex' not in pce_df.columns and 'Sex' in pce_df.columns:
        pce_df['sex'] = pce_df['Sex'].map({'Female': 0, 'Male': 1}).fillna(-1)
    if 'age' not in pce_df.columns and 'Age' in pce_df.columns:
        pce_df['age'] = pce_df['Age']

    N = Y_holdout.shape[0]
    print(f"Holdout: {N} individuals ({args.holdout_start}-{args.holdout_end})\n")

    # Pool and fit each config
    results = {}
    for config_name, ckpt_dir in CONFIG_DIRS.items():
        pattern = CONFIG_PATTERNS[config_name]
        print(f"Pooling {config_name}...")
        try:
            phi, psi, gamma, kappa = pool_from_checkpoints(ckpt_dir, pattern)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Fitting delta on holdout ({args.num_epochs} epochs)...")
        pi, loss = fit_and_extract_pi(
            Y_holdout, E_holdout, G_holdout,
            phi, psi, kappa, gamma,
            signature_refs, prevalence_t, disease_names,
            num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        )
        results[config_name] = {'pi': pi, 'loss': loss}
        print(f"  Holdout loss: {loss:.4f}\n")

    if not results:
        print("No configs succeeded.")
        return

    # AUC evaluation
    print("=" * 70)
    print("DYNAMIC 10-YEAR AUC (holdout)")
    print("=" * 70)

    auc_results = {}
    for config_name, data in results.items():
        print(f"\nEvaluating {config_name}...")
        res = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=data['pi'],
            Y_100k=Y_holdout,
            E_100k=E_holdout,
            disease_names=disease_names,
            pce_df=pce_df,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10,
        )
        auc_results[config_name] = res

    # Print comparison table
    configs = list(results.keys())
    diseases = sorted(set().union(*[set(r.keys()) for r in auc_results.values()]))
    print(f"\n{'DISEASE':<28} " + " ".join(f"{c:>14}" for c in configs) + f" {'BEST':>8}")
    print("-" * (28 + 14 * len(configs) + 10))

    for d in diseases:
        row = [f"{d:<28}"]
        vals = []
        for c in configs:
            m = auc_results.get(c, {}).get(d, {})
            auc = m.get('auc', np.nan)
            ci_l = m.get('ci_lower', np.nan)
            ci_u = m.get('ci_upper', np.nan)
            vals.append(auc)
            if np.isnan(auc):
                row.append(f"{'N/A':>14}")
            else:
                row.append(f"{auc:.3f} ({ci_l:.3f}-{ci_u:.3f})")
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        best = configs[max(valid, key=lambda x: x[1])[0]] if valid else "—"
        row.append(f"{best:>8}")
        print(" ".join(row))

    # Summary
    print("\n" + "=" * 70)
    print("HOLDOUT LOSS")
    for c, data in results.items():
        print(f"  {c}: {data['loss']:.4f}")
    best_loss = min(results.items(), key=lambda x: x[1]['loss'])
    print(f"  -> Best: {best_loss[0]} (lowest loss)")

    # Mean AUC
    rows = []
    for d, metrics in list(auc_results.get(configs[0], {}).items()):
        row = {'disease': d}
        for c in configs:
            m = auc_results.get(c, {}).get(d, {})
            row[f'{c}_auc'] = m.get('auc', np.nan)
            row[f'{c}_ci_lower'] = m.get('ci_lower', np.nan)
            row[f'{c}_ci_upper'] = m.get('ci_upper', np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in configs:
        m = df[f'{c}_auc'].mean()
        print(f"  Mean AUC {c}: {m:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = Path(args.output_dir) / 'holdout_auc_nokappa_v3_b20_30.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")


if __name__ == '__main__':
    main()
