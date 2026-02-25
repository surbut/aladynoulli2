#!/usr/bin/env python3
"""
End-to-end LOO-style slope evaluation on the full 400k, using the
same logic as the batch-0 notebook:

- For each batch b = 0..39 (10k patients each):
  - LOO-style pool: pool slope checkpoints from all OTHER batches
    (0..39 except b) → shared (gamma_level, gamma_slope, psi, epsilon, gamma_health)
  - Fit delta on batch b only → pi_b (N=10k, D, T)
- Concatenate all pi_b → pi_full (400k, D, T)
- Run static 10yr, dynamic 10yr, static 1yr, dynamic 1yr AUC
  using the same fig5utils evaluation used for holdout.

This AVVOIDS the older LOO scripts that operated via pre-saved batch
pi files, and instead mirrors the trusted holdout + batch0 code path.

Usage:
    python slope_loo_like_400k_eval.py
    python slope_loo_like_400k_eval.py --n_bootstraps 100
    python slope_loo_like_400k_eval.py --n_patients 100000  # optional 100k subset
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from slope_holdout_auc import (
    BATCH_SIZE,
    DATA_DIR,
    PCE_PATH,
    load_data,
    load_and_pool_slope_params,
    fit_slope_delta_and_extract_pi,
)


CLAUDE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CLAUDE_DIR / "results_slope_loo_like"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_from_pi(pi, Y, E, disease_names, pce_df, n_bootstraps, suffix: str):
    """
    Run static/dynamic 10yr + static/dynamic 1yr AUC using the same
    fig5utils helpers as the holdout pipeline.
    """
    from fig5utils import (
        evaluate_major_diseases_wsex_with_bootstrap_from_pi,
        evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
    )

    all_rows = []

    # Static 10yr
    print("\n[LOO-like] Static 10yr AUC...")
    res_static10 = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi,
        Y_100k=Y,
        E_100k=E,
        disease_names=disease_names,
        pce_df=pce_df,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=10,
    )
    for dg, metrics in res_static10.items():
        row = {"model": "slope_loo_like", "horizon": "static_10yr", "disease": dg}
        if isinstance(metrics, dict):
            row.update(metrics)
        all_rows.append(row)

    # Dynamic 10yr
    print("\n[LOO-like] Dynamic 10yr AUC...")
    res_dyn10 = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
        pi=pi,
        Y_100k=Y,
        E_100k=E,
        disease_names=disease_names,
        pce_df=pce_df,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=10,
    )
    for dg, metrics in res_dyn10.items():
        row = {"model": "slope_loo_like", "horizon": "dynamic_10yr", "disease": dg}
        if isinstance(metrics, dict):
            row.update(metrics)
        all_rows.append(row)

    # Static 1yr
    print("\n[LOO-like] Static 1yr AUC...")
    res_static1 = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi,
        Y_100k=Y,
        E_100k=E,
        disease_names=disease_names,
        pce_df=pce_df,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=1,
    )
    for dg, metrics in res_static1.items():
        row = {"model": "slope_loo_like", "horizon": "static_1yr", "disease": dg}
        if isinstance(metrics, dict):
            row.update(metrics)
        all_rows.append(row)

    # Dynamic 1yr
    print("\n[LOO-like] Dynamic 1yr AUC...")
    res_dyn1 = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
        pi=pi,
        Y_100k=Y,
        E_100k=E,
        disease_names=disease_names,
        pce_df=pce_df,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=1,
    )
    for dg, metrics in res_dyn1.items():
        row = {"model": "slope_loo_like", "horizon": "dynamic_1yr", "disease": dg}
        if isinstance(metrics, dict):
            row.update(metrics)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / f"loo_like_auc_{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[LOO-like] AUC results saved to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Slope LOO-like evaluation on 400k (batch-wise pooling).")
    parser.add_argument("--n_patients", type=int, default=400000,
                        help="Number of patients to evaluate (default 400000; use 100000 for first 100k).")
    parser.add_argument("--n_bootstraps", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Delta-only fitting epochs per batch (default 200, as in holdout).")
    args = parser.parse_args()

    print("=" * 70)
    print("Slope LOO-like evaluation (pool all OTHER batches per test batch)")
    print("=" * 70)

    # Load full data (same helper as holdout pipeline)
    print("Loading full data...")
    Y_full, E_full, G_full, prevalence_t, signature_refs, disease_names, pce_df_full = load_data()
    N_total = Y_full.shape[0]
    print(f"  Y: {Y_full.shape}, E: {E_full.shape}, G: {G_full.shape}, pce: {len(pce_df_full)}")

    # Optional restriction to first N patients
    N_eval = min(args.n_patients, N_total)
    n_batches = N_eval // BATCH_SIZE
    if N_eval % BATCH_SIZE != 0:
        raise ValueError(f"n_patients={N_eval} is not a multiple of BATCH_SIZE={BATCH_SIZE}")
    print(f"Evaluating first {N_eval} patients ({n_batches} batches of {BATCH_SIZE})")

    # Preallocate pi_full
    N, D, T = Y_full.shape
    # We'll infer K from signature_refs in the first batch's fit
    pi_batches = []

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        stop = start + BATCH_SIZE
        print("\n" + "-" * 70)
        print(f"[Batch {batch_idx}] patients {start}-{stop}")

        # LOO-style pool indices: all batches except this one
        train_indices = [i for i in range(n_batches) if i != batch_idx]
        print(f"  LOO-style train indices (excluding {batch_idx}): {train_indices[0]}..{train_indices[-1]} "
              f"(total {len(train_indices)})")

        # Pool slope params
        gl, gs, psi, eps, gh = load_and_pool_slope_params(train_indices)

        # Slice data for this batch
        Y_batch = Y_full[start:stop]
        E_batch = E_full[start:stop]
        G_batch = G_full[start:stop]
        pce_batch = pce_df_full.iloc[start:stop].reset_index(drop=True)

        # Fit delta and get pi for this batch
        print("  Fitting delta on this batch (LOO-style pool)...")
        pi_b, nll_b = fit_slope_delta_and_extract_pi(
            Y_batch, E_batch, G_batch, prevalence_t, signature_refs,
            gl, gs, psi, eps,
            gamma_health=gh,
            pretrained_delta=None,
            n_epochs=args.n_epochs,
        )
        print(f"  Batch {batch_idx}: NLL = {nll_b:.4f}")
        pi_batches.append(pi_b.detach().cpu())

        # Free batch-local tensors
        del gl, gs, psi, eps, gh, Y_batch, E_batch, G_batch, pce_batch, pi_b
        gc.collect()

    # Concatenate all batch π tensors
    print("\nConcatenating batch π tensors...")
    pi_full = torch.cat(pi_batches, dim=0)
    del pi_batches
    gc.collect()
    print(f"  pi_full shape: {pi_full.shape}")

    # Save π for reuse
    suffix = f"{N_eval}" if N_eval != N_total else "400k"
    pi_path = RESULTS_DIR / f"pi_slope_loo_like_{suffix}.pt"
    torch.save(pi_full, pi_path)
    print(f"Saved LOO-like π to {pi_path}")

    # Run AUC evaluation on the same subset
    Y_eval = Y_full[:N_eval]
    E_eval = E_full[:N_eval]
    pce_eval = pce_df_full.iloc[:N_eval].reset_index(drop=True)
    evaluate_from_pi(pi_full, Y_eval, E_eval, disease_names, pce_eval,
                     n_bootstraps=args.n_bootstraps, suffix=suffix)

    print("\n" + "=" * 70)
    print("Slope LOO-like evaluation COMPLETE")
    print(f"Results in {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

