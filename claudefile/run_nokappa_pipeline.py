#!/usr/bin/env python3
"""
Full nokappa pipeline: wait for training → pool → LOO predict → AUC compare.
Designed to run unattended overnight.

Usage:
    nohup env PYTHONUNBUFFERED=1 python claudefile/run_nokappa_pipeline.py > claudefile/logs/nokappa_pipeline.log 2>&1 &
"""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
NOKAPPA_TRAIN_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa'
DATA_DIR = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/'
EXPECTED_BATCHES = 40


def count_checkpoints():
    pattern = os.path.join(NOKAPPA_TRAIN_DIR, 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_*_*.pt')
    return len(glob.glob(pattern))


def wait_for_training():
    """Wait until all 40 batch checkpoints exist."""
    print("=" * 80)
    print("STEP 0: Waiting for nokappa training to complete")
    print("=" * 80)
    while True:
        n = count_checkpoints()
        if n >= EXPECTED_BATCHES:
            print(f"  All {EXPECTED_BATCHES} checkpoints found!")
            return
        print(f"  {n}/{EXPECTED_BATCHES} checkpoints found, waiting 5 min...")
        time.sleep(300)


def run_step(step_name, cmd):
    """Run a subprocess, print output, and check for errors."""
    print(f"\n{'=' * 80}")
    print(f"STEP: {step_name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 80}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"FAILED: {step_name} (exit code {result.returncode}) after {elapsed:.1f} min")
        sys.exit(1)
    print(f"DONE: {step_name} in {elapsed:.1f} min")


def main():
    t_start = time.time()
    print("=" * 80)
    print("NOKAPPA OVERNIGHT PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 0: Wait for training
    wait_for_training()

    # Step 1: Pool params
    run_step("Pool nokappa params", [
        sys.executable, str(SCRIPT_DIR / 'pool_phi_kappa_gamma_from_batches.py'),
        '--model_type', 'nokappa',
        '--max_batches', '39',
        '--output_dir', DATA_DIR,
    ])

    # Step 2: LOO predictions (5 batches, 200 epochs each)
    run_step("LOO predict nokappa", [
        sys.executable, str(SCRIPT_DIR / 'run_loo_predict_nokappa.py'),
        '--n_pred_batches', '5',
        '--n_train_batches', '40',
        '--num_epochs', '200',
        '--learning_rate', '0.1',
    ])

    # Step 3: AUC comparison
    run_step("Compare nokappa AUC", [
        sys.executable, str(SCRIPT_DIR / 'compare_nokappa_auc.py'),
        '--n_bootstraps', '100',
        '--n_batches', '5',
    ])

    total = (time.time() - t_start) / 60
    print(f"\n{'=' * 80}")
    print(f"PIPELINE COMPLETE in {total:.0f} min")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    print("\nResults:")
    print(f"  Pooled params: {DATA_DIR}pooled_phi_kappa_gamma_nokappa.pt")
    print(f"  LOO predictions: /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nokappa_loo/")
    print(f"  AUC CSV: {SCRIPT_DIR}/nokappa_auc_LOO.csv")
    print(f"  Check: tail claudefile/logs/nokappa_pipeline.log")


if __name__ == '__main__':
    main()
