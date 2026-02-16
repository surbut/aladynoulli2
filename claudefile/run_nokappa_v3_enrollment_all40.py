#!/usr/bin/env python
"""
PRODUCTION RUN: Nokappa v3 (constant LR) with E_enrollment_full on ALL 40 batches.

Same as the corrected-E run, but trains with E_enrollment_full.pt for drug matching.

Usage:
    python claudefile/run_nokappa_v3_enrollment_all40.py                # All 40 batches
    python claudefile/run_nokappa_v3_enrollment_all40.py --n_batches 2  # Quick test
    python claudefile/run_nokappa_v3_enrollment_all40.py --resume_from 15  # Resume
"""

import subprocess
import sys
import time
import argparse
import os


BATCH_SIZE = 10000
TOTAL_BATCHES = 40  # batches 0-39 = 400K patients


def run_batch(start, end, output_dir, script):
    """Run one batch. Returns (elapsed_min, success)."""
    cmd = [
        sys.executable, script,
        '--start_index', str(start),
        '--end_index', str(end),
        '--W', '0.0001',
        '--output_dir', output_dir,
        '--e_matrix', 'E_enrollment_full.pt',
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60

    lines = result.stdout.strip().split('\n')
    for line in lines[-4:]:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-400:]}")
        return elapsed, False
    return elapsed, True


def main():
    parser = argparse.ArgumentParser(
        description='PRODUCTION: Nokappa v3 (constant LR) with E_enrollment_full on all 40 batches')
    parser.add_argument('--n_batches', type=int, default=TOTAL_BATCHES,
                        help=f'Number of batches (default {TOTAL_BATCHES})')
    parser.add_argument('--resume_from', type=int, default=0,
                        help='First batch index (default 0)')
    parser.add_argument('--output_dir', type=str,
                        default='censor_e_batchrun_vectorized_REPARAM_v3_nokappa_enrollment/',
                        help='Output directory for checkpoints')
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_nokappa_v3.py')
    last_batch = min(args.resume_from + args.n_batches, TOTAL_BATCHES)

    print(f"\n{'='*70}")
    print(f"PRODUCTION: Nokappa v3 (constant LR=0.1, W=1e-4, kappa=1, 300 epochs)")
    print(f"E matrix: E_enrollment_full.pt (for drug matching)")
    print(f"Batches {args.resume_from}-{last_batch-1} "
          f"(samples {args.resume_from*BATCH_SIZE}-{last_batch*BATCH_SIZE})")
    print(f"Output: {args.output_dir}")
    print(f"Script: {script}")
    print(f"{'='*70}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    t_total = time.time()
    completed = 0
    failed = 0

    for i in range(args.resume_from, last_batch):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        print(f"\n--- Batch {i}/{last_batch-1}: samples {start}-{end} "
              f"[{completed}/{last_batch - args.resume_from} done] ---")
        elapsed, ok = run_batch(start, end, args.output_dir, script)
        if ok:
            completed += 1
            print(f"  Done in {elapsed:.1f} min")
        else:
            failed += 1
            print(f"  FAILED — continuing to next batch")

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*70}")
    print(f"PRODUCTION RUN COMPLETE (E_enrollment_full)")
    print(f"  Completed: {completed}/{last_batch - args.resume_from}")
    if failed:
        print(f"  Failed: {failed} (re-run with --resume_from for specific batches)")
    print(f"  Total time: {total_min:.0f} min ({total_min/60:.1f} hours)")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
