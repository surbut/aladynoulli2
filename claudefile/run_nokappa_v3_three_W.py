#!/usr/bin/env python
"""
Run 10 batches of nokappa v3 for three W values: 1e-5, 1e-4, 5e-4.
Runs two W values in parallel, then the third.

Usage:
    python claudefile/run_nokappa_v3_three_W.py              # All 3 W values, 10 batches
    python claudefile/run_nokappa_v3_three_W.py --n_batches 2  # Quick test (2 batches)
    python claudefile/run_nokappa_v3_three_W.py --resume_from 5  # Resume from batch 5
"""

import subprocess
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_batch(w_val, start, end):
    """Train one batch for one W value. Returns (w_val, start, end, elapsed, returncode)."""
    cmd = [
        sys.executable, 'claudefile/train_nokappa_v3.py',
        '--start_index', str(start),
        '--end_index', str(end),
        '--W', str(w_val),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60

    # Print last few lines of output
    lines = result.stdout.strip().split('\n')
    for line in lines[-3:]:
        print(f"  [W={w_val}] {line}")
    if result.returncode != 0:
        print(f"  [W={w_val}] STDERR: {result.stderr[-500:]}")

    return (w_val, start, end, elapsed, result.returncode)


def run_w_batches_sequential(w_val, n_batches, batch_size, resume_from):
    """Run all batches for one W value sequentially."""
    print(f"\n{'='*60}")
    print(f"Starting W={w_val}: {n_batches} batches")
    print(f"{'='*60}")

    t_total = time.time()
    for i in range(resume_from, n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        print(f"\n  [W={w_val}] Batch {i+1}/{n_batches}: {start}-{end}")

        w, s, e, elapsed, rc = run_single_batch(w_val, start, end)
        if rc != 0:
            print(f"  [W={w_val}] FAILED batch {i+1} (exit code {rc})")
            return False

        print(f"  [W={w_val}] Batch {i+1} done in {elapsed:.1f} min")

    total = (time.time() - t_total) / 60
    print(f"\n[W={w_val}] All {n_batches} batches complete in {total:.0f} min")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run nokappa v3 for 3 W values')
    parser.add_argument('--n_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--resume_from', type=int, default=0,
                        help='Skip batches before this index (0-based)')
    args = parser.parse_args()

    w_values = [1e-5, 1e-4, 5e-4]

    print(f"Nokappa v3 three-way W comparison")
    print(f"W values: {w_values}")
    print(f"Batches: {args.n_batches} x {args.batch_size}")
    print(f"Running first two W values in parallel, then third")

    t_total = time.time()

    # Phase 1: run W=1e-5 and W=1e-4 in parallel
    print(f"\n{'#'*60}")
    print(f"PHASE 1: W=1e-5 and W=1e-4 in parallel")
    print(f"{'#'*60}")

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_w_batches_sequential, w, args.n_batches,
                          args.batch_size, args.resume_from): w
            for w in [1e-5, 1e-4]
        }
        for future in as_completed(futures):
            w = futures[future]
            success = future.result()
            if not success:
                print(f"W={w} failed — continuing with remaining")

    # Phase 2: run W=5e-4
    print(f"\n{'#'*60}")
    print(f"PHASE 2: W=5e-4")
    print(f"{'#'*60}")

    run_w_batches_sequential(5e-4, args.n_batches, args.batch_size, args.resume_from)

    total = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE: 3 W values x {args.n_batches} batches in {total:.0f} min")
    print(f"{'='*60}")

    # Summary of output locations
    for w in w_values:
        w_str = f"{w:.0e}".replace('+', '').replace('-0', '-')
        print(f"  W={w}: ~/Dropbox/nokappa_v3_W{w_str}/")


if __name__ == '__main__':
    main()
