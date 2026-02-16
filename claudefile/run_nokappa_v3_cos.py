#!/usr/bin/env python
"""
Run 10 batches of nokappa v3 cos300 (cosine annealing LR).
Run this AFTER the constant-LR W=1e-4 training finishes.

Usage:
    python claudefile/run_nokappa_v3_cos.py              # All 10 batches
    python claudefile/run_nokappa_v3_cos.py --n_batches 2  # Quick test
    python claudefile/run_nokappa_v3_cos.py --resume_from 5  # Resume from batch 5
"""

import subprocess
import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run nokappa v3 cos300 (10 batches)')
    parser.add_argument('--n_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--resume_from', type=int, default=0)
    args = parser.parse_args()

    print(f"Nokappa v3 COS300: {args.n_batches} batches x {args.batch_size}")
    print(f"W=1e-4, LR=0.1->0.001 cosine, 300 epochs, kappa=1 fixed")
    t_total = time.time()

    for i in range(args.resume_from, args.n_batches):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        print(f"\n{'='*60}")
        print(f"Batch {i+1}/{args.n_batches}: {start}-{end}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, 'claudefile/train_nokappa_v3_cos.py',
            '--start_index', str(start),
            '--end_index', str(end),
            '--W', '0.0001',
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = (time.time() - t0) / 60

        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"  {line}")
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  STDERR: {result.stderr[-500:]}")
        else:
            print(f"  Done in {elapsed:.1f} min")

    total = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE: {args.n_batches} batches in {total:.0f} min")
    print(f"Output: ~/Dropbox/nokappa_v3_cos300/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
