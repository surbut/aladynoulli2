#!/usr/bin/env python
"""Run all 40 batches of no-kappa reparam training sequentially."""
import subprocess
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_batches', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--resume_from', type=int, default=0,
                        help='Skip batches before this index (0-based)')
    args = parser.parse_args()

    t_total = time.time()
    for i in range(args.resume_from, args.n_batches):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        print(f"\n{'='*60}")
        print(f"BATCH {i+1}/{args.n_batches}: {start}-{end}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, 'claudefile/train_reparam_v2_nokappa.py',
            '--start_index', str(start),
            '--end_index', str(end),
        ]
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = (time.time() - t0) / 60

        if result.returncode != 0:
            print(f"FAILED batch {i+1} (exit code {result.returncode})")
            sys.exit(1)
        print(f"Batch {i+1} done in {elapsed:.1f} min")

    total = (time.time() - t_total) / 60
    print(f"\nAll {args.n_batches} batches complete in {total:.0f} min")

if __name__ == '__main__':
    main()
