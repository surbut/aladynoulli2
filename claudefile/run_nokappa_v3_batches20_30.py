#!/usr/bin/env python
"""
Run nokappa v3 Constant, v3 Cos300, and v3 Clip on batches 20-30 (samples 200k-300k).
Uses separate output dirs so we don't overwrite batches 0-9.
Original nokappa (500ep, cos, clip) already exists in censor_e_batchrun_vectorized_REPARAM_v2_nokappa.

Usage:
    python claudefile/run_nokappa_v3_batches20_30.py              # All 3 configs (sequential)
    python claudefile/run_nokappa_v3_batches20_30.py --parallel   # All 3 configs in parallel
    python claudefile/run_nokappa_v3_batches20_30.py --config constant  # Constant only
    python claudefile/run_nokappa_v3_batches20_30.py --config cos300     # Cos300 only
    python claudefile/run_nokappa_v3_batches20_30.py --config clip       # Clip only
    python claudefile/run_nokappa_v3_batches20_30.py --n_batches 2       # Quick test
    python claudefile/run_nokappa_v3_batches20_30.py --resume_from 22    # Resume from batch 22
"""

import subprocess
import sys
import time
import argparse
import os


BATCH_SIZE = 10000
FIRST_BATCH = 20
LAST_BATCH = 30  # exclusive, so 20-29 = 10 batches


def run_batch(config_name, start, end, output_dir, script):
    """Run one batch. Returns (elapsed_min, success)."""
    cmd = [
        sys.executable, script,
        '--start_index', str(start),
        '--end_index', str(end),
        '--W', '0.0001',
        '--output_dir', output_dir,
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
    parser = argparse.ArgumentParser(description='Run nokappa v3 on batches 20-30')
    parser.add_argument('--config', type=str, choices=['constant', 'cos300', 'clip', 'all'],
                        default='all', help='Which config(s) to run')
    parser.add_argument('--parallel', action='store_true',
                        help='Run all 3 configs in parallel (only with --config all)')
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches (default 10 = batches 20-29)')
    parser.add_argument('--resume_from', type=int, default=FIRST_BATCH,
                        help=f'First batch index (default {FIRST_BATCH})')
    args = parser.parse_args()

    # Parallel mode: spawn 3 processes, each runs one config
    if args.parallel and args.config == 'all':
        script_path = os.path.abspath(__file__)
        procs = []
        for cfg in ['constant', 'cos300', 'clip']:
            cmd = [sys.executable, script_path, '--config', cfg,
                   '--n_batches', str(args.n_batches), '--resume_from', str(args.resume_from)]
            # Inherit stdout/stderr so output streams in real time (may interleave)
            p = subprocess.Popen(cmd)
            procs.append((cfg, p))
            print(f"Started {cfg} (PID {p.pid})")
        print(f"\nWaiting for all 3 configs to complete...")
        failed = []
        for cfg, p in procs:
            p.wait()
            if p.returncode != 0:
                failed.append((cfg, p.returncode))
            else:
                print(f"{cfg} done")
        if failed:
            for cfg, rc in failed:
                print(f"{cfg} FAILED (exit {rc})")
            sys.exit(1)
        print(f"\nALL 3 CONFIGS COMPLETE")
        return

    configs = []
    if args.config in ('constant', 'all'):
        configs.append(('constant', 'claudefile/train_nokappa_v3.py',
                       '/Users/sarahurbut/Library/CloudStorage/Dropbox/nokappa_v3_W1e-4_b20_30/'))
    if args.config in ('cos300', 'all'):
        configs.append(('cos300', 'claudefile/train_nokappa_v3_cos.py',
                       '/Users/sarahurbut/Library/CloudStorage/Dropbox/nokappa_v3_cos300_b20_30/'))
    if args.config in ('clip', 'all'):
        configs.append(('clip', 'claudefile/train_nokappa_v3_clip.py',
                       '/Users/sarahurbut/Library/CloudStorage/Dropbox/nokappa_v3_clip_b20_30/'))

    last_batch = args.resume_from + args.n_batches
    print(f"\n{'='*60}")
    print(f"Nokappa v3 on batches {args.resume_from}-{last_batch-1} (samples "
          f"{args.resume_from*BATCH_SIZE}-{last_batch*BATCH_SIZE})")
    print(f"Configs: {[c[0] for c in configs]}")
    print(f"{'='*60}\n")

    t_total = time.time()
    for config_name, script, output_dir in configs:
        print(f"\n{'#'*60}")
        print(f"{config_name.upper()}: {args.n_batches} batches")
        print(f"Output: {output_dir}")
        print(f"{'#'*60}")

        for i in range(args.resume_from, last_batch):
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            print(f"\n--- Batch {i}: {start}-{end} ---")
            elapsed, ok = run_batch(config_name, start, end, output_dir, script)
            if ok:
                print(f"  Done in {elapsed:.1f} min")
            else:
                print(f"  ABORT: batch failed")
                sys.exit(1)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_min:.0f} min")
    for _, _, out in configs:
        print(f"  {out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
