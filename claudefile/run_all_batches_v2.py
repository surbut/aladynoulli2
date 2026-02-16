#!/usr/bin/env python
"""
Run reparam v2 training for all 40 batches (0-400k).
~15 min per batch → ~10 hours total. Good for overnight.

Usage:
    PYTHONUNBUFFERED=1 python claudefile/run_all_batches_v2.py 2>&1 | tee claudefile/logs/run_all_v2.log

Resume from a specific batch (e.g., if batch 12 crashed):
    PYTHONUNBUFFERED=1 python claudefile/run_all_batches_v2.py --resume_from 12
"""

import subprocess
import sys
import time
from pathlib import Path

BATCH_SIZE = 10000
N_BATCHES = 40
SCRIPT = str(Path(__file__).parent / 'train_reparam_v2.py')

# Parse optional --resume_from
resume_from = 0
if '--resume_from' in sys.argv:
    idx = sys.argv.index('--resume_from')
    resume_from = int(sys.argv[idx + 1])

print(f"{'='*70}")
print(f"REPARAM v2: Training all {N_BATCHES} batches ({N_BATCHES * BATCH_SIZE // 1000}k patients)")
print(f"Settings: 500 epochs, cosine annealing, grad_clip=5.0, patience=75")
print(f"Estimated time: ~{N_BATCHES * 15 / 60:.0f} hours")
if resume_from > 0:
    print(f"Resuming from batch {resume_from}")
print(f"{'='*70}\n")

t_total = time.time()
completed = 0
failed = []

for batch_idx in range(resume_from, N_BATCHES):
    start = batch_idx * BATCH_SIZE
    end = (batch_idx + 1) * BATCH_SIZE

    print(f"\n{'='*70}")
    print(f"BATCH {batch_idx + 1}/{N_BATCHES}: samples {start}-{end}")
    remaining = (N_BATCHES - batch_idx) * 15
    print(f"Elapsed: {(time.time() - t_total)/60:.0f} min | "
          f"Est remaining: ~{remaining} min ({remaining/60:.1f} hrs)")
    print(f"{'='*70}")

    t_batch = time.time()

    cmd = [
        sys.executable, SCRIPT,
        '--start_index', str(start),
        '--end_index', str(end),
        '--num_epochs', '500',
        '--learning_rate', '0.1',
        '--grad_clip', '5.0',
        '--patience', '75',
    ]

    result = subprocess.run(cmd, env={**__import__('os').environ, 'PYTHONUNBUFFERED': '1'})

    elapsed_batch = (time.time() - t_batch) / 60

    if result.returncode == 0:
        completed += 1
        print(f"\nBatch {batch_idx + 1} DONE in {elapsed_batch:.1f} min "
              f"({completed}/{N_BATCHES - resume_from} completed)")
    else:
        failed.append(batch_idx)
        print(f"\nBatch {batch_idx + 1} FAILED (exit code {result.returncode}) "
              f"after {elapsed_batch:.1f} min")
        print(f"To resume: python {__file__} --resume_from {batch_idx}")

total_time = (time.time() - t_total) / 60

print(f"\n{'='*70}")
print(f"ALL DONE: {completed} completed, {len(failed)} failed in {total_time:.0f} min ({total_time/60:.1f} hrs)")
if failed:
    print(f"Failed batches: {failed}")
    print(f"Resume with: python {__file__} --resume_from {failed[0]}")
print(f"{'='*70}")
