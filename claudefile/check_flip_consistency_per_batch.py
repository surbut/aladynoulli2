#!/usr/bin/env python
"""
Check if the 13 reparam argmax flips are consistent across batches or just in a few.

For each disease that flips in the pooled reparam psi, count how many individual
batches also have that flip (argmax != initial).
"""
import numpy as np
import torch
import glob
from pathlib import Path


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _extract(ckpt, name):
    if 'model_state_dict' in ckpt and name in ckpt['model_state_dict']:
        return ckpt['model_state_dict'][name]
    if name in ckpt:
        return ckpt[name]
    return None


def main():
    data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    reparam_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
    initial_psi_path = data_dir / 'initial_psi_400k.pt'
    disease_csv = Path(__file__).parent.parent / 'claudefile/aladyn_project/pyScripts_forPublish/disease_names.csv'

    pattern = str(reparam_dir / 'enrollment_model_REPARAM_W0.0001_batch_*_*.pt')
    files = sorted(glob.glob(pattern))[:39]
    if not files:
        print(f"No files found: {pattern}")
        return

    initial_psi = torch.load(initial_psi_path, weights_only=False)
    initial_psi = to_np(initial_psi)
    K_dis = 20
    init = initial_psi[:K_dis]
    argmax_init = np.argmax(init, axis=0)

    # Load disease names
    disease_names = []
    if disease_csv.exists():
        import csv
        with open(disease_csv, 'r') as f:
            for row in csv.DictReader(f):
                disease_names.append(row.get('x', row.get('', '')))

    # Collect per-batch argmax for each disease
    n_batches = 0
    batch_argmax = []  # list of (batch_idx, argmax_per_disease)

    for i, fp in enumerate(files):
        try:
            ckpt = torch.load(fp, weights_only=False)
            psi = _extract(ckpt, 'psi')
            if psi is None:
                continue
            psi = to_np(psi)[:K_dis]
            argmax_b = np.argmax(psi, axis=0)
            batch_argmax.append((i, argmax_b))
            n_batches += 1
        except Exception as e:
            print(f"Error {Path(fp).name}: {e}")

    if n_batches == 0:
        print("No batches loaded")
        return

    # Identify diseases that flip in pooled (we need pooled for comparison)
    # Actually: we know the 13 flipped from the check_psi script. Let me recompute
    # which diseases flip in the MEAN across batches
    psi_stack = np.stack([b[1] for b in batch_argmax])  # [n_batches, K, D] - actually b[1] is argmax [D]
    # We need psi values, not argmax. Let me re-load.
    batch_argmax = []
    all_psi = []
    for fp in files:
        try:
            ckpt = torch.load(fp, weights_only=False)
            psi = _extract(ckpt, 'psi')
            if psi is None:
                continue
            psi = to_np(psi)[:K_dis]
            all_psi.append(psi)
        except Exception:
            continue

    psi_pooled = np.mean(np.stack(all_psi), axis=0)
    argmax_pooled = np.argmax(psi_pooled, axis=0)
    flipped_indices = np.where(argmax_init != argmax_pooled)[0]

    # For each flipped disease, count how many batches have same flip
    n_batches = len(all_psi)
    print(f"\nLoaded {n_batches} reparam batches")
    print(f"Pooled psi: 13 diseases flip vs initial")
    print("\n" + "="*70)
    print("PER-BATCH CONSISTENCY OF FLIPS")
    print("="*70)

    for d in flipped_indices:
        sig_init = argmax_init[d]
        sig_pooled = argmax_pooled[d]
        n_batches_flipped = sum(1 for psi in all_psi if np.argmax(psi[:, d]) != sig_init)
        n_batches_same = n_batches - n_batches_flipped
        d_name = disease_names[d] if d < len(disease_names) else f"Disease {d}"
        print(f"\n  d={d}: {d_name}  (initial sig {sig_init} -> pooled sig {sig_pooled})")
        print(f"    Batches with flip: {n_batches_flipped}/{n_batches} ({100*n_batches_flipped/n_batches:.1f}%)")
        if n_batches_flipped < n_batches:
            # Show distribution of flipped-to signatures across batches
            sigs_in_batches = [np.argmax(psi[:, d]) for psi in all_psi]
            from collections import Counter
            c = Counter(sigs_in_batches)
            print(f"    Per-batch argmax: {dict(c)}")

    print("\n" + "="*70)

    # Q2: Do the 13 diseases flip in the SAME batches?
    # Per batch: how many of the 13 flipped diseases have argmax != initial?
    n_flips_per_batch = []
    for psi in all_psi:
        argmax_b = np.argmax(psi, axis=0)
        n = sum(1 for d in flipped_indices if argmax_b[d] != argmax_init[d])
        n_flips_per_batch.append(n)

    print("\n" + "="*70)
    print("Q2: DO FLIPS CO-OCCUR IN THE SAME BATCHES?")
    print("(Per batch: how many of the 13 flipped diseases actually flip)")
    print("="*70)
    from collections import Counter
    c = Counter(n_flips_per_batch)
    for k in sorted(c.keys()):
        print(f"  Batches with exactly {k}/13 flips: {c[k]}")
    print(f"  Mean flips per batch: {np.mean(n_flips_per_batch):.1f}")
    print(f"  If independent (each disease ~72-100% flip rate): expect ~11-12 per batch")
    print(f"  If correlated: some batches would have few, others many")
    print("="*70)


if __name__ == '__main__':
    main()
