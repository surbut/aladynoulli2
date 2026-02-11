#!/usr/bin/env python
"""
Check how much psi changes from initial to final (pooled) for nolr vs reparam.

The lag in nolr (psi not in NLL path) should mean psi changes less than in reparam
(psi flows through NLL). This preserves biological interpretation (argmax stable).

Prerequisites:
    python pool_phi_kappa_gamma_from_batches.py --model_type nolr --max_batches 39
    python pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39

Usage:
    python check_psi_change_initial_vs_final.py
"""
import argparse
import csv
import numpy as np
import torch
from pathlib import Path


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# Signature names from paper Table (tab:signature_summary_actual)
SIG_NAMES = {
    0: "Cardiac Arrhythmias",
    1: "Musculoskeletal",
    2: "Upper GI/Esophageal",
    3: "Mixed/General Medical",
    4: "Upper Respiratory",
    5: "Ischemic Cardiovascular",
    6: "Metastatic Cancer",
    7: "Pain/Inflammation",
    8: "Gynecologic",
    9: "Spinal Disorders",
    10: "Ophthalmologic",
    11: "Cerebrovascular",
    12: "Renal/Urologic",
    13: "Male Urogenital",
    14: "Pulmonary/Smoking",
    15: "Metabolic/Diabetes",
    16: "Infectious/Critical Care",
    17: "Lower GI/Colon",
    18: "Hepatobiliary",
    19: "Dermatologic/Oncologic",
}


def compare_psi_change(initial_psi, final_psi, name, disease_names=None, verbose=False):
    """Compare initial vs final psi: stats and argmax stability."""
    # Use disease signatures only (exclude healthy if present)
    K_dis = 20
    init = initial_psi[:K_dis] if initial_psi.shape[0] >= K_dis else initial_psi
    fin = final_psi[:K_dis] if final_psi.shape[0] >= K_dis else final_psi

    if init.shape != fin.shape:
        min_d = min(init.shape[1], fin.shape[1])
        init = init[:, :min_d]
        fin = fin[:, :min_d]

    # Magnitude of change
    diff = fin - init
    mean_abs_diff = np.mean(np.abs(diff))
    max_abs_diff = np.max(np.abs(diff))
    std_diff = np.std(diff)

    # Correlation
    if init.size > 1 and np.std(init) > 0 and np.std(fin) > 0:
        corr = np.corrcoef(init.flatten(), fin.flatten())[0, 1]
    else:
        corr = np.nan

    # Argmax stability: how many diseases flipped primary signature?
    argmax_init = np.argmax(init, axis=0)  # [D]
    argmax_fin = np.argmax(fin, axis=0)    # [D]
    n_flipped = np.sum(argmax_init != argmax_fin)
    n_total = argmax_init.size
    pct_stable = 100 * (n_total - n_flipped) / n_total

    flipped_indices = np.where(argmax_init != argmax_fin)[0]

    print(f"\n--- {name} ---")
    print(f"  Mean |psi_final - psi_initial|: {mean_abs_diff:.6f}")
    print(f"  Max  |psi_final - psi_initial|: {max_abs_diff:.6f}")
    print(f"  Std of (psi_final - psi_initial): {std_diff:.6f}")
    print(f"  Correlation(initial, final): {corr:.6f}")
    print(f"  Argmax stability: {n_total - n_flipped}/{n_total} ({pct_stable:.2f}%) diseases kept same max sig")
    if n_flipped > 0:
        print(f"  Flipped: {n_flipped} diseases")
        if verbose and disease_names is not None:
            for idx in flipped_indices:
                d_name = disease_names[idx] if idx < len(disease_names) else f"Disease {idx}"
                sig_init = argmax_init[idx]
                sig_fin = argmax_fin[idx]
                sig_init_name = SIG_NAMES.get(sig_init, f"Sig{sig_init}")
                sig_fin_name = SIG_NAMES.get(sig_fin, f"Sig{sig_fin}")
                print(f"    d={idx}: {d_name}")
                print(f"      {sig_init_name} -> {sig_fin_name}")

    return {
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'corr': corr,
        'n_flipped': n_flipped,
        'pct_stable': pct_stable,
        'flipped_indices': flipped_indices,
        'argmax_init': argmax_init,
        'argmax_fin': argmax_fin,
    }


def load_disease_names(csv_path):
    """Load disease names from CSV. index i = model disease i."""
    names = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row.get('x', row.get('', '')))
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', help='List flipped diseases with names')
    args = parser.parse_args()

    data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    initial_psi_path = data_dir / 'initial_psi_400k.pt'
    nolr_path = data_dir / 'pooled_phi_kappa_gamma_nolr.pt'
    reparam_path = data_dir / 'pooled_phi_kappa_gamma_reparam.pt'
    disease_csv = Path(__file__).parent.parent / 'claudefile/aladyn_project/pyScripts_forPublish/disease_names.csv'

    if not initial_psi_path.exists():
        print(f"NOT FOUND: {initial_psi_path}")
        return

    disease_names = None
    if disease_csv.exists():
        disease_names = load_disease_names(disease_csv)
        print(f"Loaded {len(disease_names)} disease names")
    else:
        print(f"Disease names not found at {disease_csv}")

    print("Loading initial_psi...")
    initial_psi = torch.load(initial_psi_path, weights_only=False)
    initial_psi = to_np(initial_psi)
    print(f"  Shape: {initial_psi.shape}")

    print("\n" + "="*70)
    print("PSI CHANGE: initial vs final (pooled)")
    print("="*70)

    results = {}

    if nolr_path.exists():
        nolr = torch.load(nolr_path, weights_only=False)
        psi_nolr = nolr.get('psi')
        if psi_nolr is not None:
            psi_nolr = to_np(psi_nolr)
            results['nolr'] = compare_psi_change(
                initial_psi, psi_nolr, "NOLR",
                disease_names=disease_names, verbose=args.verbose
            )
        else:
            print(f"\n--- NOLR ---")
            print("  No psi in pooled file")
    else:
        print(f"\nNOT FOUND: {nolr_path}")
        print("Run: python pool_phi_kappa_gamma_from_batches.py --model_type nolr --max_batches 39")

    if reparam_path.exists():
        reparam = torch.load(reparam_path, weights_only=False)
        psi_reparam = reparam.get('psi')
        if psi_reparam is not None:
            psi_reparam = to_np(psi_reparam)
            results['reparam'] = compare_psi_change(
                initial_psi, psi_reparam, "REPARAM",
                disease_names=disease_names, verbose=args.verbose
            )
        else:
            print(f"\n--- REPARAM ---")
            print("  No psi in pooled file")
    else:
        print(f"\nNOT FOUND: {reparam_path}")
        print("Run: python pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39")

    # Summary comparison
    if 'nolr' in results and 'reparam' in results:
        print("\n" + "="*70)
        print("SUMMARY: nolr vs reparam (lag = less change = more stable)")
        print("="*70)
        nr = results['nolr']
        rr = results['reparam']
        print(f"  Mean |change|:  nolr={nr['mean_abs_diff']:.6f},  reparam={rr['mean_abs_diff']:.6f}  (reparam change / nolr = {rr['mean_abs_diff']/nr['mean_abs_diff']:.2f}x)")
        print(f"  Argmax stable: nolr={nr['pct_stable']:.2f}%, reparam={rr['pct_stable']:.2f}%")
        print(f"  Flipped:       nolr={nr['n_flipped']}, reparam={rr['n_flipped']}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
