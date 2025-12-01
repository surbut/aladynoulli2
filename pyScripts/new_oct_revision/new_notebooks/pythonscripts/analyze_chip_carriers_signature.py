#!/usr/bin/env python3
"""
CHIP (Clonal Hematopoiesis) carrier analysis: Signature enrichment before events.

This extends the FH analysis to work with CHIP mutation carriers.
CHIP data format: dataframe with FID/IID and mutation indicator columns (hasCH, hasDNMT3A, hasTET2, etc.)

Usage in notebook:
    %run analyze_chip_carriers_signature.py \
        --chip_file /path/to/CHIP_data.txt \
        --mutation_type "hasCH" \
        --signature_idx 5 \
        --event_indices 112,113,114,115,116
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import bootstrap
import sys
from pathlib import Path

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')


def load_chip_carriers(chip_file, processed_ids, mutation_type='hasCH'):
    """
    Load CHIP carrier data and create boolean array matching processed_ids.
    
    Parameters:
    -----------
    chip_file : str
        Path to CHIP data file (tab-separated with FID/IID and mutation columns)
    processed_ids : np.array
        Array of patient IDs matching theta/Y order
    mutation_type : str
        Column name for mutation indicator (e.g., 'hasCH', 'hasDNMT3A', 'hasTET2')
        Can also be a list of columns to combine (OR logic)
    
    Returns:
    --------
    is_carrier : np.array
        Boolean array [N] where True indicates carrier
    chip_df : pd.DataFrame
        Full CHIP dataframe for reference
    """
    print(f"Loading CHIP data from {chip_file}...")
    chip_df = pd.read_csv(chip_file, sep='\t', low_memory=False)
    
    print(f"CHIP data shape: {chip_df.shape}")
    print(f"Columns: {list(chip_df.columns[:10])}...")  # Show first 10 columns
    
    # Find IID column (could be IID, FID, or eid)
    iid_col = None
    for col in ['IID', 'FID', 'eid', 'EID', 'ID']:
        if col in chip_df.columns:
            iid_col = col
            break
    
    if iid_col is None:
        # Try first column
        iid_col = chip_df.columns[0]
        print(f"⚠️  No standard ID column found, using first column: {iid_col}")
    
    print(f"Using ID column: {iid_col}")
    
    # Handle mutation_type - can be single column or list
    if isinstance(mutation_type, str):
        mutation_cols = [mutation_type]
    else:
        mutation_cols = mutation_type
    
    # Check which mutation columns exist
    available_cols = []
    missing_cols = []
    for col in mutation_cols:
        if col in chip_df.columns:
            available_cols.append(col)
        else:
            missing_cols.append(col)
    
    if not available_cols:
        raise ValueError(f"None of the mutation columns found: {mutation_cols}\n"
                        f"Available columns: {list(chip_df.columns)}")
    
    if missing_cols:
        print(f"⚠️  Missing mutation columns: {missing_cols}")
        print(f"✓ Using available columns: {available_cols}")
    
    # Determine carrier status
    # For multiple columns: carrier if ANY is True/1
    carrier_mask = chip_df[available_cols[0]].fillna(0).astype(int) > 0
    for col in available_cols[1:]:
        carrier_mask = carrier_mask | (chip_df[col].fillna(0).astype(int) > 0)
    
    # Get carrier IIDs
    carrier_iids = set(chip_df.loc[carrier_mask, iid_col].astype(int).tolist())
    
    print(f"Found {len(carrier_iids):,} carriers based on columns: {available_cols}")
    
    # Match to processed_ids
    eids = processed_ids.astype(int)
    is_carrier = np.isin(eids, list(carrier_iids))
    
    print(f"Matched {is_carrier.sum():,} carriers to {len(eids):,} processed_ids")
    print(f"  ({is_carrier.sum()/len(eids)*100:.2f}% carrier rate)")
    
    return is_carrier, chip_df


def analyze_signature_enrichment_chip(chip_file, mutation_name, mutation_type, signature_idx, 
                                     event_indices, theta, Y, processed_ids,
                                     pre_window=5, epsilon=0.0, output_dir=None):
    """
    Analyze signature enrichment in CHIP mutation carriers before events.
    
    Parameters:
    -----------
    chip_file : str
        Path to CHIP data file
    mutation_name : str
        Name of mutation for labeling (e.g., "CHIP", "DNMT3A", "TET2")
    mutation_type : str or list
        Column name(s) in CHIP file for mutation indicator
    signature_idx : int
        Signature index to analyze (0-based)
    event_indices : list
        List of disease indices that define the event composite
    theta : np.array
        Signature loadings [N, K, T]
    Y : np.array
        Disease event matrix [N, D, T]
    processed_ids : np.array
        Patient IDs matching theta and Y
    pre_window : int
        Years before event to analyze
    epsilon : float
        Threshold for "rise" (delta > epsilon)
    output_dir : Path, optional
        Directory to save results
    """
    
    print(f"\n{'='*80}")
    print(f"CHIP MUTATION CARRIER ANALYSIS: {mutation_name}")
    print(f"{'='*80}\n")
    
    # Load carriers
    is_carrier, chip_df = load_chip_carriers(chip_file, processed_ids, mutation_type)
    
    # Convert to numpy if needed
    def to_numpy(x):
        import torch
        return x.detach().cpu().numpy() if 'torch' in str(type(x)) else x
    
    Y_np = to_numpy(Y[:len(processed_ids)])
    theta_np = to_numpy(theta[:len(processed_ids)])
    ev_idx = np.array(event_indices, dtype=int)
    
    N, K, T = theta_np.shape
    
    # Find first event time per person
    Y_sel = (Y_np[:, ev_idx, :] > 0)  # [N, |ev|, T]
    has_event = Y_sel.any(axis=(1, 2))  # [N]
    any_event_over_ev = Y_sel.any(axis=1)  # [N, T]
    first_event_t = np.full(N, -1, dtype=int)
    first_event_t[has_event] = np.argmax(any_event_over_ev[has_event], axis=1)
    
    # Compute pre-event rise in signature
    valid = (has_event) & (first_event_t >= pre_window)  # need enough history
    idx_valid = np.where(valid)[0]
    
    sig = theta_np[:, signature_idx, :]  # [N, T]
    pre_start = first_event_t[idx_valid] - pre_window
    pre_end = first_event_t[idx_valid] - 1
    
    # Delta over window (end - start)
    delta = sig[idx_valid, pre_end] - sig[idx_valid, pre_start]  # [n_valid]
    is_rise = (delta > epsilon)
    
    # Partition by carrier status
    car_valid = is_carrier[idx_valid]
    rise_car = is_rise[car_valid]
    rise_non = is_rise[~car_valid]
    
    n_car = rise_car.size
    n_non = rise_non.size
    ev_car = int(rise_car.sum())
    ev_non = int(rise_non.sum())
    
    # Statistical test
    table = [[ev_car, n_car - ev_car],
             [ev_non, n_non - ev_non]]
    OR, p = fisher_exact(table, alternative='greater')
    
    # Proportion CIs
    car_ci = proportion_confint(ev_car, n_car, method='wilson') if n_car > 0 else (np.nan, np.nan)
    non_ci = proportion_confint(ev_non, n_non, method='wilson') if n_non > 0 else (np.nan, np.nan)
    
    print(f"\n=== {mutation_name} carriers: enrichment of pre-event Signature {signature_idx} rise ===")
    print(f"Window: last {pre_window} years before first event; epsilon={epsilon}")
    print(f"Valid N with event & sufficient history: {idx_valid.size}")
    print(f"Carriers:   {ev_car}/{n_car} rising  (prop={ev_car/max(n_car,1):.3f}, CI95={car_ci})")
    print(f"Noncarriers:{ev_non}/{n_non} rising  (prop={ev_non/max(n_non,1):.3f}, CI95={non_ci})")
    print(f"Fisher exact (greater) OR={OR:.3f}, p={p:.3e}")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame({
            'mutation': [mutation_name],
            'mutation_type': [str(mutation_type)],
            'signature_idx': [signature_idx],
            'n_carriers': [n_car],
            'n_noncarriers': [n_non],
            'carriers_rising': [ev_car],
            'noncarriers_rising': [ev_non],
            'carrier_prop': [ev_car/max(n_car,1)],
            'noncarrier_prop': [ev_non/max(n_non,1)],
            'OR': [OR],
            'p_value': [p],
            'pre_window': [pre_window]
        })
        results_df.to_csv(output_dir / f'{mutation_name}_signature{signature_idx}_enrichment.csv', index=False)
        print(f"\n✓ Saved results to {output_dir}")
    
    return {
        'mutation': mutation_name,
        'signature_idx': signature_idx,
        'n_carriers': n_car,
        'n_noncarriers': n_non,
        'carriers_rising': ev_car,
        'noncarriers_rising': ev_non,
        'OR': OR,
        'p_value': p,
        'is_carrier': is_carrier,
        'idx_valid': idx_valid,
        'delta': delta,
        'is_rise': is_rise,
        'chip_df': chip_df
    }


def visualize_signature_trajectory_chip(results, theta, Y, processed_ids, event_indices,
                                       signature_idx, mutation_name, pre_window=5, post_window=3):
    """
    Visualize event-aligned signature trajectories for CHIP carriers vs noncarriers.
    """
    is_carrier = results['is_carrier']
    
    def to_numpy(x):
        import torch
        return x.detach().cpu().numpy() if 'torch' in str(type(x)) else x
    
    Y_np = to_numpy(Y[:len(processed_ids)])
    theta_np = to_numpy(theta[:len(processed_ids)])
    ev_idx = np.array(event_indices, dtype=int)
    
    # Find first event time
    Y_sel = (Y_np[:, ev_idx, :] > 0)
    has_event = Y_sel.any(axis=(1, 2))
    any_ev_time = Y_sel.any(axis=1)
    first_ev_t = np.full(len(has_event), -1, int)
    first_ev_t[has_event] = np.argmax(any_ev_time[has_event], axis=1)
    
    sig = theta_np[:, signature_idx, :]
    aligned_span = np.arange(-pre_window, post_window + 1)
    L = len(aligned_span)
    
    def build_aligned(sig, first_t, mask):
        idx = np.where(mask & (first_t >= pre_window) & (first_t < theta_np.shape[2] - post_window))[0]
        aligned = np.empty((len(idx), L), float)
        aligned[:] = np.nan
        for j, i in enumerate(idx):
            t0 = first_t[i]
            aligned[j] = sig[i, t0 - pre_window : t0 + post_window + 1]
        return idx, aligned
    
    idx_car, aligned_car = build_aligned(sig, first_ev_t, is_carrier)
    idx_non, aligned_non = build_aligned(sig, first_ev_t, ~is_carrier)
    
    # Pre-event delta distributions
    delta_car = aligned_car[:, pre_window] - aligned_car[:, 0]
    delta_non = aligned_non[:, pre_window] - aligned_non[:, 0]
    rise_car = (delta_car > 0).mean()
    rise_non = (delta_non > 0).mean()
    
    # Mean and bootstrap CI
    def mean_ci(a, n_boot=2000, alpha=0.05):
        m = np.nanmean(a, axis=0)
        ci_low, ci_high = [], []
        rng = np.random.default_rng(42)
        for t in range(a.shape[1]):
            col = a[:, t]
            col = col[~np.isnan(col)]
            if len(col) < 10:
                ci_low.append(np.nan)
                ci_high.append(np.nan)
                continue
            res = bootstrap((col,), np.mean, vectorized=False, n_resamples=n_boot,
                          paired=False, confidence_level=1-alpha, random_state=rng, method='basic')
            ci_low.append(res.confidence_interval.low)
            ci_high.append(res.confidence_interval.high)
        return m, np.array(ci_low), np.array(ci_high)
    
    m_car, lo_car, hi_car = mean_ci(aligned_car)
    m_non, lo_non, hi_non = mean_ci(aligned_non)
    
    # Plot
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.3)
    
    # (A) Event-aligned trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    x = aligned_span
    ax1.plot(x, m_car, color='#2c7fb8', lw=2.5, label=f'{mutation_name} carriers (n={aligned_car.shape[0]:,})')
    ax1.fill_between(x, lo_car, hi_car, color='#2c7fb8', alpha=0.2)
    ax1.plot(x, m_non, color='#f03b20', lw=2.5, label=f'Noncarriers (n={aligned_non.shape[0]:,})')
    ax1.fill_between(x, lo_non, hi_non, color='#f03b20', alpha=0.2)
    ax1.axvline(0, color='k', ls='--', lw=1)
    ax1.set_xlabel('Years relative to first event (0 = event)')
    ax1.set_ylabel(f'Signature {signature_idx} loading (θ)')
    ax1.set_title(f'Signature {signature_idx} trajectory aligned to first event')
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=True)
    
    # (B) Delta distributions
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(min(delta_car.min(), delta_non.min()),
                      max(delta_car.max(), delta_non.max()), 40)
    ax2.hist(delta_non, bins=bins, alpha=0.6, color='#f03b20', 
            label=f'Noncarriers (rise={rise_non:.2%})')
    ax2.hist(delta_car, bins=bins, alpha=0.6, color='#2c7fb8', 
            label=f'Carriers (rise={rise_car:.2%})')
    ax2.axvline(0, color='k', ls='--', lw=1)
    ax2.set_xlabel(f'ΔSig{signature_idx} (end-start) over last {pre_window}y pre-event')
    ax2.set_ylabel('Count')
    ax2.set_title('Pre-event change in Signature')
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=True)
    
    fig.suptitle(f'{mutation_name} carriers show pre-event signature enrichment', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Analyze signature enrichment in CHIP mutation carriers"
    )
    parser.add_argument('--chip_file', type=str, required=True,
                       help='Path to CHIP data file (tab-separated with FID/IID and mutation columns)')
    parser.add_argument('--mutation_name', type=str, required=True,
                       help='Name of mutation for labeling (e.g., "CHIP", "DNMT3A", "TET2")')
    parser.add_argument('--mutation_type', type=str, default='hasCH',
                       help='Column name(s) for mutation indicator (comma-separated for multiple)')
    parser.add_argument('--signature_idx', type=int, default=5,
                       help='Signature index to analyze (default: 5)')
    parser.add_argument('--event_indices', type=str, default='112,113,114,115,116',
                       help='Comma-separated disease indices for event composite')
    parser.add_argument('--theta_path', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt',
                       help='Path to theta tensor')
    parser.add_argument('--Y_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt',
                       help='Path to Y tensor')
    parser.add_argument('--processed_ids_path', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/processed_patient_ids.npy',
                       help='Path to processed_ids.npy')
    parser.add_argument('--pre_window', type=int, default=5,
                       help='Years before event to analyze (default: 5)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                       help='Threshold for "rise" (default: 0.0)')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization')
    
    args = parser.parse_args()
    
    # Parse mutation_type (can be comma-separated)
    mutation_type = args.mutation_type.split(',') if ',' in args.mutation_type else args.mutation_type
    
    # Load data
    print("Loading data...")
    import torch
    theta = torch.load(args.theta_path, map_location='cpu')
    if hasattr(theta, 'numpy'):
        theta = theta.numpy()
    
    Y = torch.load(args.Y_path, map_location='cpu')
    processed_ids = np.load(args.processed_ids_path)
    
    # Parse event indices
    event_indices = [int(x) for x in args.event_indices.split(',')]
    
    # Run analysis
    results = analyze_signature_enrichment_chip(
        args.chip_file, args.mutation_name, mutation_type, args.signature_idx,
        event_indices, theta, Y, processed_ids,
        args.pre_window, args.epsilon, args.output_dir
    )
    
    # Generate plot if requested
    if args.plot:
        fig = visualize_signature_trajectory_chip(
            results, theta, Y, processed_ids, event_indices,
            args.signature_idx, args.mutation_name, args.pre_window
        )
        if args.output_dir:
            output_path = Path(args.output_dir) / f'{args.mutation_name}_signature{args.signature_idx}_trajectory.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {output_path}")
        plt.show()


if __name__ == '__main__':
    main()

