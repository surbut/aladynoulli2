#!/usr/bin/env python3
"""
FH (Familial Hypercholesterolemia) mutation carrier analysis: Signature enrichment before events.

This analyzes whether FH carriers show enriched signature rise before disease events,
using the same framework as the CHIP analysis.

Usage in notebook:
    %run analyze_fh_carriers_signature.py \
        --fh_file /path/to/fh_carriers.txt \
        --signature_idx 5 \
        --event_indices 112,113,114,115,116 \
        --output_dir results/fh_analysis \
        --plot
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, bootstrap
from statsmodels.stats.proportion import proportion_confint
import sys
from pathlib import Path

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')


def load_fh_carriers(fh_file, processed_ids):
    """
    Load FH carrier data and create boolean array matching processed_ids.
    
    Parameters:
    -----------
    fh_file : str
        Path to FH carrier file (tab-separated with IID/eid column)
    processed_ids : np.array
        Array of patient IDs matching theta/Y order
    
    Returns:
    --------
    is_carrier : np.array
        Boolean array [N] where True indicates carrier
    fh_df : pd.DataFrame
        Full FH dataframe for reference
    """
    print(f"Loading FH data from {fh_file}...")
    fh_df = pd.read_csv(fh_file, sep='\t', dtype={'IID': int}, low_memory=False)
    
    print(f"FH data shape: {fh_df.shape}")
    print(f"Columns: {list(fh_df.columns[:10])}...")  # Show first 10 columns
    
    # Find IID column (could be IID, FID, or eid)
    iid_col = None
    for col in ['IID', 'FID', 'eid', 'EID', 'ID']:
        if col in fh_df.columns:
            iid_col = col
            break
    
    if iid_col is None:
        # Try first column
        iid_col = fh_df.columns[0]
        print(f"⚠️  No standard ID column found, using first column: {iid_col}")
    
    print(f"Using ID column: {iid_col}")
    
    # Get carrier eids
    carrier_eids = set(fh_df[iid_col].astype(int).tolist())
    
    # Create boolean array matching processed_ids
    eids = processed_ids.astype(int)
    is_carrier = np.isin(eids, list(carrier_eids))
    
    print(f"Loaded {is_carrier.sum():,} FH carriers out of {len(eids):,} total patients")
    return is_carrier, fh_df


def analyze_signature_enrichment_fh(fh_file, signature_idx, event_indices, 
                                    theta, Y, processed_ids,
                                    pre_window=5, epsilon=0.0, output_dir=None):
    """
    Analyze signature enrichment in FH carriers before events.
    
    Parameters:
    -----------
    fh_file : str
        Path to FH carrier file
    signature_idx : int
        Signature index to analyze (0-based)
    event_indices : list
        Disease indices for event composite
    theta : np.array
        Signature loadings [N, K, T]
    Y : np.array
        Disease outcomes [N, D, T]
    processed_ids : np.array
        Patient IDs matching theta/Y order
    pre_window : int
        Years before event to analyze
    epsilon : float
        Threshold for "rise" (default: 0.0)
    output_dir : Path, optional
        Directory to save results
    
    Returns:
    --------
    results : dict
        Analysis results with statistics
    """
    # Load FH carriers
    is_carrier, fh_df = load_fh_carriers(fh_file, processed_ids)
    
    # Convert to numpy if needed and truncate to match processed_ids length
    def to_numpy(x):
        import torch
        return x.detach().cpu().numpy() if 'torch' in str(type(x)) else x
    
    # Ensure Y and theta match processed_ids length
    N = len(processed_ids)
    Y_np = to_numpy(Y[:N])
    theta_np = to_numpy(theta[:N])
    
    # Convert event_indices to numpy array
    event_indices = np.array(event_indices)
    
    # Select relevant diseases
    Y_sel = Y_np[:, event_indices, :]  # [N, n_events, T]
    
    # Find first time index with any event
    any_event_over_ev = Y_sel.any(axis=1)  # [N, T]
    has_event = any_event_over_ev.any(axis=1)  # [N]
    first_event_t = np.full(len(has_event), -1, dtype=int)
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
    
    # 2x2 table: rows = carrier vs noncarrier, cols = rise vs no-rise
    table = [[ev_car, n_car - ev_car],
             [ev_non, n_non - ev_non]]
    OR, p = fisher_exact(table, alternative='greater')
    
    # Proportion CIs
    car_ci = proportion_confint(ev_car, n_car, method='wilson') if n_car > 0 else (np.nan, np.nan)
    non_ci = proportion_confint(ev_non, n_non, method='wilson') if n_non > 0 else (np.nan, np.nan)
    
    # Print results
    print("\n" + "="*80)
    print("FH CARRIERS: ENRICHMENT OF PRE-EVENT SIGNATURE RISE")
    print("="*80)
    print(f"Signature: {signature_idx}")
    print(f"Window: last {pre_window} years before first event; epsilon={epsilon}")
    print(f"Valid N with event & sufficient history: {idx_valid.size}")
    print(f"Carriers:   {ev_car}/{n_car} rising  (prop={ev_car/max(n_car,1):.3f}, CI95={car_ci})")
    print(f"Noncarriers:{ev_non}/{n_non} rising  (prop={ev_non/max(n_non,1):.3f}, CI95={non_ci})")
    print(f"Fisher exact (greater) OR={OR:.3f}, p={p:.3e}")
    
    # Prepare results dictionary
    results = {
        'signature_idx': signature_idx,
        'pre_window': pre_window,
        'epsilon': epsilon,
        'n_carriers': n_car,
        'n_noncarriers': n_non,
        'n_carriers_rising': ev_car,
        'n_noncarriers_rising': ev_non,
        'prop_carriers_rising': ev_car / max(n_car, 1),
        'prop_noncarriers_rising': ev_non / max(n_non, 1),
        'carrier_ci_lower': car_ci[0],
        'carrier_ci_upper': car_ci[1],
        'noncarrier_ci_lower': non_ci[0],
        'noncarrier_ci_upper': non_ci[1],
        'odds_ratio': OR,
        'p_value': p,
        'valid_n': idx_valid.size,
        'is_carrier': is_carrier,
        'idx_valid': idx_valid,
        'first_event_t': first_event_t,
        'is_rise': is_rise,
        'delta': delta
    }
    
    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame({
            'metric': ['n_carriers', 'n_noncarriers', 'n_carriers_rising', 'n_noncarriers_rising',
                      'prop_carriers_rising', 'prop_noncarriers_rising', 'odds_ratio', 'p_value'],
            'value': [n_car, n_non, ev_car, ev_non, ev_car/max(n_car,1), ev_non/max(n_non,1), OR, p]
        })
        results_df.to_csv(output_dir / f'FH_signature{signature_idx}_enrichment.csv', index=False)
        print(f"\n✓ Saved results to {output_dir}")
    
    return results


def visualize_signature_trajectory_fh(results, theta, Y, processed_ids, event_indices,
                                      signature_idx, pre_window=5, post_window=3):
    """
    Visualize signature trajectories for FH carriers vs noncarriers before events.
    Matches the original notebook visualization style.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_signature_enrichment_fh
    theta : np.array
        Signature loadings [N, K, T]
    Y : np.array
        Disease outcomes [N, D, T]
    processed_ids : np.array
        Patient IDs
    event_indices : list
        Disease indices for event composite
    signature_idx : int
        Signature index
    pre_window : int
        Years before event to show
    post_window : int
        Years after event to show
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    is_carrier = results['is_carrier']
    
    # Convert to numpy if needed and truncate to match processed_ids length
    def to_numpy(x):
        import torch
        return x.detach().cpu().numpy() if 'torch' in str(type(x)) else x
    
    N = len(processed_ids)
    theta_np = to_numpy(theta[:N])
    Y_np = to_numpy(Y[:N])
    ev_idx = np.array(event_indices, dtype=int)
    
    # Find first event time per person
    Y_sel = (Y_np[:, ev_idx, :] > 0)  # [N, |ev|, T]
    has_event = Y_sel.any(axis=(1, 2))
    any_ev_time = Y_sel.any(axis=1)  # [N, T]
    first_ev_t = np.full(N, -1, dtype=int)
    first_ev_t[has_event] = np.argmax(any_ev_time[has_event], axis=1)
    
    sig = theta_np[:, signature_idx, :]  # [N, T]
    T = sig.shape[1]
    aligned_span = np.arange(-pre_window, post_window + 1)  # e.g., -5..+3
    L = len(aligned_span)
    
    def build_aligned(sig, first_t, mask):
        idx = np.where(mask & (first_t >= pre_window) & (first_t < T - post_window))[0]
        aligned = np.empty((len(idx), L), dtype=float)
        aligned[:] = np.nan
        for j, i in enumerate(idx):
            t0 = first_t[i]
            aligned[j] = sig[i, t0 - pre_window : t0 + post_window + 1]
        return idx, aligned
    
    idx_car, aligned_car = build_aligned(sig, first_ev_t, is_carrier)
    idx_non, aligned_non = build_aligned(sig, first_ev_t, ~is_carrier)
    
    # Pre-event delta distributions (last 5y: end-start)
    delta_car = aligned_car[:, pre_window] - aligned_car[:, 0]
    delta_non = aligned_non[:, pre_window] - aligned_non[:, 0]
    rise_car = (delta_car > 0).mean()
    rise_non = (delta_non > 0).mean()
    
    # Mean and 95% bootstrap CI for trajectories
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
    
    # (A) Event-aligned mean trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    x = aligned_span
    ax1.plot(x, m_car, color='#2c7fb8', lw=2.5, label=f'FH carriers (n={aligned_car.shape[0]:,})')
    ax1.fill_between(x, lo_car, hi_car, color='#2c7fb8', alpha=0.2)
    ax1.plot(x, m_non, color='#f03b20', lw=2.5, label=f'Noncarriers (n={aligned_non.shape[0]:,})')
    ax1.fill_between(x, lo_non, hi_non, color='#f03b20', alpha=0.2)
    ax1.axvline(0, color='k', ls='--', lw=1)
    ax1.set_xlabel('Years relative to first event (0 = event)')
    ax1.set_ylabel('Signature 5 loading (θ)')
    ax1.set_title('Signature 5 trajectory aligned to first event')
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=True)
    
    # (B) Δ over last 5y before event
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(min(delta_car.min(), delta_non.min()),
                      max(delta_car.max(), delta_non.max()), 40)
    ax2.hist(delta_non, bins=bins, alpha=0.6, color='#f03b20', label=f'Noncarriers (rise={rise_non:.2%})')
    ax2.hist(delta_car, bins=bins, alpha=0.6, color='#2c7fb8', label=f'Carriers (rise={rise_car:.2%})')
    ax2.axvline(0, color='k', ls='--', lw=1)
    ax2.set_xlabel('ΔSig5 (end-start) over last 5y pre-event')
    ax2.set_ylabel('Count')
    ax2.set_title('Pre-event change in Signature 5')
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=True)
    
    fig.suptitle('FH carriers show stronger pre-event rise in Signature 5', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Analyze signature enrichment in FH mutation carriers"
    )
    parser.add_argument('--fh_file', type=str, required=True,
                       help='Path to FH carrier file (tab-separated with IID/eid column)')
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
    results = analyze_signature_enrichment_fh(
        args.fh_file, args.signature_idx, event_indices, theta, Y, processed_ids,
        args.pre_window, args.epsilon, args.output_dir
    )
    
    # Generate plot if requested
    if args.plot:
        fig = visualize_signature_trajectory_fh(
            results, theta, Y, processed_ids, event_indices,
            args.signature_idx, args.pre_window, post_window=3
        )
        if args.output_dir:
            output_path = Path(args.output_dir) / f'FH_signature{args.signature_idx}_trajectory.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {output_path}")
        plt.show()


if __name__ == '__main__':
    main()

