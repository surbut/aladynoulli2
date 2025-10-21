"""
Compute weighted smoothed prevalence for UK Biobank data
"""

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def match_weights_to_ids(weights_df, processed_ids):
    """
    Match weights to processed IDs
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        DataFrame with columns ['f.eid', 'LassoWeight']
    processed_ids : numpy array
        Array of IDs corresponding to rows in Y [N]
        
    Returns:
    --------
    matched_weights : numpy array
        Weights matched to processed_ids order [N]
    match_mask : numpy array (bool)
        Boolean mask indicating which IDs were found [N]
    """
    print("Matching weights to IDs...")
    print(f"  Weights available: {len(weights_df)}")
    print(f"  Processed IDs: {len(processed_ids)}")
    
    # Create a mapping from f.eid to LassoWeight
    weight_dict = dict(zip(weights_df['f.eid'].values, weights_df['LassoWeight'].values))
    
    # Match weights to processed_ids
    matched_weights = np.zeros(len(processed_ids))
    match_mask = np.zeros(len(processed_ids), dtype=bool)
    
    for i, pid in enumerate(processed_ids):
        if pid in weight_dict:
            matched_weights[i] = weight_dict[pid]
            match_mask[i] = True
    
    n_matched = match_mask.sum()
    print(f"  Matched: {n_matched} ({100*n_matched/len(processed_ids):.1f}%)")
    print(f"  Unmatched: {len(processed_ids) - n_matched}")
    
    if n_matched == 0:
        raise ValueError("No IDs matched! Check that f.eid values match processed_ids")
    
    return matched_weights, match_mask


def compute_weighted_prevalence(Y, weights, window_size=5, match_mask=None):
    """
    Compute weighted smoothed time-dependent prevalence
    
    Parameters:
    -----------
    Y : numpy array or torch tensor
        Binary outcome matrix of shape [N x D x T]
        N = number of individuals
        D = number of diseases
        T = number of time points
    weights : numpy array
        Inverse probability weights of shape [N]
    window_size : int
        Gaussian smoothing window size (sigma parameter)
        Default: 5 (moderate smoothing)
    match_mask : numpy array (bool), optional
        Boolean mask indicating which individuals have weights [N]
        If provided, only uses matched individuals
        
    Returns:
    --------
    prevalence_t : numpy array
        Weighted smoothed prevalence matrix of shape [D x T]
    """
    # Convert to numpy if torch tensor
    if hasattr(Y, 'numpy'):
        Y = Y.numpy()
    
    N, D, T = Y.shape
    
    # Filter to only matched individuals if mask provided
    if match_mask is not None:
        Y_matched = Y[match_mask]
        weights_matched = weights[match_mask]
        N_matched = len(weights_matched)
        print(f"\nUsing {N_matched} matched individuals (out of {N})")
    else:
        Y_matched = Y
        weights_matched = weights
        N_matched = N
    
    # Ensure weights are numpy array and normalized
    weights_matched = np.array(weights_matched)
    
    # Normalize weights to sum to N_matched (for proper prevalence calculation)
    weights_norm = weights_matched / weights_matched.sum() * N_matched
    
    print(f"\nComputing weighted prevalence:")
    print(f"  Data shape: {Y_matched.shape}")
    print(f"  Weight stats: mean={weights_matched.mean():.3f}, std={weights_matched.std():.3f}")
    print(f"  Smoothing window: {window_size}")
    
    # Compute weighted prevalence using einsum (fast vectorized operation)
    # For each disease d and time t: sum(Y[n,d,t] * weight[n]) / sum(weights)
    prevalence_t = np.einsum('n,ndt->dt', weights_norm, Y_matched) / N_matched
    
    print(f"  Raw prevalence range: [{prevalence_t.min():.4f}, {prevalence_t.max():.4f}]")
    
    # Apply Gaussian smoothing to each disease trajectory
    for d in range(D):
        prevalence_t[d, :] = gaussian_filter1d(prevalence_t[d, :], sigma=window_size)
    
    print(f"  Smoothed prevalence range: [{prevalence_t.min():.4f}, {prevalence_t.max():.4f}]")
    print(f"✓ Weighted prevalence computed\n")
    
    return prevalence_t


def compute_unweighted_prevalence(Y, window_size=5, match_mask=None):
    """
    Compute unweighted smoothed prevalence for comparison
    
    Parameters:
    -----------
    Y : numpy array or torch tensor
        Binary outcome matrix of shape [N x D x T]
    window_size : int
        Gaussian smoothing window size
    match_mask : numpy array (bool), optional
        Boolean mask indicating which individuals to use
        If provided, only uses matched individuals for fair comparison
        
    Returns:
    --------
    prevalence_t : numpy array
        Unweighted smoothed prevalence matrix of shape [D x T]
    """
    # Convert to numpy if needed
    if hasattr(Y, 'numpy'):
        Y = Y.numpy()
    
    N, D, T = Y.shape
    
    # Filter to matched individuals if mask provided
    if match_mask is not None:
        Y = Y[match_mask]
        print(f"\nUsing {len(Y)} matched individuals (out of {N})")
    
    print(f"Computing unweighted prevalence:")
    print(f"  Data shape: {Y.shape}")
    
    # Simple mean across individuals
    prevalence_t = Y.mean(axis=0)  # Shape: [D x T]
    
    print(f"  Raw prevalence range: [{prevalence_t.min():.4f}, {prevalence_t.max():.4f}]")
    
    # Apply Gaussian smoothing
    for d in range(D):
        prevalence_t[d, :] = gaussian_filter1d(prevalence_t[d, :], sigma=window_size)
    
    print(f"  Smoothed prevalence range: [{prevalence_t.min():.4f}, {prevalence_t.max():.4f}]")
    print(f"✓ Unweighted prevalence computed\n")
    
    return prevalence_t


def plot_prevalence_comparison(prevalence_weighted, prevalence_unweighted, 
                               disease_indices, disease_names=None, 
                               time_labels=None):
    """
    Plot weighted vs unweighted prevalence for selected diseases
    
    Parameters:
    -----------
    prevalence_weighted : numpy array [D x T]
        Weighted prevalence
    prevalence_unweighted : numpy array [D x T]
        Unweighted prevalence
    disease_indices : list of int
        Which diseases to plot
    disease_names : list of str, optional
        Names of diseases
    time_labels : array, optional
        Time point labels (e.g., years)
    """
    n_diseases = len(disease_indices)
    fig, axes = plt.subplots(n_diseases, 1, figsize=(12, 3*n_diseases))
    
    if n_diseases == 1:
        axes = [axes]
    
    T = prevalence_weighted.shape[1]
    if time_labels is None:
        time_labels = np.arange(T)
    
    for idx, (ax, d) in enumerate(zip(axes, disease_indices)):
        # Plot both trajectories
        ax.plot(time_labels, prevalence_unweighted[d, :], 
                label='Unweighted (Biased)', 
                color='#ef4444', linewidth=2, alpha=0.8)
        ax.plot(time_labels, prevalence_weighted[d, :], 
                label='Weighted (Population-Representative)', 
                color='#3b82f6', linewidth=2, alpha=0.8)
        
        # Calculate difference
        diff = prevalence_weighted[d, :] - prevalence_unweighted[d, :]
        mean_diff = np.abs(diff).mean()
        max_diff = np.abs(diff).max()
        
        # Add title
        if disease_names is not None:
            title = f"Disease: {disease_names[d]}"
        else:
            title = f"Disease {d}"
        title += f"\n(Mean Δ = {mean_diff:.4f}, Max Δ = {max_diff:.4f})"
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Time Point', fontsize=10)
        ax.set_ylabel('Prevalence', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Highlight if difference is substantial
        if mean_diff > 0.01:  # More than 1% difference
            ax.axhline(y=prevalence_unweighted[d, :].mean(), 
                      color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(0.02, 0.98, 'Large bias detected!', 
                   transform=ax.transAxes, 
                   fontsize=9, color='red', fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def compare_prevalences(prevalence_weighted, prevalence_unweighted):
    """
    Compute summary statistics comparing weighted vs unweighted prevalence
    
    Returns:
    --------
    summary : dict
        Dictionary with comparison statistics
    """
    diff = prevalence_weighted - prevalence_unweighted
    abs_diff = np.abs(diff)
    
    summary = {
        'mean_abs_diff': abs_diff.mean(),
        'median_abs_diff': np.median(abs_diff),
        'max_abs_diff': abs_diff.max(),
        'diseases_with_large_diff': (abs_diff.mean(axis=1) > 0.01).sum(),
        'pct_diseases_affected': (abs_diff.mean(axis=1) > 0.01).sum() / diff.shape[0] * 100
    }
    
    print("\n" + "="*60)
    print("PREVALENCE COMPARISON SUMMARY")
    print("="*60)
    print(f"Mean absolute difference: {summary['mean_abs_diff']:.4f}")
    print(f"Median absolute difference: {summary['median_abs_diff']:.4f}")
    print(f"Maximum absolute difference: {summary['max_abs_diff']:.4f}")
    print(f"Diseases with >1% mean difference: {summary['diseases_with_large_diff']}")
    print(f"Percent of diseases affected: {summary['pct_diseases_affected']:.1f}%")
    print("="*60 + "\n")
    
    return summary


# Example usage
if __name__ == '__main__':
    # Load your data
    print("Loading data...")
    Y = torch.load('data_for_running/Y_tensor.pt')
    
    # Load processed IDs
    processed_ids = np.load('data_for_running/processed_ids.npy')  # Adjust path as needed
    # OR if IDs are in a different format:
    # processed_ids = np.array([1000015, 1000023, 1000037, ...])  # Your ID array
    
    # Load weights
    weights_df = pd.read_csv('UKBSelectionWeights.csv', sep='\s+', engine='python')
    
    print(f"\nData loaded:")
    print(f"  Y shape: {Y.shape}")
    print(f"  Processed IDs: {len(processed_ids)}")
    print(f"  Available weights: {len(weights_df)}")
    
    # Match weights to IDs
    matched_weights, match_mask = match_weights_to_ids(weights_df, processed_ids)
    
    print(f"\nMatched weight statistics (for matched individuals):")
    matched_weight_values = matched_weights[match_mask]
    print(f"  Mean: {matched_weight_values.mean():.3f}")
    print(f"  Std: {matched_weight_values.std():.3f}")
    print(f"  Range: [{matched_weight_values.min():.3f}, {matched_weight_values.max():.3f}]")
    
    # Compute both versions (using only matched individuals for fair comparison)
    prevalence_weighted = compute_weighted_prevalence(Y, matched_weights, window_size=5, match_mask=match_mask)
    prevalence_unweighted = compute_unweighted_prevalence(Y, window_size=5, match_mask=match_mask)
    
    # Compare
    summary = compare_prevalences(prevalence_weighted, prevalence_unweighted)
    
    # Find diseases with largest differences
    disease_diffs = np.abs(prevalence_weighted - prevalence_unweighted).mean(axis=1)
    top_diff_indices = np.argsort(disease_diffs)[-5:][::-1]
    
    print("Top 5 diseases with largest prevalence differences:")
    for idx, d in enumerate(top_diff_indices):
        print(f"  {idx+1}. Disease {d}: mean diff = {disease_diffs[d]:.4f}")
    
    # Plot comparison for top different diseases
    print("\nGenerating comparison plots...")
    fig = plot_prevalence_comparison(
        prevalence_weighted,
        prevalence_unweighted,
        disease_indices=top_diff_indices,
        disease_names=None  # Add your disease names if available
    )
    plt.savefig('prevalence_comparison_top5.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot to prevalence_comparison_top5.png")
    
    # Save prevalence arrays
    np.save('prevalence_t_weighted.npy', prevalence_weighted)
    np.save('prevalence_t_unweighted.npy', prevalence_unweighted)
    np.save('match_mask.npy', match_mask)  # Save mask for later use
    print("\n✓ Saved prevalence arrays:")
    print("  - prevalence_t_weighted.npy")
    print("  - prevalence_t_unweighted.npy")
    print("  - match_mask.npy")
    
    plt.show()