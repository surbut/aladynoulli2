#!/usr/bin/env python3
"""
Plot individual-level signature correlations over time.

Shows how signatures correlate with each other in a single individual's trajectory,
computed from lambda values at each timepoint.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.special import expit
from typing import Optional
import matplotlib.cm as cm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def get_signature_colors(K):
    """Return a list of K distinct colors for signatures."""
    if K <= 20:
        sig_colors = cm.get_cmap('tab20')(np.linspace(0, 1, K))
        return [sig_colors[i] for i in range(K)]
    else:
        colors_20 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors_b = cm.get_cmap('tab20b')(np.linspace(0, 1, 20))
        all_colors = np.vstack([colors_20, colors_b])
        if K <= 40:
            return [all_colors[i] for i in range(K)]
        else:
            return sns.color_palette("hsv", K)


def plot_individual_signature_correlations(
    lambda_values: np.ndarray,  # (K, T) - lambda values for one individual
    time_points: np.ndarray,    # (T,) - age or time values
    age_offset: int = 30,
    output_path: Optional[str] = None,
    figsize: tuple = (16, 6)
):
    """
    Plot signature correlations over time for a single individual.
    
    Parameters:
    -----------
    lambda_values : np.ndarray, shape (K, T)
        Lambda values for one individual across all signatures and timepoints
    time_points : np.ndarray, shape (T,)
        Time points (will be converted to age if needed)
    age_offset : int
        Age offset (default: 30)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    K, T = lambda_values.shape
    
    # Convert time_points to ages if needed
    if len(time_points) == T:
        ages = time_points
    else:
        ages = np.arange(T) + age_offset
    
    # Calculate correlation matrix at each timepoint
    # For each time t, we'll use a rolling window to get stable correlations
    window_size = 5  # Use 5 timepoints for correlation calculation
    half_window = window_size // 2
    
    # Store correlations over time
    # For each pair of signatures, track correlation over time
    correlation_over_time = {}  # (sig_i, sig_j) -> array of correlations
    
    # Initialize storage
    for i in range(K):
        for j in range(i+1, K):
            correlation_over_time[(i, j)] = np.full(T, np.nan)
    
    # Calculate rolling correlations
    for t in range(half_window, T - half_window):
        # Get window of lambda values
        window_start = max(0, t - half_window)
        window_end = min(T, t + half_window + 1)
        lambda_window = lambda_values[:, window_start:window_end]  # (K, window_size)
        
        # Calculate correlation matrix for this window
        corr_matrix = np.corrcoef(lambda_window)
        
        # Store correlations for each pair
        for i in range(K):
            for j in range(i+1, K):
                correlation_over_time[(i, j)][t] = corr_matrix[i, j]
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Correlation matrix at a representative timepoint (middle)
    ax1 = axes[0]
    mid_time = T // 2
    window_start = max(0, mid_time - half_window)
    window_end = min(T, mid_time + half_window + 1)
    lambda_window = lambda_values[:, window_start:window_end]
    corr_matrix_mid = np.corrcoef(lambda_window)
    
    # Plot heatmap
    im = ax1.imshow(corr_matrix_mid, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'Signature Correlation Matrix\n(Age ~{int(ages[mid_time])} years)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Signature', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Signature', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(K))
    ax1.set_yticks(range(K))
    ax1.set_xticklabels([f'Sig {i}' for i in range(K)], fontsize=9)
    ax1.set_yticklabels([f'Sig {i}' for i in range(K)], fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Correlation', fontsize=10)
    
    # Panel 2: Time series of correlations for top correlated pairs
    ax2 = axes[1]
    
    # Find top correlated pairs (by absolute value at mid-time)
    corr_pairs = []
    for (i, j), corrs in correlation_over_time.items():
        corr_at_mid = corr_matrix_mid[i, j]
        corr_pairs.append({
            'sig1': i,
            'sig2': j,
            'corr_mid': corr_at_mid,
            'corrs_over_time': corrs
        })
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x['corr_mid']), reverse=True)
    
    # Get colors
    colors = get_signature_colors(K)
    
    # Plot top 8 most correlated pairs
    top_n = min(8, len(corr_pairs))
    for idx, pair in enumerate(corr_pairs[:top_n]):
        sig1, sig2 = pair['sig1'], pair['sig2']
        corrs = pair['corrs_over_time']
        
        # Use color based on average of the two signature colors
        color1 = colors[sig1]
        color2 = colors[sig2]
        # Blend colors
        blend_color = tuple(0.5 * np.array(color1[:3]) + 0.5 * np.array(color2[:3])) + (1.0,)
        
        ax2.plot(ages, corrs, 
                label=f'Sig {sig1}–Sig {sig2} (r={pair["corr_mid"]:.2f})',
                color=blend_color, linewidth=2, alpha=0.8)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_title('Signature Correlations Over Time\n(Top Correlated Pairs)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✓ Saved to: {output_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_individual_cumulative_risk_contributions(
    lambda_values: np.ndarray,  # (K, T) - lambda for one individual
    phi_values: np.ndarray,     # (K, D, T) - phi for all diseases
    disease_idx: int,           # Which disease to show
    disease_name: str,          # Disease name
    time_points: np.ndarray,    # (T,) - time/age values
    kappa: float = 1.0,
    age_offset: int = 30,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8)
):
    """
    Plot individual-level cumulative disease risk contributions by signature.
    
    Shows stacked area chart of how each signature contributes to disease risk over time.
    
    Parameters:
    -----------
    lambda_values : np.ndarray, shape (K, T)
        Lambda values for one individual
    phi_values : np.ndarray, shape (K, D, T)
        Phi values for all diseases
    disease_idx : int
        Index of disease to visualize
    disease_name : str
        Name of the disease
    time_points : np.ndarray, shape (T,)
        Time/age points
    kappa : float
        Kappa scaling factor
    age_offset : int
        Age offset (default: 30)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    K, T = lambda_values.shape
    
    # Convert to ages if needed
    if len(time_points) == T:
        ages = time_points
    else:
        ages = np.arange(T) + age_offset
    
    # Calculate theta (normalized lambda)
    exp_lambda = np.exp(lambda_values)  # (K, T)
    theta = exp_lambda / np.sum(exp_lambda, axis=0, keepdims=True)  # (K, T)
    
    # Get phi for this disease
    phi_disease = phi_values[:, disease_idx, :]  # (K, T)
    
    # Convert phi to probabilities (sigmoid)
    phi_probs = expit(phi_disease)  # (K, T)
    
    # Calculate contributions: theta_k * phi_kd
    contributions = theta * phi_probs  # (K, T)
    
    # Calculate total latent risk
    total_latent_risk = np.sum(contributions, axis=0)  # (T,)
    
    # Apply kappa
    total_risk = kappa * total_latent_risk  # (T,)
    total_risk = np.clip(total_risk, 0, 1)
    
    # Scale contributions to match total risk
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.divide(total_risk, total_latent_risk,
                          out=np.zeros_like(total_risk),
                          where=total_latent_risk != 0)
        scaled_contributions = contributions * scaling[None, :]  # (K, T)
    
    # Get colors
    colors = get_signature_colors(K)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stacked area plot
    ax.stackplot(ages, scaled_contributions,
                labels=[f'Sig {k}' for k in range(K)],
                colors=colors, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Also plot total risk line
    ax.plot(ages, total_risk, 'k--', linewidth=2, label='Total Risk', alpha=0.8)
    
    ax.set_title(f'Individual Cumulative Risk Contributions: {disease_name}',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Risk Contribution', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(1.0, np.max(total_risk) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
             framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✓ Saved to: {output_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


if __name__ == '__main__':
    # Example usage
    print("Individual signature correlation and risk contribution plotting functions")
    print("Import and use plot_individual_signature_correlations() and")
    print("plot_individual_cumulative_risk_contributions()")

