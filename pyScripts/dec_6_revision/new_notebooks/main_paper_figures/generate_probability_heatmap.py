#!/usr/bin/env python3
"""
Generate Panel A: Probability Heatmap showing disease probabilities across signatures.

This creates a heatmap showing sigmoid(phi) values (probabilities) for each disease
across all signatures, with arrows/labels indicating top diseases per signature.
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from pathlib import Path
from scipy.special import expit as sigmoid

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']


def plot_probability_heatmap_with_arrows(phi, clusters, disease_names,
                                         output_path=None, figsize=(14, 18),
                                         vmin=0, vmax=1, show_top_diseases_per_sig=3,
                                         time_averaging='max'):
    """
    Create probability heatmap with arrows indicating disease-signature associations.
    
    Parameters:
    -----------
    phi : np.ndarray, shape (K, D, T)
        Phi values (log hazard ratios) for each signature-disease-timepoint
    clusters : np.ndarray, shape (D,)
        Disease-to-signature assignments (which signature each disease belongs to)
    disease_names : list
        List of disease names
    output_path : str or None
        Path to save the figure. If None, displays instead.
    figsize : tuple
        Figure size (width, height)
    vmin, vmax : float
        Color scale limits for the heatmap (0-1 for probabilities)
    show_top_diseases_per_sig : int
        Number of top diseases per signature to label with arrows
    time_averaging : str
        How to aggregate over time: 'max', 'mean', or 'last'
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    K, D, T = phi.shape
    
    # Convert phi to probabilities
    phi_prob = sigmoid(phi)  # Shape: (K, D, T)
    
    # Aggregate over time dimension
    if time_averaging == 'max':
        prob_matrix = np.max(phi_prob, axis=2)  # Shape: (K, D) - max probability over time
    elif time_averaging == 'mean':
        prob_matrix = np.mean(phi_prob, axis=2)  # Shape: (K, D) - mean probability over time
    elif time_averaging == 'last':
        prob_matrix = phi_prob[:, :, -1]  # Shape: (K, D) - last timepoint
    else:
        raise ValueError(f"Unknown time_averaging: {time_averaging}")
    
    # Transpose to get (D, K) for heatmap
    prob_heatmap = prob_matrix.T  # Shape: (D, K)
    
    # Group diseases by signature
    sig_to_diseases = {}
    for d in range(len(clusters)):
        sig = clusters[d]
        if sig not in sig_to_diseases:
            sig_to_diseases[sig] = []
        sig_to_diseases[sig].append(d)
    
    # Sort diseases within each signature by their probability in that signature (highest first)
    disease_order = []
    for sig in sorted(sig_to_diseases.keys()):
        diseases_in_sig = sig_to_diseases[sig]
        # Get probability value for each disease in its own signature
        prob_values = [prob_matrix[sig, d] for d in diseases_in_sig]
        # Sort by probability (descending - highest first)
        sorted_indices = sorted(range(len(diseases_in_sig)), 
                               key=lambda i: prob_values[i], reverse=True)
        disease_order.extend([diseases_in_sig[i] for i in sorted_indices])
    
    # Reorder probability matrix and disease names
    prob_ordered = prob_heatmap[disease_order, :]
    disease_names_ordered = [disease_names[i] if i < len(disease_names) else f"Disease_{i}" 
                            for i in disease_order]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(prob_ordered, aspect='auto', cmap='YlOrRd', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Signatures', fontsize=16, fontweight='bold')
    ax.set_ylabel('Diseases (ordered by signature)', fontsize=16, fontweight='bold')
    ax.set_title(f'Panel A: Disease-Signature Probabilities ({time_averaging.capitalize()} over time)', 
                 fontsize=18, fontweight='bold', pad=25)
    
    # Set x-axis ticks (signatures 0-K)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f'Sig {i}' for i in range(K)], fontsize=11, fontweight='bold')
    
    # Set y-axis ticks (show fewer labels for readability)
    n_diseases = len(disease_order)
    y_tick_interval = max(1, n_diseases // 30)  # Show ~30 labels
    y_ticks = np.arange(0, n_diseases, y_tick_interval)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{disease_order[i]}' for i in range(0, n_diseases, y_tick_interval)], 
                      fontsize=8)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Probability', pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Probability (sigmoid(phi))', fontsize=14, fontweight='bold')
    
    # Add vertical lines to separate signatures
    for k in range(K + 1):
        ax.axvline(k - 0.5, color='white', linewidth=1.5, alpha=0.8)
    
    # Add horizontal lines to separate signature groups
    current_y = 0
    for sig in sorted(sig_to_diseases.keys()):
        n_diseases_in_sig = len(sig_to_diseases[sig])
        if current_y > 0:
            ax.axhline(current_y - 0.5, color='white', linewidth=1.5, alpha=0.8)
        current_y += n_diseases_in_sig
    
    # Add arrows and labels for top diseases in each signature
    current_y = 0
    for sig in sorted(sig_to_diseases.keys()):
        diseases_in_sig = sig_to_diseases[sig]
        # Get top diseases by probability value in this signature
        prob_values = [(prob_matrix[sig, d], d) for d in diseases_in_sig]
        prob_values.sort(reverse=True)  # Sort descending
        
        # Label top N diseases
        top_diseases = prob_values[:show_top_diseases_per_sig]
        
        for rank, (prob_val, d_idx) in enumerate(top_diseases):
            # Find position in ordered list
            y_pos = disease_order.index(d_idx)
            x_start = -0.5  # Start from left edge
            x_end = sig + 0.5  # Point to signature column
            
            # Get disease name (truncate if too long)
            disease_name = disease_names[d_idx] if d_idx < len(disease_names) else f"Disease_{d_idx}"
            if len(disease_name) > 40:
                disease_name = disease_name[:37] + "..."
            
            # Create arrow
            arrow = FancyArrowPatch(
                (x_start, y_pos), (x_end, y_pos),
                arrowstyle='->', mutation_scale=15, 
                linewidth=1.5, color='black', alpha=0.6,
                zorder=10
            )
            ax.add_patch(arrow)
            
            # Add text label on the left
            ax.text(x_start - 0.3, y_pos, disease_name, 
                   fontsize=8, ha='right', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8),
                   zorder=11)
        
        current_y += len(diseases_in_sig)
    
    # Adjust x-axis limits to accommodate labels
    ax.set_xlim(-5, K)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved probability heatmap to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function to generate probability heatmap."""
    
    print("Loading UKB data...")
    
    # Load master checkpoint with phi
    master_checkpoint = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt', map_location='cpu')
    phi = master_checkpoint['model_state_dict']['phi']
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    
    # Load UKB checkpoint for clusters and disease names
    ukb_checkpoint_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam.pt'
    ukb_checkpoint = torch.load(ukb_checkpoint_path, map_location='cpu')
    
    clusters = ukb_checkpoint['clusters']
    if torch.is_tensor(clusters):
        clusters = clusters.numpy()
    
    disease_names = ukb_checkpoint['disease_names']
    if isinstance(disease_names, (list, tuple)):
        disease_names = list(disease_names)
    elif hasattr(disease_names, 'values'):
        disease_names = disease_names.values.tolist()
    
    print(f"  Phi shape: {phi.shape}")
    print(f"  Clusters shape: {clusters.shape}")
    print(f"  Number of diseases: {len(disease_names)}")
    
    # Create output directory
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate heatmap with max probability over time
    output_path = output_dir / 'panel_a_probability_heatmap_max.pdf'
    fig = plot_probability_heatmap_with_arrows(
        phi, clusters, disease_names,
        output_path=str(output_path),
        figsize=(14, 18),
        show_top_diseases_per_sig=3,
        time_averaging='max'
    )
    
    print(f"\n✓ Probability heatmap generated successfully!")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()

