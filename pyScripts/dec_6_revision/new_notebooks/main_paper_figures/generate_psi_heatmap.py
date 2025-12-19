#!/usr/bin/env python3
"""
Generate Panel B: PSI Heatmap with arrows for disease-signature associations.

This creates a heatmap showing disease-signature associations (psi values)
with arrows/labels indicating which diseases belong to which signatures.
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

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

# Signature labels for grouping
SIGNATURE_LABELS = {
    0: 'Cardiac Arrhythmias',
    1: 'Musculoskeletal',
    2: 'Upper GI/Esophageal',
    3: 'Mixed/General Medical',
    4: 'Upper Respiratory',
    5: 'Ischemic cardiovascular',
    6: 'Metastatic Cancer',
    7: 'Pain/Inflammation',
    8: 'Gynecologic',
    9: 'Spinal Disorders',
    10: 'Ophthalmologic',
    11: 'Cerebrovascular',
    12: 'Renal/Urologic',
    13: 'Male Urogenital',
    14: 'Pulmonary/Smoking',
    15: 'Metabolic/Diabetes',
    16: 'Infectious/Critical Care',
    17: 'Lower GI/Colon',
    18: 'Hepatobiliary',
    19: 'Dermatologic/Oncologic',
    20: 'Health'
}


def plot_psi_heatmap_with_arrows(psi, clusters, disease_names, 
                                  output_path=None, figsize=(14, 18),
                                  vmin=-5, vmax=3, show_top_diseases_per_sig=3):
    """
    Create PSI heatmap with arrows indicating disease-signature associations.
    
    Parameters:
    -----------
    psi : np.ndarray, shape (K, D)
        PSI values (log odds ratios) for each signature-disease pair
    clusters : np.ndarray, shape (D,)
        Disease-to-signature assignments (which signature each disease belongs to)
    disease_names : list
        List of disease names
    output_path : str or None
        Path to save the figure. If None, displays instead.
    figsize : tuple
        Figure size (width, height)
    vmin, vmax : float
        Color scale limits for the heatmap
    show_top_diseases_per_sig : int
        Number of top diseases per signature to label with arrows
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    K, D = psi.shape
    
    # Transpose PSI to get (D, K) for heatmap
    psi_heatmap = psi.T  # Shape: (D, K)
    
    # Group diseases by signature
    sig_to_diseases = {}
    for d in range(len(clusters)):
        sig = clusters[d]
        if sig not in sig_to_diseases:
            sig_to_diseases[sig] = []
        sig_to_diseases[sig].append(d)
    
    # Sort diseases within each signature by their psi value in that signature (most positive first)
    disease_order = []
    for sig in sorted(sig_to_diseases.keys()):
        diseases_in_sig = sig_to_diseases[sig]
        # Get psi value for each disease in its own signature
        psi_values = [psi[sig, d] for d in diseases_in_sig]
        # Sort by psi (descending - most positive first)
        sorted_indices = sorted(range(len(diseases_in_sig)), 
                               key=lambda i: psi_values[i], reverse=True)
        disease_order.extend([diseases_in_sig[i] for i in sorted_indices])
    
    # Reorder PSI and disease names
    psi_ordered = psi_heatmap[disease_order, :]
    disease_names_ordered = [disease_names[i] if i < len(disease_names) else f"Disease_{i}" 
                            for i in disease_order]
    
    # Create figure (increased height to accommodate all disease names)
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], max(figsize[1], 24)))
    
    # Create heatmap
    im = ax.imshow(psi_ordered, aspect='auto', cmap='RdBu_r', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Signatures', fontsize=16, fontweight='bold')
    ax.set_ylabel('Disease index (psi kd)', fontsize=16, fontweight='bold')
    ax.set_title('Panel B: Disease-Signature Associations (Log Odds Ratio)', 
                 fontsize=18, fontweight='bold', pad=25)
    
    # Set x-axis ticks (signatures 0-K)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f'{i}' for i in range(K)], fontsize=11, fontweight='bold')
    
    # Set y-axis ticks (show all disease names - will be hard to read but comprehensive)
    n_diseases = len(disease_order)
    ax.set_yticks(np.arange(n_diseases))
    ax.set_yticklabels([disease_names_ordered[i] for i in range(n_diseases)], 
                      fontsize=6)  # Small font size to fit all names
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Log odds ratio (psi)', pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Log odds ratio (psi)', fontsize=14, fontweight='bold')
    
    # Add vertical lines to separate signatures
    for k in range(K + 1):
        ax.axvline(k - 0.5, color='white', linewidth=1.5, alpha=0.8)
    
    # Add horizontal lines to separate signature groups (diseases grouped by signature)
    # Also add signature labels as brackets on the right side
    current_y = 0
    sig_y_positions = {}  # Store y positions for each signature group

    for sig in sorted(sig_to_diseases.keys()):
        n_diseases_in_sig = len(sig_to_diseases[sig])
        if current_y > 0:
            ax.axhline(current_y - 0.5, color='white', linewidth=1.5, alpha=0.8)
        
        # Store the middle y position for this signature group
        sig_y_positions[sig] = current_y + n_diseases_in_sig / 2 - 0.5
        current_y += n_diseases_in_sig
    
    # Add signature labels on the right side with brackets
    ax2 = ax.twinx()  # Create a second y-axis for labels
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    # Get x position for labels (just to the right of the heatmap)
    x_label_pos = K + 0.5

    # Add brackets and labels for each signature group
    for sig in sorted(sig_to_diseases.keys()):
        n_diseases_in_sig = len(sig_to_diseases[sig])
        sig_start_y = sum(len(sig_to_diseases[s]) for s in sorted(sig_to_diseases.keys()) if s < sig) - 0.5
        sig_end_y = sig_start_y + n_diseases_in_sig
        
        # Get label text
        label_text = SIGNATURE_LABELS.get(sig, f'Signature {sig}')
        
        # Draw bracket (curly brace)
        y_mid = (sig_start_y + sig_end_y) / 2
        
        # Draw bracket using lines
        bracket_width = 0.3
        bracket_x_start = x_label_pos - bracket_width
        bracket_x_end = x_label_pos
        
        # Top of bracket
        ax2.plot([bracket_x_start, bracket_x_end], [sig_start_y, sig_start_y], 
                 'k-', linewidth=1.5, clip_on=False)
        # Bottom of bracket
        ax2.plot([bracket_x_start, bracket_x_end], [sig_end_y, sig_end_y], 
                 'k-', linewidth=1.5, clip_on=False)
        # Vertical line
        ax2.plot([bracket_x_start, bracket_x_start], [sig_start_y, sig_end_y], 
                 'k-', linewidth=1.5, clip_on=False)
            
        # Add label text
        ax2.text(x_label_pos + 0.1, y_mid, label_text, 
                 fontsize=9, va='center', ha='left', 
                 rotation=0, fontweight='bold',
                 clip_on=False)
    
    # Adjust x-axis limits to accommodate labels
    ax.set_xlim(-0.5, x_label_pos + 3)
    ax2.set_xlim(-0.5, x_label_pos + 3)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PSI heatmap to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function to generate PSI heatmap."""
    
    print("Loading UKB data...")
    
    # Load UKB checkpoint
    ukb_checkpoint_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt'
    ukb_checkpoint = torch.load(ukb_checkpoint_path, map_location='cpu')
    
    # Extract PSI, clusters, and disease names
    if 'model_state_dict' in ukb_checkpoint:
        psi = ukb_checkpoint['model_state_dict']['psi']
    else:
        psi = ukb_checkpoint['psi']
    
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    clusters = ukb_checkpoint['clusters']
    if torch.is_tensor(clusters):
        clusters = clusters.numpy()

    ukb_old_checkpoint_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam.pt'
    ukb_old_checkpoint = torch.load(ukb_old_checkpoint_path, map_location='cpu')
    disease_names = ukb_old_checkpoint['disease_names']
    if isinstance(disease_names, (list, tuple)):
        disease_names = list(disease_names)
    elif hasattr(disease_names, 'values'):
        disease_names = disease_names.values.tolist()
    
    print(f"  PSI shape: {psi.shape}")
    print(f"  Clusters shape: {clusters.shape}")
    print(f"  Number of diseases: {len(disease_names)}")
    
    # Create output directory
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate heatmap
    output_path = output_dir / 'panel_b_psi_heatmap.pdf'
    fig = plot_psi_heatmap_with_arrows(
        psi, clusters, disease_names,
        output_path=str(output_path),
        figsize=(14, 18),
        show_top_diseases_per_sig=3
    )
    
    print(f"\n✓ PSI heatmap generated successfully!")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()

