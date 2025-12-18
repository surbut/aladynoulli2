#!/usr/bin/env python3
"""
Generate Panel C: Disease Prevalence by Age Heatmap.

This creates a heatmap showing average disease probability (pi averaged over people)
for each disease across ages, with diseases clustered by signature.
Uses the same ordering as Panel B (PSI heatmap).
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import torch
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['ytick.labelsize'] = 10
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


def plot_pi_heatmap(pi_avg, psi, clusters, disease_names, age_offset=30,
                   output_path=None, figsize=(14, 16)):
    """
    Create Panel C: Disease prevalence by age heatmap.
    Uses the same ordering as Panel B (PSI heatmap).
    
    Parameters:
    -----------
    pi_avg : np.ndarray, shape (D, T)
        Average disease probability for each disease at each timepoint
    psi : np.ndarray, shape (K, D)
        PSI values (log odds ratios) for each signature-disease pair
    clusters : np.ndarray, shape (D,)
        Disease-to-signature assignments
    disease_names : list
        List of disease names
    age_offset : int
        Age offset (timepoint 0 = age_offset)
    output_path : str or None
        Path to save the figure. If None, displays instead.
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    D, T = pi_avg.shape
    age_points = np.arange(T) + age_offset
    
    # Group diseases by signature (same as PSI heatmap)
    sig_to_diseases = {}
    for d in range(len(clusters)):
        sig = clusters[d]
        if sig not in sig_to_diseases:
            sig_to_diseases[sig] = []
        sig_to_diseases[sig].append(d)
    
    # Sort diseases within each signature by their psi value in that signature (most positive first)
    # EXACTLY the same ordering as PSI heatmap
    disease_order = []
    for sig in sorted(sig_to_diseases.keys()):
        diseases_in_sig = sig_to_diseases[sig]
        # Get psi value for each disease in its own signature (same as PSI heatmap)
        psi_values = [psi[sig, d] for d in diseases_in_sig]
        # Sort by psi (descending - most positive first)
        sorted_indices = sorted(range(len(diseases_in_sig)), 
                               key=lambda i: psi_values[i], reverse=True)
        disease_order.extend([diseases_in_sig[i] for i in sorted_indices])
    
    # Reorder pi_avg using the same disease_order as PSI heatmap
    pi_avg_ordered = pi_avg[disease_order, :]  # Shape: (D, T)
    
    # Set vmax based on actual data (use 95th percentile to avoid outliers, or max)
    pi_max = np.max(pi_avg_ordered)
    pi_95th = np.percentile(pi_avg_ordered, 95)
    vmax_pi = min(pi_max * 1.1, pi_95th * 1.5)  # Use 95th percentile with some padding, or max
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(pi_avg_ordered, aspect='auto', cmap='Reds', 
                   vmin=0, vmax=vmax_pi, interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Age (yr)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Disease (clustered by signature)', fontsize=16, fontweight='bold')
    ax.set_title('Panel C: Average Disease Probability by Age', 
                 fontsize=18, fontweight='bold', pad=25)
    
    # Set x-axis ticks (ages)
    age_tick_interval = max(1, T // 10)
    age_tick_indices = np.arange(0, T, age_tick_interval)
    ax.set_xticks(age_tick_indices)
    ax.set_xticklabels([f'{age_points[i]:.0f}' for i in age_tick_indices], 
                      fontsize=11, fontweight='bold')
    
    # Set y-axis ticks (no disease labels - keep blank for cleaner look)
    n_diseases_show = len(disease_order)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Average probability', pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Average probability', fontsize=14, fontweight='bold')
    
    # Add horizontal lines to separate signature groups
    # Also add signature labels as brackets on the right side
    # Draw these AFTER the heatmap so they appear on top
    # Use dark lines for better contrast on red heatmap
    current_y = 0
    sig_y_positions = {}  # Store y positions for each signature group
    
    for sig in sorted(sig_to_diseases.keys()):
        n_diseases_in_sig = len(sig_to_diseases[sig])
        if current_y > 0:
            # Draw dark separator line between signature groups
            # Use zorder to ensure it appears on top of the heatmap
            ax.axhline(current_y - 0.5, color='black', linewidth=2.0, alpha=0.8, zorder=10)
        
        # Store the middle y position for this signature group
        sig_y_positions[sig] = current_y + n_diseases_in_sig / 2 - 0.5
        current_y += n_diseases_in_sig
    
    # Also add a line at the very bottom (after all diseases)
    if current_y > 0:
        ax.axhline(current_y - 0.5, color='black', linewidth=2.0, alpha=0.8, zorder=10)
    
    # Add signature labels on the right side with brackets
    ax2 = ax.twinx()  # Create a second y-axis for labels
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # Get x position for labels (just to the right of the heatmap)
    x_label_pos = T + 0.5
    
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
        print(f"✓ Saved Panel C heatmap to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function to generate Panel C heatmap."""
    
    print("Loading data...")
    
    # Load pi from enrollment predictions folder
    pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/pi_fullmode_400k.pt'
    #pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt'
    
    print(f"Loading pi from: {pi_path}")
    pi_full = torch.load(pi_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(pi_full):
        pi_full = pi_full.detach().cpu().numpy()
    
    print(f"Pi tensor shape: {pi_full.shape}")  # Should be (N, D, T)
    
    # Average over people (patients) to get (D, T) - average probability for each disease at each age
    if len(pi_full.shape) == 3:
        if pi_full.shape[1] > 50:  # Likely (N, D, T) - direct disease probabilities
            pi_avg = np.mean(pi_full, axis=0)  # Shape: (D, T)
        else:  # Likely (N, K, T) - signature probabilities
            print("Warning: pi_full appears to be signature probabilities (N, K, T)")
            print("  Need to convert to disease probabilities using clusters")
            pi_avg = np.mean(pi_full, axis=0)  # Shape: (D, T)
    else:
        raise ValueError(f"Unexpected pi_full shape: {pi_full.shape}")
    
    print(f"Average pi shape: {pi_avg.shape}")  # Should be (D, T)
    
    # Load UKB checkpoint for psi, clusters, and disease names (same source as PSI heatmap)
    ukb_checkpoint_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt'
    ukb_checkpoint = torch.load(ukb_checkpoint_path, map_location='cpu')
    
    # Extract PSI (same as PSI heatmap)
    if 'model_state_dict' in ukb_checkpoint:
        psi = ukb_checkpoint['model_state_dict']['psi']
    else:
        psi = ukb_checkpoint['psi']
    
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    # Extract clusters
    clusters = ukb_checkpoint['clusters']
    if torch.is_tensor(clusters):
        clusters = clusters.numpy()
    
    # Extract disease names (from old checkpoint for consistency)
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
    output_path = output_dir / 'panel_c_pi_heatmap.pdf'
    fig = plot_pi_heatmap(
        pi_avg, psi, clusters, disease_names,
        output_path=str(output_path),
        figsize=(14, 16)
    )
    
    print(f"\n✓ Panel C heatmap generated successfully!")
    print(f"  Saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Panel C: Disease prevalence - {pi_avg.shape[0]} diseases × {pi_avg.shape[1]} ages")
    print(f"  Diseases sorted by strongest psi within each signature (strongest at top)")
    print(f"  Same ordering as Panel B (PSI heatmap)")


if __name__ == "__main__":
    main()