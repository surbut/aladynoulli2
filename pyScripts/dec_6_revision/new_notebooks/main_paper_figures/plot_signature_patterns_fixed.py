"""
Plotting functions for signature phi patterns.

This module provides functions to visualize disease signature patterns
across different cohorts (UKB, AOU, MGB).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import torch


def plot_signature_phi_patterns(phi_np, clusters_np, disease_names, selected_signatures=None, 
                                  age_offset=30, figsize=(8, 5), top_n_diseases=None, 
                                  plot_probability=True, prevalence_t_np=None, logit_prev_t_np=None):
    """
    Plot phi values (as probability or log hazard ratio) for signature-specific diseases (colored) 
    vs background diseases (in grey), arranged in a grid. Optionally overlay prevalence and logit prevalence.
    
    Parameters:
    -----------
    phi_np : np.ndarray, shape (K, D, T)
        Phi values (log hazard ratios) for each signature, disease, and timepoint
    clusters_np : np.ndarray, shape (D,)
        Disease-to-signature assignments
    disease_names : list
        List of disease names
    selected_signatures : list or None
        List of signature indices to plot. If None, plot all signatures.
    age_offset : int
        Age offset (timepoint 0 = age_offset)
    figsize : tuple
        Figure size per signature subplot (width, height)
    top_n_diseases : int or None
        If specified, plot only top N signature-specific diseases based on max phi value
    plot_probability : bool
        If True, convert phi to probability using sigmoid. If False, plot log hazard ratio.
    prevalence_t_np : np.ndarray or None, shape (D, T)
        Smoothed prevalence for each disease and timepoint. If provided, will be overlaid.
    logit_prev_t_np : np.ndarray or None, shape (D, T)
        Logit prevalence for each disease and timepoint. If provided, will be overlaid as probability.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing all subplots
    """
    K, D, T = phi_np.shape
    
    if selected_signatures is None:
        selected_signatures = list(range(K))
    
    n_sigs = len(selected_signatures)
    age_points = np.arange(T) + age_offset
    
    # Convert to probability if requested
    if plot_probability:
        plot_values = sigmoid(phi_np)
        y_label = 'Prob (disease | sig k, age)'
    else:
        plot_values = phi_np
        y_label = 'Log hazard ratio'
    
    # Create subplots in a grid (5 cols, variable rows)
    n_cols = 5
    n_rows = (n_sigs + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), 
                             sharex=True, sharey=False)
    
    # Access axes directly by row/col - most reliable method
    for idx, k in enumerate(selected_signatures):
        # Calculate row and column
        row = idx // n_cols
        col = idx % n_cols
        
        # Get the correct axis
        if n_sigs == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[col]  # Single row, axes is 1D
        else:
            ax = axes[row, col]  # Multiple rows, axes is 2D
        
        # Find diseases assigned to this signature
        sig_disease_indices = np.where(clusters_np == k)[0]
        
        # Find all other diseases (background)
        background_disease_indices = np.where(clusters_np != k)[0]
        
        # Plot background diseases in grey
        for d in background_disease_indices:
            ax.plot(age_points, plot_values[k, d, :], 
                   color='grey', alpha=0.2, linewidth=0.3, zorder=1)
        
        # Plot signature-specific diseases in colored lines
        if len(sig_disease_indices) > 0:
            # Optionally select top N diseases
            if top_n_diseases is not None and len(sig_disease_indices) > top_n_diseases:
                # Rank by max value
                max_values = np.max(plot_values[k, sig_disease_indices, :], axis=1)
                top_indices = np.argsort(max_values)[-top_n_diseases:]
                sig_disease_indices = sig_disease_indices[top_indices]
            
            # Use different colors for signature-specific diseases
            # Use tab20 for more color options, or hsv for even more distinct colors
            n_diseases = len(sig_disease_indices)
            if n_diseases <= 10:
                colors = plt.cm.tab10(np.arange(n_diseases))
            elif n_diseases <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, n_diseases))
            else:
                # For many diseases, use hsv colormap which provides more distinct colors
                colors = plt.cm.hsv(np.linspace(0, 0.9, n_diseases))  # 0 to 0.9 to avoid red wrapping to red
            
            for i, d in enumerate(sig_disease_indices):
                color = colors[i]
                disease_name = disease_names[d] if d < len(disease_names) else f"Disease_{d}"
                
                # Plot model's learned phi (as probability)
                ax.plot(age_points, plot_values[k, d, :], 
                       color=color, linewidth=2.5, label=disease_name, zorder=2)
                
                # Overlay smoothed prevalence if available
                if prevalence_t_np is not None and d < prevalence_t_np.shape[0]:
                    # Convert to numpy if it's a torch tensor
                    if isinstance(prevalence_t_np, torch.Tensor):
                        prev_values = prevalence_t_np[d, :].detach().cpu().numpy()
                    else:
                        prev_values = prevalence_t_np[d, :]
                    ax.plot(age_points, prev_values, 
                           color=color, linewidth=1.5, linestyle='--', 
                           alpha=0.6, label=f'{disease_name} (prev)', zorder=3)
                
                # Overlay logit prevalence (converted to probability) if available
                if logit_prev_t_np is not None and d < logit_prev_t_np.shape[0]:
                    # Convert to numpy if it's a torch tensor
                    if isinstance(logit_prev_t_np, torch.Tensor):
                        logit_prev_values = logit_prev_t_np[d, :].detach().cpu().numpy()
                    else:
                        logit_prev_values = logit_prev_t_np[d, :]
                    logit_prev_prob = sigmoid(logit_prev_values)
                    ax.plot(age_points, logit_prev_prob, 
                           color=color, linewidth=1.5, linestyle=':', 
                           alpha=0.6, label=f'{disease_name} (logit prev)', zorder=3)
        
        ax.set_title(f'Signature {k}', fontsize=12, fontweight='bold', pad=5)
        if idx >= (n_rows - 1) * n_cols:  # Only label x-axis on bottom row
            ax.set_xlabel('Age (yr)', fontsize=10)
        if idx % n_cols == 0:  # Only label y-axis on left column
            ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(age_points[0], age_points[-1])
        
        # Set y-axis limits based on data range for this signature
        if len(sig_disease_indices) > 0:
            sig_data = plot_values[k, sig_disease_indices, :]
            y_min = max(0, np.min(sig_data) * 0.9) if plot_probability else np.min(sig_data) * 1.1
            y_max = np.max(sig_data) * 1.1
            
            # Also consider prevalence and logit prevalence if available
            if prevalence_t_np is not None:
                # Convert to numpy if it's a torch tensor
                if isinstance(prevalence_t_np, torch.Tensor):
                    prev_data = prevalence_t_np[sig_disease_indices, :].detach().cpu().numpy()
                else:
                    prev_data = prevalence_t_np[sig_disease_indices, :]
                y_min = min(y_min, max(0, np.min(prev_data) * 0.9))
                y_max = max(y_max, np.max(prev_data) * 1.1)
            
            if logit_prev_t_np is not None:
                # Convert to numpy if it's a torch tensor
                if isinstance(logit_prev_t_np, torch.Tensor):
                    logit_prev_t_np_numpy = logit_prev_t_np.detach().cpu().numpy()
                else:
                    logit_prev_t_np_numpy = logit_prev_t_np
                logit_prev_prob_data = sigmoid(logit_prev_t_np_numpy[sig_disease_indices, :])
                y_min = min(y_min, max(0, np.min(logit_prev_prob_data) * 0.9))
                y_max = max(y_max, np.max(logit_prev_prob_data) * 1.1)
            
            ax.set_ylim(y_min, y_max)
        
        # Add legend if there are signature-specific diseases (only for first few to avoid clutter)
        if len(sig_disease_indices) > 0 and len(sig_disease_indices) <= 10 and idx < 3:  # Only show legend for first 3 signatures
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, 
                     framealpha=0.9, edgecolor='gray', fancybox=True)
    
    # Hide unused subplots
    for idx in range(n_sigs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for legend
    return fig

