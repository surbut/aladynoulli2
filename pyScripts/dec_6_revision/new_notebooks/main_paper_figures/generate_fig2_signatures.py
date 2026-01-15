#!/usr/bin/env python3
"""
Generate Figure 2 signature plots - one signature per panel, saved as high-resolution PDFs.

Usage:
    python generate_fig2_signatures.py
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import torch
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']


def plot_single_signature_multi_cohort(phi_ukb, clusters_ukb, disease_names_ukb,
                                       phi_aou, clusters_aou, disease_names_aou,
                                       phi_mgb, clusters_mgb, disease_names_mgb,
                                       signature_idx_ukb, signature_idx_aou=None, signature_idx_mgb=None,
                                       age_offset=30, figsize=(24, 8), 
                                       top_n_diseases=None, overlapping_diseases=None,
                                       plot_probability=False, 
                                       prevalence_ukb=None, prevalence_aou=None, prevalence_mgb=None):
    """
    Plot a single signature across UKB, AOU, and MGB cohorts side by side.
    
    Parameters:
    -----------
    phi_ukb, phi_aou, phi_mgb : np.ndarray, shape (K, D, T)
        Phi values for each cohort
    clusters_ukb, clusters_aou, clusters_mgb : np.ndarray, shape (D,)
        Disease-to-signature assignments for each cohort
    disease_names_ukb, disease_names_aou, disease_names_mgb : list
        Disease names for each cohort
    signature_idx_ukb : int
        Signature index in UKB to plot
    signature_idx_aou : int or None
        Signature index in AOU (if None, uses signature_idx_ukb)
    signature_idx_mgb : int or None
        Signature index in MGB (if None, uses signature_idx_ukb)
    age_offset : int
        Age offset (timepoint 0 = age_offset)
    figsize : tuple
        Figure size (width, height)
    top_n_diseases : int or None
        If specified, plot only top N signature-specific diseases based on max phi value
    plot_probability : bool
        If True, convert phi to probability using sigmoid. If False, plot log hazard ratio.
    prevalence_ukb, prevalence_aou, prevalence_mgb : np.ndarray or None, shape (D, T)
        Smoothed prevalence for each cohort
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    K, D_ukb, T_ukb = phi_ukb.shape
    D_aou, T_aou = phi_aou.shape[1], phi_aou.shape[2]
    D_mgb, T_mgb = phi_mgb.shape[1], phi_mgb.shape[2]
    
    # Use minimum T across all cohorts to ensure alignment
    T = min(T_ukb, T_aou, T_mgb)
    print(f"  Using T={T} (UKB: {T_ukb}, AOU: {T_aou}, MGB: {T_mgb})")
    
    # Truncate phi arrays to same T if needed
    if T_ukb > T:
        phi_ukb = phi_ukb[:, :, :T]
    if T_aou > T:
        phi_aou = phi_aou[:, :, :T]
    if T_mgb > T:
        phi_mgb = phi_mgb[:, :, :T]
    
    # Truncate prevalence if needed
    if prevalence_ukb is not None and prevalence_ukb.shape[1] > T:
        prevalence_ukb = prevalence_ukb[:, :T]
    if prevalence_aou is not None and prevalence_aou.shape[1] > T:
        prevalence_aou = prevalence_aou[:, :T]
    if prevalence_mgb is not None and prevalence_mgb.shape[1] > T:
        prevalence_mgb = prevalence_mgb[:, :T]
    
    age_points = np.arange(T) + age_offset
    
    # Convert to probability if requested (after truncation)
    if plot_probability:
        plot_values_ukb = sigmoid(phi_ukb)
        plot_values_aou = sigmoid(phi_aou)
        plot_values_mgb = sigmoid(phi_mgb)
        y_label = 'Probability'
    else:
        plot_values_ukb = phi_ukb
        plot_values_aou = phi_aou
        plot_values_mgb = phi_mgb
        y_label = 'Log Hazard Ratio'
    
    # Map signature indices for each cohort
    k_ukb = signature_idx_ukb
    k_aou = signature_idx_aou if signature_idx_aou is not None else signature_idx_ukb
    k_mgb = signature_idx_mgb if signature_idx_mgb is not None else signature_idx_ukb
    
    # Map signature indices for each cohort (already done above, but ensure they're set)
    cohorts = [
        ('UKB', plot_values_ukb, clusters_ukb, disease_names_ukb, prevalence_ukb, D_ukb, k_ukb),
        ('AOU', plot_values_aou, clusters_aou, disease_names_aou, prevalence_aou, D_aou, k_aou),
        ('MGB', plot_values_mgb, clusters_mgb, disease_names_mgb, prevalence_mgb, D_mgb, k_mgb)
    ]
    
    # Find diseases assigned to signature in UKB
    sig_disease_indices_ukb_all = np.where(clusters_ukb == k_ukb)[0]
    
    # Filter to only overlapping diseases if specified
    if overlapping_diseases is not None:
        # Find indices of overlapping diseases in UKB
        sig_disease_indices_ukb = []
        for disease_name in overlapping_diseases:
            # Try to find disease by name (case-insensitive, partial match)
            for idx in sig_disease_indices_ukb_all:
                if idx < len(disease_names_ukb):
                    ukb_name = disease_names_ukb[idx]
                    # Check for exact match or if disease name contains the target name
                    if (disease_name.lower() in ukb_name.lower() or 
                        ukb_name.lower() in disease_name.lower()):
                        sig_disease_indices_ukb.append(idx)
                        break
        sig_disease_indices_ukb = np.array(sig_disease_indices_ukb)
        print(f"  Filtered to {len(sig_disease_indices_ukb)} overlapping diseases: {[disease_names_ukb[i] for i in sig_disease_indices_ukb]}")
    else:
        sig_disease_indices_ukb = sig_disease_indices_ukb_all
        if top_n_diseases is not None and len(sig_disease_indices_ukb) > top_n_diseases:
            max_values = np.max(plot_values_ukb[k_ukb, sig_disease_indices_ukb, :], axis=1)
            top_indices = np.argsort(max_values)[-top_n_diseases:]
            sig_disease_indices_ukb = sig_disease_indices_ukb[top_indices]
    
    # Calculate shared y-axis range across all cohorts for overlapping diseases
    # First, find overlapping disease indices in each cohort
    all_sig_data = []
    all_prev_data = []
    
    for cohort_name, plot_values, clusters, disease_names, prevalence, D, k in cohorts:
        sig_disease_indices_all = np.where(clusters == k)[0]
        
        # Filter to overlapping diseases if specified
        if overlapping_diseases is not None:
            sig_disease_indices = []
            for disease_name in overlapping_diseases:
                for idx in sig_disease_indices_all:
                    if idx < len(disease_names):
                        cohort_name_str = disease_names[idx]
                        if (disease_name.lower() in cohort_name_str.lower() or 
                            cohort_name_str.lower() in disease_name.lower()):
                            sig_disease_indices.append(idx)
                            break
            sig_disease_indices = np.array(sig_disease_indices)
        else:
            sig_disease_indices = sig_disease_indices_all
        
        if len(sig_disease_indices) > 0:
            # Collect phi data
            cohort_sig_data = plot_values[k, sig_disease_indices, :]
            all_sig_data.append(cohort_sig_data)
            
            # Collect prevalence data if available (only when plotting in probability space)
            if plot_probability and prevalence is not None:
                if isinstance(prevalence, torch.Tensor):
                    prev_data = prevalence[sig_disease_indices, :].detach().cpu().numpy()
                else:
                    prev_data = prevalence[sig_disease_indices, :]
                all_prev_data.append(prev_data)
    
    # Calculate global y-axis limits
    if len(all_sig_data) > 0:
        global_sig_min = min(np.min(data) for data in all_sig_data)
        global_sig_max = max(np.max(data) for data in all_sig_data)
        
        if len(all_prev_data) > 0:
            global_prev_min = min(np.min(data) for data in all_prev_data)
            global_prev_max = max(np.max(data) for data in all_prev_data)
            y_min = min(global_sig_min, max(0, global_prev_min)) * (0.95 if plot_probability else 1.05)
            y_max = max(global_sig_max, global_prev_max) * 1.05
        else:
            y_min = max(0, global_sig_min * 0.95) if plot_probability else global_sig_min * 1.05
            y_max = global_sig_max * 1.05
        
        shared_ylim = (y_min, y_max)
    else:
        shared_ylim = None
    
    # Create figure with 3 subplots (one per cohort)
    # Use sharey=True if we have overlapping diseases to ensure same scale
    share_y_axis = (overlapping_diseases is not None and len(sig_disease_indices_ukb) > 0)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=share_y_axis)
    
    # Create color mapping based on UKB diseases
    if len(sig_disease_indices_ukb) > 0:
        n_colors = len(sig_disease_indices_ukb)
        if n_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
        disease_color_map = {disease_names_ukb[d]: colors[i] for i, d in enumerate(sig_disease_indices_ukb)}
    else:
        disease_color_map = {}
    
    # Plot each cohort
    for ax_idx, (cohort_name, plot_values, clusters, disease_names, prevalence, D, k) in enumerate(cohorts):
        ax = axes[ax_idx]
        
        # Find diseases assigned to this signature in this cohort
        sig_disease_indices_all = np.where(clusters == k)[0]
        background_disease_indices = np.where(clusters != k)[0]
        
        # Plot background diseases in light grey
        for d in background_disease_indices:
            if d < D:
                ax.plot(age_points, plot_values[k, d, :], 
                       color='lightgray', alpha=0.15, linewidth=0.5, zorder=1)
        
        # Filter to overlapping diseases if specified
        if overlapping_diseases is not None:
            sig_disease_indices = []
            for disease_name in overlapping_diseases:
                # Try to find disease by name in this cohort
                for idx in sig_disease_indices_all:
                    if idx < len(disease_names):
                        cohort_name_str = disease_names[idx]
                        # Check for exact match or if disease name contains the target name
                        if (disease_name.lower() in cohort_name_str.lower() or 
                            cohort_name_str.lower() in disease_name.lower()):
                            sig_disease_indices.append(idx)
                            break
            sig_disease_indices = np.array(sig_disease_indices)
        else:
            sig_disease_indices = sig_disease_indices_all
            # Optionally select top N diseases
            if top_n_diseases is not None and len(sig_disease_indices) > top_n_diseases:
                max_values = np.max(plot_values[k, sig_disease_indices, :], axis=1)
                top_indices = np.argsort(max_values)[-top_n_diseases:]
                sig_disease_indices = sig_disease_indices[top_indices]
        
        # Plot signature-specific diseases
        if len(sig_disease_indices) > 0:
            
            for i, d in enumerate(sig_disease_indices):
                if d >= D:
                    continue
                    
                disease_name = disease_names[d] if d < len(disease_names) else f"Disease_{d}"
                
                # Get color from UKB mapping if available, otherwise use default
                color = disease_color_map.get(disease_name, plt.cm.tab10(i % 10))
                
                # Plot model's learned phi
                ax.plot(age_points, plot_values[k, d, :], 
                       color=color, linewidth=3.0, label=disease_name, zorder=3, alpha=0.9)
                
                # Overlay smoothed prevalence if available (only when plotting in probability space)
                if plot_probability and prevalence is not None and d < prevalence.shape[0]:
                    if isinstance(prevalence, torch.Tensor):
                        prev_values = prevalence[d, :].detach().cpu().numpy()
                    else:
                        prev_values = prevalence[d, :]
                    ax.plot(age_points, prev_values, 
                           color=color, linewidth=2.2, linestyle='--', 
                           alpha=0.5, label=f'{disease_name} (prev)', zorder=2)
        
        # Styling
        ax.set_title(cohort_name, fontsize=16, fontweight='bold', pad=15)
        if ax_idx == 0:  # Only label y-axis on left plot
            ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
        ax.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax.set_xlim(age_points[0], age_points[-1])
        
        # Set y-axis limits
        if shared_ylim is not None and overlapping_diseases is not None:
            # Use shared y-axis limits for overlapping diseases
            ax.set_ylim(shared_ylim)
        elif len(sig_disease_indices) > 0:
            # Use cohort-specific limits if not using overlapping diseases
            sig_data = plot_values[k, sig_disease_indices, :]
            y_min = max(0, np.min(sig_data) * 0.95) if plot_probability else np.min(sig_data) * 1.05
            y_max = np.max(sig_data) * 1.05
            
            # Also consider prevalence if available (only when plotting in probability space)
            if plot_probability and prevalence is not None:
                if isinstance(prevalence, torch.Tensor):
                    prev_data = prevalence[sig_disease_indices, :].detach().cpu().numpy()
                else:
                    prev_data = prevalence[sig_disease_indices, :]
                y_min = min(y_min, max(0, np.min(prev_data) * 0.95))
                y_max = max(y_max, np.max(prev_data) * 1.05)
            
            ax.set_ylim(y_min, y_max)
        
        # Add legend only on rightmost plot
        if ax_idx == 2 and len(sig_disease_indices) > 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
                     framealpha=0.95, edgecolor='gray', fancybox=True, frameon=True)
    
    # Add overall title
    fig.suptitle(f'Signature {k_ukb} (UKB) / {k_aou} (AOU) / {k_mgb} (MGB)', fontsize=22, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.98, 0.98])  # Leave space for suptitle
    return fig


def plot_single_signature(phi_np, clusters_np, disease_names, signature_idx,
                          age_offset=30, figsize=(10, 7), top_n_diseases=None, 
                          plot_probability=False, prevalence_t_np=None, logit_prev_t_np=None):
    """
    Plot a single signature with prettier styling for publication.
    
    Parameters:
    -----------
    phi_np : np.ndarray, shape (K, D, T)
        Phi values (log hazard ratios) for each signature, disease, and timepoint
    clusters_np : np.ndarray, shape (D,)
        Disease-to-signature assignments
    disease_names : list
        List of disease names
    signature_idx : int
        Signature index to plot
    age_offset : int
        Age offset (timepoint 0 = age_offset)
    figsize : tuple
        Figure size (width, height)
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
        The figure object
    """
    K, D, T = phi_np.shape
    age_points = np.arange(T) + age_offset
    
    # Convert to probability if requested
    if plot_probability:
        plot_values = sigmoid(phi_np)
        y_label = 'Probability'
    else:
        plot_values = phi_np
        y_label = 'Log Hazard Ratio'
    
    # Create single figure
    fig, ax = plt.subplots(figsize=figsize)
    
    k = signature_idx
    
    # Find diseases assigned to this signature
    sig_disease_indices = np.where(clusters_np == k)[0]
    
    # Find all other diseases (background)
    background_disease_indices = np.where(clusters_np != k)[0]
    
    # Plot background diseases in light grey
    for d in background_disease_indices:
        ax.plot(age_points, plot_values[k, d, :], 
               color='lightgray', alpha=0.15, linewidth=0.5, zorder=1)
    
    # Plot signature-specific diseases in colored lines
    if len(sig_disease_indices) > 0:
        # Optionally select top N diseases
        if top_n_diseases is not None and len(sig_disease_indices) > top_n_diseases:
            # Rank by max value
            max_values = np.max(plot_values[k, sig_disease_indices, :], axis=1)
            top_indices = np.argsort(max_values)[-top_n_diseases:]
            sig_disease_indices = sig_disease_indices[top_indices]
        
        # Use a nice color palette
        n_colors = len(sig_disease_indices)
        if n_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
        
        for i, d in enumerate(sig_disease_indices):
            color = colors[i]
            disease_name = disease_names[d] if d < len(disease_names) else f"Disease_{d}"
            
            # Plot model's learned phi
            ax.plot(age_points, plot_values[k, d, :], 
                   color=color, linewidth=2.5, label=disease_name, zorder=3, alpha=0.9)
            
            # Overlay smoothed prevalence if available (only when plotting in probability space)
            if plot_probability and prevalence_t_np is not None and d < prevalence_t_np.shape[0]:
                if isinstance(prevalence_t_np, torch.Tensor):
                    prev_values = prevalence_t_np[d, :].detach().cpu().numpy()
                else:
                    prev_values = prevalence_t_np[d, :]
                ax.plot(age_points, prev_values, 
                       color=color, linewidth=1.8, linestyle='--', 
                       alpha=0.5, label=f'{disease_name} (prev)', zorder=2)
    
    # Styling
    ax.set_title(f'Signature {k}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlim(age_points[0], age_points[-1])
    
    # Set y-axis limits based on data range
    if len(sig_disease_indices) > 0:
        sig_data = plot_values[k, sig_disease_indices, :]
        y_min = max(0, np.min(sig_data) * 0.95) if plot_probability else np.min(sig_data) * 1.05
        y_max = np.max(sig_data) * 1.05
        
        # Also consider prevalence if available (only when plotting in probability space)
        if plot_probability and prevalence_t_np is not None:
            if isinstance(prevalence_t_np, torch.Tensor):
                prev_data = prevalence_t_np[sig_disease_indices, :].detach().cpu().numpy()
            else:
                prev_data = prevalence_t_np[sig_disease_indices, :]
            y_min = min(y_min, max(0, np.min(prev_data) * 0.95))
            y_max = max(y_max, np.max(prev_data) * 1.05)
        
        ax.set_ylim(y_min, y_max)
    
    # Add legend
    if len(sig_disease_indices) > 0:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                 framealpha=0.95, edgecolor='gray', fancybox=True, frameon=True)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to generate all signature plots."""
    
    print("Loading data from all cohorts...")
    
    # Load UKB data
    # UKB phi: Pooled from batches in censor_e_batchrun_vectorized (corrected E batches)
    # Source: master_for_fitting_pooled_correctedE.pt contains pooled phi from 40 batches
    print("\n1. Loading UKB data...")
    master_checkpoint = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt', map_location='cpu')
    phi_ukb = master_checkpoint['model_state_dict']['phi'].numpy()
    
    ukb_checkpoint_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam.pt'
    ukb_checkpoint = torch.load(ukb_checkpoint_path, map_location='cpu')
    clusters_ukb = ukb_checkpoint['clusters']
    if torch.is_tensor(clusters_ukb):
        clusters_ukb = clusters_ukb.numpy()
    
    prevalence_ukb = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/prevalence_t_corrected.pt', map_location='cpu')
    if torch.is_tensor(prevalence_ukb):
        prevalence_ukb = prevalence_ukb.numpy()
    
    disease_names_ukb = ukb_checkpoint['disease_names']
    if isinstance(disease_names_ukb, (list, tuple)):
        disease_names_ukb = list(disease_names_ukb)
    elif hasattr(disease_names_ukb, 'values'):
        disease_names_ukb = disease_names_ukb.values.tolist()
    
    print(f"  UKB: phi shape {phi_ukb.shape}, clusters shape {clusters_ukb.shape}, {len(disease_names_ukb)} diseases")
    
    # Load AOU data
    # AoU phi: Pooled from batches in Dropbox/aou_batches (created by train_aou_batches.py with corrected E)
    # Source: aou_model_master_correctedE.pt contains pooled phi from 25 batches
    print("\n2. Loading AOU data...")
    aou_checkpoint = torch.load('/Users/sarahurbut/aladynoulli2/aou_model_master_correctedE.pt', map_location='cpu')
    phi_aou = aou_checkpoint['model_state_dict']['phi']
    if torch.is_tensor(phi_aou):
        phi_aou = phi_aou.detach().cpu().numpy()
    
    clusters_aou = aou_checkpoint['clusters']
    if torch.is_tensor(clusters_aou):
        clusters_aou = clusters_aou.detach().cpu().numpy()
    
    disease_names_aou = aou_checkpoint.get('disease_names', None)
    if disease_names_aou is None:
        # Try to load from old checkpoint
        try:
            aou_old = torch.load('/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_AOU.pt', map_location='cpu')
            disease_names_aou = aou_old.get('disease_names', [f"Disease_{i}" for i in range(phi_aou.shape[1])])
        except:
            disease_names_aou = [f"Disease_{i}" for i in range(phi_aou.shape[1])]
    if isinstance(disease_names_aou, (list, tuple)):
        disease_names_aou = list(disease_names_aou)
    elif hasattr(disease_names_aou, 'values'):
        disease_names_aou = disease_names_aou.values.tolist()
    
    # Load AOU prevalence if available
    try:
        prevalence_aou = torch.load('/Users/sarahurbut/aladynoulli2/aou_prevalence_corrected_E.pt', map_location='cpu')
        if torch.is_tensor(prevalence_aou):
            prevalence_aou = prevalence_aou.numpy()
    except:
        prevalence_aou = None
        print("  Warning: AOU prevalence not found, skipping overlay")
    
    print(f"  AOU: phi shape {phi_aou.shape}, clusters shape {clusters_aou.shape}, {len(disease_names_aou)} diseases")
    
    # Load MGB data
    # MGB phi: From mgb_model_initialized.pt (trained with corrected E, not from batches)
    # Source: Single trained model with corrected E matrix
    print("\n3. Loading MGB data...")
    mgb_checkpoint = torch.load('/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt', map_location='cpu')
    phi_mgb = mgb_checkpoint['model_state_dict']['phi']
    if torch.is_tensor(phi_mgb):
        phi_mgb = phi_mgb.detach().cpu().numpy()
    
    clusters_mgb = mgb_checkpoint['clusters']
    if torch.is_tensor(clusters_mgb):
        clusters_mgb = clusters_mgb.detach().cpu().numpy()
    
    disease_names_mgb = mgb_checkpoint.get('disease_names', None)
    if disease_names_mgb is None:
        # Try to load from old checkpoint
        try:
            mgb_old = torch.load('/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt', map_location='cpu')
            disease_names_mgb = mgb_old.get('disease_names', [f"Disease_{i}" for i in range(phi_mgb.shape[1])])
        except:
            disease_names_mgb = [f"Disease_{i}" for i in range(phi_mgb.shape[1])]
    if isinstance(disease_names_mgb, (list, tuple)):
        disease_names_mgb = list(disease_names_mgb)
    elif hasattr(disease_names_mgb, 'values'):
        disease_names_mgb = disease_names_mgb.values.tolist()
    
    # Load MGB prevalence if available
    try:
        prevalence_mgb = torch.load('/Users/sarahurbut/aladynoulli2/mgb_prevalence_corrected_E.pt', map_location='cpu')
        if torch.is_tensor(prevalence_mgb):
            prevalence_mgb = prevalence_mgb.numpy()
    except:
        prevalence_mgb = None
        print("  Warning: MGB prevalence not found, skipping overlay")
    
    print(f"  MGB: phi shape {phi_mgb.shape}, clusters shape {clusters_mgb.shape}, {len(disease_names_mgb)} diseases")
    
    # Create output directory
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig2')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Selected signatures to plot (one per panel)
    # Mapping: UKB sig -> (AOU sig, MGB sig)
    signature_mappings = {
        5: (16, 5),   # UKB sig 5 -> AOU sig 16, MGB sig 5 (Cardiovascular)
        6: (11, 11),  # UKB sig 6 -> AOU sig 11, MGB sig 11 (Malignancy)
    }
    
    # Define overlapping diseases for each signature (by UKB disease names)
    # Cardiovascular diseases (Signature 5)
    cardiovascular_diseases = [
        "Coronary atherosclerosis",
        "Myocardial infarction", 
        "Other chronic ischemic heart disease, unspecified",
        "Unstable angina (intermediate coronary syndrome)"
    ]
    
    # Malignancy diseases (Signature 6)
    malignancy_diseases = [
        "Malignant neoplasm, other",
        "Secondary malignant neoplasm",
        "Secondary malignancy of lymph nodes",
        "Secondary malignancy of respiratory organs",
        "Secondary malignant neoplasm of digestive systems",
        "Secondary malignant neoplasm of liver",
        "Secondary malignancy of bone"
    ]
    
    # Map signature to disease list
    signature_disease_lists = {
        5: cardiovascular_diseases,
        6: malignancy_diseases
    }
    
    # Plot one signature at a time across all cohorts and save as high-resolution PDF
    for sig_idx_ukb in signature_mappings.keys():
        sig_idx_aou, sig_idx_mgb = signature_mappings[sig_idx_ukb]
        print(f"\nPlotting Signature {sig_idx_ukb} (UKB) / {sig_idx_aou} (AOU) / {sig_idx_mgb} (MGB)...")
        
        # Get list of overlapping diseases for this signature
        overlapping_diseases = signature_disease_lists.get(sig_idx_ukb, None)
        
        # Create figure with 3 panels (one per cohort)
        fig = plot_single_signature_multi_cohort(
            phi_ukb, clusters_ukb, disease_names_ukb,
            phi_aou, clusters_aou, disease_names_aou,
            phi_mgb, clusters_mgb, disease_names_mgb,
            signature_idx_ukb=sig_idx_ukb,
            signature_idx_aou=sig_idx_aou,
            signature_idx_mgb=sig_idx_mgb,
            overlapping_diseases=overlapping_diseases,  # Filter to only overlapping diseases
            plot_probability=False,  # Plot log hazard ratio (not probability)
            prevalence_ukb=prevalence_ukb,
            prevalence_aou=prevalence_aou,
            prevalence_mgb=prevalence_mgb,
            figsize=(24, 8)  # Larger figure for 3 panels - increased from (18, 6)
        )
        
        # Save as high-resolution PDF
        output_path = output_dir / f'fig2_signature_{sig_idx_ukb}_multi_cohort.pdf'
        fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {output_path}")
        
        plt.close(fig)  # Close figure to free memory
    
    print(f"\n✓ All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

