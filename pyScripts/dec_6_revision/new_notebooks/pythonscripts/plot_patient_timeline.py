#!/usr/bin/env python3
"""
Plot Patient Timeline - Individual Trajectory Visualization

Creates a multi-panel plot showing:
- Panel 1: Signature loadings (θ) vs Age with stacked average theta bar
- Panel 2: Disease timeline (scatter plot) with diseases labeled by chronological order
- Panel 3: Disease probabilities (π) over time, stopping after diagnosis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_patient_timeline(patient_idx, 
                          theta_path='/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective_correctE.pt',
                          checkpoint_path='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt',
                          pi_path='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/pi_fullmode_400k.pt',
                          Y_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt',
                          initial_clusters_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt',
                          disease_names_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv',
                          output_path=None,
                          figsize=(14, 10)):
    """
    Plot patient timeline with signature trajectories, disease timeline, and disease probabilities.
    
    Parameters:
    -----------
    patient_idx : int
        Patient index to plot
    theta_path : str
        Path to theta file (N, K, T)
    checkpoint_path : str
        Path to checkpoint file (for reference, not all data loaded from here)
    pi_path : str
        Path to pi file (N, D, T)
    Y_path : str
        Path to Y file (N, D, T)
    initial_clusters_path : str
        Path to initial clusters file
    disease_names_path : str
        Path to disease names CSV
    output_path : str or None
        Path to save the figure. If None, displays instead.
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    
    # Load data
    initial_clusters = torch.load(initial_clusters_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(initial_clusters):
        initial_clusters = initial_clusters.numpy()
    K = int(initial_clusters.max() + 1)

    # Load theta from the specified file
    theta_full = torch.load(theta_path, map_location='cpu', weights_only=False)

    # Check structure of theta file
    if isinstance(theta_full, dict):
        # If it's a dict, try common keys
        if 'theta' in theta_full:
            theta = theta_full['theta']
        elif 'thetas' in theta_full:
            theta = theta_full['thetas']
        elif 'lambda_' in theta_full:
            theta = torch.softmax(theta_full['lambda_'], dim=1)
        else:
            # Try first tensor value
            theta = list(theta_full.values())[0]
            if torch.is_tensor(theta) and theta.dim() == 3:
                theta = torch.softmax(theta, dim=1)
    else:
        theta = theta_full

    # Convert to numpy if needed
    if torch.is_tensor(theta):
        theta = theta.numpy()
    elif isinstance(theta, list):
        theta = np.array(theta)

    # Load Y and pi
    Y = torch.load(Y_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(Y):
        Y_np = Y.numpy()
    else:
        Y_np = Y

    pi_full = torch.load(pi_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(pi_full):
        pi_np = pi_full.numpy()
    else:
        pi_np = pi_full

    # Load disease names
    disease_names = pd.read_csv(disease_names_path)['x'].tolist()

    # Check dimensions
    N_theta, K_total, T_theta = theta.shape
    N_y, D, T_y = Y_np.shape
    N_pi, D_pi, T_pi = pi_np.shape

    print(f"Theta shape: ({N_theta}, {K_total}, {T_theta})")
    print(f"Y shape: ({N_y}, {D}, {T_y})")
    print(f"Pi shape: ({N_pi}, {D_pi}, {T_pi})")
    print(f"\nRequested patient index: {patient_idx}")

    # Verify patient index is valid
    if patient_idx >= N_theta:
        print(f"WARNING: Patient {patient_idx} not in theta array (max index: {N_theta-1})")
        patient_idx = min(patient_idx, N_theta - 1)
        print(f"Using patient index: {patient_idx}")

    if patient_idx >= N_y:
        print(f"WARNING: Patient {patient_idx} not in Y array (max index: {N_y-1})")

    if patient_idx >= N_pi:
        print(f"WARNING: Patient {patient_idx} not in pi array (max index: {N_pi-1})")

    # Get patient data
    patient_theta = theta[patient_idx, :, :]  # (K, T)
    patient_Y = Y_np[patient_idx, :, :] if patient_idx < N_y else np.zeros((D, T_y))  # (D, T)
    patient_pi = pi_np[patient_idx, :, :] if patient_idx < N_pi else np.zeros((D_pi, T_pi))  # (D, T)

    # Ensure T dimensions match (use minimum)
    T = min(T_theta, T_y, T_pi)
    if T_theta != T:
        patient_theta = patient_theta[:, :T]
    if T_y != T:
        patient_Y = patient_Y[:, :T]
    if T_pi != T:
        patient_pi = patient_pi[:, :T]

    print(f"\nUsing T = {T} (aligned across all arrays)")

    # Find diagnoses
    diagnosis_times = {}
    for d in range(patient_Y.shape[0]):
        event_times = np.where(patient_Y[d, :] == 1)[0]
        if len(event_times) > 0:
            diagnosis_times[d] = event_times.tolist()

    # Print summary
    n_diseases = len(diagnosis_times)
    all_times = []
    for times in diagnosis_times.values():
        all_times.extend(times)
    time_range = (min(all_times), max(all_times)) if all_times else (0, 0)
    print(f"\nPatient {patient_idx} Summary:")
    print(f"  Number of diseases: {n_diseases}")
    print(f"  Diagnosis timepoints: {sorted(set(all_times))}")
    print(f"  Time range: {time_range[0]} to {time_range[1]} (ages {30+time_range[0]} to {30+time_range[1]})")

    # Calculate average theta (K vector) - average across time
    avg_theta = patient_theta.mean(axis=1)  # Shape: (K,)

    # Find diseases with events
    diseases_with_events = []
    for d in range(patient_Y.shape[0]):
        event_times = np.where(patient_Y[d, :] == 1)[0]
        if len(event_times) > 0:
            diseases_with_events.append(d)
            if d not in diagnosis_times:
                diagnosis_times[d] = event_times.tolist()

    # Convert timepoints to ages
    ages = np.arange(30, 30 + T)

    # ============================================================================
    # Create Plot
    # ============================================================================

    # Use larger figure size for better readability
    if figsize == (14, 10):  # Default size
        figsize = (24, 18)  # Make it wider to accommodate larger legend
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(3, 1, height_ratios=[2.5, 1.5, 2], hspace=0.6)

    colors = sns.color_palette("tab20", K_total)
    sig_colors = sns.color_palette("tab20", K_total)

    # Panel 1: Signature loadings (θ) vs Age
    ax1 = fig.add_subplot(gs[0])

    for k in range(K_total):
        ax1.plot(ages, patient_theta[k, :], 
                 label=f'Signature {k}', linewidth=3, color=colors[k], alpha=0.8)

    # Add horizontal lines at diagnosis times
    for d, times in diagnosis_times.items():
        for t in times:
            if t >= T:
                continue
            age_at_diag = 30 + t
            # Get the signature for this disease
            sig_for_disease = initial_clusters[d] if d < len(initial_clusters) else -1
            if sig_for_disease < K_total:
                # Draw horizontal line at the theta value for this signature at diagnosis time
                theta_at_diag = patient_theta[sig_for_disease, t]
            ax1.axhline(y=theta_at_diag, xmin=(age_at_diag - 30) / (81 - 30), 
                       xmax=(age_at_diag - 30 + 1) / (81 - 30),
                       color=colors[sig_for_disease], linestyle='--', 
                       alpha=0.6, linewidth=2)

    # Add thin stacked bar showing average theta (single bar, stacked)
    # Position it at the top right - make it bigger
    ax1_bar = ax1.inset_axes([0.7, 0.7, 0.25, 0.18])  # [x, y, width, height] in axes coordinates

    # Sort signatures by average theta (largest first) for better visualization
    sorted_indices = np.argsort(avg_theta)[::-1]
    sorted_avg_theta = avg_theta[sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # Create stacked bar (single bar)
    bottom = 0
    for i, (val, color) in enumerate(zip(sorted_avg_theta, sorted_colors)):
        if val > 0.01:  # Only show if > 1% to avoid clutter
            ax1_bar.barh(0, val, left=bottom, color=color, height=0.5, alpha=0.8)
            bottom += val

    ax1_bar.set_xlim([0, 1])
    ax1_bar.set_ylim([-0.5, 0.5])
    ax1_bar.set_xticks([0, 0.5, 1.0])
    ax1_bar.set_xticklabels(['0', '0.5', '1'], fontsize=11)
    ax1_bar.set_yticks([])
    ax1_bar.set_title('Avg θ (stacked)', fontsize=12)
    ax1_bar.spines['top'].set_visible(False)
    ax1_bar.spines['right'].set_visible(False)
    ax1_bar.spines['left'].set_visible(False)

    ax1.set_ylabel('Signature loadings (θ)', fontsize=16)
    ax1.set_title(f'Patient {patient_idx}: Signature Trajectories', fontsize=18, fontweight='bold')
    # Make legend much more readable - larger font, better spacing
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, ncol=2, 
               columnspacing=1.2, handlelength=3, handletextpad=0.8, 
               framealpha=0.95, prop={'size': 16})
    ax1.tick_params(labelsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([30, 81])
    ax1.set_ylim([0, None])

    # Panel 2: Disease timeline (scatter plot)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    if len(diagnosis_times) > 0:
        # Sort diagnoses by time
        diag_order = sorted([(d, times[0]) for d, times in diagnosis_times.items()], 
                           key=lambda x: x[1])
        disease_rows = {d: i for i, (d, _) in enumerate(diag_order)}
        
        for d, t_diag in diag_order:
            if t_diag >= T:
                continue
            sig_for_disease = initial_clusters[d] if d < len(initial_clusters) else -1
            color = sig_colors[sig_for_disease] if sig_for_disease < K_total else 'gray'
            age_at_diag = 30 + t_diag
            y = disease_rows[d]
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            ax2.scatter(age_at_diag, y, s=150, color=color, alpha=0.7, zorder=10, 
                       edgecolors='black', linewidths=2)
            # Add disease name label on the right side - make it much more readable
            ax2.text(82, y, f'{disease_name} (Sig {sig_for_disease})', 
                    fontsize=16, fontweight='bold', verticalalignment='center', ha='left')
        
        ax2.set_yticks(range(len(diag_order)))
        # Label diseases by chronological order (1, 2, 3, ...)
        ax2.set_yticklabels([f'{i+1}' for i in range(len(diag_order))], fontsize=14)
        # Extend x-axis further to accommodate larger right-side labels
        ax2.set_xlim([30, 95])
    else:
        ax2.text(0.5, 0.5, 'No diagnoses', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12)
        ax2.set_xlim([30, 81])

    ax2.set_ylabel('Disease', fontsize=15)
    ax2.set_title('Disease timeline', fontsize=16, fontweight='bold')
    ax2.tick_params(labelsize=13)
    ax2.grid(True, alpha=0.3, axis='x')

    # Panel 3: Disease probabilities (π) - stop after diagnosis
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Sort diseases by chronological order (by diagnosis time) - same order as Panel 2
    if len(diagnosis_times) > 0:
        diseases_sorted_by_time = sorted(diseases_with_events, 
                                         key=lambda d: min(diagnosis_times[d]) if d in diagnosis_times else float('inf'))
    else:
        diseases_sorted_by_time = diseases_with_events

    # Plot diseases with events, colored by signature, stopping after diagnosis
    # Plot in chronological order (same as Panel 2) - no legend since labels are in Panel 2
    for d in diseases_sorted_by_time:
        disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
        sig_for_disease = initial_clusters[d] if d < len(initial_clusters) else -1
        color = sig_colors[sig_for_disease] if sig_for_disease < K_total else 'gray'
        
        # Get diagnosis timepoint
        if d in diagnosis_times:
            first_diag_t = min(diagnosis_times[d])
            if first_diag_t >= T:
                first_diag_t = T - 1
            # Only plot up to and including diagnosis timepoint
            plot_ages = ages[:first_diag_t + 1]
            plot_pi = patient_pi[d, :first_diag_t + 1]
        else:
            plot_ages = ages
            plot_pi = patient_pi[d, :]
        
        # Plot probability curve (stops after diagnosis)
        # No label since we're not showing a legend
        ax3.plot(plot_ages, plot_pi, 
                 color=color, linewidth=3, alpha=0.7)
        
        # Mark diagnosis timepoint
        if d in diagnosis_times:
            for t in diagnosis_times[d]:
                if t >= T:
                    continue
                age_at_diag = 30 + t
                ax3.scatter(age_at_diag, patient_pi[d, t], 
                           color=color, s=150, zorder=10, marker='o', 
                           edgecolors='black', linewidths=2.5)

    ax3.set_xlabel('Age (yr)', fontsize=15)
    ax3.set_ylabel('Disease Probability (π)', fontsize=15)
    ax3.set_title('Disease Probabilities Over Time (colored by primary signature)', fontsize=16, fontweight='bold')
    ax3.tick_params(labelsize=13)
    # No legend - diseases are labeled on the right side of Panel 2
    ax3.grid(True, alpha=0.3)
    # Match Panel 2 xlim (95 if diagnoses exist, 81 otherwise)
    if len(diagnosis_times) > 0:
        ax3.set_xlim([30, 95])
    else:
        ax3.set_xlim([30, 81])

    plt.suptitle(f'Patient {patient_idx} Timeline', fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {output_path}")
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    patient_idx = 148745
    
    output_path = f'/Users/sarahurbut/aladynoulli2/patient_{patient_idx}_timeline_panel_style.pdf'
    
    fig = plot_patient_timeline(
        patient_idx=patient_idx,
        output_path=output_path
    )

