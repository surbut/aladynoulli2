#!/usr/bin/env python3
"""
Plot Patient Timeline V3 - Optimized layout with fully visible disease details

Creates a multi-panel plot showing:
- Panel 1: Signature loadings (θ) vs Age 
- Panel 2: Disease timeline with ALL disease labels visible
- Panel 3: Disease probabilities (π) over time
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import textwrap


def plot_patient_timeline(patient_idx, 
                          theta_path='/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective_correctE.pt',
                          checkpoint_path='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt',
                          pi_path='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/pi_fullmode_400k.pt',
                          Y_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt',
                          initial_clusters_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt',
                          disease_names_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv',
                          output_path=None,
                          figsize=(20, 14)):
    """
    Plot patient timeline with signature trajectories, disease timeline, and disease probabilities.
    """
    
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Load data
    initial_clusters = torch.load(initial_clusters_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(initial_clusters):
        initial_clusters = initial_clusters.numpy()
    K = int(initial_clusters.max() + 1)

    # Load theta from the specified file
    theta_full = torch.load(theta_path, map_location='cpu', weights_only=False)

    # Check structure of theta file
    if isinstance(theta_full, dict):
        if 'theta' in theta_full:
            theta = theta_full['theta']
        elif 'thetas' in theta_full:
            theta = theta_full['thetas']
        elif 'lambda_' in theta_full:
            theta = torch.softmax(theta_full['lambda_'], dim=1)
        else:
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
    # Create Plot with optimized layout for disease visibility
    # ============================================================================

    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Modified grid layout:
    # - Smaller top panel for signatures
    # - Larger middle section for disease timeline and details
    # - Bottom panel for probabilities
    # Using more columns to accommodate disease details
    gs = plt.GridSpec(3, 4, width_ratios=[1.5, 1.5, 1.2, 1.2], 
                      height_ratios=[2, 2.8, 2], 
                      hspace=0.35, wspace=0.25)

    # Use distinct colors for signatures
    colors = sns.color_palette("tab20", K_total)
    sig_colors = colors

    # ============================================================================
    # Panel 1: Signature loadings (θ) vs Age - spans full width
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Plot signature trajectories
    for k in range(K_total):
        ax1.plot(ages, patient_theta[k, :], 
                 label=f'Signature {k}', linewidth=2.3, color=colors[k], alpha=0.85)

    # Add subtle vertical lines at diagnosis times
    for d, times in diagnosis_times.items():
        for t in times:
            if t >= T:
                continue
            age_at_diag = 30 + t
            ax1.axvline(x=age_at_diag, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)

    # Add inset for average theta composition
    ax1_bar = ax1.inset_axes([0.84, 0.62, 0.14, 0.32])
    
    # Sort signatures by average theta
    sorted_indices = np.argsort(avg_theta)[::-1]
    sorted_avg_theta = avg_theta[sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # Create stacked bar
    bottom = 0
    for i, (val, color) in enumerate(zip(sorted_avg_theta, sorted_colors)):
        if val > 0.005:  # Only show if > 0.5%
            ax1_bar.barh(0, val, left=bottom, color=color, height=0.7, alpha=0.85, edgecolor='none')
            bottom += val

    ax1_bar.set_xlim([0, 1])
    ax1_bar.set_ylim([-0.5, 0.5])
    ax1_bar.set_xticks([0, 0.5, 1])
    ax1_bar.set_xticklabels(['0', '0.5', '1'], fontsize=8)
    ax1_bar.set_yticks([])
    ax1_bar.set_title('Avg θ', fontsize=9, fontweight='bold', pad=3)
    for spine in ax1_bar.spines.values():
        spine.set_visible(False)
    ax1_bar.tick_params(length=0)

    ax1.set_ylabel('Signature Loading (θ)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Patient {patient_idx}: Signature Trajectories Over Time', 
                  fontsize=15, fontweight='bold', pad=10)
    
    # Create a compact legend
    ncols = min(4, (K_total + 3) // 4)  # Dynamic columns based on K
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
               ncol=ncols, columnspacing=1.0, handlelength=1.8, handletextpad=0.5,
               framealpha=0.95, borderpad=0.4, borderaxespad=0.2)
    
    ax1.tick_params(labelsize=11)
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax1.set_xlim([30, 81])
    ax1.set_ylim([0, max(patient_theta.max() * 1.08, 0.5)])
    ax1.set_xlabel('Age (years)', fontsize=12)

    # ============================================================================
    # Panel 2: Disease timeline - left half of middle row
    # ============================================================================
    ax2 = fig.add_subplot(gs[1, :2])

    if len(diagnosis_times) > 0:
        # Sort diagnoses by time
        diag_order = sorted([(d, times[0]) for d, times in diagnosis_times.items()], 
                           key=lambda x: x[1])
        
        # Show all diseases (or reasonable max)
        max_diseases_shown = min(30, len(diag_order))  # Increased limit
        diag_order_shown = diag_order[:max_diseases_shown]
        
        disease_rows = {d: i for i, (d, _) in enumerate(diag_order_shown)}
        
        # Plot disease events as a timeline
        for i, (d, t_diag) in enumerate(diag_order_shown):
            if t_diag >= T:
                continue
            sig_for_disease = initial_clusters[d] if d < len(initial_clusters) else -1
            color = sig_colors[sig_for_disease] if sig_for_disease < K_total else 'gray'
            age_at_diag = 30 + t_diag
            y_pos = len(diag_order_shown) - i - 1  # Reverse order so earliest is at top
            
            # Draw horizontal line from age 30 to diagnosis
            ax2.plot([30, age_at_diag], [y_pos, y_pos], color=color, linewidth=1, alpha=0.3)
            
            # Plot the diagnosis event
            ax2.scatter(age_at_diag, y_pos, s=90, color=color, alpha=0.85, zorder=10, 
                       edgecolors='black', linewidths=1.2)
            
            # Add disease number on the left
            ax2.text(29.5, y_pos, f'{i+1}', fontsize=8, fontweight='bold', 
                    verticalalignment='center', ha='right')
        
        ax2.set_yticks(range(len(diag_order_shown)))
        ax2.set_yticklabels([])
        ax2.set_ylim([-0.5, len(diag_order_shown) - 0.5])
        
        # Add note if more diseases exist
        if len(diag_order) > max_diseases_shown:
            ax2.text(0.5, -0.02, f'(Showing first {max_diseases_shown} of {len(diag_order)} diseases)', 
                    transform=ax2.transAxes, ha='center', fontsize=9, style='italic', color='gray')
        
    else:
        ax2.text(0.5, 0.5, 'No diagnoses recorded', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12, color='gray')

    ax2.set_ylabel('Disease Order\n(chronological)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age (years)', fontsize=12)
    ax2.set_title('Disease Timeline', fontsize=14, fontweight='bold', pad=8)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, alpha=0.15, axis='x', linestyle='-', linewidth=0.5)
    ax2.set_xlim([30, 81])

    # ============================================================================
    # Panel 2b: Disease legend - right half of middle row (TWO COLUMNS)
    # ============================================================================
    # Use two subplot areas for two columns of disease details
    ax2_legend1 = fig.add_subplot(gs[1, 2])
    ax2_legend2 = fig.add_subplot(gs[1, 3])
    ax2_legend1.axis('off')
    ax2_legend2.axis('off')
    
    if len(diagnosis_times) > 0:
        # Sort diagnoses by time
        diag_order_shown = diag_order[:max_diseases_shown]
        
        # Split diseases into two columns
        mid_point = (len(diag_order_shown) + 1) // 2
        first_column = diag_order_shown[:mid_point]
        second_column = diag_order_shown[mid_point:]
        
        # Format first column
        legend_text1 = []
        for i, (d, t_diag) in enumerate(first_column):
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            sig_for_disease = initial_clusters[d] if d < len(initial_clusters) else -1
            
            # Smart truncation
            max_len = 24
            if len(disease_name) > max_len:
                truncated = disease_name[:max_len-3]
                last_space = truncated.rfind(' ')
                if last_space > 12:
                    disease_name = truncated[:last_space] + '...'
                else:
                    disease_name = truncated + '...'
            
            age_at_diag = 30 + t_diag
            legend_text1.append(f'{i+1:2d}. {disease_name[:24]:<24s}\n    Sig {sig_for_disease:2d}, Age {age_at_diag:2d}')
        
        # Format second column
        legend_text2 = []
        for i, (d, t_diag) in enumerate(second_column, start=mid_point):
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            sig_for_disease = initial_clusters[d] if d < len(initial_clusters) else -1
            
            # Smart truncation
            max_len = 24
            if len(disease_name) > max_len:
                truncated = disease_name[:max_len-3]
                last_space = truncated.rfind(' ')
                if last_space > 12:
                    disease_name = truncated[:last_space] + '...'
                else:
                    disease_name = truncated + '...'
            
            age_at_diag = 30 + t_diag
            legend_text2.append(f'{i+1:2d}. {disease_name[:24]:<24s}\n    Sig {sig_for_disease:2d}, Age {age_at_diag:2d}')
        
        # Display first column
        ax2_legend1.text(0.05, 0.98, 'Disease Details (1/2):', fontsize=10, fontweight='bold',
                        transform=ax2_legend1.transAxes, va='top')
        legend_str1 = '\n'.join(legend_text1)
        ax2_legend1.text(0.05, 0.93, legend_str1, fontsize=7,
                        transform=ax2_legend1.transAxes, va='top', 
                        fontfamily='monospace', linespacing=1.25)
        
        # Display second column
        if second_column:
            ax2_legend2.text(0.05, 0.98, 'Disease Details (2/2):', fontsize=10, fontweight='bold',
                            transform=ax2_legend2.transAxes, va='top')
            legend_str2 = '\n'.join(legend_text2)
            ax2_legend2.text(0.05, 0.93, legend_str2, fontsize=7,
                            transform=ax2_legend2.transAxes, va='top', 
                            fontfamily='monospace', linespacing=1.25)

    # ============================================================================
    # Panel 3: Disease probabilities - full width bottom
    # ============================================================================
    ax3 = fig.add_subplot(gs[2, :])

    # Sort diseases by max probability for better visualization
    if len(diagnosis_times) > 0:
        max_probs = {}
        for d in diseases_with_events[:50]:  # Consider top 50 for selection
            if d in diagnosis_times:
                first_diag_t = min(diagnosis_times[d])
                if first_diag_t >= T:
                    first_diag_t = T - 1
                max_probs[d] = patient_pi[d, :first_diag_t + 1].max()
            else:
                max_probs[d] = patient_pi[d, :].max()
        
        # Select top diseases by probability
        n_diseases_to_plot = min(20, len(max_probs))
        top_diseases = sorted(max_probs.keys(), key=lambda x: max_probs[x], reverse=True)[:n_diseases_to_plot]
        
        # Group diseases by signature for visual clarity
        diseases_by_sig = {}
        for d in top_diseases:
            sig = initial_clusters[d] if d < len(initial_clusters) else -1
            if sig not in diseases_by_sig:
                diseases_by_sig[sig] = []
            diseases_by_sig[sig].append(d)
        
        # Plot disease probabilities grouped by signature
        plotted_count = 0
        for sig in sorted(diseases_by_sig.keys()):
            for d in diseases_by_sig[sig]:
                color = sig_colors[sig] if sig < K_total and sig >= 0 else 'gray'
                
                # Get diagnosis timepoint
                if d in diagnosis_times:
                    first_diag_t = min(diagnosis_times[d])
                    if first_diag_t >= T:
                        first_diag_t = T - 1
                    # Plot up to diagnosis
                    plot_ages = ages[:first_diag_t + 1]
                    plot_pi = patient_pi[d, :first_diag_t + 1]
                else:
                    plot_ages = ages
                    plot_pi = patient_pi[d, :]
                
                # Vary line style for diseases in same signature
                linestyle = '-' if plotted_count % 3 == 0 else ('--' if plotted_count % 3 == 1 else '-.')
                
                # Plot probability curve
                ax3.plot(plot_ages, plot_pi, 
                         color=color, linewidth=1.8, alpha=0.7, linestyle=linestyle)
                
                # Mark diagnosis points
                if d in diagnosis_times:
                    for t in diagnosis_times[d]:
                        if t >= T:
                            continue
                        age_at_diag = 30 + t
                        ax3.scatter(age_at_diag, patient_pi[d, t], 
                                   color=color, s=80, zorder=10, marker='o', 
                                   edgecolors='black', linewidths=1.2, alpha=0.9)
                
                plotted_count += 1

        # Add annotation about what's shown
        ax3.text(0.98, 0.98, f'Top {n_diseases_to_plot} diseases by max probability\nGrouped by signature, diagnoses marked with ●', 
                transform=ax3.transAxes, ha='right', va='top', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax3.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Disease Probability (π)', fontsize=13, fontweight='bold')
    ax3.set_title('Disease Risk Trajectories (stopping at diagnosis)', fontsize=14, fontweight='bold', pad=8)
    ax3.tick_params(labelsize=11)
    ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax3.set_xlim([30, 81])
    
    # Set y-axis limit intelligently
    if len(diagnosis_times) > 0:
        y_max = min(0.1, ax3.get_ylim()[1] * 1.1)  # Cap at 10% or current max
        ax3.set_ylim([0, y_max])
    
    # Format y-axis as percentage
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

    # Add main title
    fig.suptitle(f'Patient {patient_idx} - Comprehensive Disease Trajectory Analysis', 
                 fontsize=17, fontweight='bold', y=0.98)
    
    # Add subtitle with summary stats
    subtitle = f'Total diseases: {n_diseases} | Age range: {30+time_range[0]}-{30+time_range[1]} | Signatures: {K_total}'
    fig.text(0.5, 0.95, subtitle, ha='center', fontsize=11, style='italic', color='#666666')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ Saved improved plot to: {output_path}")
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    patient_idx = 148745
    
    output_path = f'/Users/sarahurbut/aladynoulli2/patient_{patient_idx}_timeline_v3.pdf'
    
    fig = plot_patient_timeline(
        patient_idx=patient_idx,
        output_path=output_path,
        figsize=(20, 14)  # Wider format for better readability
    )
