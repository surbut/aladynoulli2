#!/usr/bin/env python3
"""
Analyze early vs late onset disease patterns across batch results.

Usage:
    python analyze_age_onset.py --results_dir /path/to/censor_e_batchrun_vectorized --disease_index 113
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import glob
import argparse

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


def get_signature_colors(K):
    """Return a list of K distinct colors for signatures."""
    import matplotlib.cm as cm
    if K <= 20:
        # Use tab20 colormap for up to 20 signatures
        # Use np.linspace to properly sample distinct colors
        sig_colors = cm.get_cmap('tab20')(np.linspace(0, 1, K))
        # Convert to list of tuples for matplotlib
        return [sig_colors[i] for i in range(K)]
    else:
        # For more than 20, combine tab20 and tab20b
        colors_20 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors_b = cm.get_cmap('tab20b')(np.linspace(0, 1, 20))
        all_colors = np.vstack([colors_20, colors_b])
        if K <= 40:
            return [all_colors[i] for i in range(K)]
        else:
            # For more than 40, use HSV palette
            return sns.color_palette("hsv", K)


def find_model_files(results_base_dir: str, pattern: str = "enrollment_model_VECTORIZED_W0.0001_batch_*.pt") -> List[Dict]:
    """
    Find all model files in the results directory.
    
    Returns list of dicts with keys: 'path', 'start', 'end', 'filename'
    """
    base_path = Path(results_base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base_dir}")
    
    # Find all matching model files
    model_files = sorted(base_path.glob(pattern))
    
    if not model_files:
        # Try alternative pattern
        model_files = sorted(base_path.glob("enrollment_model_*_batch_*.pt"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern in {results_base_dir}")
    
    batch_info = []
    for model_file in model_files:
        # Extract start and end indices from filename
        # Pattern: enrollment_model_VECTORIZED_W0.0001_batch_0_10000.pt
        # Or: enrollment_model_W0.0001_batch_0_10000.pt
        parts = model_file.stem.split('_')
        try:
            # Find the index of 'batch' and get next two numbers
            batch_idx_list = [i for i, p in enumerate(parts) if p == 'batch']
            if not batch_idx_list:
                print(f"Warning: 'batch' not found in filename {model_file.name}")
                continue
            batch_idx = batch_idx_list[0]
            start = int(parts[batch_idx + 1])
            end = int(parts[batch_idx + 2])
            batch_info.append({
                'path': str(model_file),
                'start': start,
                'end': end,
                'filename': model_file.name
            })
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse filename {model_file.name}: {e}")
            print(f"  Parts: {parts}")
            continue
    
    return sorted(batch_info, key=lambda x: x['start'])


def analyze_age_onset_patterns(
    results_base_dir: str,
    disease_index: int = 113,  # MI is typically index 113 (0-indexed)
    early_threshold: int = 55,
    late_threshold: int = 70,
    age_offset: int = 30,
    output_path: Optional[str] = None,
    return_stats: bool = True
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Analyze early vs late onset disease patterns across batch results.
    
    Parameters:
    -----------
    results_base_dir : str
        Base directory containing batch model files
    disease_index : int
        Index of the disease to analyze (default: 113 for MI)
    early_threshold : int
        Age threshold for early onset (default: 55)
    late_threshold : int
        Age threshold for late onset (default: 70)
    age_offset : int
        Age offset (timepoint 0 = age_offset, default: 30)
    output_path : str, optional
        Path to save the figure
    return_stats : bool
        Whether to return statistics dictionary
    
    Returns:
    --------
    early_onset_indices : List[int]
        Global indices of early-onset patients
    late_onset_indices : List[int]
        Global indices of late-onset patients
    stats : Dict[str, Any]
        Statistics dictionary (if return_stats=True)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Age Onset Patterns")
    print(f"{'='*80}")
    print(f"Results directory: {results_base_dir}")
    print(f"Disease index: {disease_index}")
    print(f"Early threshold: <{early_threshold} years")
    print(f"Late threshold: >{late_threshold} years")
    print(f"{'='*80}\n")
    
    # Find all model files
    batch_files = find_model_files(results_base_dir)
    print(f"Found {len(batch_files)} batch files\n")
    
    if not batch_files:
        raise ValueError("No batch files found. Cannot proceed.")
    
    # Initialize storage
    all_early_onset = []  # (global_idx, lambda_values, diagnosis_age)
    all_late_onset = []
    total_patients_scanned = 0
    
    # Process each batch
    for batch_info in batch_files:
        batch_model_path = batch_info['path']
        if not os.path.exists(batch_model_path):
            print(f"Warning: File not found: {batch_model_path}")
            continue
        
        try:
            print(f"Processing {batch_info['filename']} (patients {batch_info['start']} to {batch_info['end']})...")
            model_data = torch.load(batch_model_path, map_location='cpu', weights_only=False)
            
            # Extract Y and lambda
            Y_batch = model_data['Y']
            if 'model_state_dict' in model_data and 'lambda_' in model_data['model_state_dict']:
                lambda_batch = model_data['model_state_dict']['lambda_']
            else:
                print(f"  Warning: lambda_ not found in model_state_dict, skipping...")
                continue
            
            # Convert to numpy
            if torch.is_tensor(Y_batch):
                Y_batch = Y_batch.cpu().numpy()
            if torch.is_tensor(lambda_batch):
                lambda_batch = lambda_batch.cpu().numpy()
            
            current_N = Y_batch.shape[0]
            total_patients_scanned += current_N
            
            # Check disease index is valid
            if disease_index >= Y_batch.shape[1]:
                print(f"  Warning: disease_index {disease_index} >= D ({Y_batch.shape[1]}), skipping...")
                continue
            
            # Process all patients in this batch
            batch_early_count = 0
            batch_late_count = 0
            for local_idx in range(current_N):
                diagnosis_times = np.where(Y_batch[local_idx, disease_index] > 0.5)[0]
                if len(diagnosis_times) > 0:
                    diagnosis_time = diagnosis_times[0]
                    diagnosis_age = age_offset + diagnosis_time
                    global_idx = batch_info['start'] + local_idx
                    
                    # Store lambda values with the patient info
                    lambda_values = lambda_batch[local_idx]
                    if diagnosis_age < early_threshold:
                        all_early_onset.append((global_idx, lambda_values, diagnosis_age))
                        batch_early_count += 1
                    elif diagnosis_age > late_threshold:
                        all_late_onset.append((global_idx, lambda_values, diagnosis_age))
                        batch_late_count += 1
            
            if batch_early_count > 0 or batch_late_count > 0:
                print(f"  Found: {batch_early_count} early, {batch_late_count} late")
                        
        except Exception as e:
            print(f"  Error processing {batch_info['filename']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Results Summary")
    print(f"{'='*80}")
    print(f"Total patients scanned: {total_patients_scanned}")
    print(f"Early-onset cases (<{early_threshold} years): {len(all_early_onset)}")
    print(f"Late-onset cases (>{late_threshold} years): {len(all_late_onset)}")
    print(f"{'='*80}\n")
    
    if not all_early_onset or not all_late_onset:
        print("Insufficient data for analysis")
        return [], [], {}
    
    # Calculate statistics
    early_ages = [age for _, _, age in all_early_onset]
    late_ages = [age for _, _, age in all_late_onset]
    print(f"Early onset mean age: {np.mean(early_ages):.1f} ± {np.std(early_ages):.1f} years (range: {min(early_ages):.1f}-{max(early_ages):.1f})")
    print(f"Late onset mean age: {np.mean(late_ages):.1f} ± {np.std(late_ages):.1f} years (range: {min(late_ages):.1f}-{max(late_ages):.1f})")
    
    # Get dimensions from the stored lambda values
    K = all_early_onset[0][1].shape[0]
    T = all_early_onset[0][1].shape[1]
    time_points = np.arange(T) + age_offset
    
    # Calculate theta values directly from stored lambda values
    early_theta = np.zeros((len(all_early_onset), K, T))
    late_theta = np.zeros((len(all_late_onset), K, T))
    
    for i, (_, lambda_values, _) in enumerate(all_early_onset):
        exp_lambda = np.exp(lambda_values)
        early_theta[i] = exp_lambda / np.sum(exp_lambda, axis=0, keepdims=True)
    
    for i, (_, lambda_values, _) in enumerate(all_late_onset):
        exp_lambda = np.exp(lambda_values)
        late_theta[i] = exp_lambda / np.sum(exp_lambda, axis=0, keepdims=True)
    
    # Create visualization with improved styling
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.flat
    
    # Get consistent colors ONCE for all plots (so same signature = same color across all subplots)
    signature_colors = get_signature_colors(K)
    
    def plot_group_patterns(theta_values, mean_age, axes, title_prefix, colors):
        ax_prop, ax_vel = axes
        
        # Calculate mean and SEM
        mean_theta = np.mean(theta_values, axis=0)
        sem_theta = np.std(theta_values, axis=0) / np.sqrt(len(theta_values))
        
        # Calculate velocities
        velocities = np.gradient(mean_theta, axis=1)
        vel_sem = np.std(np.gradient(theta_values, axis=2), axis=0) / np.sqrt(len(theta_values))
        
        # Plot proportions with improved styling
        for k in range(K):
            color = colors[k]
            ax_prop.plot(time_points, mean_theta[k], 
                        label=f'Sig {k}', color=color, linewidth=2.5, alpha=0.9, zorder=3)
            ax_prop.fill_between(time_points, 
                               mean_theta[k] - sem_theta[k],
                               mean_theta[k] + sem_theta[k],
                               color=color, alpha=0.15, zorder=2)
        
        # Plot velocities with improved styling
        for k in range(K):
            color = colors[k]
            ax_vel.plot(time_points, velocities[k], 
                       label=f'Sig {k}', color=color, linewidth=2.5, alpha=0.9, zorder=3)
            ax_vel.fill_between(time_points,
                              velocities[k] - vel_sem[k],
                              velocities[k] + vel_sem[k],
                              color=color, alpha=0.15, zorder=2)
        
        # Add vertical lines for diagnosis age
        ax_prop.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
                       label=f'Avg Diagnosis: {mean_age:.1f} yr', alpha=0.8, zorder=4)
        ax_vel.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
                      label=f'Avg Diagnosis: {mean_age:.1f} yr', alpha=0.8, zorder=4)
        
        # Customize plots
        ax_prop.set_title(f'{title_prefix}\nAverage Signature Loading', 
                         fontsize=14, fontweight='bold', pad=10)
        ax_vel.set_title(f'{title_prefix}\nVelocity', 
                        fontsize=14, fontweight='bold', pad=10)
        ax_prop.set_ylabel('Average Signature Loading', fontsize=12, fontweight='bold')
        ax_vel.set_ylabel('Velocity (change/year)', fontsize=12, fontweight='bold')
        ax_prop.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        ax_vel.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        
        # Grid styling
        ax_prop.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax_vel.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        
        # Set axis limits
        ax_prop.set_xlim(time_points[0], time_points[-1])
        ax_prop.set_ylim(0, None)  # Start from 0 for proportions
        ax_vel.set_xlim(time_points[0], time_points[-1])
        ax_vel.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3, zorder=1)
    
    # Plot patterns
    early_mean_age = np.mean(early_ages)
    late_mean_age = np.mean(late_ages)
    
    plot_group_patterns(
        early_theta, 
        early_mean_age,
        (ax1, ax2),
        f'Early Onset (<{early_threshold} yr, n={len(all_early_onset)})',
        signature_colors
    )
    
    plot_group_patterns(
        late_theta,
        late_mean_age,
        (ax3, ax4),
        f'Late Onset (>{late_threshold} yr, n={len(all_late_onset)})',
        signature_colors
    )
    
    # Add shared legend for signatures (only show first 10 to avoid clutter)
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Filter to only signature labels (not diagnosis age)
    sig_handles = [h for h, l in zip(handles1, labels1) if l.startswith('Sig ')]
    sig_labels = [l for l in labels1 if l.startswith('Sig ')]
    diag_handles = [h for h, l in zip(handles1, labels1) if l.startswith('Avg Diagnosis')]
    diag_labels = [l for l in labels1 if l.startswith('Avg Diagnosis')]
    
    # Show all signatures in legend (or limit to first 15 if too many)
    max_sigs_in_legend = 15
    if len(sig_handles) > max_sigs_in_legend:
        sig_handles = sig_handles[:max_sigs_in_legend]
        sig_labels = sig_labels[:max_sigs_in_legend]
    
    # Create combined legend
    all_handles = sig_handles + diag_handles
    all_labels = sig_labels + diag_labels
    
    fig.legend(all_handles, all_labels, 
              bbox_to_anchor=(1.02, 0.5), loc='center left', 
              fontsize=9, framealpha=0.95, edgecolor='gray', 
              fancybox=True, frameon=True, ncol=1)
    
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"\n✓ Saved figure to: {output_path}")
        plt.close()
    else:
        plt.show()
    
    # Calculate statistics if requested
    stats = {}
    if return_stats:
        # Calculate mean theta for each group
        early_mean_theta = np.mean(early_theta, axis=0)
        late_mean_theta = np.mean(late_theta, axis=0)
        
        # Find the 5-year window before diagnosis for each group
        early_diag_idx = int(early_mean_age - age_offset)
        late_diag_idx = int(late_mean_age - age_offset)
        
        # Get the 5-year window before diagnosis
        early_start_idx = max(0, early_diag_idx - 5)
        late_start_idx = max(0, late_diag_idx - 5)
        
        # Calculate peak contributions in the pre-diagnosis window
        early_sig5_peak = np.max(early_mean_theta[5, early_start_idx:early_diag_idx]) if early_diag_idx > early_start_idx else 0
        late_sig5_peak = np.max(late_mean_theta[5, late_start_idx:late_diag_idx]) if late_diag_idx > late_start_idx else 0
        
        # Find ages at peak within the window
        if early_diag_idx > early_start_idx:
            early_peak_idx = early_start_idx + np.argmax(early_mean_theta[5, early_start_idx:early_diag_idx])
            early_peak_age = time_points[early_peak_idx]
        else:
            early_peak_age = early_mean_age
        
        if late_diag_idx > late_start_idx:
            late_peak_idx = late_start_idx + np.argmax(late_mean_theta[5, late_start_idx:late_diag_idx])
            late_peak_age = time_points[late_peak_idx]
        else:
            late_peak_age = late_mean_age
        
        # Calculate velocities in the 5 years before diagnosis
        early_velocities = np.gradient(early_mean_theta, axis=1)
        late_velocities = np.gradient(late_mean_theta, axis=1)
        
        # Get the velocity right before diagnosis (last 2-3 time points)
        early_vel_pre_mi = np.mean(early_velocities[5, max(0, early_diag_idx-3):early_diag_idx]) if early_diag_idx > 0 else 0
        late_vel_pre_mi = np.mean(late_velocities[5, max(0, late_diag_idx-3):late_diag_idx]) if late_diag_idx > 0 else 0
        
        # Also calculate the maximum velocity in the 5-year window
        if early_diag_idx > early_start_idx:
            early_max_vel = np.max(early_velocities[5, early_start_idx:early_diag_idx])
        else:
            early_max_vel = 0
        
        if late_diag_idx > late_start_idx:
            late_max_vel = np.max(late_velocities[5, late_start_idx:late_diag_idx])
        else:
            late_max_vel = 0
        
        stats = {
            'early_peak_contribution': early_sig5_peak,
            'late_peak_contribution': late_sig5_peak,
            'early_peak_age': early_peak_age,
            'late_peak_age': late_peak_age,
            'early_velocity_pre_mi': early_vel_pre_mi,
            'late_velocity_pre_mi': late_vel_pre_mi,
            'early_max_velocity': early_max_vel,
            'late_max_velocity': late_max_vel,
            'early_diagnosis_age': early_mean_age,
            'late_diagnosis_age': late_mean_age,
            'early_n': len(all_early_onset),
            'late_n': len(all_late_onset)
        }
        
        print("\n" + "="*80)
        print("Quantitative Results for Signature 5 (Cardiovascular):")
        print("="*80)
        print(f"Early-onset (n={len(all_early_onset)}):")
        print(f"  Peak contribution: {stats['early_peak_contribution']*100:.1f}% at age {stats['early_peak_age']:.1f}")
        print(f"  Mean diagnosis age: {stats['early_diagnosis_age']:.1f}")
        print(f"  Velocity before diagnosis (last 3 years): {stats['early_velocity_pre_mi']:.4f}/year")
        print(f"  Max velocity in 5-year window: {stats['early_max_velocity']:.4f}/year")
        print(f"\nLate-onset (n={len(all_late_onset)}):")
        print(f"  Peak contribution: {stats['late_peak_contribution']*100:.1f}% at age {stats['late_peak_age']:.1f}")
        print(f"  Mean diagnosis age: {stats['late_diagnosis_age']:.1f}")
        print(f"  Velocity before diagnosis (last 3 years): {stats['late_velocity_pre_mi']:.4f}/year")
        print(f"  Max velocity in 5-year window: {stats['late_max_velocity']:.4f}/year")
        if stats['late_velocity_pre_mi'] != 0:
            print(f"\nVelocity ratio (early/late): {stats['early_velocity_pre_mi']/stats['late_velocity_pre_mi']:.2f}-fold higher in early onset")
        print("="*80)
        
        return [idx for idx, _, _ in all_early_onset], [idx for idx, _, _ in all_late_onset], stats
    
    return [idx for idx, _, _ in all_early_onset], [idx for idx, _, _ in all_late_onset], {}


def main():
    parser = argparse.ArgumentParser(description='Analyze early vs late onset disease patterns')
    parser.add_argument('--results_dir', type=str, 
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized',
                       help='Base directory containing batch model files')
    parser.add_argument('--disease_index', type=int, default=113,
                       help='Disease index to analyze (default: 113 for MI)')
    parser.add_argument('--early_threshold', type=int, default=55,
                       help='Age threshold for early onset (default: 55)')
    parser.add_argument('--late_threshold', type=int, default=70,
                       help='Age threshold for late onset (default: 70)')
    parser.add_argument('--age_offset', type=int, default=30,
                       help='Age offset (default: 30)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save output figure')
    
    args = parser.parse_args()
    
    early_indices, late_indices, stats = analyze_age_onset_patterns(
        results_base_dir=args.results_dir,
        disease_index=args.disease_index,
        early_threshold=args.early_threshold,
        late_threshold=args.late_threshold,
        age_offset=args.age_offset,
        output_path=args.output_path,
        return_stats=True
    )
    
    return early_indices, late_indices, stats


if __name__ == '__main__':
    main()

