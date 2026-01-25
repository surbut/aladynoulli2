"""
Plot cumulative incidence curves from observed (Y tensor) and predicted (pi) data.

Cumulative incidence accounts for competing risks (censoring) and shows the probability
of experiencing a disease event by time t.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_data(data_dir, pi_path):
    """Load all required data"""
    print("Loading data...")
    
    # Load Y tensor
    Y = torch.load(Path(data_dir) / 'Y_tensor.pt', weights_only=False)
    print(f"  Y tensor: {Y.shape}")
    
    # Load censor info
    censor_df = pd.read_csv(Path(data_dir) / 'censor_info.csv')
    print(f"  Censor info: {len(censor_df)} patients")
    
    # Load predicted pi
    pi = torch.load(pi_path, weights_only=False)
    print(f"  Predicted pi: {pi.shape}")
    
    # Load E matrix (corrected) for at-risk calculation
    E_corrected = torch.load(Path(data_dir) / 'E_matrix_corrected.pt', weights_only=False)
    print(f"  E matrix (corrected): {E_corrected.shape}")
    
    # Load disease names if available
    try:
        essentials = torch.load(Path(data_dir) / 'model_essentials.pt', weights_only=False)
        disease_names = essentials.get('disease_names', None)
    except:
        disease_names = None
    
    return Y, censor_df, pi, E_corrected, disease_names


def calculate_cumulative_incidence_observed(Y, E_corrected, disease_idx, timepoint_ages):
    """
    Calculate observed cumulative incidence for a disease.
    
    Cumulative incidence at time t = number of events by time t / number at risk at time 0
    
    Accounts for censoring: only includes people who are still at risk at each timepoint.
    """
    N, D, T = Y.shape
    
    # Convert to numpy
    if torch.is_tensor(Y):
        Y_np = Y.numpy()
    else:
        Y_np = Y
    
    if torch.is_tensor(E_corrected):
        E_np = E_corrected.numpy()
    else:
        E_np = E_corrected
    
    # Initialize arrays
    cumulative_incidence = np.zeros(T)
    n_at_risk = np.zeros(T)
    n_events = np.zeros(T)
    
    # At time 0, all patients are at risk (those with E >= 0)
    n_at_risk[0] = (E_np[:, disease_idx] >= 0).sum()
    
    # Calculate cumulative incidence at each timepoint
    for t in range(T):
        # Number at risk at time t: patients with E >= t
        at_risk_mask = E_np[:, disease_idx] >= t
        n_at_risk[t] = at_risk_mask.sum()
        
        if n_at_risk[t] > 0:
            # Number of events by time t (first occurrence)
            # An event occurred if Y[i, d, t] == 1 and it's the first occurrence
            events_by_t = np.zeros(N, dtype=bool)
            for i in range(N):
                if at_risk_mask[i]:
                    # Check if event occurred at or before time t
                    if t > 0:
                        # Check if event occurred in [0, t]
                        if (Y_np[i, disease_idx, :t+1] == 1).any():
                            events_by_t[i] = True
                    else:
                        # At time 0, check if event at time 0
                        if Y_np[i, disease_idx, 0] == 1:
                            events_by_t[i] = True
            
            n_events[t] = events_by_t.sum()
            cumulative_incidence[t] = n_events[t] / n_at_risk[0]  # Denominator is initial at-risk population
        else:
            cumulative_incidence[t] = cumulative_incidence[t-1] if t > 0 else 0.0
    
    return cumulative_incidence, n_at_risk, n_events


def calculate_cumulative_incidence_predicted(pi, E_corrected, disease_idx, timepoint_ages):
    """
    Calculate predicted cumulative incidence for a disease.
    
    Uses predicted probabilities (pi) and accounts for censoring.
    Cumulative incidence is the cumulative probability of experiencing the event.
    """
    N, D, T = pi.shape
    
    # Convert to numpy
    if torch.is_tensor(pi):
        pi_np = pi.numpy()
    else:
        pi_np = pi
    
    if torch.is_tensor(E_corrected):
        E_np = E_corrected.numpy()
    else:
        E_np = E_corrected
    
    # Initialize arrays
    cumulative_incidence = np.zeros(T)
    n_at_risk = np.zeros(T)
    
    # Get initial at-risk population (at time 0)
    initial_at_risk_mask = E_np[:, disease_idx] >= 0
    n_initial = initial_at_risk_mask.sum()
    
    if n_initial == 0:
        return cumulative_incidence, n_at_risk
    
    # Calculate cumulative incidence at each timepoint
    for t in range(T):
        # Number at risk at time t: patients with E >= t
        at_risk_mask = E_np[:, disease_idx] >= t
        n_at_risk[t] = at_risk_mask.sum()
        
        if n_at_risk[t] > 0:
            # For each patient in the initial population, calculate their cumulative probability
            # of having the event by time t, accounting for their censoring time
            cumulative_probs = np.zeros(N)
            
            for i in range(N):
                if initial_at_risk_mask[i]:
                    # Patient's censoring time for this disease
                    censor_time = E_np[i, disease_idx]
                    
                    # Only consider timepoints up to min(t, censor_time)
                    max_time = min(t + 1, censor_time + 1)
                    
                    if max_time > 0:
                        # Calculate cumulative probability: 1 - product(1 - pi[s]) for s=0 to max_time-1
                        pi_values = pi_np[i, disease_idx, :max_time]
                        # Avoid numerical issues with very small probabilities
                        pi_values = np.clip(pi_values, 1e-10, 1 - 1e-10)
                        cumulative_probs[i] = 1 - np.prod(1 - pi_values)
                    else:
                        cumulative_probs[i] = 0.0
            
            # Average over initial at-risk population (not just those still at risk at time t)
            cumulative_incidence[t] = cumulative_probs[initial_at_risk_mask].mean()
        else:
            cumulative_incidence[t] = cumulative_incidence[t-1] if t > 0 else 0.0
    
    return cumulative_incidence, n_at_risk


def plot_cumulative_incidence_curves(Y, pi, E_corrected, disease_names, disease_indices, 
                                     timepoint_ages, output_path=None):
    """
    Plot cumulative incidence curves for specified diseases.
    
    Args:
        Y: Observed disease tensor (N, D, T)
        pi: Predicted probability tensor (N, D, T)
        E_corrected: Corrected E matrix (N, D)
        disease_names: List of disease names
        disease_indices: List of disease indices to plot
        timepoint_ages: Array of ages corresponding to timepoints
        output_path: Path to save figure
    """
    n_diseases = len(disease_indices)
    
    # Create subplots
    fig, axes = plt.subplots(n_diseases, 1, figsize=(12, 4 * n_diseases))
    if n_diseases == 1:
        axes = [axes]
    
    for idx, disease_idx in enumerate(disease_indices):
        ax = axes[idx]
        
        # Calculate observed cumulative incidence
        ci_observed, n_at_risk_obs, n_events = calculate_cumulative_incidence_observed(
            Y, E_corrected, disease_idx, timepoint_ages
        )
        
        # Calculate predicted cumulative incidence
        ci_predicted, n_at_risk_pred = calculate_cumulative_incidence_predicted(
            pi, E_corrected, disease_idx, timepoint_ages
        )
        
        # Plot
        ax.plot(timepoint_ages, ci_observed, 'o-', color='#1f77b4', 
                label='Observed', linewidth=2, markersize=4, alpha=0.8)
        ax.plot(timepoint_ages, ci_predicted, 's-', color='#ff7f0e', 
                label='Predicted', linewidth=2, markersize=4, alpha=0.8)
        
        # Formatting
        disease_name = disease_names[disease_idx] if disease_names else f'Disease {disease_idx}'
        ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Incidence', fontsize=12, fontweight='bold')
        ax.set_title(f'{disease_name}\nObserved vs Predicted Cumulative Incidence', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        
        # Add text with summary statistics
        max_ci_obs = ci_observed.max()
        max_ci_pred = ci_predicted.max()
        stats_text = f'Max Observed CI: {max_ci_obs:.4f}\nMax Predicted CI: {max_ci_pred:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot cumulative incidence curves')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/',
                       help='Directory containing Y_tensor.pt, censor_info.csv, E_matrix_corrected.pt')
    parser.add_argument('--pi_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt',
                       help='Path to predicted pi tensor')
    parser.add_argument('--diseases', type=str, default=None,
                       help='Comma-separated list of disease indices or names (e.g., "0,1,2" or "ASCVD,Diabetes")')
    parser.add_argument('--output_path', type=str,
                       default='/Users/sarahurbut/aladynoulli2/claudefile/cumulative_incidence_curves.pdf',
                       help='Output path for figure')
    parser.add_argument('--max_diseases', type=int, default=5,
                       help='Maximum number of diseases to plot if --diseases not specified')
    args = parser.parse_args()
    
    # Load data
    Y, censor_df, pi, E_corrected, disease_names = load_data(args.data_dir, args.pi_path)
    
    # Determine which diseases to plot
    if args.diseases:
        # Parse disease list
        disease_list = [d.strip() for d in args.diseases.split(',')]
        disease_indices = []
        for d in disease_list:
            try:
                # Try as index
                idx = int(d)
                disease_indices.append(idx)
            except ValueError:
                # Try as name
                if disease_names is not None:
                    try:
                        idx = list(disease_names).index(d)
                        disease_indices.append(idx)
                    except ValueError:
                        print(f"Warning: Disease '{d}' not found, skipping...")
    else:
        # Plot first few diseases with events
        N, D, T = Y.shape
        # Find diseases with at least some events
        disease_event_counts = []
        for d in range(min(D, 100)):  # Check first 100 diseases
            event_count = (Y[:, d, :] == 1).sum().item()
            disease_event_counts.append((d, event_count))
        
        # Sort by event count and take top diseases
        disease_event_counts.sort(key=lambda x: x[1], reverse=True)
        disease_indices = [d for d, _ in disease_event_counts[:args.max_diseases]]
        print(f"\nPlotting top {len(disease_indices)} diseases by event count...")
    
    # Create timepoint ages (assuming timepoint 0 = age 30)
    T = Y.shape[2]
    timepoint_ages = np.arange(T) + 30
    
    # Plot
    print(f"\nPlotting cumulative incidence curves for {len(disease_indices)} diseases...")
    fig = plot_cumulative_incidence_curves(
        Y, pi, E_corrected, disease_names, disease_indices, 
        timepoint_ages, args.output_path
    )
    
    print("\n✓ Complete!")
    plt.show()


if __name__ == '__main__':
    main()

