import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import os
import glob

# Add path for model import
sys.path.insert(0, str(Path(__file__).parent.parent / 'claudefile' / 'aws_offsetmaster'))
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma import AladynSurvivalFixedPhiFixedGamma

# Set page config
st.set_page_config(layout="wide", page_title="Real-Time Patient Refitting")

@st.cache_data
def load_fixed_components(data_dir, master_checkpoint_name="master_for_fitting_pooled_correctedE_nolr.pt"):
    """Load fixed components (phi, psi, gamma) from master checkpoint
    
    Args:
        data_dir: Directory containing master checkpoint and essentials
        master_checkpoint_name: Name of the master checkpoint file (default: _nolr version with pooled gamma)
    """
    checkpoint_path = os.path.join(data_dir, master_checkpoint_name)
    essentials_path = os.path.join(data_dir, 'model_essentials.pt')
    refs_path = os.path.join(data_dir, 'reference_trajectories.pt')
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Master checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    phi = checkpoint['model_state_dict']['phi'].cpu().numpy()
    psi = checkpoint['model_state_dict']['psi'].cpu().numpy()
    
    # Load gamma from master checkpoint (should be there for _nolr version)
    gamma = None
    if 'gamma' in checkpoint.get('model_state_dict', {}):
        gamma = checkpoint['model_state_dict']['gamma'].cpu().numpy()
        print(f"✓ Loaded gamma from master checkpoint: {checkpoint_path}")
        print(f"  Gamma shape: {gamma.shape}")
        if 'n_batches' in checkpoint:
            print(f"  Pooled from {checkpoint['n_batches']} batches")
    elif 'gamma' in checkpoint:
        gamma = checkpoint['gamma']
        if torch.is_tensor(gamma):
            gamma = gamma.cpu().numpy()
        print(f"✓ Loaded gamma from master checkpoint: {checkpoint_path}")
        print(f"  Gamma shape: {gamma.shape}")
    
    if gamma is None:
        raise ValueError(f"Gamma not found in master checkpoint: {checkpoint_path}. Please ensure you're using the _nolr version that contains pooled gamma.")
    
    # Load essentials
    essentials = torch.load(essentials_path, map_location='cpu', weights_only=False)
    
    # Load signature refs
    refs = torch.load(refs_path, map_location='cpu', weights_only=False)
    signature_refs = refs['signature_refs']
    
    # Load corrected prevalence (computed with corrected E_corrected)
    prevalence_corrected_path = os.path.join(data_dir, 'prevalence_t_corrected.pt')
    if os.path.exists(prevalence_corrected_path):
        print(f"Loading corrected prevalence from: {prevalence_corrected_path}")
        prevalence_t = torch.load(prevalence_corrected_path, map_location='cpu', weights_only=False)
        if torch.is_tensor(prevalence_t):
            prevalence_t = prevalence_t.cpu().numpy()
        print(f"  Loaded corrected prevalence shape: {prevalence_t.shape}")
    else:
        print(f"Warning: Corrected prevalence not found at {prevalence_corrected_path}")
        print(f"  Falling back to prevalence_t from model_essentials.pt")
        prevalence_t = essentials['prevalence_t']
        if torch.is_tensor(prevalence_t):
            prevalence_t = prevalence_t.cpu().numpy()
    
    return {
        'phi': phi,
        'psi': psi,
        'gamma': gamma,  # Pooled gamma from _nolr batches (unshrunken)
        'prevalence_t': prevalence_t,
        'signature_refs': signature_refs,
        'disease_names': essentials['disease_names']
    }

def find_interesting_patients(Y_full, G_full=None, min_diseases=5, min_time_spread=8, max_patients_to_check=5000, top_n=100, prioritize_high_prs=True):
    """
    Find patients with many diagnoses and signature shifts
    
    Criteria:
    - Many diagnoses (>= min_diseases)
    - Diagnoses spread over time (>= min_time_spread years)
    - Multiple signature transitions
    - Optionally prioritize high PRS patients
    """
    N = min(Y_full.shape[0], max_patients_to_check)
    D, T = Y_full.shape[1], Y_full.shape[2]
    
    # Compute PRS magnitude if G is provided
    prs_magnitudes = None
    if G_full is not None and prioritize_high_prs:
        # Use first 36 components (PRS only, before sex/PCs)
        P_prs = min(36, G_full.shape[1])
        prs_values = G_full[:N, :P_prs] if len(G_full.shape) > 1 else G_full[:N]
        prs_magnitudes = np.linalg.norm(prs_values, axis=1)  # [N] - magnitude of PRS vector
    
    patient_scores = []
    
    for n in range(N):
        # Count diagnoses
        n_diagnoses = 0
        diagnosis_times = []
        for d in range(D):
            diag_times = np.where(Y_full[n, d, :] > 0.5)[0]
            if len(diag_times) > 0:
                n_diagnoses += 1
                diagnosis_times.append(int(diag_times[0]))
        
        if n_diagnoses < min_diseases:
            continue
        
        if len(diagnosis_times) < 2:
            continue
        
        time_spread = max(diagnosis_times) - min(diagnosis_times)
        if time_spread < min_time_spread:
            continue
        
        # Score: prioritize more diseases, more time spread, earlier start (more data)
        score = (
            n_diagnoses * 2.0 +  # More diseases = better
            time_spread * 0.5 +   # More spread = better
            (T - max(diagnosis_times)) * 0.3  # More follow-up after last diagnosis = better
        )
        
        # Add PRS bonus if available
        prs_bonus = 0.0
        if prs_magnitudes is not None:
            # Normalize PRS magnitude (percentile-based bonus)
            prs_percentile = (prs_magnitudes[n] - prs_magnitudes.min()) / (prs_magnitudes.max() - prs_magnitudes.min() + 1e-10)
            prs_bonus = prs_percentile * 5.0  # Up to 5x bonus for high PRS
            score += prs_bonus
        
        patient_scores.append({
            'patient_id': n,
            'score': score,
            'n_diagnoses': n_diagnoses,
            'time_spread': time_spread,
            'first_diag': min(diagnosis_times),
            'last_diag': max(diagnosis_times),
            'prs_magnitude': prs_magnitudes[n] if prs_magnitudes is not None else None
        })
    
    # Sort by score descending
    patient_scores.sort(key=lambda x: x['score'], reverse=True)
    
    return patient_scores[:top_n]

@st.cache_data
def load_sample_patients(data_dir, n_patients=50, find_interesting=True):
    """Load sample patients, optionally finding those with many diagnoses and signature shifts"""
    Y_path = os.path.join(data_dir, 'Y_tensor.pt')
    G_path = os.path.join(data_dir, 'G_matrix.pt')
    
    Y_full = torch.load(Y_path, map_location='cpu', weights_only=False)
    G_full = torch.load(G_path, map_location='cpu', weights_only=False)
    
    # Try to load E_corrected from full retrospective data
    E_corrected = None
    possible_E_paths = [
        os.path.join(data_dir, 'E_matrix_corrected.pt'),
        os.path.join(data_dir, 'E_corrected.pt'),
        os.path.join(data_dir, 'aou_E_corrected.pt'),
    ]
    
    for E_path in possible_E_paths:
        if os.path.exists(E_path):
            print(f"Loading E_corrected (full retrospective) from: {E_path}")
            E_corrected = torch.load(E_path, map_location='cpu', weights_only=False)
            if torch.is_tensor(E_corrected):
                E_corrected = E_corrected.numpy()
            print(f"  Loaded E_corrected shape: {E_corrected.shape}")
            break
    
    if E_corrected is None:
        print("Warning: E_corrected (full retrospective) not found. Will create E from Y (may be inaccurate for censoring).")
    
    # Find interesting patients if requested
    if find_interesting:
        print(f"Searching for interesting patients (many diagnoses, signature shifts, high PRS)...")
        Y_np = Y_full.numpy() if torch.is_tensor(Y_full) else Y_full
        G_np = G_full.numpy() if torch.is_tensor(G_full) else G_full
        interesting_patients = find_interesting_patients(
            Y_np,
            G_full=G_np,
            min_diseases=5, 
            min_time_spread=8, 
            max_patients_to_check=min(5000, Y_np.shape[0]),
            top_n=n_patients,
            prioritize_high_prs=True
        )
        
        if interesting_patients:
            patient_indices = [p['patient_id'] for p in interesting_patients]
            print(f"Found {len(interesting_patients)} interesting patients:")
            for i, p in enumerate(interesting_patients[:5]):  # Show first 5
                print(f"  Patient {p['patient_id']}: {p['n_diagnoses']} diagnoses, {p['time_spread']} year spread")
            
            Y_sample = Y_full[patient_indices].numpy() if torch.is_tensor(Y_full) else Y_full[patient_indices]
            G_sample = G_full[patient_indices].numpy() if torch.is_tensor(G_full) else G_full[patient_indices]
            if E_corrected is not None:
                E_sample = E_corrected[patient_indices]
            else:
                E_sample = None
            
            return Y_sample, G_sample, E_sample, interesting_patients
        else:
            print("No interesting patients found, using first n_patients")
    
    # Fallback: take first n_patients
    Y_sample = Y_full[:n_patients].numpy() if torch.is_tensor(Y_full) else Y_full[:n_patients]
    G_sample = G_full[:n_patients].numpy() if torch.is_tensor(G_full) else G_full[:n_patients]
    if E_corrected is not None:
        E_sample = E_corrected[:n_patients]
        if torch.is_tensor(E_sample):
            E_sample = E_sample.numpy()
    else:
        E_sample = None
    
    return Y_sample, G_sample, E_sample, None

def fit_patient_model(patient_G, patient_Y, patient_E, fixed_components, num_epochs=30, learning_rate=0.1):
    """Fit model for a single patient in real time using fixed-gamma model"""
    # Check that gamma is available
    if fixed_components.get('gamma') is None:
        raise ValueError("Gamma must be provided for fixed-gamma model.")
    
    # Get expected P from gamma shape
    gamma = fixed_components['gamma']
    expected_P = gamma.shape[0]  # Gamma is [P, K_total]
    
    # Handle both 2D [D, T] and 3D [N, D, T] inputs - use N=1 for single patient
    if len(patient_Y.shape) == 2:
        D, T = patient_Y.shape
        patient_Y = patient_Y[np.newaxis, :, :]  # [1, D, T]
    else:
        patient_Y = patient_Y[0:1, :, :]  # [1, D, T]
    
    D, T = patient_Y.shape[1], patient_Y.shape[2]
    
    # Handle patient_G: use N=1 and pad if necessary
    if len(patient_G.shape) == 1:
        P_input = patient_G.shape[0]
        patient_G = patient_G[np.newaxis, :]  # [1, P_input]
    else:
        P_input = patient_G.shape[1]
        patient_G = patient_G[0:1, :]  # [1, P_input]
    
    # Pad patient_G if it has fewer components than expected (e.g., missing PCs)
    if P_input < expected_P:
        print(f"Warning: Patient G has {P_input} components but gamma expects {expected_P}. Padding with zeros for missing components (likely PCs).")
        padding = np.zeros((1, expected_P - P_input))
        patient_G = np.concatenate([patient_G, padding], axis=1)  # [1, expected_P]
        P = expected_P
    elif P_input > expected_P:
        raise ValueError(f"Patient G has {P_input} components but gamma expects {expected_P}. Cannot subset gamma.")
    else:
        P = P_input
    
    # Handle event times: use N=1
    if len(patient_E.shape) == 1:
        patient_E = patient_E[np.newaxis, :]  # [1, D]
    else:
        patient_E = patient_E[0:1, :]  # [1, D]
    
    # Create model with N=1 (single patient) using fixed-gamma version
    model = AladynSurvivalFixedPhiFixedGamma(
        N=1,
        D=D,
        T=T,
        K=20,
        P=P,
        G=patient_G,
        Y=patient_Y,
        R=0,
        W=0.0001,
        prevalence_t=fixed_components['prevalence_t'],
        init_sd_scaler=0.1,
        genetic_scale=1.0,
        pretrained_phi=fixed_components['phi'],
        pretrained_psi=fixed_components['psi'],
        pretrained_gamma=fixed_components['gamma'],  # Fixed gamma from pooled _nolr batches
        signature_references=fixed_components['signature_refs'],
        healthy_reference=True,
        disease_names=fixed_components['disease_names']
    )
    
    # Suppress print statements during training
    import io
    import contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        # Fast training (only lambda is trained, gamma is fixed)
        losses, _ = model.fit(patient_E, num_epochs=num_epochs, learning_rate=learning_rate)
    
    # Get predictions - single patient (N=1)
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Return results (already N=1, so no need to index)
    return pi[0].cpu().numpy(), theta[0].cpu().numpy(), model, losses

def create_event_times_from_Y(Y):
    """Create event times from Y matrix (disease codes over time)"""
    if len(Y.shape) == 3:
        Y = Y[0]  # Take first patient if 3D
    
    D, T = Y.shape
    event_times = np.full(D, T - 1)  # Default to censoring at end
    
    for d in range(D):
        times = np.where(Y[d, :] > 0.5)[0]
        if len(times) > 0:
            event_times[d] = times[0]
    
    return event_times

def get_diagnosis_progression(Y, disease_names, age_offset=30):
    """Create a narrative of diagnosis progression over time"""
    D, T = Y.shape
    progression = []
    
    # Collect all diagnosis events
    events = []
    for d in range(D):
        diag_times = np.where(Y[d, :] > 0.5)[0]
        for t in diag_times:
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            events.append({
                'time': int(t),
                'age': age_offset + int(t),
                'disease': disease_name,
                'disease_idx': d
            })
    
    # Sort by time
    events.sort(key=lambda x: x['time'])
    
    # Create narrative
    if len(events) == 0:
        return "No diagnoses recorded yet."
    
    narrative_parts = []
    for i, event in enumerate(events):
        if i == 0:
            narrative_parts.append(f"**At age {event['age']}:** {event['disease']} was diagnosed.")
        else:
            narrative_parts.append(f"**At age {event['age']}:** {event['disease']} appeared.")
    
    return "\n\n".join(narrative_parts)

def load_cluster_assignments(data_dir):
    """Load disease-to-signature cluster assignments from initial_clusters_400k.pt"""
    clusters_path = Path(data_dir) / 'initial_clusters_400k.pt'
    
    if not clusters_path.exists():
        alt_paths = [
            Path(data_dir) / 'initial_clusters.pt',
            Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt')
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                clusters_path = alt_path
                break
    
    if not clusters_path.exists():
        return None
    
    try:
        clusters = torch.load(clusters_path, map_location='cpu', weights_only=False)
        
        if isinstance(clusters, torch.Tensor):
            cluster_assignments = clusters.cpu().numpy()
        elif isinstance(clusters, dict):
            if 'clusters' in clusters:
                cluster_assignments = clusters['clusters']
                if isinstance(cluster_assignments, torch.Tensor):
                    cluster_assignments = cluster_assignments.cpu().numpy()
            elif 'initial_clusters' in clusters:
                cluster_assignments = clusters['initial_clusters']
                if isinstance(cluster_assignments, torch.Tensor):
                    cluster_assignments = cluster_assignments.cpu().numpy()
            else:
                return None
        elif isinstance(clusters, np.ndarray):
            cluster_assignments = clusters
        else:
            return None
        
        return cluster_assignments
    except Exception as e:
        print(f"Warning: Could not load cluster assignments: {e}")
        return None

def plot_predictions(pi, theta, Y, disease_names, time_window=None, age_offset=30, cluster_assignments=None, 
                     max_diseases_to_show=15, min_prob_threshold=0.001):
    """Plot disease probabilities and signature proportions with signature-based colors
    
    Args:
        max_diseases_to_show: Maximum number of diseases to display (prevents overcrowding)
        min_prob_threshold: Minimum probability threshold to show a disease
    """
    D, T = pi.shape
    if time_window is None:
        time_window = range(T)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Get signature colors
    K = theta.shape[0]
    sig_colors = sns.color_palette("tab20", K)
    
    # Plot 1: Disease probabilities
    ax1 = axes[0]
    
    # Get diseases that occurred
    diseases_with_events = []
    for d in range(D):
        if np.any(Y[d, :] > 0.5):
            diseases_with_events.append(d)
    
    # Smart filtering: prioritize diseases with events, then by max probability
    disease_scores = []
    for d in range(D):
        has_event = np.any(Y[d, :] > 0.5)
        max_prob = np.max(pi[d, :])
        mean_prob = np.mean(pi[d, :])
        
        # Score: events get priority, then by max probability
        score = (1000 if has_event else 0) + max_prob * 100 + mean_prob * 10
        
        if max_prob >= min_prob_threshold:  # Only consider diseases above threshold
            disease_scores.append((d, score, has_event, max_prob))
    
    # Sort by score and take top N
    disease_scores.sort(key=lambda x: x[1], reverse=True)
    diseases_to_plot = [d for d, _, _, _ in disease_scores[:max_diseases_to_show]]
    
    # Map diseases to signature colors
    disease_color_map = {}
    if cluster_assignments is not None and len(cluster_assignments) >= D:
        for d in diseases_to_plot:
            sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            if 0 <= sig < K:
                disease_color_map[d] = sig_colors[sig]
            else:
                disease_color_map[d] = 'gray'
    else:
        if len(diseases_to_plot) > 0:
            colors = sns.color_palette("husl", len(diseases_to_plot))
            disease_color_map = {d: colors[i] for i, d in enumerate(diseases_to_plot)}
    
    # Plot selected diseases
    for d in diseases_to_plot:
        disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
        color = disease_color_map.get(d, 'gray')
        
        # Get signature assignment for label
        sig_label = ""
        if cluster_assignments is not None and d < len(cluster_assignments):
            sig = int(cluster_assignments[d])
            sig_label = f" (Sig {sig})"
        
        # Truncate long disease names
        display_name = disease_name
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        # Plot probability curve
        ax1.plot(time_window, pi[d, time_window], 
                label=f"{display_name}{sig_label}",
                color=color, linewidth=2, alpha=0.7)
        
        # Mark diagnosis times
        diag_times = np.where(Y[d, time_window] > 0.5)[0]
        if len(diag_times) > 0:
            for t in diag_times:
                ax1.axvline(x=t, color=color, linestyle='--', alpha=0.5)
                ax1.scatter(t, pi[d, t], color=color, s=80, zorder=10, marker='o', edgecolors='black', linewidths=1)
    
    # Add note if diseases were filtered
    if len(disease_scores) > max_diseases_to_show:
        n_with_events = sum(1 for _, _, has_event, _ in disease_scores[:max_diseases_to_show] if has_event)
        ax1.text(0.98, 0.02, f'Showing top {max_diseases_to_show} of {len(disease_scores)} diseases\n({n_with_events} with events, sorted by max probability)',
                transform=ax1.transAxes, ha='right', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    elif len(disease_scores) > 0:
        n_with_events = sum(1 for _, _, has_event, _ in disease_scores if has_event)
        ax1.text(0.98, 0.02, f'Showing {len(disease_scores)} diseases ({n_with_events} with events)',
                transform=ax1.transAxes, ha='right', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    if len(diseases_to_plot) == 0:
        ax1.text(0.5, 0.5, 'No diseases meet the filtering criteria\n(Try lowering the probability threshold)', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='gray')
    
    ax1.set_title('Disease Probabilities Over Time (colored by primary signature)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (years from baseline)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signature proportions (show top signatures)
    ax2 = axes[1]
    
    # Find top signatures by average loading
    avg_theta = theta.mean(axis=1)
    top_sig_indices = np.argsort(avg_theta)[::-1][:min(15, K)]  # Show top 15 signatures
    
    # Plot top signatures
    for k in top_sig_indices:
        ax2.plot(time_window, theta[k, time_window], 
                label=f'Sig {k}', color=sig_colors[k], linewidth=2, alpha=0.7)
    
    # Plot remaining signatures as light gray
    remaining_sigs = [k for k in range(K) if k not in top_sig_indices]
    if len(remaining_sigs) > 0:
        for k in remaining_sigs:
            ax2.plot(time_window, theta[k, time_window], 
                    color='lightgray', linewidth=1, alpha=0.3)
    
    ax2.set_title(f'Signature Proportions Over Time (top {len(top_sig_indices)} shown, {len(remaining_sigs)} others in gray)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (years from baseline)', fontsize=12)
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_patient_timeline_comprehensive(pi, theta, Y, disease_names, age_offset=30, cluster_assignments=None, figsize=(20, 14)):
    """
    Create comprehensive multi-panel timeline visualization similar to plot_patient_timeline_v3.py
    
    Shows:
    - Panel 1: Signature loadings (θ) vs Age
    - Panel 2: Disease timeline with chronological ordering
    - Panel 2b: Disease details (two columns)
    - Panel 3: Disease probability trajectories (π) over time, stopping at diagnosis
    """
    D, T = pi.shape
    K = theta.shape[0]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    
    # Find diagnoses
    diagnosis_times = {}
    for d in range(D):
        event_times = np.where(Y[d, :] > 0.5)[0]
        if len(event_times) > 0:
            diagnosis_times[d] = event_times.tolist()
    
    n_diseases = len(diagnosis_times)
    all_times = []
    for times in diagnosis_times.values():
        all_times.extend(times)
    time_range = (min(all_times), max(all_times)) if all_times else (0, T-1)
    
    # Calculate average theta
    avg_theta = theta.mean(axis=1)  # Shape: (K,)
    
    # Get signature colors
    sig_colors = sns.color_palette("tab20", K)
    
    # Get cluster assignments for diseases
    if cluster_assignments is None:
        cluster_assignments = np.zeros(D, dtype=int)
    
    # Convert timepoints to ages
    ages = np.arange(age_offset, age_offset + T)
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = plt.GridSpec(3, 4, width_ratios=[1.5, 1.5, 1.2, 1.2], 
                      height_ratios=[2, 2.8, 2], 
                      hspace=0.35, wspace=0.25)
    
    # ============================================================================
    # Panel 1: Signature loadings (θ) vs Age
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    for k in range(K):
        ax1.plot(ages, theta[k, :], 
                 label=f'Signature {k}', linewidth=2.3, color=sig_colors[k], alpha=0.85)
    
    # Add vertical lines at diagnosis times
    for d, times in diagnosis_times.items():
        for t in times:
            if t >= T:
                continue
            age_at_diag = age_offset + t
            ax1.axvline(x=age_at_diag, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)
    
    # Add inset for average theta
    ax1_bar = ax1.inset_axes([0.84, 0.62, 0.14, 0.32])
    sorted_indices = np.argsort(avg_theta)[::-1]
    sorted_avg_theta = avg_theta[sorted_indices]
    sorted_colors = [sig_colors[i] for i in sorted_indices]
    
    bottom = 0
    for val, color in zip(sorted_avg_theta, sorted_colors):
        if val > 0.005:
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
    ax1.set_title('Signature Trajectories Over Time', fontsize=15, fontweight='bold', pad=10)
    ncols = min(4, (K + 3) // 4)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
               ncol=ncols, columnspacing=1.0, handlelength=1.8, handletextpad=0.5,
               framealpha=0.95, borderpad=0.4, borderaxespad=0.2)
    ax1.tick_params(labelsize=11)
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax1.set_xlim([age_offset, age_offset + T])
    ax1.set_ylim([0, max(theta.max() * 1.08, 0.5)])
    ax1.set_xlabel('Age (years)', fontsize=12)
    
    # ============================================================================
    # Panel 2: Disease timeline
    # ============================================================================
    ax2 = fig.add_subplot(gs[1, :2])
    
    if len(diagnosis_times) > 0:
        diag_order = sorted([(d, times[0]) for d, times in diagnosis_times.items()], 
                           key=lambda x: x[1])
        max_diseases_shown = min(30, len(diag_order))
        diag_order_shown = diag_order[:max_diseases_shown]
        
        for i, (d, t_diag) in enumerate(diag_order_shown):
            if t_diag >= T:
                continue
            sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            color = sig_colors[sig_for_disease] if 0 <= sig_for_disease < K else 'gray'
            age_at_diag = age_offset + t_diag
            y_pos = len(diag_order_shown) - i - 1
            
            ax2.plot([age_offset, age_at_diag], [y_pos, y_pos], color=color, linewidth=1, alpha=0.3)
            ax2.scatter(age_at_diag, y_pos, s=90, color=color, alpha=0.85, zorder=10, 
                       edgecolors='black', linewidths=1.2)
            ax2.text(age_offset - 1, y_pos, f'{i+1}', fontsize=8, fontweight='bold', 
                    verticalalignment='center', ha='right')
        
        ax2.set_yticks(range(len(diag_order_shown)))
        ax2.set_yticklabels([])
        ax2.set_ylim([-0.5, len(diag_order_shown) - 0.5])
        
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
    ax2.set_xlim([age_offset, age_offset + T])
    
    # ============================================================================
    # Panel 2b: Disease details (two columns)
    # ============================================================================
    ax2_legend1 = fig.add_subplot(gs[1, 2])
    ax2_legend2 = fig.add_subplot(gs[1, 3])
    ax2_legend1.axis('off')
    ax2_legend2.axis('off')
    
    if len(diagnosis_times) > 0:
        mid_point = (len(diag_order_shown) + 1) // 2
        first_column = diag_order_shown[:mid_point]
        second_column = diag_order_shown[mid_point:]
        
        legend_text1 = []
        for i, (d, t_diag) in enumerate(first_column):
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            
            max_len = 24
            if len(disease_name) > max_len:
                truncated = disease_name[:max_len-3]
                last_space = truncated.rfind(' ')
                if last_space > 12:
                    disease_name = truncated[:last_space] + '...'
                else:
                    disease_name = truncated + '...'
            
            age_at_diag = age_offset + t_diag
            legend_text1.append(f'{i+1:2d}. {disease_name[:24]:<24s}\n    Sig {sig_for_disease:2d}, Age {age_at_diag:2d}')
        
        legend_text2 = []
        for i, (d, t_diag) in enumerate(second_column, start=mid_point):
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            
            max_len = 24
            if len(disease_name) > max_len:
                truncated = disease_name[:max_len-3]
                last_space = truncated.rfind(' ')
                if last_space > 12:
                    disease_name = truncated[:last_space] + '...'
                else:
                    disease_name = truncated + '...'
            
            age_at_diag = age_offset + t_diag
            legend_text2.append(f'{i+1:2d}. {disease_name[:24]:<24s}\n    Sig {sig_for_disease:2d}, Age {age_at_diag:2d}')
        
        ax2_legend1.text(0.05, 0.98, 'Disease Details (1/2):', fontsize=10, fontweight='bold',
                        transform=ax2_legend1.transAxes, va='top')
        legend_str1 = '\n'.join(legend_text1)
        ax2_legend1.text(0.05, 0.93, legend_str1, fontsize=7,
                        transform=ax2_legend1.transAxes, va='top', 
                        fontfamily='monospace', linespacing=1.25)
        
        if second_column:
            ax2_legend2.text(0.05, 0.98, 'Disease Details (2/2):', fontsize=10, fontweight='bold',
                            transform=ax2_legend2.transAxes, va='top')
            legend_str2 = '\n'.join(legend_text2)
            ax2_legend2.text(0.05, 0.93, legend_str2, fontsize=7,
                            transform=ax2_legend2.transAxes, va='top', 
                            fontfamily='monospace', linespacing=1.25)
    
    # ============================================================================
    # Panel 3: Disease probabilities
    # ============================================================================
    ax3 = fig.add_subplot(gs[2, :])
    
    if len(diagnosis_times) > 0:
        # Get diseases with events
        diseases_with_events = list(diagnosis_times.keys())
        
        # Smart filtering: prioritize diseases with events, then by max probability
        disease_scores = []
        for d in range(D):
            has_event = d in diagnosis_times
            if d in diagnosis_times:
                first_diag_t = min(diagnosis_times[d])
                if first_diag_t >= T:
                    first_diag_t = T - 1
                max_prob = pi[d, :first_diag_t + 1].max()
            else:
                max_prob = pi[d, :].max()
            
            # Score: events get priority, then by max probability
            score = (1000 if has_event else 0) + max_prob * 100
            
            if max_prob > 0.0001:  # Only consider diseases with some probability
                disease_scores.append((d, score, has_event, max_prob))
        
        # Sort and take top N
        disease_scores.sort(key=lambda x: x[1], reverse=True)
        n_diseases_to_plot = min(20, len(disease_scores))
        top_diseases = [d for d, _, _, _ in disease_scores[:n_diseases_to_plot]]
        
        # Group by signature
        diseases_by_sig = {}
        for d in top_diseases:
            sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            if sig not in diseases_by_sig:
                diseases_by_sig[sig] = []
            diseases_by_sig[sig].append(d)
        
        plotted_count = 0
        for sig in sorted(diseases_by_sig.keys()):
            for d in diseases_by_sig[sig]:
                color = sig_colors[sig] if 0 <= sig < K else 'gray'
                
                if d in diagnosis_times:
                    first_diag_t = min(diagnosis_times[d])
                    if first_diag_t >= T:
                        first_diag_t = T - 1
                    plot_ages = ages[:first_diag_t + 1]
                    plot_pi = pi[d, :first_diag_t + 1]
                else:
                    plot_ages = ages
                    plot_pi = pi[d, :]
                
                linestyle = '-' if plotted_count % 3 == 0 else ('--' if plotted_count % 3 == 1 else '-.')
                ax3.plot(plot_ages, plot_pi, 
                         color=color, linewidth=1.8, alpha=0.7, linestyle=linestyle)
                
                if d in diagnosis_times:
                    for t in diagnosis_times[d]:
                        if t >= T:
                            continue
                        age_at_diag = age_offset + t
                        ax3.scatter(age_at_diag, pi[d, t], 
                                   color=color, s=80, zorder=10, marker='o', 
                                   edgecolors='black', linewidths=1.2, alpha=0.9)
                
                plotted_count += 1
        
        if len(disease_scores) > n_diseases_to_plot:
            ax3.text(0.98, 0.98, f'Top {n_diseases_to_plot} of {len(disease_scores)} diseases\n(prioritizing events + max probability)\nGrouped by signature, diagnoses marked with ●', 
                    transform=ax3.transAxes, ha='right', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        else:
            ax3.text(0.98, 0.98, f'Top {n_diseases_to_plot} diseases by max probability\nGrouped by signature, diagnoses marked with ●', 
                    transform=ax3.transAxes, ha='right', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Disease Probability (π)', fontsize=13, fontweight='bold')
    ax3.set_title('Disease Risk Trajectories (stopping at diagnosis)', fontsize=14, fontweight='bold', pad=8)
    ax3.tick_params(labelsize=11)
    ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax3.set_xlim([age_offset, age_offset + T])
    
    if len(diagnosis_times) > 0:
        y_max = min(0.1, ax3.get_ylim()[1] * 1.1)
        ax3.set_ylim([0, y_max])
    
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    
    # Add title
    fig.suptitle('Comprehensive Disease Trajectory Analysis', 
                 fontsize=17, fontweight='bold', y=0.98)
    subtitle = f'Total diseases: {n_diseases} | Age range: {age_offset+time_range[0]}-{age_offset+time_range[1]} | Signatures: {K}'
    fig.text(0.5, 0.95, subtitle, ha='center', fontsize=11, style='italic', color='#666666')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def pad_G_to_match_gamma(patient_G, gamma):
    """Pad patient_G to match gamma's expected P dimension (handles missing PCs/sex)."""
    expected_P = gamma.shape[0]  # Gamma is [P, K_total]
    
    G_flat = np.array(patient_G).flatten() if isinstance(patient_G, (list, np.ndarray)) else patient_G
    if len(G_flat.shape) > 1:
        G_flat = G_flat[0] if G_flat.shape[0] == 1 else G_flat.flatten()
    
    P_input = len(G_flat)
    
    if P_input < expected_P:
        padding = np.zeros(expected_P - P_input)
        G_flat = np.concatenate([G_flat, padding])
    elif P_input > expected_P:
        raise ValueError(f"Patient G has {P_input} components but gamma expects {expected_P}.")
    
    return G_flat

def diagnose_genetic_penalty(model, fixed_components, sample_G=None):
    """Diagnose whether the genetic penalty W is appropriate."""
    gamma = fixed_components['gamma']
    if gamma is None:
        return {
            'status': 'error',
            'message': 'Gamma not available for diagnosis'
        }
    
    gamma_np = gamma if isinstance(gamma, np.ndarray) else gamma.cpu().numpy()
    P, K = gamma_np.shape
    
    # 1. Gamma magnitude statistics
    gamma_abs = np.abs(gamma_np)
    gamma_mean_abs = np.mean(gamma_abs)
    gamma_std_abs = np.std(gamma_abs)
    gamma_max_abs = np.max(gamma_abs)
    gamma_percentiles = np.percentile(gamma_abs, [25, 50, 75, 95, 99])
    
    # 2. Estimate GP prior variance
    lambda_amplitude = model.lambda_amplitude if hasattr(model, 'lambda_amplitude') else 1.0
    gp_prior_var = lambda_amplitude ** 2
    
    # 3. Genetic effect scale
    typical_genetic_effect_std = np.sqrt(np.sum(gamma_np**2, axis=0))  # [K] - per signature
    
    # 4. Signal-to-noise ratio
    snr_per_signature = typical_genetic_effect_std / np.sqrt(gp_prior_var)
    
    # 5. Counterfactual effect assessment
    counterfactual_effects = None
    if sample_G is not None:
        G_padded = pad_G_to_match_gamma(sample_G, gamma)
        genetic_effects = model.genetic_scale * (G_padded @ gamma_np)  # [K]
        counterfactual_effects = {
            'mean_abs': np.mean(np.abs(genetic_effects)),
            'max_abs': np.max(np.abs(genetic_effects)),
            'std': np.std(genetic_effects),
            'per_signature': genetic_effects
        }
    
    # 6. Recommendations
    recommendations = []
    warnings = []
    
    # Check if gamma values are too small
    if gamma_max_abs < 0.001:
        warnings.append("⚠️ **W may be too high**: Maximum |gamma| < 0.001. Genetic effects are very small.")
        recommendations.append("Consider reducing W (e.g., try W/10) to allow larger genetic effects.")
    elif gamma_max_abs < 0.01:
        warnings.append("⚠️ **W may be moderately high**: Maximum |gamma| < 0.01. Genetic effects are small.")
        recommendations.append("Consider reducing W (e.g., try W/2) to allow larger genetic effects.")
    
    # Check if gamma values are too large
    if gamma_max_abs > 1.0:
        warnings.append("⚠️ **W may be too low**: Maximum |gamma| > 1.0. Genetic effects are very large.")
        recommendations.append("Consider increasing W (e.g., try W*10) to regularize genetic effects.")
    elif gamma_max_abs > 0.5:
        warnings.append("⚠️ **W may be moderately low**: Maximum |gamma| > 0.5. Genetic effects are large.")
        recommendations.append("Consider increasing W (e.g., try W*2) to regularize genetic effects.")
    
    # Check SNR
    mean_snr = np.mean(snr_per_signature)
    if mean_snr < 0.1:
        warnings.append("⚠️ **Low signal-to-noise**: Genetic effects are <10% of GP prior std. Effects may be negligible.")
        recommendations.append("Consider reducing W to increase genetic signal strength.")
    elif mean_snr > 2.0:
        warnings.append("⚠️ **High signal-to-noise**: Genetic effects are >2x GP prior std. May be overfitting.")
        recommendations.append("Consider increasing W to prevent overfitting.")
    
    # Check counterfactual effects
    if counterfactual_effects is not None:
        cf_max = counterfactual_effects['max_abs']
        if cf_max < 0.01:
            warnings.append("⚠️ **Counterfactual effects negligible**: Max genetic effect < 0.01. PRS=0 vs actual makes little difference.")
            recommendations.append("This suggests W may be too high - genetic effects are too small to be meaningful.")
        elif cf_max > 1.0:
            warnings.append("⚠️ **Counterfactual effects very large**: Max genetic effect > 1.0. PRS dominates predictions.")
            recommendations.append("This suggests W may be too low - genetic effects may be overfitting.")
    
    # Overall assessment
    if len(warnings) == 0:
        status = 'good'
        status_message = "✓ **W appears appropriate**: Genetic effects are in a reasonable range."
    elif any('too high' in w for w in warnings):
        status = 'too_high'
        status_message = "🔴 **W likely too high**: Genetic effects are being over-penalized."
    elif any('too low' in w for w in warnings):
        status = 'too_low'
        status_message = "🟡 **W likely too low**: Genetic effects may be overfitting."
    else:
        status = 'moderate'
        status_message = "🟠 **W may need adjustment**: Some concerns about genetic effect scale."
    
    return {
        'status': status,
        'status_message': status_message,
        'gamma_stats': {
            'mean_abs': gamma_mean_abs,
            'std_abs': gamma_std_abs,
            'max_abs': gamma_max_abs,
            'percentiles': gamma_percentiles,
            'shape': (P, K)
        },
        'gp_prior_var': gp_prior_var,
        'lambda_amplitude': lambda_amplitude,
        'typical_genetic_effect_std': typical_genetic_effect_std,
        'snr_per_signature': snr_per_signature,
        'mean_snr': mean_snr,
        'counterfactual_effects': counterfactual_effects,
        'warnings': warnings,
        'recommendations': recommendations
    }

def plot_genetic_penalty_diagnostics(diagnosis):
    """Plot diagnostic visualizations for genetic penalty assessment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Gamma distribution
    ax1 = axes[0, 0]
    gamma_abs = diagnosis['gamma_stats']['percentiles']
    ax1.barh(['P25', 'P50', 'P75', 'P95', 'P99'], gamma_abs, color='steelblue', alpha=0.7)
    ax1.axvline(x=0.001, color='red', linestyle='--', alpha=0.5, label='Very Small')
    ax1.axvline(x=0.01, color='orange', linestyle='--', alpha=0.5, label='Small')
    ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Moderate')
    ax1.set_xlabel('|Gamma| Value')
    ax1.set_title('Gamma Magnitude Distribution', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. SNR per signature
    ax2 = axes[0, 1]
    snr = diagnosis['snr_per_signature']
    ax2.bar(range(len(snr)), snr, color='coral', alpha=0.7)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Low (0.1)')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good (1.0)')
    ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='High (2.0)')
    ax2.set_xlabel('Signature Index')
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_title('Genetic Effect SNR per Signature', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Counterfactual effects
    ax3 = axes[1, 0]
    if diagnosis['counterfactual_effects'] is not None:
        cf_effects = diagnosis['counterfactual_effects']['per_signature']
        colors = ['red' if abs(e) < 0.01 else 'orange' if abs(e) < 0.1 else 'green' for e in cf_effects]
        ax3.bar(range(len(cf_effects)), cf_effects, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Negligible')
        ax3.axhline(y=-0.01, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Signature Index')
        ax3.set_ylabel('Genetic Effect (PRS contribution)')
        ax3.set_title('Counterfactual Genetic Effects', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No sample patient data\nfor counterfactual analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Counterfactual Genetic Effects', fontweight='bold')
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    **Gamma Statistics:**
    Mean |γ|: {diagnosis['gamma_stats']['mean_abs']:.6f}
    Std |γ|: {diagnosis['gamma_stats']['std_abs']:.6f}
    Max |γ|: {diagnosis['gamma_stats']['max_abs']:.6f}
    
    **GP Prior:**
    Lambda Amplitude: {diagnosis['lambda_amplitude']:.4f}
    GP Prior Variance: {diagnosis['gp_prior_var']:.4f}
    
    **Signal-to-Noise:**
    Mean SNR: {diagnosis['mean_snr']:.4f}
    Min SNR: {np.min(diagnosis['snr_per_signature']):.4f}
    Max SNR: {np.max(diagnosis['snr_per_signature']):.4f}
    
    **Assessment:**
    {diagnosis['status_message']}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='center', family='monospace')
    
    plt.suptitle('Genetic Penalty (W) Diagnostic Assessment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_counterfactual_signature_trajectory(model, signature_idx, patient_G, time_window=None):
    """Plot actual vs counterfactual signature trajectory (PRS=0) for a patient."""
    if time_window is None:
        time_window = range(model.T)
    
    # Get actual lambda and theta
    lambda_actual = model.lambda_[0, signature_idx, :].detach().cpu().numpy()  # [T]
    theta_actual = torch.softmax(model.lambda_[0], dim=0)[signature_idx, :].detach().cpu().numpy()  # [T]
    
    # Compute counterfactual: remove genetic effect
    G_padded = pad_G_to_match_gamma(patient_G, model.gamma)
    gamma_k = model.gamma[:, signature_idx].detach().cpu().numpy()
    genetic_effect = float(model.genetic_scale * np.dot(G_padded, gamma_k))
    
    print(f"  Genetic effect for signature {signature_idx}: {genetic_effect:.4f}")
    print(f"  G_padded range: [{G_padded.min():.4f}, {G_padded.max():.4f}]")
    print(f"  gamma_k range: [{gamma_k.min():.4f}, {gamma_k.max():.4f}]")
    
    lambda_cf = lambda_actual - genetic_effect
    
    # Recompute theta for counterfactual
    lambda_all_cf = model.lambda_[0].detach().cpu().numpy().copy()  # [K_total, T]
    lambda_all_cf[signature_idx, :] = lambda_cf
    theta_cf = torch.softmax(torch.tensor(lambda_all_cf), dim=0)[signature_idx, :].numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot lambda trajectories
    ax1.plot(time_window, lambda_actual[time_window], label="Actual λ", linewidth=2, color='#e74c3c')
    ax1.plot(time_window, lambda_cf[time_window], label="Counterfactual λ (PRS=0)", 
             linestyle='--', linewidth=2, color='#3498db')
    # Handle signature_refs
    ref_val = model.signature_refs[signature_idx]
    if isinstance(ref_val, torch.Tensor):
        ref_val = ref_val.detach().cpu().numpy()
        if ref_val.ndim > 0:
            ref_val = float(ref_val.mean() if len(ref_val) > 1 else ref_val[0])
        else:
            ref_val = float(ref_val.item())
    else:
        ref_val = float(ref_val)
    ax1.axhline(y=ref_val, color='gray', 
                linestyle=':', alpha=0.5, label='Reference')
    ax1.set_title(f'Signature {signature_idx} Lambda Trajectory: Actual vs Counterfactual', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Lambda Value', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot theta (signature proportions)
    ax2.plot(time_window, theta_actual[time_window], label="Actual θ", linewidth=2, color='#e74c3c')
    ax2.plot(time_window, theta_cf[time_window], label="Counterfactual θ (PRS=0)", 
             linestyle='--', linewidth=2, color='#3498db')
    ax2.set_title(f'Signature {signature_idx} Proportion: Actual vs Counterfactual', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Signature Proportion', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_gp_projection(model, signature_idx, patient_G, time_window=None):
    """Plot patient's lambda trajectory vs GP prior mean (reference + genetic effect)."""
    if time_window is None:
        time_window = range(model.T)
    
    # Patient's actual lambda
    lambda_patient = model.lambda_[0, signature_idx, :].detach().cpu().numpy()  # [T]
    
    # GP prior mean: reference + genetic effect
    ref_val = model.signature_refs[signature_idx]
    if isinstance(ref_val, torch.Tensor):
        ref_val = ref_val.detach().cpu().numpy()
        if ref_val.ndim > 0:
            reference = float(ref_val.mean() if len(ref_val) > 1 else ref_val[0])
        else:
            reference = float(ref_val.item())
    else:
        reference = float(ref_val)
    
    G_padded = pad_G_to_match_gamma(patient_G, model.gamma)
    gamma_k = model.gamma[:, signature_idx].detach().cpu().numpy()
    genetic_effect = float(model.genetic_scale * np.dot(G_padded, gamma_k))
    gp_mean = reference + genetic_effect  # Constant over time
    
    # Compute deviation from GP mean
    deviation = lambda_patient - gp_mean
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot lambda vs GP mean
    ax1.plot(time_window, lambda_patient[time_window], label=f"Patient λ (Signature {signature_idx})", 
             linewidth=2, color='#e74c3c')
    ax1.axhline(y=gp_mean, label=f"GP Prior Mean (ref + genetic)", 
                linestyle='--', linewidth=2, color='#3498db')
    ax1.axhline(y=reference, label="Reference Only", 
                linestyle=':', linewidth=1.5, color='gray', alpha=0.7)
    ax1.fill_between(time_window, gp_mean - 1.96, gp_mean + 1.96, 
                     alpha=0.2, color='#3498db', label='95% GP Prior (approx)')
    ax1.set_title(f'GP Projection: Patient Lambda vs Prior Mean (Signature {signature_idx})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Lambda Value', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot deviation from GP mean
    ax2.plot(time_window, deviation[time_window], linewidth=2, color='#9b59b6')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(time_window, -1.96, 1.96, alpha=0.2, color='gray', label='95% Prior (approx)')
    ax2.set_title(f'Deviation from GP Prior Mean (Signature {signature_idx})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Deviation (λ - μ)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_signature_contribution_heatmap(pi, theta, phi_prob, disease_names, diseases_to_show=None, time_window=None):
    """Plot heatmap showing signature contributions to diseases."""
    D, T = pi.shape
    K = theta.shape[0]
    
    if time_window is None:
        time_window = range(T)
    if diseases_to_show is None:
        diseases_to_show = range(min(20, D))
    
    # Compute average contribution
    contributions = np.zeros((K, len(diseases_to_show)))
    for i, d in enumerate(diseases_to_show):
        for k in range(K):
            contrib_t = theta[k, time_window] * phi_prob[k, d, time_window]
            contributions[k, i] = np.mean(contrib_t)
    
    fig, ax = plt.subplots(figsize=(max(12, len(diseases_to_show) * 0.5), max(6, K * 0.3)))
    
    disease_labels = [disease_names[d] if d < len(disease_names) else f'Disease {d}' 
                      for d in diseases_to_show]
    
    sns.heatmap(contributions, 
                xticklabels=disease_labels,
                yticklabels=[f'Sig {k}' for k in range(K)],
                cmap='YlOrRd', 
                annot=False,
                fmt='.3f',
                cbar_kws={'label': 'Average Contribution'})
    
    ax.set_title('Signature Contributions to Diseases (θ × φ)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Disease', fontsize=11)
    ax.set_ylabel('Signature', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_genetic_effect_decomposition(model, patient_G, top_n=10):
    """Plot which PRS features contribute most to each signature."""
    K = model.K
    G_padded = pad_G_to_match_gamma(patient_G, model.gamma)
    P = len(G_padded)
    
    # Compute genetic effects
    genetic_effects = np.zeros((P, K))
    for k in range(K):
        gamma_k = model.gamma[:, k].detach().cpu().numpy()
        genetic_effects[:, k] = G_padded * gamma_k
    
    # Get top contributing features for each signature
    fig, axes = plt.subplots(2, (K + 1) // 2, figsize=(16, 10))
    axes = axes.flatten() if K > 1 else [axes]
    
    for k in range(K):
        ax = axes[k]
        effects = genetic_effects[:, k]
        top_indices = np.argsort(np.abs(effects))[-top_n:][::-1]
        
        colors = ['#e74c3c' if e > 0 else '#3498db' for e in effects[top_indices]]
        bars = ax.barh(range(len(top_indices)), effects[top_indices], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax.set_title(f'Sig {k} (Total: {np.sum(effects):.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Genetic Effect', fontsize=9)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused subplots
    for k in range(K, len(axes)):
        axes[k].axis('off')
    
    plt.suptitle('Genetic Effect Decomposition by Signature (Top Features)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_signature_transitions(theta, Y, disease_names, cluster_assignments, age_offset=30, figsize=(16, 10)):
    """
    Plot signature transitions: when signatures become dominant/decline
    Shows signature dominance over time and disease events that trigger transitions
    """
    K, T = theta.shape
    ages = np.arange(age_offset, age_offset + T)
    
    # Find dominant signature at each timepoint
    dominant_sig = np.argmax(theta, axis=0)  # [T]
    
    # Find signature transitions (when dominant signature changes)
    transitions = []
    current_dom = dominant_sig[0]
    for t in range(1, T):
        if dominant_sig[t] != current_dom:
            transitions.append({
                'time': t,
                'age': age_offset + t,
                'from_sig': int(current_dom),
                'to_sig': int(dominant_sig[t])
            })
            current_dom = dominant_sig[t]
    
    # Find disease events
    diagnosis_times = {}
    for d in range(len(disease_names)):
        event_times = np.where(Y[d, :] > 0.5)[0]
        if len(event_times) > 0:
            diagnosis_times[d] = event_times.tolist()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1.5, 1])
    
    # Panel 1: Signature trajectories with transitions
    ax1 = axes[0]
    sig_colors = sns.color_palette("tab20", K)
    
    for k in range(K):
        ax1.plot(ages, theta[k, :], label=f'Sig {k}', linewidth=2, 
                color=sig_colors[k], alpha=0.7)
    
    # Highlight transitions
    for trans in transitions:
        ax1.axvline(x=trans['age'], color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.text(trans['age'], ax1.get_ylim()[1] * 0.95, 
                f"Sig {trans['from_sig']}→{trans['to_sig']}", 
                rotation=90, va='top', ha='right', fontsize=8, fontweight='bold')
    
    # Mark disease events
    for d, times in diagnosis_times.items():
        sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
        color = sig_colors[sig] if 0 <= sig < K else 'gray'
        for t in times:
            if t < T:
                ax1.axvline(x=age_offset + t, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
    
    ax1.set_ylabel('Signature Loading (θ)', fontsize=12, fontweight='bold')
    ax1.set_title('Signature Transitions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim([age_offset, age_offset + T])
    
    # Panel 2: Dominant signature over time
    ax2 = axes[1]
    for k in range(K):
        mask = dominant_sig == k
        if np.any(mask):
            ax2.fill_between(ages[mask], k-0.4, k+0.4, color=sig_colors[k], alpha=0.7, label=f'Sig {k}')
    
    # Mark transitions
    for trans in transitions:
        ax2.axvline(x=trans['age'], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax2.set_ylabel('Dominant Signature', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age (years)', fontsize=12)
    ax2.set_yticks(range(K))
    ax2.set_yticklabels([f'Sig {k}' for k in range(K)])
    ax2.set_ylim([-0.5, K-0.5])
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_xlim([age_offset, age_offset + T])
    
    # Panel 3: Disease events timeline
    ax3 = axes[2]
    if diagnosis_times:
        y_pos = 0
        for d, times in sorted(diagnosis_times.items(), key=lambda x: min(x[1])):
            sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            color = sig_colors[sig] if 0 <= sig < K else 'gray'
            disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            
            for t in times:
                if t < T:
                    ax3.scatter(age_offset + t, y_pos, s=60, color=color, 
                              alpha=0.7, edgecolors='black', linewidths=0.5)
            y_pos += 1
        
        ax3.set_ylabel('Disease Events', fontsize=10, fontweight='bold')
        ax3.set_ylim([-0.5, len(diagnosis_times) - 0.5])
        ax3.set_yticks([])
    
    ax3.set_xlabel('Age (years)', fontsize=12)
    ax3.grid(True, alpha=0.2, axis='x')
    ax3.set_xlim([age_offset, age_offset + T])
    
    plt.tight_layout()
    return fig

def plot_signature_interactions(theta, age_offset=30, top_n=5, figsize=(14, 8)):
    """
    Plot multi-signature interactions: show correlations and co-occurrences
    """
    K, T = theta.shape
    ages = np.arange(age_offset, age_offset + T)
    
    # Find top signatures by average loading
    avg_theta = theta.mean(axis=1)
    top_sigs = np.argsort(avg_theta)[-top_n:][::-1]
    
    # Compute correlation matrix between top signatures
    corr_matrix = np.corrcoef(theta[top_sigs, :])
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Correlation heatmap
    ax1 = axes[0]
    sig_colors = sns.color_palette("tab20", K)
    top_colors = [sig_colors[i] for i in top_sigs]
    
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(top_sigs)))
    ax1.set_xticklabels([f'Sig {i}' for i in top_sigs], rotation=45, ha='right')
    ax1.set_yticks(range(len(top_sigs)))
    ax1.set_yticklabels([f'Sig {i}' for i in top_sigs])
    ax1.set_title('Signature Correlations (Top Signatures)', fontsize=12, fontweight='bold')
    
    # Add correlation values
    for i in range(len(top_sigs)):
        for j in range(len(top_sigs)):
            text = ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Panel 2: Co-occurrence plot (when signatures are both high)
    ax2 = axes[1]
    threshold = 0.1  # Both signatures above this threshold
    
    cooccurrence_times = []
    for t in range(T):
        high_sigs = []
        for k in top_sigs:
            if theta[k, t] > threshold:
                high_sigs.append(k)
        if len(high_sigs) >= 2:
            cooccurrence_times.append(t)
    
    if cooccurrence_times:
        ax2.fill_between([age_offset + t for t in cooccurrence_times], 
                        0, 1, alpha=0.3, color='green', label='Multiple signatures high')
    
    # Plot top signatures
    for i, k in enumerate(top_sigs):
        ax2.plot(ages, theta[k, :], label=f'Sig {k}', 
                color=sig_colors[k], linewidth=2, alpha=0.7)
        ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('Age (years)', fontsize=12)
    ax2.set_ylabel('Signature Loading', fontsize=12)
    ax2.set_title('Signature Co-occurrence (Top Signatures)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig

def plot_signature_importance_ranking(theta, pi, Y, disease_names, cluster_assignments, age_offset=30, figsize=(12, 8)):
    """
    Rank signatures by their importance to this patient:
    - Average loading over time
    - Contribution to disease risk
    - Association with actual diagnoses
    """
    K, T = theta.shape
    D = len(disease_names)
    
    # Compute importance metrics
    importance_scores = {}
    
    for k in range(K):
        # 1. Average loading
        avg_loading = theta[k, :].mean()
        
        # 2. Peak loading
        peak_loading = theta[k, :].max()
        
        # 3. Contribution to disease risk (weighted by pi)
        risk_contribution = 0.0
        diseases_in_sig = 0
        for d in range(D):
            sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            if sig == k:
                diseases_in_sig += 1
                # Weight by max probability
                risk_contribution += pi[d, :].max()
        
        # 4. Association with actual diagnoses
        diagnosis_association = 0.0
        for d in range(D):
            sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            if sig == k:
                if np.any(Y[d, :] > 0.5):
                    diagnosis_association += 1
        
        # Combined score
        score = (
            avg_loading * 2.0 +
            peak_loading * 1.0 +
            risk_contribution * 0.5 +
            diagnosis_association * 1.5
        )
        
        importance_scores[k] = {
            'score': score,
            'avg_loading': avg_loading,
            'peak_loading': peak_loading,
            'risk_contribution': risk_contribution,
            'diagnosis_association': diagnosis_association,
            'n_diseases': diseases_in_sig
        }
    
    # Sort by score
    sorted_sigs = sorted(importance_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1.5])
    
    # Panel 1: Importance ranking bar chart
    ax1 = axes[0]
    sig_colors = sns.color_palette("tab20", K)
    
    sigs_ranked = [k for k, _ in sorted_sigs]
    scores_ranked = [v['score'] for _, v in sorted_sigs]
    colors_ranked = [sig_colors[k] for k in sigs_ranked]
    
    bars = ax1.barh(range(K), scores_ranked, color=colors_ranked, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_yticks(range(K))
    ax1.set_yticklabels([f'Sig {k}' for k in sigs_ranked])
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Signature Importance Ranking', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores_ranked)):
        ax1.text(score * 1.02, i, f'{score:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Panel 2: Detailed metrics
    ax2 = axes[1]
    x = np.arange(K)
    width = 0.2
    
    avg_loadings = [importance_scores[k]['avg_loading'] for k in sigs_ranked]
    peak_loadings = [importance_scores[k]['peak_loading'] for k in sigs_ranked]
    risk_contribs = [importance_scores[k]['risk_contribution'] for k in sigs_ranked]
    diag_assocs = [importance_scores[k]['diagnosis_association'] for k in sigs_ranked]
    
    # Normalize for visualization
    max_val = max(max(avg_loadings), max(peak_loadings), max(risk_contribs) if risk_contribs else 0, max(diag_assocs) if diag_assocs else 0)
    if max_val > 0:
        avg_loadings_norm = [v / max_val for v in avg_loadings]
        peak_loadings_norm = [v / max_val for v in peak_loadings]
        risk_contribs_norm = [v / max_val if max_val > 0 else 0 for v in risk_contribs]
        diag_assocs_norm = [v / max_val if max_val > 0 else 0 for v in diag_assocs]
    else:
        avg_loadings_norm = avg_loadings
        peak_loadings_norm = peak_loadings
        risk_contribs_norm = risk_contribs
        diag_assocs_norm = diag_assocs
    
    ax2.bar(x - 1.5*width, avg_loadings_norm, width, label='Avg Loading', alpha=0.7, color='steelblue')
    ax2.bar(x - 0.5*width, peak_loadings_norm, width, label='Peak Loading', alpha=0.7, color='coral')
    ax2.bar(x + 0.5*width, risk_contribs_norm, width, label='Risk Contribution', alpha=0.7, color='green')
    ax2.bar(x + 1.5*width, diag_assocs_norm, width, label='Diagnosis Association', alpha=0.7, color='purple')
    
    ax2.set_xlabel('Signature (ranked by importance)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Metric', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Sig {k}' for k in sigs_ranked], rotation=45, ha='right')
    ax2.set_title('Signature Importance Components', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, sorted_sigs

def main():
    st.title("🔬 Real-Time Patient Refitting")
    st.markdown("**Fit model in real time for individual patients with editable disease history**")
    st.info("ℹ️ **Using pooled gamma from _nolr batches (unshrunken, no lambda_reg)** - counterfactuals should show larger genetic effects! 💡 **Tip:** High PRS patients are prioritized - look for patients with high PRS values for more dramatic genetic effects.")
    
    # Sidebar for data directory
    st.sidebar.header("📁 Data Configuration")
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/",
        help="Path to directory containing model checkpoints and data"
    )
    
    # Master checkpoint selection
    st.sidebar.subheader("Master Checkpoint")
    master_checkpoint_name = st.sidebar.selectbox(
        "Master Checkpoint File",
        ["master_for_fitting_pooled_correctedE_nolr.pt", "master_for_fitting_pooled_correctedE.pt"],
        help="Select which master checkpoint to use. The _nolr version contains pooled gamma (unshrunken, no lambda_reg)."
    )
    
    if master_checkpoint_name == "master_for_fitting_pooled_correctedE_nolr.pt":
        st.sidebar.success("✓ Using _nolr version with pooled unshrunken gamma")
    else:
        st.sidebar.warning("⚠️ Using old version - gamma may not be in checkpoint. Use _nolr version for unshrunken gamma.")
    
    # Load fixed components
    if st.sidebar.button("Load Model Components", type="primary"):
        try:
            with st.spinner("Loading fixed components..."):
                fixed_components = load_fixed_components(data_dir, master_checkpoint_name=master_checkpoint_name)
                
                gamma = fixed_components['gamma']
                gamma_max_abs = np.abs(gamma).max()
                gamma_mean_abs = np.abs(gamma).mean()
                
                # Determine if gamma is shrunken or unshrunken
                if gamma_max_abs < 0.01:
                    gamma_status = "🔴 Shrunken (regularized)"
                    gamma_note = "Max |γ| < 0.01 - likely from batches with penalty"
                elif gamma_max_abs < 0.1:
                    gamma_status = "🟡 Moderately shrunken"
                    gamma_note = "Max |γ| < 0.1 - may be from batches with penalty"
                else:
                    gamma_status = "🟢 Unshrunken (no lambda_reg)"
                    gamma_note = "Max |γ| ≥ 0.1 - from _nolr batches (no penalty)"
                
                st.sidebar.success(f"✓ Components loaded! Gamma shape: {gamma.shape}")
                st.sidebar.info(f"""
                **Gamma Status:** {gamma_status}
                - Max |γ|: {gamma_max_abs:.6f}
                - Mean |γ|: {gamma_mean_abs:.6f}
                - {gamma_note}
                """)
                st.session_state['fixed_components'] = fixed_components
                st.session_state['disease_names'] = fixed_components['disease_names']
        except Exception as e:
            st.sidebar.error(f"Error loading components: {e}")
            import traceback
            st.sidebar.code(traceback.format_exc())
            st.stop()
    
    if 'fixed_components' not in st.session_state:
        st.info("👆 Please load model components from the sidebar first.")
        st.stop()
    
    fixed_components = st.session_state['fixed_components']
    disease_names = st.session_state['disease_names']
    D = len(disease_names)
    T = fixed_components['phi'].shape[2]
    
    # Load cluster assignments for signature-based coloring
    if 'cluster_assignments' not in st.session_state:
        cluster_assignments = load_cluster_assignments(data_dir)
        st.session_state['cluster_assignments'] = cluster_assignments
        if cluster_assignments is not None:
            st.sidebar.success(f"✓ Loaded cluster assignments ({len(cluster_assignments)} diseases)")
        else:
            st.sidebar.warning("⚠️ Could not load cluster assignments - using default colors")
    else:
        cluster_assignments = st.session_state['cluster_assignments']
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 Sample Patient", "✏️ Custom Patient"])
    
    with tab1:
        st.header("Load Sample Patient")
        
        # Load sample patients
        col1, col2 = st.columns(2)
        with col1:
            find_interesting = st.checkbox("Find Interesting Patients", value=True, 
                                          help="Find patients with many diagnoses and signature shifts")
        with col2:
            n_patients_to_load = st.slider("Number of Patients", 10, 100, 50, 10)
        
        if st.button("Load Sample Patients", type="primary"):
            try:
                with st.spinner("Loading sample patients..."):
                    result = load_sample_patients(data_dir, n_patients=n_patients_to_load, find_interesting=find_interesting)
                    Y_sample, G_sample, E_sample, interesting_info = result
                    
                    st.session_state['Y_sample'] = Y_sample
                    st.session_state['G_sample'] = G_sample
                    st.session_state['E_sample'] = E_sample
                    st.session_state['interesting_patients'] = interesting_info
                    
                    if E_sample is not None:
                        st.success(f"✓ Loaded {len(Y_sample)} sample patients with E_corrected!")
                    else:
                        st.warning(f"Loaded {len(Y_sample)} sample patients, but E_corrected not found. Will create E from Y.")
                    
                    if interesting_info:
                        st.info(f"📊 Found {len(interesting_info)} interesting patients with many diagnoses and signature shifts!")
            except Exception as e:
                st.error(f"Error loading sample patients: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        if 'Y_sample' in st.session_state:
            # Get age offset first (default 30)
            age_offset_default = 30
            
            # Create patient selection with info
            patient_options = []
            patient_info_text = []
            
            for i in range(len(st.session_state['Y_sample'])):
                Y_p = st.session_state['Y_sample'][i]
                n_diag = np.sum([np.any(Y_p[d, :] > 0.5) for d in range(Y_p.shape[0])])
                
                if 'interesting_patients' in st.session_state and st.session_state['interesting_patients']:
                    if i < len(st.session_state['interesting_patients']):
                        info = st.session_state['interesting_patients'][i]
                        prs_info = ""
                        if info.get('prs_magnitude') is not None:
                            prs_mag = info['prs_magnitude']
                            prs_info = f", PRS={prs_mag:.1f}"
                        label = f"Patient {info['patient_id']} ({info['n_diagnoses']} diag, {info['time_spread']}yr{prs_info})"
                        info_text = f"Patient {info['patient_id']}: {info['n_diagnoses']} diagnoses, "
                        info_text += f"spread over {info['time_spread']} years (timepoints {info['first_diag']}-{info['last_diag']})"
                        if info.get('prs_magnitude') is not None:
                            info_text += f" | PRS magnitude: {info['prs_magnitude']:.2f}"
                        patient_info_text.append(info_text)
                    else:
                        label = f"Patient {i} ({n_diag} diagnoses)"
                        patient_info_text.append(f"Patient {i}: {n_diag} diagnoses")
                else:
                    label = f"Patient {i} ({n_diag} diagnoses)"
                    patient_info_text.append(f"Patient {i}: {n_diag} diagnoses")
                
                patient_options.append(label)
            
            # Add helper text about PRS
            if 'interesting_patients' in st.session_state and st.session_state['interesting_patients']:
                high_prs_patients = [i for i, p in enumerate(st.session_state['interesting_patients']) 
                                    if i < len(patient_options) and p.get('prs_magnitude', 0) > 0]
                if high_prs_patients:
                    st.caption(f"💡 **Tip:** Patients with high PRS are prioritized in the list. Look for patients with higher PRS values for more dramatic genetic effects!")
            
            patient_idx = st.selectbox(
                "Select Patient",
                range(len(st.session_state['Y_sample'])),
                format_func=lambda x: patient_options[x] if x < len(patient_options) else f"Patient {x}",
                help="Patients are sorted by interestingness (many diagnoses, time spread, high PRS)"
            )
            
            # Show patient info
            if patient_idx < len(patient_info_text):
                info_text = patient_info_text[patient_idx]
                # Add PRS info if available
                if 'interesting_patients' in st.session_state and st.session_state['interesting_patients']:
                    if patient_idx < len(st.session_state['interesting_patients']):
                        info = st.session_state['interesting_patients'][patient_idx]
                        if info.get('prs_magnitude') is not None:
                            prs_mag = info['prs_magnitude']
                            info_text += f" | PRS magnitude: {prs_mag:.2f}"
                st.info(f"ℹ️ **{info_text}**")
            
            Y_patient = st.session_state['Y_sample'][patient_idx]
            G_patient = st.session_state['G_sample'][patient_idx]
            
            # Show PRS summary
            if len(G_patient.shape) == 1:
                prs_values = G_patient[:36] if len(G_patient) >= 36 else G_patient
                prs_magnitude = np.linalg.norm(prs_values)
                prs_mean = np.mean(np.abs(prs_values))
                st.caption(f"📊 **PRS Summary:** Magnitude = {prs_magnitude:.2f}, Mean |PRS| = {prs_mean:.3f} (first {len(prs_values)} components)")
            elif len(G_patient.shape) == 2 and G_patient.shape[0] == 1:
                prs_values = G_patient[0, :36] if G_patient.shape[1] >= 36 else G_patient[0, :]
                prs_magnitude = np.linalg.norm(prs_values)
                prs_mean = np.mean(np.abs(prs_values))
                st.caption(f"📊 **PRS Summary:** Magnitude = {prs_magnitude:.2f}, Mean |PRS| = {prs_mean:.3f} (first {len(prs_values)} components)")
            if 'E_sample' in st.session_state and st.session_state['E_sample'] is not None:
                E_patient = st.session_state['E_sample'][patient_idx]
                using_E_corrected = True
            else:
                E_patient = None
                using_E_corrected = False
            
            st.subheader("Patient Disease History")
            
            if using_E_corrected:
                st.info(f"ℹ️ Using E_corrected for this patient (shape: {E_patient.shape}).")
                if len(E_patient.shape) == 1:
                    events = (E_patient < T - 1).sum()
                    censored = (E_patient == T - 1).sum()
                    mean_E = E_patient.mean()
                    st.caption(f"E_corrected summary: {events} diseases with events, {censored} censored at max time. Mean E: {mean_E:.2f}")
            
            age_offset = st.number_input("Patient Age at Baseline", min_value=0, max_value=100, value=age_offset_default, 
                                         help="Age at time 0 (baseline)")
            progression_text = get_diagnosis_progression(Y_patient, disease_names, age_offset=age_offset)
            
            st.markdown("**📖 Diagnosis Progression:**")
            st.markdown(progression_text)
            
            # Also show as table
            st.markdown("**📋 Detailed Timeline:**")
            diagnoses_df = []
            for d in range(D):
                times = np.where(Y_patient[d, :] > 0.5)[0]
                if len(times) > 0:
                    ages = [age_offset + int(t) for t in times]
                    diagnoses_df.append({
                        'Disease': disease_names[d] if d < len(disease_names) else f'Disease {d}',
                        'Time Points': ', '.join(map(str, times)),
                        'Ages': ', '.join(map(str, ages))
                    })
            
            if diagnoses_df:
                st.dataframe(pd.DataFrame(diagnoses_df), use_container_width=True)
            else:
                st.info("No diagnoses recorded for this patient.")
            
            # Editable Y matrix
            st.subheader("✏️ Edit Disease History")
            st.markdown("**Modify disease codes over time (0 = no disease, 1 = disease present):**")
            
            time_cols = [f"t={t}" for t in range(min(20, T))]
            edit_data = {}
            for d in range(min(10, D)):
                disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
                edit_data[disease_name] = Y_patient[d, :min(20, T)]
            
            edited_df = st.data_editor(
                pd.DataFrame(edit_data, index=time_cols).T,
                use_container_width=True,
                num_rows="fixed"
            )
            
            for i, disease_name in enumerate(edited_df.index):
                if i < D:
                    Y_patient[i, :min(20, T)] = edited_df.loc[disease_name].values
            
            # Training parameters
            st.subheader("⚙️ Training Parameters")
            col1, col2 = st.columns(2)
            with col1:
                num_epochs = st.slider("Number of Epochs", 10, 100, 30, help="More epochs = better fit but slower")
            with col2:
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, help="Higher = faster convergence but may be unstable")
            
            # Fit model
            if st.button("🔄 Refit Model in Real Time", type="primary"):
                with st.spinner("Fitting model... This may take a few seconds."):
                    if E_patient is None or not using_E_corrected:
                        st.info("ℹ️ Creating E from Y (E_corrected not available).")
                        E_patient_for_fit = create_event_times_from_Y(Y_patient)
                    else:
                        E_patient_for_fit = E_patient
                        st.success("✓ Using E_corrected for fitting")
                    
                    pi, theta, model, losses = fit_patient_model(
                        G_patient, Y_patient, E_patient_for_fit, 
                        fixed_components, num_epochs, learning_rate
                    )
                    
                    st.session_state['pi'] = pi
                    st.session_state['theta'] = theta
                    st.session_state['losses'] = losses
                    st.session_state['Y_current'] = Y_patient
                    st.session_state['model'] = model
                    st.session_state['G_current'] = G_patient
                    
                    st.success("✓ Model fitted successfully!")
            
            # Display results
            if 'pi' in st.session_state:
                st.subheader("📈 Predictions")
                
                age_offset = st.number_input("Age at Baseline for Plot", min_value=0, max_value=100, value=30, 
                                             key="age_plot", help="Age at time 0 (baseline)")
                
                # Disease filtering options
                col1, col2 = st.columns(2)
                with col1:
                    max_diseases_show = st.slider("Max Diseases to Show", 5, 30, 15, 
                                                 help="Limit number of diseases shown to prevent overcrowding")
                with col2:
                    min_prob_threshold = st.slider("Min Probability Threshold", 0.0, 0.01, 0.001, 0.0001,
                                                  format="%.4f", help="Only show diseases with max probability above this")
                
                # Show timeline visualization option
                show_timeline = st.checkbox("Show Comprehensive Timeline View", value=False, key="show_timeline_sample")
                
                if show_timeline:
                    fig_timeline = plot_patient_timeline_comprehensive(
                        st.session_state['pi'],
                        st.session_state['theta'],
                        st.session_state['Y_current'],
                        disease_names,
                        age_offset=age_offset,
                        cluster_assignments=cluster_assignments,
                        figsize=(20, 14)
                    )
                    st.pyplot(fig_timeline)
                    plt.close(fig_timeline)
                else:
                    fig = plot_predictions(
                        st.session_state['pi'],
                        st.session_state['theta'],
                        st.session_state['Y_current'],
                        disease_names,
                        age_offset=age_offset,
                        cluster_assignments=cluster_assignments,
                        max_diseases_to_show=max_diseases_show,
                        min_prob_threshold=min_prob_threshold
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Loss curve
                if 'losses' in st.session_state:
                    fig_loss, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(st.session_state['losses'])
                    ax.set_title('Training Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig_loss)
                
                # Summary table
                st.subheader("📋 Risk Summary")
                pi = st.session_state['pi']
                summary_data = []
                for d in range(D):
                    if np.any(st.session_state['Y_current'][d, :] > 0.5):
                        max_prob = np.max(pi[d, :])
                        max_time = np.argmax(pi[d, :])
                        mean_prob = np.mean(pi[d, :])
                        summary_data.append({
                            'Disease': disease_names[d] if d < len(disease_names) else f'Disease {d}',
                            'Mean Risk': f"{mean_prob:.4f}",
                            'Max Risk': f"{max_prob:.4f}",
                            'Time of Max': int(max_time)
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                # Advanced visualizations
                if 'model' in st.session_state:
                    st.subheader("🔬 Advanced Visualizations")
                    
                    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5, viz_tab6, viz_tab7, viz_tab8, viz_tab9 = st.tabs([
                        "📈 Disease Timeline", 
                        "🔄 Signature Transitions",
                        "🔗 Signature Interactions",
                        "⭐ Signature Importance",
                        "Counterfactual Genetics", 
                        "GP Projection", 
                        "Signature Contributions",
                        "Genetic Decomposition",
                        "W Penalty Diagnostics"
                    ])
                    
                    with viz_tab1:
                        st.markdown("**Comprehensive disease trajectory visualization showing signature evolution, disease timeline, and risk trajectories**")
                        age_offset_timeline = st.number_input("Age at Baseline", min_value=0, max_value=100, value=30, 
                                                              key="age_timeline", help="Age at time 0 (baseline)")
                        fig_timeline = plot_patient_timeline_comprehensive(
                            st.session_state['pi'],
                            st.session_state['theta'],
                            st.session_state['Y_current'],
                            disease_names,
                            age_offset=age_offset_timeline,
                            cluster_assignments=cluster_assignments,
                            figsize=(20, 14)
                        )
                        st.pyplot(fig_timeline)
                        plt.close(fig_timeline)
                    
                    model_current = st.session_state['model']
                    G_from_model = model_current.G[0].detach().cpu().numpy()
                    
                    with viz_tab2:
                        st.markdown("**Signature Transitions:** See when signatures become dominant or decline, and how disease events trigger transitions**")
                        st.info("""
                        **What to look for:**
                        - **Red dashed lines**: Signature transitions (when dominant signature changes)
                        - **Colored dots**: Disease events, colored by their primary signature
                        - **Panel 2**: Shows which signature is dominant at each timepoint
                        - **Panel 3**: Timeline of all disease events
                        """)
                        age_offset_transitions = st.number_input("Age at Baseline", min_value=0, max_value=100, value=30, 
                                                                key="age_transitions", help="Age at time 0 (baseline)")
                        fig_transitions = plot_signature_transitions(
                            st.session_state['theta'],
                            st.session_state['Y_current'],
                            disease_names,
                            cluster_assignments,
                            age_offset=age_offset_transitions,
                            figsize=(16, 10)
                        )
                        st.pyplot(fig_transitions)
                        plt.close(fig_transitions)
                    
                    with viz_tab3:
                        st.markdown("**Signature Interactions:** See how signatures correlate and co-occur over time**")
                        st.info("""
                        **What to look for:**
                        - **Left panel**: Correlation matrix between top signatures (red = positive, blue = negative)
                        - **Right panel**: When multiple signatures are high simultaneously (green shaded regions)
                        """)
                        top_n_interactions = st.slider("Number of Top Signatures to Analyze", 3, 10, 5, 
                                                      key="top_n_interactions")
                        age_offset_interactions = st.number_input("Age at Baseline", min_value=0, max_value=100, value=30, 
                                                                 key="age_interactions", help="Age at time 0 (baseline)")
                        fig_interactions = plot_signature_interactions(
                            st.session_state['theta'],
                            age_offset=age_offset_interactions,
                            top_n=top_n_interactions,
                            figsize=(14, 8)
                        )
                        st.pyplot(fig_interactions)
                        plt.close(fig_interactions)
                    
                    with viz_tab4:
                        st.markdown("**Signature Importance Ranking:** Which signatures matter most for this patient?**")
                        st.info("""
                        **Ranking based on:**
                        - Average loading over time
                        - Peak loading
                        - Contribution to disease risk
                        - Association with actual diagnoses
                        """)
                        age_offset_importance = st.number_input("Age at Baseline", min_value=0, max_value=100, value=30, 
                                                               key="age_importance", help="Age at time 0 (baseline)")
                        fig_importance, sig_ranking = plot_signature_importance_ranking(
                            st.session_state['theta'],
                            st.session_state['pi'],
                            st.session_state['Y_current'],
                            disease_names,
                            cluster_assignments,
                            age_offset=age_offset_importance,
                            figsize=(12, 8)
                        )
                        st.pyplot(fig_importance)
                        plt.close(fig_importance)
                        
                        # Show ranking details
                        with st.expander("📊 Detailed Signature Ranking"):
                            ranking_df = []
                            for rank, (sig, metrics) in enumerate(sig_ranking, 1):
                                ranking_df.append({
                                    'Rank': rank,
                                    'Signature': f'Sig {sig}',
                                    'Total Score': f"{metrics['score']:.3f}",
                                    'Avg Loading': f"{metrics['avg_loading']:.4f}",
                                    'Peak Loading': f"{metrics['peak_loading']:.4f}",
                                    'Risk Contribution': f"{metrics['risk_contribution']:.4f}",
                                    'Diagnosis Association': f"{metrics['diagnosis_association']:.1f}",
                                    'N Diseases in Sig': metrics['n_diseases']
                                })
                            st.dataframe(pd.DataFrame(ranking_df), use_container_width=True)
                    
                    with viz_tab5:
                        st.markdown("**Counterfactual Genetics:** What if this patient had PRS=0?** See how signature trajectories would change.")
                        gamma_current = fixed_components.get('gamma')
                        if gamma_current is not None:
                            gamma_max_abs = np.abs(gamma_current).max()
                            gamma_mean_abs = np.abs(gamma_current).mean()
                            if gamma_max_abs < 0.01:
                                gamma_type = "**regularized (shrunken) gamma**"
                                gamma_note = "⚠️ Small effects are expected with regularized gamma."
                            else:
                                gamma_type = "**unshrunken gamma (from _nolr batches)**"
                                gamma_note = "✓ This uses unshrunken gamma from _nolr batches (no lambda_reg). Counterfactuals should show larger genetic effects!"
                            
                            with st.expander("📊 Gamma Statistics (click to view)"):
                                st.write(f"**Max |γ|:** {gamma_max_abs:.6f}")
                                st.write(f"**Mean |γ|:** {gamma_mean_abs:.6f}")
                                st.write(f"**Shape:** {gamma_current.shape}")
                                st.write(f"**Status:** {gamma_type}")
                        else:
                            gamma_type = "gamma"
                            gamma_note = "⚠️ Gamma not loaded!"
                        
                        st.info(f"""
                        **ℹ️ About the Counterfactual:**
                        - This visualization uses {gamma_type}
                        - {gamma_note}
                        - Check the 'W Penalty Diagnostics' tab to see detailed metrics about genetic effect magnitudes
                        """)
                        sig_idx_cf = st.selectbox(
                            "Select Signature for Counterfactual",
                            range(model_current.K),
                            format_func=lambda x: f"Signature {x}",
                            key="cf_sig"
                        )
                        if st.button("Generate Counterfactual Plot", key="cf_btn"):
                            G_padded = pad_G_to_match_gamma(G_from_model, model_current.gamma)
                            gamma_k = model_current.gamma[:, sig_idx_cf].detach().cpu().numpy()
                            genetic_effect = float(model_current.genetic_scale * np.dot(G_padded, gamma_k))
                            
                            fig_cf = plot_counterfactual_signature_trajectory(
                                model_current, sig_idx_cf, G_from_model
                            )
                            st.pyplot(fig_cf)
                            
                            gamma_current = fixed_components.get('gamma')
                            if gamma_current is not None:
                                gamma_max_abs = np.abs(gamma_current).max()
                                if gamma_max_abs < 0.01:
                                    gamma_note = "Small effects (<0.01) are expected with regularized gamma."
                                else:
                                    gamma_note = "This uses unshrunken gamma from _nolr batches (no lambda_reg). Genetic effects should be larger than with regularized gamma!"
                            else:
                                gamma_note = ""
                            
                            st.caption(f"""
                            **Interpretation:** The counterfactual shows what would happen if all PRS values were set to zero. 
                            Genetic effect magnitude: {genetic_effect:.6f}. 
                            
                            **Note:** {gamma_note}
                            """)
                    
                    with viz_tab6:
                        st.markdown("**GP Projection:** See how the patient's lambda trajectory compares to the Gaussian Process prior mean (reference + genetic effect).")
                        sig_idx_gp = st.selectbox(
                            "Select Signature for GP Projection",
                            range(model_current.K),
                            format_func=lambda x: f"Signature {x}",
                            key="gp_sig"
                        )
                        if st.button("Generate GP Projection Plot", key="gp_btn"):
                            fig_gp = plot_gp_projection(
                                model_current, sig_idx_gp, G_from_model
                            )
                            st.pyplot(fig_gp)
                            st.caption("**Interpretation:** The GP prior mean is the expected lambda value (reference trajectory + genetic effect). Deviations show how the patient's disease history has shifted their signature trajectory.")
                    
                    with viz_tab7:
                        st.markdown("**Signature Contribution Heatmap:** Which signatures drive risk for each disease?")
                        n_diseases_heatmap = st.slider(
                            "Number of Diseases to Show",
                            5, min(50, D), 20,
                            key="n_diseases_heatmap"
                        )
                        mean_risks = np.mean(st.session_state['pi'], axis=1)
                        top_diseases = np.argsort(mean_risks)[-n_diseases_heatmap:][::-1]
                        
                        if st.button("Generate Contribution Heatmap", key="heatmap_btn"):
                            with torch.no_grad():
                                _, theta_current, phi_prob_current = model_current.forward()
                            theta_np = theta_current[0].cpu().numpy()
                            phi_prob_np = phi_prob_current.cpu().numpy()
                            K_disease = phi_prob_np.shape[0]
                            theta_disease = theta_np[:K_disease, :]
                            fig_heatmap = plot_signature_contribution_heatmap(
                                st.session_state['pi'],
                                theta_disease,
                                phi_prob_np,
                                disease_names,
                                diseases_to_show=top_diseases
                            )
                            st.pyplot(fig_heatmap)
                            st.caption("**Interpretation:** Each cell shows how much a signature contributes to a disease's risk (θ × φ). Darker colors = higher contribution.")
                    
                    with viz_tab8:
                        st.markdown("**Genetic Effect Decomposition:** Which PRS features matter most for each signature?")
                        top_n_features = st.slider(
                            "Top N Features per Signature",
                            5, 20, 10,
                            key="top_n_features"
                        )
                        if st.button("Generate Genetic Decomposition", key="decomp_btn"):
                            fig_decomp = plot_genetic_effect_decomposition(
                                model_current, G_from_model, top_n=top_n_features
                            )
                            st.pyplot(fig_decomp)
                            st.caption("**Interpretation:** Shows which genetic features (PRS components) contribute most to each signature. Red = positive effect, Blue = negative effect.")
                    
                    with viz_tab9:
                        st.markdown("**Genetic Penalty (W) Diagnostics:** Assess whether W is too high, too low, or appropriate.")
                        st.markdown("""
                        **How to interpret:**
                        - **W too high**: Gamma values are very small, genetic effects negligible, counterfactuals show no difference
                        - **W too low**: Gamma values are very large, genetic effects dominate, risk of overfitting
                        - **W appropriate**: Genetic effects are meaningful but not dominant, good balance between data fit and regularization
                        
                        **Note:** With _nolr batches (no lambda_reg), gamma values should be larger than with regularization.
                        """)
                        
                        if st.button("Run W Penalty Diagnostics", key="w_diag_btn"):
                            sample_G_for_diag = G_from_model if 'G_current' in st.session_state else None
                            diagnosis = diagnose_genetic_penalty(
                                model_current, 
                                fixed_components,
                                sample_G=sample_G_for_diag
                            )
                            
                            if diagnosis['status'] == 'error':
                                st.error(diagnosis['message'])
                            else:
                                if diagnosis['status'] == 'good':
                                    st.success(diagnosis['status_message'])
                                elif diagnosis['status'] == 'too_high':
                                    st.error(diagnosis['status_message'])
                                elif diagnosis['status'] == 'too_low':
                                    st.warning(diagnosis['status_message'])
                                else:
                                    st.warning(diagnosis['status_message'])
                                
                                if diagnosis['warnings']:
                                    st.subheader("⚠️ Warnings")
                                    for warning in diagnosis['warnings']:
                                        st.markdown(warning)
                                
                                if diagnosis['recommendations']:
                                    st.subheader("💡 Recommendations")
                                    for rec in diagnosis['recommendations']:
                                        st.markdown(f"- {rec}")
                                
                                fig_diag = plot_genetic_penalty_diagnostics(diagnosis)
                                st.pyplot(fig_diag)
                                
                                with st.expander("📊 Detailed Statistics"):
                                    st.json({
                                        'gamma_stats': {
                                            'mean_abs': float(diagnosis['gamma_stats']['mean_abs']),
                                            'std_abs': float(diagnosis['gamma_stats']['std_abs']),
                                            'max_abs': float(diagnosis['gamma_stats']['max_abs']),
                                            'percentiles': {
                                                'P25': float(diagnosis['gamma_stats']['percentiles'][0]),
                                                'P50': float(diagnosis['gamma_stats']['percentiles'][1]),
                                                'P75': float(diagnosis['gamma_stats']['percentiles'][2]),
                                                'P95': float(diagnosis['gamma_stats']['percentiles'][3]),
                                                'P99': float(diagnosis['gamma_stats']['percentiles'][4])
                                            }
                                        },
                                        'gp_prior': {
                                            'lambda_amplitude': float(diagnosis['lambda_amplitude']),
                                            'variance': float(diagnosis['gp_prior_var'])
                                        },
                                        'signal_to_noise': {
                                            'mean': float(diagnosis['mean_snr']),
                                            'per_signature': [float(x) for x in diagnosis['snr_per_signature']]
                                        },
                                        'counterfactual_effects': diagnosis['counterfactual_effects'] if diagnosis['counterfactual_effects'] else None
                                    })
                else:
                    st.info("👆 Please fit the model first to see advanced visualizations.")
    
    with tab2:
        st.header("Create Custom Patient")
        st.markdown("**Manually create a patient with custom genetic data and disease history**")
        
        # Genetic data input
        st.subheader("🧬 Genetic Data")
        st.markdown("**Enter PRS values (36 PRS + 1 sex + 10 PCs = 47 values):**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Generate Random Genetic Data"):
                G_custom = np.random.randn(47)
                st.session_state['G_custom'] = G_custom
        with col2:
            if st.button("Generate High PRS (2x std)"):
                # Generate high PRS: mean=0, std=2 for PRS components
                G_custom = np.zeros(47)
                G_custom[:36] = np.random.randn(36) * 2.0  # High PRS
                st.session_state['G_custom'] = G_custom
        with col3:
            if st.button("Generate Low PRS (-2x std)"):
                # Generate low PRS: mean=0, std=-2 for PRS components
                G_custom = np.zeros(47)
                G_custom[:36] = np.random.randn(36) * -2.0  # Low PRS
                st.session_state['G_custom'] = G_custom
        
        if 'G_custom' not in st.session_state:
            # Default: high PRS patient (set seed for reproducibility of default)
            np.random.seed(42)
            G_custom = np.zeros(47)
            G_custom[:36] = np.random.randn(36) * 2.0  # High PRS by default (2x std)
            # Add sex (component 36) - random
            G_custom[36] = np.random.choice([0, 1])  # Sex: 0=Female, 1=Male
            # Add PCs (components 37-46) - small random values
            G_custom[37:47] = np.random.randn(10) * 0.5
            st.session_state['G_custom'] = G_custom
            st.info("💡 **Default:** High PRS patient (PRS ~2x std) - genetic effects should be more visible in counterfactuals!")
        
        # Show PRS summary for custom patient
        if 'G_custom' in st.session_state:
            prs_custom = st.session_state['G_custom'][:36]
            prs_magnitude = np.linalg.norm(prs_custom)
            prs_mean_abs = np.mean(np.abs(prs_custom))
            prs_max_abs = np.max(np.abs(prs_custom))
            
            if prs_magnitude > 3.0:
                prs_status = "🔥 **Very High PRS**"
            elif prs_magnitude > 2.0:
                prs_status = "⭐ **High PRS**"
            elif prs_magnitude > 1.0:
                prs_status = "📊 **Moderate PRS**"
            else:
                prs_status = "📉 **Low PRS**"
            
            st.caption(f"{prs_status} | Magnitude: {prs_magnitude:.2f} | Mean |PRS|: {prs_mean_abs:.3f} | Max |PRS|: {prs_max_abs:.3f}")
        
        G_df = pd.DataFrame({
            'Value': st.session_state['G_custom']
        }, index=[f'Feature {i}' for i in range(47)])
        
        edited_G = st.data_editor(G_df, use_container_width=True, num_rows="fixed")
        st.session_state['G_custom'] = edited_G['Value'].values
        
        # Custom disease history
        st.subheader("📋 Disease History")
        st.markdown("**Create custom disease timeline:**")
        st.info("💡 **Note:** When a disease appears, it only contributes to the loss at the **first occurrence** (event time). Times after the event don't contribute to the loss - the patient is censored after the event occurs.")
        
        if 'Y_custom' not in st.session_state:
            st.session_state['Y_custom'] = np.zeros((D, T))
        
        age_offset_custom_input = st.number_input("Patient Age at Baseline", min_value=0, max_value=100, value=30, 
                                                  key="age_custom_input", help="Age at time 0 (baseline)")
        progression_text_custom_display = get_diagnosis_progression(
            st.session_state['Y_custom'], disease_names, age_offset=age_offset_custom_input
        )
        st.markdown("**📖 Current Diagnosis Progression:**")
        st.markdown(progression_text_custom_display)
        
        selected_disease = st.selectbox(
            "Select Disease to Add/Edit",
            range(D),
            format_func=lambda x: disease_names[x] if x < len(disease_names) else f'Disease {x}'
        )
        
        if cluster_assignments is not None and selected_disease < len(cluster_assignments):
            sig = int(cluster_assignments[selected_disease])
            st.caption(f"📌 This disease belongs to **Signature {sig}** (will be colored accordingly in plots)")
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.slider("Start Time (Event Time)", 0, T-1, 0, key="start", 
                                   help="First time point where disease appears (this becomes the event time)")
        with col2:
            end_time = st.slider("End Time (Visualization)", start_time, T-1, min(20, T-1), key="end",
                                help="Last time point for visualization (only start_time contributes to loss)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Diagnosis", key="add"):
                st.session_state['Y_custom'][selected_disease, start_time:end_time+1] = 1.0
                st.success(f"Added {disease_names[selected_disease]} at times {start_time}-{end_time} (event time: {start_time})")
        with col2:
            if st.button("➖ Remove Diagnosis", key="remove"):
                st.session_state['Y_custom'][selected_disease, start_time:end_time+1] = 0.0
                st.success(f"Removed {disease_names[selected_disease]} at times {start_time}-{end_time}")
        
        st.markdown("**Current Disease Matrix (first 20 diseases × first 20 time points):**")
        Y_display = st.session_state['Y_custom'][:min(20, D), :min(20, T)]
        
        display_df = pd.DataFrame(
            Y_display,
            index=[disease_names[i] if i < len(disease_names) else f'Disease {i}' 
                   for i in range(min(20, D))],
            columns=[f't={t}' for t in range(min(20, T))]
        )
        
        if cluster_assignments is not None:
            K = fixed_components['phi'].shape[0] if 'phi' in fixed_components else 20
            sig_colors = sns.color_palette("tab20", K)
            
            def highlight_row(row):
                row_name = row.name
                disease_idx = None
                for i in range(min(20, D)):
                    disease_name = disease_names[i] if i < len(disease_names) else f'Disease {i}'
                    if disease_name == row_name:
                        disease_idx = i
                        break
                
                if disease_idx is not None and disease_idx < len(cluster_assignments):
                    sig = int(cluster_assignments[disease_idx])
                    if 0 <= sig < len(sig_colors):
                        color = sig_colors[sig]
                        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color)
                        return [f'background-color: {hex_color}30' for _ in row]
                return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            st.caption("💡 Row colors indicate the primary signature for each disease (matching plot colors)")
        else:
            st.dataframe(display_df, use_container_width=True)
        
        st.subheader("⚙️ Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            num_epochs_custom = st.slider("Number of Epochs", 10, 100, 30, key="epochs_custom")
        with col2:
            learning_rate_custom = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lr_custom")
        
        if st.button("🔄 Fit Custom Patient Model", type="primary"):
            with st.spinner("Fitting model for custom patient..."):
                E_custom = create_event_times_from_Y(st.session_state['Y_custom'])
                
                pi, theta, model, losses = fit_patient_model(
                    st.session_state['G_custom'],
                    st.session_state['Y_custom'],
                    E_custom,
                    fixed_components,
                    num_epochs_custom,
                    learning_rate_custom
                )
                
                st.session_state['pi_custom'] = pi
                st.session_state['theta_custom'] = theta
                st.session_state['losses_custom'] = losses
                
                st.success("✓ Custom patient model fitted!")
        
        if 'pi_custom' in st.session_state:
            st.subheader("📈 Custom Patient Predictions")
            
            age_offset_custom_display = st.number_input("Patient Age at Baseline for Plot", min_value=0, max_value=100, value=30, 
                                                        key="age_custom_display", help="Age at time 0 (baseline)")
            
            progression_text_custom = get_diagnosis_progression(
                st.session_state['Y_custom'], disease_names, age_offset=age_offset_custom_display
            )
            st.markdown("**📖 Diagnosis Progression:**")
            st.markdown(progression_text_custom)
            
            E_custom_display = create_event_times_from_Y(st.session_state['Y_custom'])
            event_info = []
            for d in range(D):
                if E_custom_display[d] < T - 1:
                    event_info.append({
                        'Disease': disease_names[d] if d < len(disease_names) else f'Disease {d}',
                        'Event Time': int(E_custom_display[d]),
                        'Age at Event': age_offset_custom_display + int(E_custom_display[d])
                    })
            
            if event_info:
                st.markdown("**⏱️ Event Times (used in loss calculation):**")
                st.caption("Only the first occurrence of each disease contributes to the loss. Times after the event are censored.")
                st.dataframe(pd.DataFrame(event_info), use_container_width=True)
            
            # Show timeline visualization option
            show_timeline = st.checkbox("Show Comprehensive Timeline View", value=False, key="show_timeline_custom")
            
            if show_timeline:
                st.subheader("📈 Comprehensive Disease Timeline")
                fig_timeline = plot_patient_timeline_comprehensive(
                    st.session_state['pi_custom'],
                    st.session_state['theta_custom'],
                    st.session_state['Y_custom'],
                    disease_names,
                    age_offset=age_offset_custom_display,
                    cluster_assignments=cluster_assignments,
                    figsize=(20, 14)
                )
                st.pyplot(fig_timeline)
                plt.close(fig_timeline)
            else:
                # Disease filtering for custom patient too
                col1, col2 = st.columns(2)
                with col1:
                    max_diseases_custom = st.slider("Max Diseases to Show", 5, 30, 15, 
                                                    key="max_diseases_custom",
                                                    help="Limit number of diseases shown")
                with col2:
                    min_prob_custom = st.slider("Min Probability Threshold", 0.0, 0.01, 0.001, 0.0001,
                                               format="%.4f", key="min_prob_custom",
                                               help="Only show diseases above this threshold")
                
                fig = plot_predictions(
                    st.session_state['pi_custom'],
                    st.session_state['theta_custom'],
                    st.session_state['Y_custom'],
                    disease_names,
                    age_offset=age_offset_custom_display,
                    cluster_assignments=cluster_assignments,
                    max_diseases_to_show=max_diseases_custom,
                    min_prob_threshold=min_prob_custom
                )
                st.pyplot(fig)
                plt.close(fig)
            
            if 'losses_custom' in st.session_state:
                fig_loss, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state['losses_custom'])
                ax.set_title('Training Loss (Custom Patient)')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_loss)

if __name__ == "__main__":
    main()