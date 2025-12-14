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
def load_averaged_gamma_from_batches(batch_dir, pattern="enrollment_model_W0.0001_batch_*_*.pt"):
    """Load and average gamma from multiple batch checkpoints
    
    Args:
        batch_dir: Directory containing batch model checkpoints
        pattern: Glob pattern to match batch checkpoint files
        
    Returns:
        Averaged gamma array or None if no checkpoints found
    """
    checkpoint_files = glob.glob(os.path.join(batch_dir, pattern))
    
    if not checkpoint_files:
        print(f"No batch checkpoints found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(checkpoint_files)} batch checkpoints. Loading gamma...")
    all_gammas = []
    
    for checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract gamma
            gamma = None
            if 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
            
            if gamma is not None:
                # Detach if tensor and convert to numpy
                if torch.is_tensor(gamma):
                    gamma = gamma.detach().cpu().numpy()
                elif not isinstance(gamma, np.ndarray):
                    gamma = np.array(gamma)
                
                # Check if gamma is all zeros (might indicate untrained model)
                if np.allclose(gamma, 0):
                    print(f"  Warning: {os.path.basename(checkpoint_path)} has gamma=0 (possibly untrained)")
                else:
                all_gammas.append(gamma)
                print(f"  Loaded gamma from {os.path.basename(checkpoint_path)} (shape: {gamma.shape})")
            else:
                print(f"  Warning: No gamma found in {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"  Warning: Could not load gamma from {os.path.basename(checkpoint_path)}: {e}")
            continue
    
    if not all_gammas:
        print("No gamma values found in any checkpoint!")
        return None
    
    # Debug: Check gamma values before averaging
    print(f"\nDebugging gamma values:")
    for i, gamma in enumerate(all_gammas):
        gamma_min = np.min(gamma)
        gamma_max = np.max(gamma)
        gamma_mean = np.mean(gamma)
        gamma_std = np.std(gamma)
        non_zero_count = np.count_nonzero(gamma)
        total_count = gamma.size
        print(f"  Batch {i+1}: shape={gamma.shape}, min={gamma_min:.6f}, max={gamma_max:.6f}, "
              f"mean={gamma_mean:.6f}, std={gamma_std:.6f}, non-zero={non_zero_count}/{total_count}")
    
    # Stack and average
    gamma_stack = np.stack(all_gammas)
    gamma_mean = np.mean(gamma_stack, axis=0)
    
    # Debug: Check averaged gamma
    print(f"\nAveraged gamma stats:")
    print(f"  Shape: {gamma_mean.shape}")
    print(f"  Min: {np.min(gamma_mean):.6f}, Max: {np.max(gamma_mean):.6f}")
    print(f"  Mean: {np.mean(gamma_mean):.6f}, Std: {np.std(gamma_mean):.6f}")
    print(f"  Non-zero: {np.count_nonzero(gamma_mean)}/{gamma_mean.size}")
    
    print(f"✓ Averaged gamma from {len(all_gammas)} batches. Final shape: {gamma_mean.shape}")
    return gamma_mean

@st.cache_data
def load_averaged_gamma_file(gamma_file_path):
    """Load pre-saved averaged gamma from file
    
    Args:
        gamma_file_path: Path to saved gamma file (.pt or .npy)
        
    Returns:
        Gamma array or None if file not found
    """
    if not os.path.exists(gamma_file_path):
        return None
    
    try:
        if gamma_file_path.endswith('.npy'):
            # Load from numpy file
            gamma = np.load(gamma_file_path)
            print(f"✓ Loaded gamma from numpy file: {gamma_file_path}")
            print(f"  Shape: {gamma.shape}")
            return gamma
        elif gamma_file_path.endswith('.pt'):
            # Load from PyTorch file
            checkpoint = torch.load(gamma_file_path, map_location='cpu', weights_only=False)
            if 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
                if torch.is_tensor(gamma):
                    gamma = gamma.cpu().numpy()
                print(f"✓ Loaded gamma from PyTorch file: {gamma_file_path}")
                print(f"  Shape: {gamma.shape}")
                if 'n_batches' in checkpoint:
                    print(f"  Averaged from {checkpoint['n_batches']} batches")
                return gamma
            else:
                print(f"Warning: 'gamma' key not found in checkpoint file")
                return None
        else:
            print(f"Warning: Unsupported file format. Use .npy or .pt")
            return None
    except Exception as e:
        print(f"Error loading gamma from {gamma_file_path}: {e}")
        return None

@st.cache_data
def load_fixed_components(data_dir, gamma_checkpoint_path=None, batch_dir=None, gamma_pattern="enrollment_model_W0.0001_batch_*_*.pt", gamma_file_path=None):
    """Load fixed components (phi, psi, gamma, etc.) from checkpoint
    
    Args:
        data_dir: Directory containing master checkpoint and essentials
        gamma_checkpoint_path: Optional path to a single trained batch model checkpoint to load gamma from.
        batch_dir: Optional directory containing multiple batch checkpoints. If provided, will average gamma from all batches.
        gamma_pattern: Glob pattern to match batch checkpoint files (used if batch_dir is provided).
    """
    checkpoint_path = os.path.join(data_dir, 'master_for_fitting_pooled_correctedE.pt')
    essentials_path = os.path.join(data_dir, 'model_essentials.pt')
    refs_path = os.path.join(data_dir, 'reference_trajectories.pt')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    phi = checkpoint['model_state_dict']['phi'].cpu().numpy()
    psi = checkpoint['model_state_dict']['psi'].cpu().numpy()
    
    # Load gamma - prioritize: saved file > batch_dir (averaged) > single checkpoint > master checkpoint
    gamma = None
    
    if gamma_file_path and os.path.exists(gamma_file_path):
        # Try to load pre-saved averaged gamma file first
        print(f"Attempting to load pre-saved gamma from file: {gamma_file_path}")
        gamma = load_averaged_gamma_file(gamma_file_path)
    
    if gamma is None and batch_dir and os.path.exists(batch_dir):
        # Try to load averaged gamma from multiple batches
        print(f"Attempting to load averaged gamma from batch directory: {batch_dir}")
        gamma = load_averaged_gamma_from_batches(batch_dir, gamma_pattern)
    
    if gamma is None and gamma_checkpoint_path and os.path.exists(gamma_checkpoint_path):
        # Fallback to single checkpoint
        print(f"Loading gamma from single trained model: {gamma_checkpoint_path}")
        gamma_checkpoint = torch.load(gamma_checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in gamma_checkpoint and 'gamma' in gamma_checkpoint['model_state_dict']:
            gamma = gamma_checkpoint['model_state_dict']['gamma'].cpu().numpy()
        elif 'gamma' in gamma_checkpoint:
            gamma = gamma_checkpoint['gamma']
            if torch.is_tensor(gamma):
                gamma = gamma.cpu().numpy()
    
    if gamma is None:
        # Try to load gamma from master checkpoint (unlikely to be there)
        if 'gamma' in checkpoint.get('model_state_dict', {}):
            gamma = checkpoint['model_state_dict']['gamma'].cpu().numpy()
        elif 'gamma' in checkpoint:
            gamma = checkpoint['gamma'].cpu().numpy() if torch.is_tensor(checkpoint['gamma']) else checkpoint['gamma']
    
    if gamma is None:
        st.warning("⚠️ Gamma not found. Please provide either a batch directory or a trained batch model checkpoint path.")
    
    # Load essentials
    essentials = torch.load(essentials_path, map_location='cpu', weights_only=False)
    
    # Load signature refs
    refs = torch.load(refs_path, map_location='cpu', weights_only=False)
    signature_refs = refs['signature_refs']
    
    return {
        'phi': phi,
        'psi': psi,
        'gamma': gamma,  # May be None if not available
        'prevalence_t': essentials['prevalence_t'],
        'signature_refs': signature_refs,
        'disease_names': essentials['disease_names']
    }

@st.cache_data
def load_sample_patients(data_dir, n_patients=10):
    """Load a small subset of Y and G for sample patients"""
    Y_path = os.path.join(data_dir, 'Y_tensor.pt')
    G_path = os.path.join(data_dir, 'G_matrix.pt')
    
    Y_full = torch.load(Y_path, map_location='cpu', weights_only=False)
    G_full = torch.load(G_path, map_location='cpu', weights_only=False)
    
    # Take first n_patients
    Y_sample = Y_full[:n_patients].numpy()
    G_sample = G_full[:n_patients].numpy()
    
    return Y_sample, G_sample

def fit_patient_model(patient_G, patient_Y, patient_E, fixed_components, num_epochs=30, learning_rate=0.1):
    """Fit model for a single patient in real time using fixed-gamma model"""
    # Check that gamma is available
    if fixed_components.get('gamma') is None:
        raise ValueError("Gamma must be provided for fixed-gamma model. Please load gamma from a trained batch model checkpoint.")
    
    # Get expected P from gamma shape
    gamma = fixed_components['gamma']
    expected_P = gamma.shape[0]  # Gamma is [P, K_total]
    
    # Handle both 2D [D, T] and 3D [N, D, T] inputs - use N=1 for single patient
    if len(patient_Y.shape) == 2:
        # 2D input: keep as [1, D, T] for single patient
        D, T = patient_Y.shape
        patient_Y = patient_Y[np.newaxis, :, :]  # [1, D, T]
    else:
        # 3D input: use first patient only
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
        N=1,  # Single patient - normalization will be skipped
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
        pretrained_gamma=fixed_components['gamma'],  # Fixed gamma from trained model
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
    # Handle both 2D [D, T] and 3D [N, D, T] inputs
    if len(Y.shape) == 3:
        Y = Y[0]  # Take first patient if 3D
    
    D, T = Y.shape
    event_times = np.full(D, T - 1)  # Default to censoring at end (T-1 since 0-indexed)
    
    for d in range(D):
        # Find first occurrence of disease
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
        # Try alternative paths
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
        
        # Handle different formats
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

def plot_predictions(pi, theta, Y, disease_names, time_window=None, age_offset=30, cluster_assignments=None):
    """Plot disease probabilities and signature proportions with signature-based colors"""
    D, T = pi.shape
    if time_window is None:
        time_window = range(T)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Get signature colors (same as used in signature plot)
    K = theta.shape[0]
    sig_colors = sns.color_palette("tab20", K)
    
    # Plot 1: Disease probabilities
    ax1 = axes[0]
    
    # Get diseases that occurred
    diseases_with_events = []
    for d in range(D):
        if np.any(Y[d, :] > 0.5):
            diseases_with_events.append(d)
    
    # Map diseases to signature colors
    disease_color_map = {}
    if cluster_assignments is not None and len(cluster_assignments) >= D:
        # Use signature-based colors
        for d in diseases_with_events:
            sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
            # Ensure signature index is valid
            if 0 <= sig < K:
                disease_color_map[d] = sig_colors[sig]
            else:
                # Fallback: use a default color
                disease_color_map[d] = 'gray'
    else:
        # Fallback: use distinct colors if cluster assignments not available
        if len(diseases_with_events) > 0:
            colors = sns.color_palette("husl", len(diseases_with_events))
            disease_color_map = {d: colors[i] for i, d in enumerate(diseases_with_events)}
    
    for d in diseases_with_events:
        disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
        color = disease_color_map.get(d, 'gray')
        
        # Get signature assignment for label
        sig_label = ""
        if cluster_assignments is not None and d < len(cluster_assignments):
            sig = int(cluster_assignments[d])
            sig_label = f" (Sig {sig})"
        
        # Plot probability curve
        ax1.plot(time_window, pi[d, time_window], 
                label=f"{disease_name}{sig_label}",
                color=color, linewidth=2, alpha=0.7)
        
        # Mark diagnosis times
        diag_times = np.where(Y[d, time_window] > 0.5)[0]
        if len(diag_times) > 0:
            for t in diag_times:
                ax1.axvline(x=t, color=color, linestyle='--', alpha=0.5)
                ax1.scatter(t, pi[d, t], color=color, s=80, zorder=10, marker='o', edgecolors='black', linewidths=1)
    
    ax1.set_title('Disease Probabilities Over Time (colored by primary signature)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (years from baseline)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signature proportions
    ax2 = axes[1]
    for k in range(K):
        ax2.plot(time_window, theta[k, time_window], 
                label=f'Signature {k}', color=sig_colors[k], linewidth=2, alpha=0.7)
    
    ax2.set_title('Signature Proportions Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (years from baseline)', fontsize=12)
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def pad_G_to_match_gamma(patient_G, gamma):
    """Pad patient_G to match gamma's expected P dimension (handles missing PCs/sex)."""
    # Get expected P from gamma shape
    expected_P = gamma.shape[0]  # Gamma is [P, K_total]
    
    # Flatten G to 1D
    G_flat = np.array(patient_G).flatten() if isinstance(patient_G, (list, np.ndarray)) else patient_G
    if len(G_flat.shape) > 1:
        G_flat = G_flat[0] if G_flat.shape[0] == 1 else G_flat.flatten()
    
    P_input = len(G_flat)
    
    # Pad if necessary (missing PCs/sex)
    if P_input < expected_P:
        padding = np.zeros(expected_P - P_input)
        G_flat = np.concatenate([G_flat, padding])
    elif P_input > expected_P:
        raise ValueError(f"Patient G has {P_input} components but gamma expects {expected_P}.")
    
    return G_flat

def plot_counterfactual_signature_trajectory(model, signature_idx, patient_G, time_window=None):
    """Plot actual vs counterfactual signature trajectory (PRS=0) for a patient."""
    if time_window is None:
        time_window = range(model.T)
    
    # Get actual lambda and theta
    lambda_actual = model.lambda_[0, signature_idx, :].detach().cpu().numpy()  # [T]
    theta_actual = torch.softmax(model.lambda_[0], dim=0)[signature_idx, :].detach().cpu().numpy()  # [T]
    
    # Compute counterfactual: remove genetic effect
    # Pad G to match gamma dimensions (handles missing PCs/sex)
    G_padded = pad_G_to_match_gamma(patient_G, model.gamma)
    gamma_k = model.gamma[:, signature_idx].detach().cpu().numpy()
    genetic_effect = float(model.genetic_scale * np.dot(G_padded, gamma_k))  # Ensure scalar
    
    # Debug: print genetic effect to verify it's non-zero
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
    # Handle signature_refs - could be scalar or time-varying
    ref_val = model.signature_refs[signature_idx]
    if isinstance(ref_val, torch.Tensor):
        ref_val = ref_val.detach().cpu().numpy()
        if ref_val.ndim > 0:
            # If time-varying, use mean or first value
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
    # Handle signature_refs - could be scalar or time-varying
    ref_val = model.signature_refs[signature_idx]
    if isinstance(ref_val, torch.Tensor):
        ref_val = ref_val.detach().cpu().numpy()
        if ref_val.ndim > 0:
            # If time-varying, use mean or first value
            reference = float(ref_val.mean() if len(ref_val) > 1 else ref_val[0])
        else:
            reference = float(ref_val.item())
    else:
        reference = float(ref_val)
    
    # Pad G to match gamma dimensions (handles missing PCs/sex)
    G_padded = pad_G_to_match_gamma(patient_G, model.gamma)
    gamma_k = model.gamma[:, signature_idx].detach().cpu().numpy()
    genetic_effect = float(model.genetic_scale * np.dot(G_padded, gamma_k))  # Ensure scalar
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
        diseases_to_show = range(min(20, D))  # Show top 20 diseases
    
    # Compute average contribution: theta[k, t] * phi_prob[k, d, t] averaged over time
    contributions = np.zeros((K, len(diseases_to_show)))
    for i, d in enumerate(diseases_to_show):
        for k in range(K):
            contrib_t = theta[k, time_window] * phi_prob[k, d, time_window]
            contributions[k, i] = np.mean(contrib_t)
    
    fig, ax = plt.subplots(figsize=(max(12, len(diseases_to_show) * 0.5), max(6, K * 0.3)))
    
    # Create labels
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
    # Pad G to match gamma dimensions (handles missing PCs/sex)
    G_padded = pad_G_to_match_gamma(patient_G, model.gamma)
    P = len(G_padded)
    
    # Compute genetic effects: G * gamma for each signature (element-wise)
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

def main():
    st.title("🔬 Real-Time Patient Refitting")
    st.markdown("**Fit model in real time for individual patients with editable disease history**")
    
    # Sidebar for data directory
    st.sidebar.header("📁 Data Configuration")
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/",
        help="Path to directory containing model checkpoints and data"
    )
    
    # Gamma loading options
    st.sidebar.subheader("Gamma Loading")
    gamma_option = st.sidebar.radio(
        "Gamma Source",
        ["Pre-saved Averaged Gamma", "Average from Batch Directory", "Single Checkpoint", "Try Master Checkpoint"],
        help="Choose how to load gamma. Recommended: Pre-saved Averaged Gamma (fastest) or Average from Batch Directory."
    )
    
    batch_dir = None
    gamma_checkpoint_path = None
    gamma_file_path = None
    gamma_pattern = "enrollment_model_W0.0001_batch_*_*.pt"  # Default pattern
    
    if gamma_option == "Pre-saved Averaged Gamma":
        gamma_file_path = st.sidebar.text_input(
            "Gamma File Path",
            value="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/averaged_gamma_from_batches.pt",
            help="Path to pre-saved averaged gamma file (.pt or .npy)"
        )
    elif gamma_option == "Average from Batch Directory":
        batch_dir = st.sidebar.text_input(
            "Batch Directory",
            value="",
            help="Directory containing batch model checkpoints (e.g., enrollment_retrospective_full/)"
        )
        gamma_pattern = st.sidebar.text_input(
            "Checkpoint Pattern",
            value="enrollment_model_W0.0001_batch_*_*.pt",
            help="Glob pattern to match batch checkpoint files"
        )
    elif gamma_option == "Single Checkpoint":
        gamma_checkpoint_path = st.sidebar.text_input(
            "Gamma Checkpoint Path",
            value="",
            help="Path to a single trained batch model checkpoint (.pt file)"
        )
    
    # Load fixed components
    if st.sidebar.button("Load Model Components", type="primary"):
        try:
            with st.spinner("Loading fixed components..."):
                if gamma_option == "Pre-saved Averaged Gamma":
                    fixed_components = load_fixed_components(
                        data_dir,
                        gamma_file_path=gamma_file_path if gamma_file_path else None
                    )
                elif gamma_option == "Average from Batch Directory":
                    fixed_components = load_fixed_components(
                        data_dir, 
                        batch_dir=batch_dir if batch_dir else None,
                        gamma_pattern=gamma_pattern
                    )
                elif gamma_option == "Single Checkpoint":
                    fixed_components = load_fixed_components(
                        data_dir, 
                        gamma_checkpoint_path=gamma_checkpoint_path if gamma_checkpoint_path else None
                    )
                else:  # Try Master Checkpoint
                    fixed_components = load_fixed_components(data_dir)
                
                if fixed_components.get('gamma') is None:
                    st.sidebar.warning("⚠️ Gamma not loaded. Please provide a batch directory or checkpoint path.")
                else:
                    st.sidebar.success(f"✓ Components loaded! Gamma shape: {fixed_components['gamma'].shape}")
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
    T = fixed_components['phi'].shape[2]  # Get T from phi shape
    
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
        if st.button("Load Sample Patients"):
            try:
                with st.spinner("Loading sample patients..."):
                    Y_sample, G_sample = load_sample_patients(data_dir, n_patients=10)
                    st.session_state['Y_sample'] = Y_sample
                    st.session_state['G_sample'] = G_sample
                    st.success(f"Loaded {len(Y_sample)} sample patients!")
            except Exception as e:
                st.error(f"Error loading sample patients: {e}")
        
        if 'Y_sample' in st.session_state:
            patient_idx = st.selectbox(
                "Select Patient",
                range(len(st.session_state['Y_sample'])),
                format_func=lambda x: f"Patient {x}"
            )
            
            Y_patient = st.session_state['Y_sample'][patient_idx]  # [D, T]
            G_patient = st.session_state['G_sample'][patient_idx]  # [P]
            
            st.subheader("Patient Disease History")
            
            # Show diagnosis progression narrative
            age_offset = st.number_input("Patient Age at Baseline", min_value=0, max_value=100, value=30, 
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
            
            # Create editable dataframe
            time_cols = [f"t={t}" for t in range(min(20, T))]  # Show first 20 time points
            edit_data = {}
            for d in range(min(10, D)):  # Show first 10 diseases
                disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
                edit_data[disease_name] = Y_patient[d, :min(20, T)]
            
            edited_df = st.data_editor(
                pd.DataFrame(edit_data, index=time_cols).T,
                use_container_width=True,
                num_rows="fixed"
            )
            
            # Update Y_patient with edited values
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
                    # Create event times
                    E_patient = create_event_times_from_Y(Y_patient)
                    
                    # Fit model
                    pi, theta, model, losses = fit_patient_model(
                        G_patient, Y_patient, E_patient, 
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
                
                # Plot predictions
                age_offset = st.number_input("Age at Baseline for Plot", min_value=0, max_value=100, value=30, 
                                             key="age_plot", help="Age at time 0 (baseline)")
                fig = plot_predictions(
                    st.session_state['pi'],
                    st.session_state['theta'],
                    st.session_state['Y_current'],
                    disease_names,
                    age_offset=age_offset,
                    cluster_assignments=cluster_assignments
                )
                st.pyplot(fig)
                
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
                    
                    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                        "Counterfactual Genetics", 
                        "GP Projection", 
                        "Signature Contributions",
                        "Genetic Decomposition"
                    ])
                    
                    model_current = st.session_state['model']
                    # Use model's G (the actual G used for fitting, properly padded)
                    G_from_model = model_current.G[0].detach().cpu().numpy()  # [P]
                    
                    with viz_tab1:
                        st.markdown("**What if this patient had PRS=0?** See how signature trajectories would change.")
                        sig_idx_cf = st.selectbox(
                            "Select Signature for Counterfactual",
                            range(model_current.K),
                            format_func=lambda x: f"Signature {x}",
                            key="cf_sig"
                        )
                        if st.button("Generate Counterfactual Plot", key="cf_btn"):
                            fig_cf = plot_counterfactual_signature_trajectory(
                                model_current, sig_idx_cf, G_from_model
                            )
                            st.pyplot(fig_cf)
                            st.caption("**Interpretation:** The counterfactual shows what would happen if all PRS values were set to zero. The difference shows the genetic contribution to this signature's trajectory.")
                    
                    with viz_tab2:
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
                    
                    with viz_tab3:
                        st.markdown("**Signature Contribution Heatmap:** Which signatures drive risk for each disease?")
                        n_diseases_heatmap = st.slider(
                            "Number of Diseases to Show",
                            5, min(50, D), 20,
                            key="n_diseases_heatmap"
                        )
                        # Show diseases with highest mean risk
                        mean_risks = np.mean(st.session_state['pi'], axis=1)
                        top_diseases = np.argsort(mean_risks)[-n_diseases_heatmap:][::-1]
                        
                        if st.button("Generate Contribution Heatmap", key="heatmap_btn"):
                            with torch.no_grad():
                                _, theta_current, phi_prob_current = model_current.forward()
                            fig_heatmap = plot_signature_contribution_heatmap(
                                st.session_state['pi'],
                                theta_current[0].cpu().numpy(),
                                phi_prob_current[0].cpu().numpy(),
                                disease_names,
                                diseases_to_show=top_diseases
                            )
                            st.pyplot(fig_heatmap)
                            st.caption("**Interpretation:** Each cell shows how much a signature contributes to a disease's risk (θ × φ). Darker colors = higher contribution.")
                    
                    with viz_tab4:
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
                else:
                    st.info("👆 Please fit the model first to see advanced visualizations.")
    
    with tab2:
        st.header("Create Custom Patient")
        st.markdown("**Manually create a patient with custom genetic data and disease history**")
        
        # Genetic data input
        st.subheader("🧬 Genetic Data")
        st.markdown("**Enter PRS values (36 PRS + 1 sex + 10 PCs = 47 values):**")
        
        # Default: zeros or random
        if st.button("Generate Random Genetic Data"):
            G_custom = np.random.randn(47)
            st.session_state['G_custom'] = G_custom
        elif 'G_custom' not in st.session_state:
            G_custom = np.zeros(47)
            st.session_state['G_custom'] = G_custom
        
        # Editable genetic data
        G_df = pd.DataFrame({
            'Value': st.session_state['G_custom']
        }, index=[f'Feature {i}' for i in range(47)])
        
        edited_G = st.data_editor(G_df, use_container_width=True, num_rows="fixed")
        st.session_state['G_custom'] = edited_G['Value'].values
        
        # Custom disease history
        st.subheader("📋 Disease History")
        st.markdown("**Create custom disease timeline:**")
        st.info("💡 **Note:** When a disease appears, it only contributes to the loss at the **first occurrence** (event time). Times after the event don't contribute to the loss - the patient is censored after the event occurs. You can set a disease to persist in the visualization, but only the first time point will be used as the event time.")
        
        # Initialize Y
        if 'Y_custom' not in st.session_state:
            st.session_state['Y_custom'] = np.zeros((D, T))
        
        # Show current diagnosis progression
        age_offset_custom_input = st.number_input("Patient Age at Baseline", min_value=0, max_value=100, value=30, 
                                                  key="age_custom_input", help="Age at time 0 (baseline)")
        progression_text_custom_display = get_diagnosis_progression(
            st.session_state['Y_custom'], disease_names, age_offset=age_offset_custom_input
        )
        st.markdown("**📖 Current Diagnosis Progression:**")
        st.markdown(progression_text_custom_display)
        
        # Disease selector
        selected_disease = st.selectbox(
            "Select Disease to Add/Edit",
            range(D),
            format_func=lambda x: disease_names[x] if x < len(disease_names) else f'Disease {x}'
        )
        
        # Show signature for selected disease
        if cluster_assignments is not None and selected_disease < len(cluster_assignments):
            sig = int(cluster_assignments[selected_disease])
            st.caption(f"📌 This disease belongs to **Signature {sig}** (will be colored accordingly in plots)")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.slider("Start Time (Event Time)", 0, T-1, 0, key="start", 
                                   help="First time point where disease appears (this becomes the event time)")
        with col2:
            end_time = st.slider("End Time (Visualization)", start_time, T-1, min(20, T-1), key="end",
                                help="Last time point for visualization (only start_time contributes to loss)")
        
        # Add/remove diagnosis
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Diagnosis", key="add"):
                st.session_state['Y_custom'][selected_disease, start_time:end_time+1] = 1.0
                st.success(f"Added {disease_names[selected_disease]} at times {start_time}-{end_time} (event time: {start_time})")
        with col2:
            if st.button("➖ Remove Diagnosis", key="remove"):
                st.session_state['Y_custom'][selected_disease, start_time:end_time+1] = 0.0
                st.success(f"Removed {disease_names[selected_disease]} at times {start_time}-{end_time}")
        
        # Show current Y matrix with signature-based styling
        st.markdown("**Current Disease Matrix (first 20 diseases × first 20 time points):**")
        Y_display = st.session_state['Y_custom'][:min(20, D), :min(20, T)]
        
        # Create styled dataframe with signature colors
        display_df = pd.DataFrame(
            Y_display,
            index=[disease_names[i] if i < len(disease_names) else f'Disease {i}' 
                   for i in range(min(20, D))],
            columns=[f't={t}' for t in range(min(20, T))]
        )
        
        # Apply signature-based row colors if cluster assignments available
        if cluster_assignments is not None:
            K = fixed_components['phi'].shape[0] if 'phi' in fixed_components else 20
            sig_colors = sns.color_palette("tab20", K)
            
            # Create a styled dataframe with signature-based row colors
            def highlight_row(row):
                # Find the disease index by matching the row name
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
                        # Convert RGB to hex with light transparency
                        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color)
                        return [f'background-color: {hex_color}30' for _ in row]
                return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            st.caption("💡 Row colors indicate the primary signature for each disease (matching plot colors)")
        else:
            st.dataframe(display_df, use_container_width=True)
        
        # Training parameters
        st.subheader("⚙️ Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            num_epochs_custom = st.slider("Number of Epochs", 10, 100, 30, key="epochs_custom")
        with col2:
            learning_rate_custom = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lr_custom")
        
        # Fit custom patient
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
        
        # Display custom patient results
        if 'pi_custom' in st.session_state:
            st.subheader("📈 Custom Patient Predictions")
            
            # Get age offset for display (use the same one from input section)
            age_offset_custom_display = st.number_input("Patient Age at Baseline for Plot", min_value=0, max_value=100, value=30, 
                                                        key="age_custom_display", help="Age at time 0 (baseline)")
            
            # Show diagnosis progression for custom patient
            progression_text_custom = get_diagnosis_progression(
                st.session_state['Y_custom'], disease_names, age_offset=age_offset_custom_display
            )
            st.markdown("**📖 Diagnosis Progression:**")
            st.markdown(progression_text_custom)
            
            # Show event times info
            E_custom_display = create_event_times_from_Y(st.session_state['Y_custom'])
            event_info = []
            for d in range(D):
                if E_custom_display[d] < T - 1:  # Disease occurred (not censored at end)
                    event_info.append({
                        'Disease': disease_names[d] if d < len(disease_names) else f'Disease {d}',
                        'Event Time': int(E_custom_display[d]),
                        'Age at Event': age_offset_custom_display + int(E_custom_display[d])
                    })
            
            if event_info:
                st.markdown("**⏱️ Event Times (used in loss calculation):**")
                st.caption("Only the first occurrence of each disease contributes to the loss. Times after the event are censored.")
                st.dataframe(pd.DataFrame(event_info), use_container_width=True)
            
            fig = plot_predictions(
                st.session_state['pi_custom'],
                st.session_state['theta_custom'],
                st.session_state['Y_custom'],
                disease_names,
                age_offset=age_offset_custom_display,
                cluster_assignments=cluster_assignments
            )
            st.pyplot(fig)
            
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

