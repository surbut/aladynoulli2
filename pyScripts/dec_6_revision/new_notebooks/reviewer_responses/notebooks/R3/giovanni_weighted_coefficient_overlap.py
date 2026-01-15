"""
Giovanni's Weighted Coefficient Overlap Analysis

Implements Giovanni's suggestion: 
  sum(coefficients in overlap) / sum(all coefficients in UKBB)
  
This is bounded to [0, 1] and represents what fraction of the UKB signature's
total coefficient weight is in the overlap with another cohort's signature.

Key advantages:
  - Bounded to [0, 1] (unlike the previous weighted Jaccard that could exceed 1.0)
  - Uses only UKB coefficients, avoiding scale differences across cohorts
  - Interpretable: "What fraction of UKB signature's weight overlaps with other signature?"
  
Note: For cross-cohort comparison, normalization by max psi within each cohort
      might be considered, but Giovanni's formula avoids this by using only UKB coefficients.
"""

import numpy as np
import pandas as pd
import torch
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.special import expit  # sigmoid function

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def pool_psi_from_batches(batch_dir, pattern, max_batches=None):
    """
    Load and pool psi from all batch files (average across batches).
    
    Args:
        batch_dir: Directory containing batch files
        pattern: Glob pattern for batch files (e.g., "aou_model_batch_*.pt")
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled psi (mean across batches) as numpy array, shape (K, D)
        Pooled psi std (std across batches) as numpy array, shape (K, D)
    """
    batch_dir = Path(batch_dir)
    all_psis = []
    
    # Find all matching files
    files = sorted(glob.glob(str(batch_dir / pattern)))
    print(f"  Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Extract psi
            if 'model_state_dict' in checkpoint and 'psi' in checkpoint['model_state_dict']:
                psi = checkpoint['model_state_dict']['psi']
            elif 'psi' in checkpoint:
                psi = checkpoint['psi']
            else:
                continue
            
            if torch.is_tensor(psi):
                psi = psi.detach().cpu().numpy()
            
            all_psis.append(psi)
        except Exception as e:
            print(f"    Warning: Could not load {Path(file_path).name}: {e}")
            continue
    
    if len(all_psis) == 0:
        raise ValueError(f"No valid psi found in {batch_dir} matching {pattern}")
    
    # Stack and compute mean/std
    all_psis = np.stack(all_psis, axis=0)  # Shape: (n_batches, K, D)
    psi_mean = np.mean(all_psis, axis=0)
    psi_std = np.std(all_psis, axis=0)
    
    print(f"  ✓ Pooled {len(all_psis)} batches: shape {psi_mean.shape}")
    return psi_mean, psi_std


def load_psi_from_checkpoint(checkpoint_path):
    """Load psi from a single checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint and 'psi' in checkpoint['model_state_dict']:
        psi = checkpoint['model_state_dict']['psi']
    elif 'psi' in checkpoint:
        psi = checkpoint['psi']
    else:
        raise ValueError(f"No psi found in {checkpoint_path}")
    
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    return psi


def pool_phi_from_batches(batch_dir, pattern, max_batches=None):
    """
    Load and pool phi from all batch files (average across batches, then average over time).
    
    Args:
        batch_dir: Directory containing batch files
        pattern: Glob pattern for batch files (e.g., "aou_model_batch_*.pt")
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        Pooled phi (mean across batches, then mean over time) as numpy array, shape (K, D)
        Pooled phi std (std across batches) as numpy array, shape (K, D, T)
    """
    batch_dir = Path(batch_dir)
    all_phis = []
    
    # Find all matching files
    files = sorted(glob.glob(str(batch_dir / pattern)))
    print(f"  Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Extract phi
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                continue
            
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            all_phis.append(phi)
        except Exception as e:
            print(f"    Warning: Could not load {Path(file_path).name}: {e}")
            continue
    
    if len(all_phis) == 0:
        raise ValueError(f"No valid phi found in {batch_dir} matching {pattern}")
    
    # Stack and compute mean/std
    all_phis = np.stack(all_phis, axis=0)  # Shape: (n_batches, K, D, T)
    phi_mean_batches = np.mean(all_phis, axis=0)  # Shape: (K, D, T)
    phi_std_batches = np.std(all_phis, axis=0)  # Shape: (K, D, T)
    
    # Average over time dimension
    phi_mean = np.mean(phi_mean_batches, axis=2)  # Shape: (K, D)
    
    print(f"  ✓ Pooled {len(all_phis)} batches: phi shape {phi_mean_batches.shape}, time-averaged shape {phi_mean.shape}")
    return phi_mean, phi_std_batches


def load_phi_from_checkpoint(checkpoint_path):
    """Load phi from a single checkpoint file and average over time dimension."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
        phi = checkpoint['model_state_dict']['phi']
    elif 'phi' in checkpoint:
        phi = checkpoint['phi']
    else:
        raise ValueError(f"No phi found in {checkpoint_path}")
    
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    
    # Average over time dimension (axis 2)
    phi_time_avg = np.mean(phi, axis=2)  # Shape: (K, D)
    
    return phi_time_avg


def compute_batch_consistency_stats(batch_dir, pattern, posterior_clusters, disease_names, cohort_name, max_batches=None):
    """
    Compute batch-level consistency: For each disease, count how many batches
    assigned it to the same signature as the final posterior assignment.
    
    Args:
        batch_dir: Directory containing batch files
        pattern: Glob pattern for batch files
        posterior_clusters: Final cluster assignments (from averaged psi), shape (D,)
        disease_names: List of disease names, length D
        cohort_name: Name of cohort (for printing)
        max_batches: Maximum number of batches to load (None = all)
    
    Returns:
        DataFrame with columns: Disease, Final_Sig, N_Batches_Match, N_Batches_Total, Pct_Consistent
    """
    batch_dir = Path(batch_dir)
    files = sorted(glob.glob(str(batch_dir / pattern)))
    
    if max_batches is not None:
        files = files[:max_batches]
    
    if len(files) == 0:
        raise ValueError(f"No files found matching {pattern} in {batch_dir}")
    
    # Load all batch psis and compute clusters for each
    batch_clusters = []
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint and 'psi' in checkpoint['model_state_dict']:
                psi = checkpoint['model_state_dict']['psi']
            elif 'psi' in checkpoint:
                psi = checkpoint['psi']
            else:
                continue
            
            if torch.is_tensor(psi):
                psi = psi.detach().cpu().numpy()
            
            # Compute clusters from this batch's psi
            batch_cluster = np.argmax(psi, axis=0)  # Shape (D,)
            batch_clusters.append(batch_cluster)
        except Exception as e:
            print(f"    Warning: Could not load {Path(file_path).name}: {e}")
            continue
    
    if len(batch_clusters) == 0:
        raise ValueError(f"No valid psi found in batches matching {pattern}")
    
    # Stack: shape (n_batches, D)
    batch_clusters = np.stack(batch_clusters, axis=0)
    n_batches = batch_clusters.shape[0]
    
    # For each disease, count how many batches matched the final assignment
    n_matches = np.sum(batch_clusters == posterior_clusters[np.newaxis, :], axis=0)  # Shape (D,)
    
    # Create DataFrame
    results = pd.DataFrame({
        'Disease': disease_names,
        'Final_Sig': posterior_clusters,
        'N_Batches_Match': n_matches,
        'N_Batches_Total': n_batches,
        'Pct_Consistent': 100 * n_matches / n_batches
    })
    
    return results


def compare_initial_vs_posterior_clusters(initial_clusters, posterior_clusters, disease_names, cohort_name):
    """
    Compare initial vs posterior cluster assignments for MGB.
    
    Args:
        initial_clusters: Initial cluster assignments, shape (D,)
        posterior_clusters: Posterior cluster assignments, shape (D,)
        disease_names: List of disease names, length D
        cohort_name: Name of cohort (for printing)
    
    Returns:
        DataFrame with columns: Disease, Initial_Sig, Posterior_Sig, Changed
    """
    changed_mask = initial_clusters != posterior_clusters
    
    results = pd.DataFrame({
        'Disease': disease_names,
        'Initial_Sig': initial_clusters,
        'Posterior_Sig': posterior_clusters,
        'Changed': changed_mask
    })
    
    return results


def compute_binary_jaccard_posterior(ukb_clusters, other_clusters,
                                     ukb_disease_names, other_disease_names, common_diseases):
    """
    Compute binary modified Jaccard using posterior cluster assignments.
    Same as original binary Jaccard but uses posterior clusters instead of initial.
    """
    # Convert to numpy arrays if needed
    if isinstance(ukb_clusters, torch.Tensor):
        ukb_clusters = ukb_clusters.numpy()
    if isinstance(other_clusters, torch.Tensor):
        other_clusters = other_clusters.numpy()
    
    # Convert disease names to lists
    if isinstance(ukb_disease_names, np.ndarray):
        ukb_disease_names = ukb_disease_names.tolist()
    if isinstance(other_disease_names, np.ndarray):
        other_disease_names = other_disease_names.tolist()
    
    # Create mapping from disease name to indices
    ukb_disease_to_idx = {d: i for i, d in enumerate(ukb_disease_names)}
    other_disease_to_idx = {d: i for i, d in enumerate(other_disease_names)}
    
    K_ukb = int(ukb_clusters.max() + 1)
    K_other = int(other_clusters.max() + 1)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((K_ukb, K_other))
    
    jaccard_scores = []
    cluster_details = {}
    
    for ukb_sig in range(K_ukb):
        # Get diseases in UKB signature (from common diseases)
        ukb_sig_diseases = [
            d for d in common_diseases
            if d in ukb_disease_to_idx and ukb_clusters[ukb_disease_to_idx[d]] == ukb_sig
        ]
        
        if len(ukb_sig_diseases) == 0:
            continue
        
        best_match_score = 0
        best_match_cluster = None
        best_intersection = set()
        
        for other_sig in range(K_other):
            # Get diseases in other signature (from common diseases)
            other_sig_diseases = [
                d for d in common_diseases
                if d in other_disease_to_idx and other_clusters[other_disease_to_idx[d]] == other_sig
            ]
            
            if len(other_sig_diseases) == 0:
                continue
            
            # Find overlap
            intersection = set(ukb_sig_diseases) & set(other_sig_diseases)
            
            # Modified Jaccard: intersection / UKB signature size
            jaccard_k = len(intersection) / len(ukb_sig_diseases) if len(ukb_sig_diseases) > 0 else 0.0
            
            # Store in similarity matrix
            similarity_matrix[ukb_sig, other_sig] = jaccard_k
            
            if jaccard_k > best_match_score:
                best_match_score = jaccard_k
                best_match_cluster = other_sig
                best_intersection = intersection
        
        jaccard_scores.append(best_match_score)
        cluster_details[ukb_sig] = {
            'jaccard': best_match_score,
            'best_match': best_match_cluster,
            'intersection': best_intersection
        }
    
    # Create best matches DataFrame
    best_matches = []
    for ukb_sig in range(K_ukb):
        if ukb_sig in cluster_details:
            best_matches.append({
                'UKB': ukb_sig,
                'Other': cluster_details[ukb_sig]['best_match'],
                'Jaccard': cluster_details[ukb_sig]['jaccard']
            })
    
    best_matches_df = pd.DataFrame(best_matches)
    
    return similarity_matrix, best_matches_df, cluster_details


def compute_giovanni_weighted_overlap_normalized(ukb_psi, other_psi, ukb_clusters, other_clusters,
                                                 ukb_disease_names, other_disease_names, common_diseases):
    """
    Compute Giovanni's weighted coefficient overlap using sigmoid(psi).
    
    Formula:
        Weighted Overlap(UKB_sig_k, Other_sig_k') = 
            Σ_{d ∈ overlap} sigmoid(ψ_{k,d}) / Σ_{d ∈ UKB_sig_k} sigmoid(ψ_{k,d})
    
    where:
        - overlap = {diseases in both UKB_sig_k and Other_sig_k'}
        - UKB_sig_k = {all diseases assigned to UKB signature k}
        - sigmoid(ψ_{k,d}) = expit(ψ_{k,d}) = 1 / (1 + exp(-ψ_{k,d}))
    
    This metric is bounded to [0, 1] and represents the fraction of UKB signature k's
    total coefficient weight (normalized via sigmoid) that overlaps with the other cohort's signature k'.
    
    Since we only use UKB psi values, no cross-cohort normalization is needed.
    """
    # Apply sigmoid to UKB psi to get positive weights
    # sigmoid maps to [0, 1]: -5 -> ~0.007, 0 -> 0.5, 5 -> ~0.993
    # Since we're only using UKB psi, no cross-cohort normalization needed
    ukb_psi_normalized = expit(ukb_psi)  # Shape: (K_ukb, D_ukb)
    
    # Convert to numpy arrays if needed
    if isinstance(ukb_clusters, torch.Tensor):
        ukb_clusters = ukb_clusters.numpy()
    if isinstance(other_clusters, torch.Tensor):
        other_clusters = other_clusters.numpy()
    
    # Convert disease names to lists
    if isinstance(ukb_disease_names, np.ndarray):
        ukb_disease_names = ukb_disease_names.tolist()
    if isinstance(other_disease_names, np.ndarray):
        other_disease_names = other_disease_names.tolist()
    
    # Create mapping from disease name to indices
    ukb_disease_to_idx = {d: i for i, d in enumerate(ukb_disease_names)}
    other_disease_to_idx = {d: i for i, d in enumerate(other_disease_names)}
    
    # Get number of signatures
    K_ukb = ukb_psi.shape[0]
    K_other = other_psi.shape[0]
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((K_ukb, K_other))
    
    # For each UKB signature
    for ukb_sig in range(K_ukb):
        # Get ALL diseases in UKB signature (for denominator)
        ukb_sig_all_indices = np.where(ukb_clusters == ukb_sig)[0]
        
        if len(ukb_sig_all_indices) == 0:
            continue
        
        # Compute total UKB coefficient weight for this signature (ALL diseases, normalized)
        ukb_total_weight = np.sum(ukb_psi_normalized[ukb_sig, ukb_sig_all_indices])
        
        # Skip if total weight is exactly zero (to avoid division by zero)
        if ukb_total_weight == 0:
            continue
        
        # Get diseases in UKB signature that are in common_diseases (for overlap computation)
        ukb_sig_common_diseases = [
            d for d in common_diseases
            if d in ukb_disease_to_idx and ukb_clusters[ukb_disease_to_idx[d]] == ukb_sig
        ]
        
        if len(ukb_sig_common_diseases) == 0:
            continue
        
        # For each other cohort signature
        for other_sig in range(K_other):
            # Get diseases in other signature (from common diseases only)
            other_sig_diseases = [
                d for d in common_diseases
                if d in other_disease_to_idx and other_clusters[other_disease_to_idx[d]] == other_sig
            ]
            
            if len(other_sig_diseases) == 0:
                continue
            
            # Find overlap (diseases in both signatures)
            overlap_diseases = set(ukb_sig_common_diseases) & set(other_sig_diseases)
            
            if len(overlap_diseases) == 0:
                continue
            
            # Compute overlap coefficient weight (sum of normalized UKB psi for diseases in overlap)
            overlap_indices = [ukb_disease_to_idx[d] for d in overlap_diseases]
            overlap_weight = np.sum(ukb_psi_normalized[ukb_sig, overlap_indices])
            
            # Giovanni's formula: overlap_weight / ukb_total_weight
            similarity_matrix[ukb_sig, other_sig] = overlap_weight / ukb_total_weight
    
    # Create best matches DataFrame
    best_matches = []
    for ukb_sig in range(K_ukb):
        best_match_idx = np.argmax(similarity_matrix[ukb_sig, :])
        best_match_score = similarity_matrix[ukb_sig, best_match_idx]
        best_matches.append({
            'UKB': ukb_sig,
            'Other': best_match_idx,
            'Weighted_Overlap': best_match_score
        })
    
    best_matches_df = pd.DataFrame(best_matches)
    
    return similarity_matrix, best_matches_df


def compute_giovanni_weighted_overlap_phi(ukb_phi, ukb_psi, other_psi, ukb_clusters, other_clusters,
                                         ukb_disease_names, other_disease_names, common_diseases):
    """
    Compute Giovanni's weighted coefficient overlap using sigmoid(time-averaged phi).
    
    Formula:
        Weighted Overlap(UKB_sig_k, Other_sig_k') = 
            Σ_{d ∈ overlap} sigmoid(φ̄_{k,d}) / Σ_{d ∈ UKB_sig_k} sigmoid(φ̄_{k,d})
    
    where:
        - overlap = {diseases in both UKB_sig_k and Other_sig_k'}
        - UKB_sig_k = {all diseases assigned to UKB signature k}
        - φ̄_{k,d} = mean_t(φ_{k,d,t}) (time-averaged phi)
        - sigmoid(φ̄_{k,d}) = expit(φ̄_{k,d}) = 1 / (1 + exp(-φ̄_{k,d}))
    
    Note: Cluster assignments are still based on argmax(psi), but weights use phi.
    """
    # Apply sigmoid to UKB phi (time-averaged) to get positive weights
    ukb_phi_normalized = expit(ukb_phi)  # Shape: (K_ukb, D_ukb)
    
    # Convert to numpy arrays if needed
    if isinstance(ukb_clusters, torch.Tensor):
        ukb_clusters = ukb_clusters.numpy()
    if isinstance(other_clusters, torch.Tensor):
        other_clusters = other_clusters.numpy()
    
    # Convert disease names to lists
    if isinstance(ukb_disease_names, np.ndarray):
        ukb_disease_names = ukb_disease_names.tolist()
    if isinstance(other_disease_names, np.ndarray):
        other_disease_names = other_disease_names.tolist()
    
    # Create mapping from disease name to indices
    ukb_disease_to_idx = {d: i for i, d in enumerate(ukb_disease_names)}
    other_disease_to_idx = {d: i for i, d in enumerate(other_disease_names)}
    
    # Get number of signatures
    K_ukb = ukb_phi.shape[0]
    K_other = other_psi.shape[0]
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((K_ukb, K_other))
    
    # For each UKB signature
    for ukb_sig in range(K_ukb):
        # Get ALL diseases in UKB signature (for denominator)
        ukb_sig_all_indices = np.where(ukb_clusters == ukb_sig)[0]
        
        if len(ukb_sig_all_indices) == 0:
            continue
        
        # Compute total UKB coefficient weight for this signature (ALL diseases, normalized)
        ukb_total_weight = np.sum(ukb_phi_normalized[ukb_sig, ukb_sig_all_indices])
        
        # Skip if total weight is exactly zero (to avoid division by zero)
        if ukb_total_weight == 0:
            continue
        
        # Get diseases in UKB signature that are in common_diseases (for overlap computation)
        ukb_sig_common_diseases = [
            d for d in common_diseases
            if d in ukb_disease_to_idx and ukb_clusters[ukb_disease_to_idx[d]] == ukb_sig
        ]
        
        if len(ukb_sig_common_diseases) == 0:
            continue
        
        # For each other cohort signature
        for other_sig in range(K_other):
            # Get diseases in other signature (from common diseases only)
            other_sig_diseases = [
                d for d in common_diseases
                if d in other_disease_to_idx and other_clusters[other_disease_to_idx[d]] == other_sig
            ]
            
            if len(other_sig_diseases) == 0:
                continue
            
            # Find overlap (diseases in both signatures)
            overlap_diseases = set(ukb_sig_common_diseases) & set(other_sig_diseases)
            
            if len(overlap_diseases) == 0:
                continue
            
            # Compute overlap coefficient weight (sum of normalized UKB phi for diseases in overlap)
            overlap_indices = [ukb_disease_to_idx[d] for d in overlap_diseases]
            overlap_weight = np.sum(ukb_phi_normalized[ukb_sig, overlap_indices])
            
            # Giovanni's formula: overlap_weight / ukb_total_weight
            similarity_matrix[ukb_sig, other_sig] = overlap_weight / ukb_total_weight
    
    # Create best matches DataFrame
    best_matches = []
    for ukb_sig in range(K_ukb):
        best_match_idx = np.argmax(similarity_matrix[ukb_sig, :])
        best_match_score = similarity_matrix[ukb_sig, best_match_idx]
        best_matches.append({
            'UKB': ukb_sig,
            'Other': best_match_idx,
            'Weighted_Overlap': best_match_score
        })
    
    best_matches_df = pd.DataFrame(best_matches)
    
    return similarity_matrix, best_matches_df


def compute_giovanni_weighted_overlap(ukb_psi, other_psi, ukb_clusters, other_clusters,
                                     ukb_disease_names, other_disease_names, common_diseases):
    """
    Compute Giovanni's weighted coefficient overlap:
      sum(psi_ukb in overlap) / sum(psi_ukb for all diseases in UKB signature)
    
    This is bounded to [0, 1] and represents the fraction of UKB signature's
    total coefficient weight that overlaps with another cohort's signature.
    
    Args:
        ukb_psi: UKB psi matrix, shape (K_ukb, D_ukb)
        other_psi: Other cohort psi matrix, shape (K_other, D_other)
        ukb_clusters: UKB cluster assignments, length D_ukb
        other_clusters: Other cohort cluster assignments, length D_other
        ukb_disease_names: List of UKB disease names, length D_ukb
        other_disease_names: List of other cohort disease names, length D_other
        common_diseases: List of common disease names
    
    Returns:
        similarity_matrix: Matrix of weighted overlaps, shape (K_ukb, K_other)
        best_matches: DataFrame with best matches for each UKB signature
    """
    # Convert to numpy arrays if needed
    if isinstance(ukb_clusters, torch.Tensor):
        ukb_clusters = ukb_clusters.numpy()
    if isinstance(other_clusters, torch.Tensor):
        other_clusters = other_clusters.numpy()
    
    # Convert disease names to lists
    if isinstance(ukb_disease_names, np.ndarray):
        ukb_disease_names = ukb_disease_names.tolist()
    if isinstance(other_disease_names, np.ndarray):
        other_disease_names = other_disease_names.tolist()
    
    # Create mapping from disease name to indices
    ukb_disease_to_idx = {d: i for i, d in enumerate(ukb_disease_names)}
    other_disease_to_idx = {d: i for i, d in enumerate(other_disease_names)}
    
    # Get number of signatures
    K_ukb = ukb_psi.shape[0]
    K_other = other_psi.shape[0]
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((K_ukb, K_other))
    
    # For each UKB signature
    for ukb_sig in range(K_ukb):
        # Get ALL diseases in UKB signature (for denominator)
        # Giovanni's formula: "sum(all coefficients in UKBB)" means ALL diseases in the signature
        ukb_sig_all_indices = np.where(ukb_clusters == ukb_sig)[0]
        
        if len(ukb_sig_all_indices) == 0:
            continue
        
        # Compute total UKB coefficient weight for this signature (ALL diseases)
        ukb_total_weight = np.sum(ukb_psi[ukb_sig, ukb_sig_all_indices])
        
        # Skip if total weight is exactly zero (to avoid division by zero)
        if ukb_total_weight == 0:
            continue
        
        # Get diseases in UKB signature that are in common_diseases (for overlap computation)
        ukb_sig_common_diseases = [
            d for d in common_diseases
            if d in ukb_disease_to_idx and ukb_clusters[ukb_disease_to_idx[d]] == ukb_sig
        ]
        
        if len(ukb_sig_common_diseases) == 0:
            continue
        
        # For each other cohort signature
        for other_sig in range(K_other):
            # Get diseases in other signature (from common diseases only)
            other_sig_diseases = [
                d for d in common_diseases
                if d in other_disease_to_idx and other_clusters[other_disease_to_idx[d]] == other_sig
            ]
            
            if len(other_sig_diseases) == 0:
                continue
            
            # Find overlap (diseases in both signatures)
            overlap_diseases = set(ukb_sig_common_diseases) & set(other_sig_diseases)
            
            if len(overlap_diseases) == 0:
                continue
            
            # Compute overlap coefficient weight (sum of UKB psi for diseases in overlap)
            overlap_indices = [ukb_disease_to_idx[d] for d in overlap_diseases]
            overlap_weight = np.sum(ukb_psi[ukb_sig, overlap_indices])
            
            # Giovanni's formula: overlap_weight / ukb_total_weight
            similarity_matrix[ukb_sig, other_sig] = overlap_weight / ukb_total_weight
    
    # Create best matches DataFrame
    best_matches = []
    for ukb_sig in range(K_ukb):
        best_match_idx = np.argmax(similarity_matrix[ukb_sig, :])
        best_match_score = similarity_matrix[ukb_sig, best_match_idx]
        best_matches.append({
            'UKB': ukb_sig,
            'Other': best_match_idx,
            'Weighted_Overlap': best_match_score
        })
    
    best_matches_df = pd.DataFrame(best_matches)
    
    return similarity_matrix, best_matches_df


def print_psi_summary(psi, cohort_name):
    """Print summary statistics for psi values."""
    print(f"\n{cohort_name} Psi Summary:")
    print(f"  Shape: {psi.shape}")
    print(f"  Min: {np.min(psi):.4f}")
    print(f"  Max: {np.max(psi):.4f}")
    print(f"  Mean: {np.mean(psi):.4f}")
    print(f"  Median: {np.median(psi):.4f}")
    print(f"  Std: {np.std(psi):.4f}")


def plot_binary_jaccard_heatmap(similarity_matrix_ukb_mgb, similarity_matrix_ukb_aou,
                                best_matches_mgb, best_matches_aou,
                                output_path=None):
    """
    Plot heatmaps of binary modified Jaccard similarity.
    UKB rows are standardized to 0-20 in order.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Standardize UKB rows to 0-20, reorder columns by best matching signature
    def reorder_matrix(sim_matrix, best_matches):
        # UKB order is always 0, 1, 2, ..., 20 (standardized)
        K_ukb = sim_matrix.shape[0]
        ukb_order = np.arange(K_ukb)
        
        # Sort by UKB signature to get best matches in order
        sorted_best = best_matches.sort_values('UKB')
        # Extract best matching signatures as column order
        best_match_order = sorted_best['Other'].values
        
        # Handle any unmatched columns (if other cohort has more signatures)
        K_other = sim_matrix.shape[1]
        all_other_sigs = set(range(K_other))
        matched_sigs = set(best_match_order)
        unmatched_sigs = sorted(all_other_sigs - matched_sigs)
        
        # Combine: best matches first, then unmatched signatures
        other_order = list(best_match_order) + unmatched_sigs
        
        # Reorder matrix (keep UKB rows in order, reorder columns by best match)
        reordered = sim_matrix[ukb_order, :][:, other_order]
        return reordered, ukb_order, np.array(other_order)
    
    # Reorder MGB matrix
    sim_mgb_reordered, ukb_order_mgb, mgb_order = reorder_matrix(
        similarity_matrix_ukb_mgb, best_matches_mgb
    )
    
    # Reorder AoU matrix
    sim_aou_reordered, ukb_order_aou, aou_order = reorder_matrix(
        similarity_matrix_ukb_aou, best_matches_aou
    )
    
    # Plot MGB
    im1 = axes[0].imshow(sim_mgb_reordered, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Composition Preservation Probability: UKB vs MGB\n(Using Posterior Clusters)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('MGB Signature', fontsize=10)
    axes[0].set_ylabel('UKB Signature', fontsize=10)
    axes[0].set_xticks(range(len(mgb_order)))
    axes[0].set_xticklabels(mgb_order, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(ukb_order_mgb)))
    axes[0].set_yticklabels(ukb_order_mgb, fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='Jaccard Similarity', fraction=0.046, pad=0.04)
    
    # Plot AoU
    im2 = axes[1].imshow(sim_aou_reordered, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Binary Modified Jaccard: UKB vs AoU\n(Using Posterior Clusters)',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('AoU Signature', fontsize=10)
    axes[1].set_ylabel('UKB Signature', fontsize=10)
    axes[1].set_xticks(range(len(aou_order)))
    axes[1].set_xticklabels(aou_order, rotation=90, fontsize=8)
    axes[1].set_yticks(range(len(ukb_order_aou)))
    axes[1].set_yticklabels(ukb_order_aou, fontsize=8)
    plt.colorbar(im2, ax=axes[1], label='Jaccard Similarity', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\n✓ Saved heatmap to: {output_path}")
    
    plt.show()  # Display in notebook
    return fig


def plot_giovanni_weighted_heatmap(similarity_matrix_ukb_mgb, similarity_matrix_ukb_aou,
                                   best_matches_mgb, best_matches_aou,
                                   output_path=None):
    """
    Plot heatmaps of Giovanni's weighted coefficient overlap.
    UKB rows are standardized to 0-20 in order.
    
    Args:
        similarity_matrix_ukb_mgb: Similarity matrix UKB vs MGB, shape (K_ukb, K_mgb)
        similarity_matrix_ukb_aou: Similarity matrix UKB vs AoU, shape (K_ukb, K_aou)
        best_matches_mgb: DataFrame with best matches for MGB
        best_matches_aou: DataFrame with best matches for AoU
        output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Standardize UKB rows to 0-20, reorder columns by best matching signature
    def reorder_matrix(sim_matrix, best_matches):
        # UKB order is always 0, 1, 2, ..., 20 (standardized)
        K_ukb = sim_matrix.shape[0]
        ukb_order = np.arange(K_ukb)
        
        # Sort by UKB signature to get best matches in order
        sorted_best = best_matches.sort_values('UKB')
        # Extract best matching signatures as column order
        best_match_order = sorted_best['Other'].values
        
        # Handle any unmatched columns (if other cohort has more signatures)
        K_other = sim_matrix.shape[1]
        all_other_sigs = set(range(K_other))
        matched_sigs = set(best_match_order)
        unmatched_sigs = sorted(all_other_sigs - matched_sigs)
        
        # Combine: best matches first, then unmatched signatures
        other_order = list(best_match_order) + unmatched_sigs
        
        # Reorder matrix (keep UKB rows in order, reorder columns by best match)
        reordered = sim_matrix[ukb_order, :][:, other_order]
        return reordered, ukb_order, np.array(other_order)
    
    # Reorder MGB matrix
    sim_mgb_reordered, ukb_order_mgb, mgb_order = reorder_matrix(
        similarity_matrix_ukb_mgb, best_matches_mgb
    )
    
    # Reorder AoU matrix
    sim_aou_reordered, ukb_order_aou, aou_order = reorder_matrix(
        similarity_matrix_ukb_aou, best_matches_aou
    )
    
    # Plot MGB (use actual data range, but clamp to [0,1] for normalized version)
    vmin_actual = min(0, np.min(sim_mgb_reordered))
    vmax_actual = max(1, np.max(sim_mgb_reordered))
    im1 = axes[0].imshow(sim_mgb_reordered, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Giovanni Weighted Overlap: UKB vs MGB\n(Overlap coefficient weight / Total UKB coefficient weight)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('MGB Signature', fontsize=10)
    axes[0].set_ylabel('UKB Signature', fontsize=10)
    axes[0].set_xticks(range(len(mgb_order)))
    axes[0].set_xticklabels(mgb_order, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(ukb_order_mgb)))
    axes[0].set_yticklabels(ukb_order_mgb, fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='Weighted Overlap', fraction=0.046, pad=0.04)
    
    # Plot AoU (use actual data range, but clamp to [0,1] for normalized version)
    im2 = axes[1].imshow(sim_aou_reordered, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Giovanni Weighted Overlap: UKB vs AoU\n(Overlap coefficient weight / Total UKB coefficient weight)',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('AoU Signature', fontsize=10)
    axes[1].set_ylabel('UKB Signature', fontsize=10)
    axes[1].set_xticks(range(len(aou_order)))
    axes[1].set_xticklabels(aou_order, rotation=90, fontsize=8)
    axes[1].set_yticks(range(len(ukb_order_aou)))
    axes[1].set_yticklabels(ukb_order_aou, fontsize=8)
    plt.colorbar(im2, ax=axes[1], label='Weighted Overlap', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\n✓ Saved heatmap to: {output_path}")
    
    plt.show()  # Display in notebook
    return fig


def main():
    """Main function to run the analysis."""
    print("="*80)
    print("GIOVANNI'S WEIGHTED COEFFICIENT OVERLAP ANALYSIS")
    print("="*80)
    
    # Paths
    ukb_batch_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full'
    ukb_pattern = 'enrollment_model_W0.0001_batch_*_*.pt'
    
    aou_batch_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/aou_batches'
    aou_pattern = 'aou_model_batch_*.pt'
    
    mgb_checkpoint_path = '/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt'
    
    ukb_disease_names_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv'
    
    # Load UKB psi (pooled from batches)
    print("\n1. Loading UKB psi (pooled from batches)...")
    ukb_psi, ukb_psi_std = pool_psi_from_batches(ukb_batch_dir, ukb_pattern, max_batches=None)
    print_psi_summary(ukb_psi, "UKB")
    
    # Load AoU psi (pooled from batches)
    print("\n2. Loading AoU psi (pooled from batches)...")
    aou_psi, aou_psi_std = pool_psi_from_batches(aou_batch_dir, aou_pattern, max_batches=None)
    print_psi_summary(aou_psi, "AoU")
    
    # Load MGB psi (from checkpoint)
    print("\n3. Loading MGB psi (from checkpoint)...")
    mgb_psi = load_psi_from_checkpoint(mgb_checkpoint_path)
    print_psi_summary(mgb_psi, "MGB")
    
    # Load phi from batches and checkpoints (for phi-based weighted overlap)
    print("\n4a. Loading UKB phi (pooled from batches, time-averaged)...")
    ukb_phi, ukb_phi_std = pool_phi_from_batches(ukb_batch_dir, ukb_pattern, max_batches=None)
    
    print("\n4b. Loading AoU phi (pooled from batches, time-averaged)...")
    aou_phi, aou_phi_std = pool_phi_from_batches(aou_batch_dir, aou_pattern, max_batches=None)
    
    print("\n4c. Loading MGB phi (from checkpoint, time-averaged)...")
    mgb_phi = load_phi_from_checkpoint(mgb_checkpoint_path)
    
    # Load disease names
    print("\n5. Loading disease names...")
    ukb_disease_names = pd.read_csv(ukb_disease_names_path)['x'].tolist()
    
    # Load disease names from checkpoints
    mgb_checkpoint = torch.load(mgb_checkpoint_path, map_location='cpu', weights_only=False)
    mgb_disease_names = mgb_checkpoint['disease_names']
    
    aou_checkpoint = torch.load('/Users/sarahurbut/aladynoulli2/aou_model_initialized.pt', map_location='cpu', weights_only=False)
    aou_disease_names = aou_checkpoint['disease_names']
    
    # Load initial clusters (old approach)
    print("\n6. Loading initial clusters (old approach)...")
    ukb_clusters_initial = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt', map_location='cpu')
    if isinstance(ukb_clusters_initial, torch.Tensor):
        ukb_clusters_initial = ukb_clusters_initial.numpy()
    print(f"  ✓ Loaded initial UKB clusters")
    
    # Get initial clusters for MGB and AoU from checkpoints
    mgb_clusters_initial = mgb_checkpoint['clusters']
    aou_clusters_initial = aou_checkpoint['clusters']
    if isinstance(mgb_clusters_initial, torch.Tensor):
        mgb_clusters_initial = mgb_clusters_initial.numpy()
    if isinstance(aou_clusters_initial, torch.Tensor):
        aou_clusters_initial = aou_clusters_initial.numpy()
    print(f"  ✓ Loaded initial MGB and AoU clusters")
    
    # EXPERIMENT 1: Compute clusters from posterior psi (max sig per disease)
    print("\n7. Computing clusters from posterior psi (max signature per disease)...")
    
    # UKB: argmax across signatures for each disease
    ukb_clusters_posterior = np.argmax(ukb_psi, axis=0)
    print(f"  ✓ UKB: Computed posterior clusters from averaged psi")
    
    # MGB: argmax across signatures for each disease
    mgb_clusters_posterior = np.argmax(mgb_psi, axis=0)
    print(f"  ✓ MGB: Computed posterior clusters from psi")
    
    # AoU: argmax across signatures for each disease
    aou_clusters_posterior = np.argmax(aou_psi, axis=0)
    print(f"  ✓ AoU: Computed posterior clusters from averaged psi")
    
    # Use posterior clusters for main analysis
    ukb_clusters = ukb_clusters_posterior
    mgb_clusters = mgb_clusters_posterior
    aou_clusters = aou_clusters_posterior
    
    print(f"\n  UKB: {ukb_clusters.max()+1} signatures")
    print(f"  MGB: {mgb_clusters.max()+1} signatures")
    print(f"  AoU: {aou_clusters.max()+1} signatures")
    
    # Compute batch consistency statistics
    print("\n" + "="*80)
    print("BATCH CONSISTENCY STATISTICS")
    print("="*80)
    
    # UKB batch consistency
    print("\n  UKB: Computing batch consistency...")
    ukb_batch_stats = compute_batch_consistency_stats(
        ukb_batch_dir, ukb_pattern, ukb_clusters, ukb_disease_names, 'UKB'
    )
    print(f"    ✓ Computed batch consistency for {len(ukb_batch_stats)} diseases")
    print(f"    Median consistency: {ukb_batch_stats['Pct_Consistent'].median():.1f}%")
    print(f"    Mean consistency: {ukb_batch_stats['Pct_Consistent'].mean():.1f}%")
    print(f"    Range: [{ukb_batch_stats['Pct_Consistent'].min():.1f}%, {ukb_batch_stats['Pct_Consistent'].max():.1f}%]")
    
    # Show diseases with <100% consistency
    inconsistent = ukb_batch_stats[ukb_batch_stats['Pct_Consistent'] < 100.0].sort_values('Pct_Consistent')
    if len(inconsistent) > 0:
        print(f"\n    Diseases with <100% consistency ({len(inconsistent)} total):")
        for _, row in inconsistent.iterrows():
            print(f"      {row['Disease']}: {row['Pct_Consistent']:.1f}% ({row['N_Batches_Match']}/{row['N_Batches_Total']} batches)")
    
    # Summary by signature
    print("\n    UKB batch consistency by signature:")
    ukb_sig_summary = ukb_batch_stats.groupby('Final_Sig').agg({
        'Pct_Consistent': ['mean', 'median', 'min', 'max', 'count']
    }).round(1)
    print(ukb_sig_summary)
    
    # AoU batch consistency
    print("\n  AoU: Computing batch consistency...")
    aou_batch_stats = compute_batch_consistency_stats(
        aou_batch_dir, aou_pattern, aou_clusters, aou_disease_names, 'AoU'
    )
    print(f"    ✓ Computed batch consistency for {len(aou_batch_stats)} diseases")
    print(f"    Median consistency: {aou_batch_stats['Pct_Consistent'].median():.1f}%")
    print(f"    Mean consistency: {aou_batch_stats['Pct_Consistent'].mean():.1f}%")
    print(f"    Range: [{aou_batch_stats['Pct_Consistent'].min():.1f}%, {aou_batch_stats['Pct_Consistent'].max():.1f}%]")
    
    # Show diseases with <100% consistency
    inconsistent = aou_batch_stats[aou_batch_stats['Pct_Consistent'] < 100.0].sort_values('Pct_Consistent')
    if len(inconsistent) > 0:
        print(f"\n    Diseases with <100% consistency ({len(inconsistent)} total):")
        for _, row in inconsistent.iterrows():
            print(f"      {row['Disease']}: {row['Pct_Consistent']:.1f}% ({row['N_Batches_Match']}/{row['N_Batches_Total']} batches)")
    
    # Summary by signature
    print("\n    AoU batch consistency by signature:")
    aou_sig_summary = aou_batch_stats.groupby('Final_Sig').agg({
        'Pct_Consistent': ['mean', 'median', 'min', 'max', 'count']
    }).round(1)
    print(aou_sig_summary)
    
    # UKB initial vs posterior comparison
    print("\n  UKB: Comparing initial vs posterior cluster assignments...")
    ukb_initial_vs_posterior = compare_initial_vs_posterior_clusters(
        ukb_clusters_initial, ukb_clusters_posterior, ukb_disease_names, 'UKB'
    )
    n_changed_ukb = ukb_initial_vs_posterior['Changed'].sum()
    pct_changed_ukb = 100 * n_changed_ukb / len(ukb_initial_vs_posterior)
    print(f"    ✓ Compared {len(ukb_initial_vs_posterior)} diseases")
    print(f"    Changed signatures: {n_changed_ukb} / {len(ukb_initial_vs_posterior)} ({pct_changed_ukb:.1f}%)")
    print(f"    Unchanged: {len(ukb_initial_vs_posterior) - n_changed_ukb} ({100-pct_changed_ukb:.1f}%)")
    
    # AoU initial vs posterior comparison
    print("\n  AoU: Comparing initial vs posterior cluster assignments...")
    aou_initial_vs_posterior = compare_initial_vs_posterior_clusters(
        aou_clusters_initial, aou_clusters_posterior, aou_disease_names, 'AoU'
    )
    n_changed_aou = aou_initial_vs_posterior['Changed'].sum()
    pct_changed_aou = 100 * n_changed_aou / len(aou_initial_vs_posterior)
    print(f"    ✓ Compared {len(aou_initial_vs_posterior)} diseases")
    print(f"    Changed signatures: {n_changed_aou} / {len(aou_initial_vs_posterior)} ({pct_changed_aou:.1f}%)")
    print(f"    Unchanged: {len(aou_initial_vs_posterior) - n_changed_aou} ({100-pct_changed_aou:.1f}%)")
    
    # MGB initial vs posterior comparison
    print("\n  MGB: Comparing initial vs posterior cluster assignments...")
    mgb_initial_vs_posterior = compare_initial_vs_posterior_clusters(
        mgb_clusters_initial, mgb_clusters_posterior, mgb_disease_names, 'MGB'
    )
    n_changed = mgb_initial_vs_posterior['Changed'].sum()
    pct_changed = 100 * n_changed / len(mgb_initial_vs_posterior)
    print(f"    ✓ Compared {len(mgb_initial_vs_posterior)} diseases")
    print(f"    Changed signatures: {n_changed} / {len(mgb_initial_vs_posterior)} ({pct_changed:.1f}%)")
    print(f"    Unchanged: {len(mgb_initial_vs_posterior) - n_changed} ({100-pct_changed:.1f}%)")
    
    # Summary by initial signature
    print("\n    MGB changes by initial signature:")
    mgb_change_summary = mgb_initial_vs_posterior.groupby('Initial_Sig').agg({
        'Changed': ['sum', 'count']
    })
    mgb_change_summary.columns = ['N_Changed', 'N_Total']
    mgb_change_summary['Pct_Changed'] = 100 * mgb_change_summary['N_Changed'] / mgb_change_summary['N_Total']
    print(mgb_change_summary.round(1))
    
    # Find common diseases
    print("\n8. Finding common diseases...")
    ukb_disease_set = set(ukb_disease_names)
    mgb_disease_set = set(mgb_disease_names)
    aou_disease_set = set(aou_disease_names)
    
    common_ukb_mgb = sorted(list(ukb_disease_set & mgb_disease_set))
    common_ukb_aou = sorted(list(ukb_disease_set & aou_disease_set))
    
    print(f"  Common UKB-MGB: {len(common_ukb_mgb)} diseases")
    print(f"  Common UKB-AoU: {len(common_ukb_aou)} diseases")
    
    # Compute binary Jaccard with INITIAL clusters (old approach)
    print("\n" + "="*80)
    print("BINARY JACCARD USING INITIAL CLUSTERS (OLD APPROACH)")
    print("="*80)
    print("\n  UKB ↔ MGB:")
    jaccard_matrix_mgb_initial, best_matches_mgb_initial, details_mgb_initial = compute_binary_jaccard_posterior(
        ukb_clusters_initial, mgb_clusters_initial,
        ukb_disease_names, mgb_disease_names, common_ukb_mgb
    )
    jaccard_scores_mgb_initial = [details_mgb_initial[k]['jaccard'] for k in details_mgb_initial.keys()]
    print(f"    Median Jaccard: {np.median(jaccard_scores_mgb_initial):.4f}")
    print(f"    Range: [{np.min(jaccard_scores_mgb_initial):.4f}, {np.max(jaccard_scores_mgb_initial):.4f}]")
    
    print("\n  UKB ↔ AoU:")
    jaccard_matrix_aou_initial, best_matches_aou_initial, details_aou_initial = compute_binary_jaccard_posterior(
        ukb_clusters_initial, aou_clusters_initial,
        ukb_disease_names, aou_disease_names, common_ukb_aou
    )
    jaccard_scores_aou_initial = [details_aou_initial[k]['jaccard'] for k in details_aou_initial.keys()]
    print(f"    Median Jaccard: {np.median(jaccard_scores_aou_initial):.4f}")
    print(f"    Range: [{np.min(jaccard_scores_aou_initial):.4f}, {np.max(jaccard_scores_aou_initial):.4f}]")
    
    # EXPERIMENT 1: Binary Jaccard using posterior clusters
    print("\n" + "="*80)
    print("EXPERIMENT 1: BINARY JACCARD USING POSTERIOR CLUSTERS")
    print("="*80)
    print("\n  UKB ↔ MGB:")
    jaccard_matrix_mgb, best_matches_mgb_posterior, details_mgb_posterior = compute_binary_jaccard_posterior(
        ukb_clusters, mgb_clusters,
        ukb_disease_names, mgb_disease_names, common_ukb_mgb
    )
    jaccard_scores_mgb = [details_mgb_posterior[k]['jaccard'] for k in details_mgb_posterior.keys()]
    print(f"    Median Jaccard: {np.median(jaccard_scores_mgb):.4f}")
    print(f"    Range: [{np.min(jaccard_scores_mgb):.4f}, {np.max(jaccard_scores_mgb):.4f}]")
    
    print("\n  UKB ↔ AoU:")
    jaccard_matrix_aou, best_matches_aou_posterior, details_aou_posterior = compute_binary_jaccard_posterior(
        ukb_clusters, aou_clusters,
        ukb_disease_names, aou_disease_names, common_ukb_aou
    )
    jaccard_scores_aou = [details_aou_posterior[k]['jaccard'] for k in details_aou_posterior.keys()]
    print(f"    Median Jaccard: {np.median(jaccard_scores_aou):.4f}")
    print(f"    Range: [{np.min(jaccard_scores_aou):.4f}, {np.max(jaccard_scores_aou):.4f}]")
    
    # Plot and save posterior binary Jaccard heatmap
    print("\n  Plotting binary Jaccard heatmaps (posterior clusters)...")
    output_dir = Path(__file__).parent
    binary_posterior_output_path = output_dir / 'binary_jaccard_posterior_heatmaps.pdf'
    plot_binary_jaccard_heatmap(
        jaccard_matrix_mgb, jaccard_matrix_aou,
        best_matches_mgb_posterior, best_matches_aou_posterior,
        output_path=binary_posterior_output_path
    )
    
    # Plot binary heatmaps (both initial and posterior)
    print("\n  Plotting binary Jaccard heatmaps (initial vs posterior)...")
    output_dir = Path(__file__).parent
    binary_output_path = output_dir / 'binary_jaccard_comparison_heatmaps.pdf'
    
    # Create 2x2 subplot: initial (top row) vs posterior (bottom row)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Reorder function: standardize UKB rows to 0-20, order columns by best matching signature
    def reorder_matrix(sim_matrix, best_matches):
        # UKB order is always 0, 1, 2, ..., 20 (standardized)
        K_ukb = sim_matrix.shape[0]
        ukb_order = np.arange(K_ukb)
        
        # Sort by UKB signature to get best matches in order
        sorted_best = best_matches.sort_values('UKB')
        # Extract best matching signatures as column order
        best_match_order = sorted_best['Other'].values
        
        # Handle any unmatched columns (if other cohort has more signatures)
        K_other = sim_matrix.shape[1]
        all_other_sigs = set(range(K_other))
        matched_sigs = set(best_match_order)
        unmatched_sigs = sorted(all_other_sigs - matched_sigs)
        
        # Combine: best matches first, then unmatched signatures
        other_order = list(best_match_order) + unmatched_sigs
        
        # Reorder matrix (keep UKB rows in order, reorder columns by best match)
        reordered = sim_matrix[ukb_order, :][:, other_order]
        return reordered, ukb_order, np.array(other_order)
    
    # Top row: Initial clusters
    # MGB initial
    sim_mgb_initial, ukb_order_mgb_initial, mgb_order_initial = reorder_matrix(
        jaccard_matrix_mgb_initial, best_matches_mgb_initial
    )
    im1 = axes[0, 0].imshow(sim_mgb_initial, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0, 0].set_title('Composition Preservation Probability: UKB vs MGB\n(Initial Clusters)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('MGB Signature', fontsize=10)
    axes[0, 0].set_ylabel('UKB Signature', fontsize=10)
    axes[0, 0].set_xticks(range(len(mgb_order_initial)))
    axes[0, 0].set_xticklabels(mgb_order_initial, rotation=90, fontsize=8)
    axes[0, 0].set_yticks(range(len(ukb_order_mgb_initial)))
    axes[0, 0].set_yticklabels(ukb_order_mgb_initial, fontsize=8)
    plt.colorbar(im1, ax=axes[0, 0], label='Jaccard', fraction=0.046, pad=0.04)
    
    # AoU initial
    sim_aou_initial, ukb_order_aou_initial, aou_order_initial = reorder_matrix(
        jaccard_matrix_aou_initial, best_matches_aou_initial
    )
    im2 = axes[0, 1].imshow(sim_aou_initial, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Composition Preservation Probability: UKB vs AoU\n(Initial Clusters)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('AoU Signature', fontsize=10)
    axes[0, 1].set_ylabel('UKB Signature', fontsize=10)
    axes[0, 1].set_xticks(range(len(aou_order_initial)))
    axes[0, 1].set_xticklabels(aou_order_initial, rotation=90, fontsize=8)
    axes[0, 1].set_yticks(range(len(ukb_order_aou_initial)))
    axes[0, 1].set_yticklabels(ukb_order_aou_initial, fontsize=8)
    plt.colorbar(im2, ax=axes[0, 1], label='Jaccard', fraction=0.046, pad=0.04)
    
    # Bottom row: Posterior clusters
    # MGB posterior
    sim_mgb_posterior, ukb_order_mgb_posterior, mgb_order_posterior = reorder_matrix(
        jaccard_matrix_mgb, best_matches_mgb_posterior
    )
    im3 = axes[1, 0].imshow(sim_mgb_posterior, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Composition Preservation Probability: UKB vs MGB\n(Posterior Clusters)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('MGB Signature', fontsize=10)
    axes[1, 0].set_ylabel('UKB Signature', fontsize=10)
    axes[1, 0].set_xticks(range(len(mgb_order_posterior)))
    axes[1, 0].set_xticklabels(mgb_order_posterior, rotation=90, fontsize=8)
    axes[1, 0].set_yticks(range(len(ukb_order_mgb_posterior)))
    axes[1, 0].set_yticklabels(ukb_order_mgb_posterior, fontsize=8)
    plt.colorbar(im3, ax=axes[1, 0], label='Jaccard', fraction=0.046, pad=0.04)
    
    # AoU posterior
    sim_aou_posterior, ukb_order_aou_posterior, aou_order_posterior = reorder_matrix(
        jaccard_matrix_aou, best_matches_aou_posterior
    )
    im4 = axes[1, 1].imshow(sim_aou_posterior, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title('Composition Preservation Probability: UKB vs AoU\n(Posterior Clusters)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('AoU Signature', fontsize=10)
    axes[1, 1].set_ylabel('UKB Signature', fontsize=10)
    axes[1, 1].set_xticks(range(len(aou_order_posterior)))
    axes[1, 1].set_xticklabels(aou_order_posterior, rotation=90, fontsize=8)
    axes[1, 1].set_yticks(range(len(ukb_order_aou_posterior)))
    axes[1, 1].set_yticklabels(ukb_order_aou_posterior, fontsize=8)
    plt.colorbar(im4, ax=axes[1, 1], label='Jaccard', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(binary_output_path, bbox_inches='tight', dpi=300)
    print(f"    ✓ Saved comparison heatmaps to: {binary_output_path}")
    plt.show()  # Display in notebook
    plt.close()
    
    # EXPERIMENT 2: Weighted overlap using normalized psi (sigmoid)
    print("\n" + "="*80)
    print("EXPERIMENT 2: WEIGHTED OVERLAP USING NORMALIZED PSI (SIGMOID)")
    print("="*80)
    print("\n  Formula:")
    print("    Weighted Overlap(UKB_sig_k, Other_sig_k') = ")
    print("        Σ_{d ∈ overlap} sigmoid(ψ_{k,d}) / Σ_{d ∈ UKB_sig_k} sigmoid(ψ_{k,d})")
    print("\n  where:")
    print("    - overlap = diseases in both UKB signature k and Other signature k'")
    print("    - UKB_sig_k = all diseases assigned to UKB signature k")
    print("    - sigmoid(ψ_{k,d}) = expit(ψ_{k,d}) = 1 / (1 + exp(-ψ_{k,d}))")
    print("    - This metric is bounded to [0, 1]")
    print("\n  UKB ↔ MGB:")
    similarity_ukb_mgb_norm, best_matches_mgb_norm = compute_giovanni_weighted_overlap_normalized(
        ukb_psi, mgb_psi, ukb_clusters, mgb_clusters,
        ukb_disease_names, mgb_disease_names, common_ukb_mgb
    )
    print(f"    Similarity matrix shape: {similarity_ukb_mgb_norm.shape}")
    print(f"    Range: [{np.min(similarity_ukb_mgb_norm):.4f}, {np.max(similarity_ukb_mgb_norm):.4f}]")
    print(f"    Median best match: {best_matches_mgb_norm['Weighted_Overlap'].median():.4f}")
    
    print("\n  UKB ↔ AoU:")
    similarity_ukb_aou_norm, best_matches_aou_norm = compute_giovanni_weighted_overlap_normalized(
        ukb_psi, aou_psi, ukb_clusters, aou_clusters,
        ukb_disease_names, aou_disease_names, common_ukb_aou
    )
    print(f"    Similarity matrix shape: {similarity_ukb_aou_norm.shape}")
    print(f"    Range: [{np.min(similarity_ukb_aou_norm):.4f}, {np.max(similarity_ukb_aou_norm):.4f}]")
    print(f"    Median best match: {best_matches_aou_norm['Weighted_Overlap'].median():.4f}")
    
    # Original weighted overlap (for comparison)
    print("\n" + "="*80)
    print("ORIGINAL: WEIGHTED OVERLAP USING RAW PSI")
    print("="*80)
    print("\n  UKB ↔ MGB:")
    print("\n  UKB ↔ MGB:")
    similarity_ukb_mgb, best_matches_mgb = compute_giovanni_weighted_overlap(
        ukb_psi, mgb_psi, ukb_clusters, mgb_clusters,
        ukb_disease_names, mgb_disease_names, common_ukb_mgb
    )
    print(f"    Similarity matrix shape: {similarity_ukb_mgb.shape}")
    print(f"    Range: [{np.min(similarity_ukb_mgb):.4f}, {np.max(similarity_ukb_mgb):.4f}]")
    print(f"    Median best match: {best_matches_mgb['Weighted_Overlap'].median():.4f}")
    
    print("\n  UKB ↔ AoU:")
    similarity_ukb_aou, best_matches_aou = compute_giovanni_weighted_overlap(
        ukb_psi, aou_psi, ukb_clusters, aou_clusters,
        ukb_disease_names, aou_disease_names, common_ukb_aou
    )
    print(f"    Similarity matrix shape: {similarity_ukb_aou.shape}")
    print(f"    Range: [{np.min(similarity_ukb_aou):.4f}, {np.max(similarity_ukb_aou):.4f}]")
    print(f"    Median best match: {best_matches_aou['Weighted_Overlap'].median():.4f}")
    
    # EXPERIMENT 3: Weighted overlap using time-averaged phi (sigmoid)
    print("\n" + "="*80)
    print("EXPERIMENT 3: WEIGHTED OVERLAP USING TIME-AVERAGED PHI (SIGMOID)")
    print("="*80)
    print("\n  Formula:")
    print("    Weighted Overlap(UKB_sig_k, Other_sig_k') = ")
    print("        Σ_{d ∈ overlap} sigmoid(φ̄_{k,d}) / Σ_{d ∈ UKB_sig_k} sigmoid(φ̄_{k,d})")
    print("\n  where:")
    print("    - overlap = diseases in both UKB signature k and Other signature k'")
    print("    - UKB_sig_k = all diseases assigned to UKB signature k")
    print("    - φ̄_{k,d} = mean_t(φ_{k,d,t}) (time-averaged phi)")
    print("    - sigmoid(φ̄_{k,d}) = expit(φ̄_{k,d}) = 1 / (1 + exp(-φ̄_{k,d}))")
    print("    - Cluster assignments use argmax(psi), but weights use phi")
    print("    - This metric is bounded to [0, 1]")
    print("\n  UKB ↔ MGB:")
    similarity_ukb_mgb_phi, best_matches_mgb_phi = compute_giovanni_weighted_overlap_phi(
        ukb_phi, ukb_psi, mgb_psi, ukb_clusters, mgb_clusters,
        ukb_disease_names, mgb_disease_names, common_ukb_mgb
    )
    print(f"    Similarity matrix shape: {similarity_ukb_mgb_phi.shape}")
    print(f"    Range: [{np.min(similarity_ukb_mgb_phi):.4f}, {np.max(similarity_ukb_mgb_phi):.4f}]")
    print(f"    Median best match: {best_matches_mgb_phi['Weighted_Overlap'].median():.4f}")
    
    print("\n  UKB ↔ AoU:")
    similarity_ukb_aou_phi, best_matches_aou_phi = compute_giovanni_weighted_overlap_phi(
        ukb_phi, ukb_psi, aou_psi, ukb_clusters, aou_clusters,
        ukb_disease_names, aou_disease_names, common_ukb_aou
    )
    print(f"    Similarity matrix shape: {similarity_ukb_aou_phi.shape}")
    print(f"    Range: [{np.min(similarity_ukb_aou_phi):.4f}, {np.max(similarity_ukb_aou_phi):.4f}]")
    print(f"    Median best match: {best_matches_aou_phi['Weighted_Overlap'].median():.4f}")
    
    # Plot heatmaps for normalized psi version
    print("\n9a. Plotting heatmaps (normalized psi version)...")
    output_dir = Path(__file__).parent
    output_path_psi = output_dir / 'giovanni_weighted_overlap_psi_heatmaps.pdf'
    
    plot_giovanni_weighted_heatmap(
        similarity_ukb_mgb_norm, similarity_ukb_aou_norm,
        best_matches_mgb_norm, best_matches_aou_norm,
        output_path=output_path_psi
    )
    
    # Plot heatmaps for phi version
    print("\n9b. Plotting heatmaps (time-averaged phi version)...")
    output_path_phi = output_dir / 'giovanni_weighted_overlap_phi_heatmaps.pdf'
    
    plot_giovanni_weighted_heatmap(
        similarity_ukb_mgb_phi, similarity_ukb_aou_phi,
        best_matches_mgb_phi, best_matches_aou_phi,
        output_path=output_path_phi
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'ukb_psi': ukb_psi,
        'mgb_psi': mgb_psi,
        'aou_psi': aou_psi,
        'ukb_clusters_posterior': ukb_clusters,
        'mgb_clusters_posterior': mgb_clusters,
        'aou_clusters_posterior': aou_clusters,
        # Experiment 1: Binary with posterior clusters
        'jaccard_matrix_mgb': jaccard_matrix_mgb,
        'jaccard_matrix_aou': jaccard_matrix_aou,
        'best_matches_mgb_posterior': best_matches_mgb_posterior,
        'best_matches_aou_posterior': best_matches_aou_posterior,
        'details_mgb_posterior': details_mgb_posterior,
        'details_aou_posterior': details_aou_posterior,
        # Experiment 2: Weighted with normalized psi
        'similarity_ukb_mgb_norm': similarity_ukb_mgb_norm,
        'similarity_ukb_aou_norm': similarity_ukb_aou_norm,
        'best_matches_mgb_norm': best_matches_mgb_norm,
        'best_matches_aou_norm': best_matches_aou_norm,
        # Original weighted (for comparison)
        'similarity_ukb_mgb': similarity_ukb_mgb,
        'similarity_ukb_aou': similarity_ukb_aou,
        'best_matches_mgb': best_matches_mgb,
        'best_matches_aou': best_matches_aou,
        # Experiment 3: Weighted with time-averaged phi
        'ukb_phi': ukb_phi,
        'mgb_phi': mgb_phi,
        'aou_phi': aou_phi,
        'similarity_ukb_mgb_phi': similarity_ukb_mgb_phi,
        'similarity_ukb_aou_phi': similarity_ukb_aou_phi,
        'best_matches_mgb_phi': best_matches_mgb_phi,
        'best_matches_aou_phi': best_matches_aou_phi
    }


if __name__ == '__main__':
    results = main()

