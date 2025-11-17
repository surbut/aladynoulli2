import numpy as np
import torch
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _se_kernel(T: int, length_scale: float, amplitude: float) -> np.ndarray:
    t = np.arange(T)
    dt = t[:, None] - t[None, :]
    K = (amplitude ** 2) * np.exp(-0.5 * (dt / max(length_scale, 1e-6)) ** 2)
    # small jitter for numerical stability
    return K + 1e-6 * np.eye(T)


def _estimate_length_scale_from_acf(acf: np.ndarray) -> float:
    """Half-correlation time mapped to SE length-scale."""
    idx = np.where(acf <= 0.5)[0]
    if idx.size == 0:
        return 5.0
    dt = float(idx[0])
    return dt / np.sqrt(2.0 * np.log(2.0))


def _mean_acf_over_individuals(resid_sig: np.ndarray, maxlag: int = 25) -> np.ndarray:
    """resid_sig: [N, T] -> mean ACF over individuals, length maxlag+1."""
    def acf_1d(x: np.ndarray, L: int) -> np.ndarray:
        x = x - x.mean()
        c = np.correlate(x, x, mode="full")
        c = c[c.size // 2 :]
        c /= (c[0] + 1e-12)
        return c[: L + 1]

    N, T = resid_sig.shape
    L = min(maxlag, T - 1)
    acfs = np.stack([acf_1d(resid_sig[i], L) for i in range(N)], axis=0)
    return acfs.mean(axis=0)


def _calculate_signature_refs_from_Y(Y: np.ndarray, clusters: np.ndarray, K: int, T: int, 
                                     smooth: bool = True, frac: float = 0.3, return_noisy: bool = False):
    """
    Calculate signature reference trajectories from Y data using cluster assignments.
    Similar to create_reference_trajectories in the actual fit.
    
    Parameters:
    -----------
    Y : np.ndarray
        Disease data [N, D, T]
    clusters : np.ndarray
        Cluster assignment for each disease [D]
    K : int
        Number of signatures
    T : int
        Number of timepoints
    smooth : bool
        Whether to apply LOWESS smoothing
    frac : float
        LOWESS smoothing fraction
    return_noisy : bool
        If True, return both noisy and smoothed refs (for visualization)
        
    Returns:
    --------
    signature_refs : np.ndarray or tuple
        If return_noisy=False: Reference trajectories [K, T] on logit scale (smoothed if smooth=True)
        If return_noisy=True: Tuple of (noisy_refs, smoothed_refs) both [K, T] on logit scale
    """
    from scipy.special import logit
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    # Get raw counts and proportions
    Y_counts = Y.sum(axis=0)  # [D, T] - sum over individuals
    
    # Only count diseases that belong to one of the K signatures (exclude -1 assignments)
    valid_mask = (clusters >= 0) & (clusters < K)  # Only diseases assigned to one of K signatures
    total_counts = Y_counts[valid_mask].sum(axis=0) + 1e-8  # [T] - only count valid diseases
    
    signature_props = np.zeros((K, T))
    for k in range(K):
        cluster_mask = (clusters == k)
        if cluster_mask.sum() > 0:
            signature_props[k] = Y_counts[cluster_mask].sum(axis=0) / total_counts
        else:
            signature_props[k] = 1e-8
    
    # Normalize and clamp (ensure proportions sum to 1 at each timepoint)
    signature_props = np.clip(signature_props, 1e-8, 1-1e-8)
    props_sum_before = signature_props.sum(axis=0)
    signature_props = signature_props / (signature_props.sum(axis=0, keepdims=True) + 1e-8)
    props_sum_after = signature_props.sum(axis=0)
    
    # Debug: check if normalization worked
    if not np.allclose(props_sum_after, 1.0, atol=0.01):
        print(f"WARNING in _calculate_signature_refs_from_Y: Proportions don't sum to 1!")
        print(f"  Before norm: {props_sum_before[0]:.3f}, After norm: {props_sum_after[0]:.3f}")
        print(f"  Valid diseases: {valid_mask.sum()}/{len(clusters)}, K={K}")
    
    # Convert to logit (noisy signature refs)
    logit_props = logit(signature_props)
    
    # Store noisy refs if requested
    if return_noisy:
        signature_refs_noisy = logit_props.copy()
        signature_props_noisy = signature_props.copy()
    
    # Optionally smooth
    signature_refs = np.zeros_like(logit_props)
    times = np.arange(T)
    for k in range(K):
        if smooth:
            smoothed = lowess(
                logit_props[k],
                times,
                frac=frac,
                it=3,
                delta=0.0,
                return_sorted=False
            )
            signature_refs[k] = smoothed
        else:
            signature_refs[k] = logit_props[k]
    
    # Re-normalize after smoothing to ensure proportions sum to 1 at each timepoint
    # Convert back to probability scale, normalize, then convert back to logit
    from scipy.special import expit
    props_smoothed = expit(signature_refs)  # Convert logit to probability
    props_smoothed = np.clip(props_smoothed, 1e-8, 1-1e-8)
    props_smoothed = props_smoothed / (props_smoothed.sum(axis=0, keepdims=True) + 1e-8)  # Re-normalize to sum to 1
    signature_refs = logit(props_smoothed)  # Convert back to logit scale
    
    if return_noisy:
        return signature_refs_noisy, signature_refs, signature_props_noisy
    else:
        return signature_refs


def choose_sample_diseases(phi: np.ndarray, k_per_sig: int = 1, fallback_max: int = 5):
    """
    Pick a small, meaningful disease set if the caller doesn't provide one.
    - prefer MI(112), Depression(67), AFib(127), Breast(17), Colon(10) if in range
    - otherwise choose top diseases per signature by max eta over time
    """
    preferred = [112, 47, 127, 17, 10]
    K, D, T = phi.shape
    fixed = [d for d in preferred if 0 <= d < D]
    if len(fixed) >= min(fallback_max, D):
        return fixed[:fallback_max]

    eta = expit(phi)  # [K, D, T]
    scores = eta.max(axis=2)  # [K, D]

    chosen = []
    for k in range(K):
        for d in np.argsort(-scores[k])[:k_per_sig]:
            if int(d) not in chosen:
                chosen.append(int(d))
            if len(chosen) >= fallback_max:
                return chosen

    # fill remaining from overall
    overall = np.argsort(-scores.max(axis=0))
    for d in overall:
        if int(d) not in chosen:
            chosen.append(int(d))
        if len(chosen) >= fallback_max:
            break
    return chosen


def generate_clustered_survival_data_external_phi_lam(
    N: int = 1000, D: int = None, T: int = None, K: int = None, P: int = None
):
    """
    Simulate a cohort using population structure from a fitted model.
    - Uses fitted phi and time-varying signature references
    - Uses TRUE initial psi and cluster assignments (from initial_psi_400k.pt and initial_clusters_400k.pt)
    - Samples new individuals with genetics G and GP-driven lambda
    - Computes theta, pi and simulates first-event-at-risk Y
    """
    ckpt = torch.load(
        "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_model_W0.0001_fulldata_sexspecific.pt",
        map_location="cpu",
    )
    st = ckpt.get("model_state_dict", ckpt)

    # Load TRUE initial psi and cluster assignments (not fitted)
    initial_psi = torch.load(
        "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_psi_400k.pt",
        map_location="cpu"
    )
    initial_clusters = torch.load(
        "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt",
        map_location="cpu"
    )
    
    # Convert to numpy if needed
    if torch.is_tensor(initial_psi):
        initial_psi = initial_psi.cpu().numpy()
    if torch.is_tensor(initial_clusters):
        initial_clusters = initial_clusters.cpu().numpy()
    
    # Fitted components (we'll use phi from fit, but psi/clusters from initial)
    phi = st["phi"].cpu().numpy()  # [K, D, T]
    lam_fit = st["lambda_"].cpu().numpy()  # [N_fit, K, T]
    gamma = st.get("gamma", None)
    gamma = gamma.cpu().numpy() if gamma is not None else None

    # Dimensions from fit (unless overridden by args)
    K_fit, D_fit, T_fit = phi.shape
    D = D or D_fit
    T = T or T_fit
    
    # Handle dimension mismatch: initial_psi might have K signatures, but fit might have K+1 (with healthy)
    K_initial = initial_psi.shape[0]
    if initial_psi.shape[1] != D_fit:
        raise ValueError(f"initial_psi D dimension {initial_psi.shape[1]} doesn't match D_fit {D_fit}")
    if len(initial_clusters) != D_fit:
        raise ValueError(f"initial_clusters length {len(initial_clusters)} doesn't match D_fit {D_fit}")
    
    # Check if fitted model has healthy signature (K_fit = K_initial + 1)
    has_healthy = (K_fit == K_initial + 1)
    
    # Verify that fitted psi's max per disease matches initial_clusters
    fitted_psi = st.get("psi", None)
    if fitted_psi is not None:
        fitted_psi = fitted_psi.cpu().numpy()
        if fitted_psi.shape[0] == K_fit and fitted_psi.shape[1] == D_fit:
            # Check max psi per disease (excluding healthy if present)
            disease_sigs = fitted_psi[:K_initial, :] if has_healthy else fitted_psi
            max_sig_per_disease = np.argmax(disease_sigs, axis=0)
            matches = (max_sig_per_disease == initial_clusters).sum()
            print(f"Fitted psi max matches initial clusters for {matches}/{D_fit} diseases")
            if matches < D_fit * 0.8:  # Warn if < 80% match
                print(f"Warning: Only {matches}/{D_fit} diseases have max psi matching initial cluster")
    
    # Use TRUE psi (not fitted) - pad with healthy if needed
    if has_healthy:
        # Add healthy signature row (typically all negative/low values)
        healthy_psi = np.full((1, D_fit), -5.0)  # Healthy signature has low psi for all diseases
        psi = np.vstack([initial_psi, healthy_psi])  # [K_fit, D_fit]
    else:
        psi = initial_psi  # [K_fit, D_fit]

    # Choose sample diseases first to identify required signatures
    sample_diseases = choose_sample_diseases(phi, k_per_sig=1, fallback_max=5)
    
    # Identify required signatures using TRUE cluster assignments
    # Each disease d belongs to cluster initial_clusters[d]
    required_sigs = set()
    for d in sample_diseases:
        if d < D_fit:
            true_cluster = int(initial_clusters[d])
            required_sigs.add(true_cluster)
    required_sigs = sorted(list(required_sigs))
    
    # Determine K: if specified and < K_fit, randomly sample but include required
    K_target = K or K_fit
    if K_target < K_fit:
        # Randomly sample K signatures, ensuring required ones are included
        all_sigs = set(range(K_fit))
        remaining = list(all_sigs - set(required_sigs))
        n_needed = K_target - len(required_sigs)
        if n_needed > 0:
            if len(remaining) >= n_needed:
                additional = np.random.choice(remaining, size=n_needed, replace=False).tolist()
            else:
                additional = remaining  # take all remaining if not enough
            selected_sigs = sorted(required_sigs + additional)
        else:
            selected_sigs = required_sigs[:K_target]  # if K_target < len(required), just take first K_target
        print(f"Selected {len(selected_sigs)} signatures from {K_fit} total: {selected_sigs}")
        print(f"Required signatures (for sample diseases): {required_sigs}")
    else:
        selected_sigs = list(range(K_fit))
    
    K = len(selected_sigs)
    
    # Find all diseases that belong to the selected signatures
    # Map selected_sigs (which are indices in K_fit) to indices in K_initial (disease signatures only)
    # Exclude healthy signature (index K_initial) from disease selection
    selected_sigs_initial = [s for s in selected_sigs if s < K_initial]
    
    if len(selected_sigs_initial) == 0:
        raise ValueError(f"No disease signatures selected! selected_sigs={selected_sigs}, K_initial={K_initial}")
    
    # Find diseases that belong to these signatures (using initial_clusters)
    candidate_diseases = []
    for d in range(D_fit):
        if int(initial_clusters[d]) in selected_sigs_initial:
            candidate_diseases.append(d)
    
    # Select up to D diseases from candidates
    if D and len(candidate_diseases) >= D:
        # Prioritize sample_diseases, then randomly sample rest
        selected_diseases = list(set(sample_diseases) & set(candidate_diseases))
        remaining = [d for d in candidate_diseases if d not in selected_diseases]
        n_needed = D - len(selected_diseases)
        if n_needed > 0 and len(remaining) > 0:
            additional = np.random.choice(remaining, size=min(n_needed, len(remaining)), replace=False).tolist()
            selected_diseases = sorted(selected_diseases + additional)
        else:
            selected_diseases = sorted(selected_diseases[:D])
    else:
        selected_diseases = sorted(candidate_diseases[:D] if D else candidate_diseases)
    
    print(f"Selected {len(selected_diseases)} diseases from signatures {selected_sigs}: {selected_diseases[:10]}..." if len(selected_diseases) > 10 else f"Selected {len(selected_diseases)} diseases: {selected_diseases}")
    
    # Subset all arrays to selected signatures AND selected diseases
    phi = phi[selected_sigs, :, :]  # [K, D_fit, T_fit]
    psi = psi[selected_sigs, :]  # [K, D_fit]
    lam_fit = lam_fit[:, selected_sigs, :]  # [N_fit, K, T_fit]
    if gamma is not None:
        gamma = gamma[:, selected_sigs]  # [P, K]
    
    # Now subset to selected diseases
    phi = phi[:, selected_diseases, :]  # [K, len(selected_diseases), T_fit]
    psi = psi[:, selected_diseases]  # [K, len(selected_diseases)]
    initial_clusters_subset = initial_clusters[selected_diseases]  # [len(selected_diseases)]

    # Map cluster assignments to position indices for calculating signature_refs from Y later
    sig_to_position = {orig_sig: k_idx for k_idx, orig_sig in enumerate(selected_sigs_initial)}
    clusters_mapped = np.array([sig_to_position.get(int(c), -1) for c in initial_clusters_subset])
    
    # Align dimensions to T
    phi = phi[:, :, :T]  # [K, len(selected_diseases), T]
    lam_fit = lam_fit[:, :, :T]  # [N_fit, K, T]
    
    # Update D to actual number of selected diseases
    D = len(selected_diseases)

    # Get G_fit from checkpoint (for matching individuals to lambda)
    G_fit = ckpt.get("G", None)
    if G_fit is not None:
        G_fit = G_fit.cpu().numpy() if torch.is_tensor(G_fit) else np.asarray(G_fit)
    
    # Use TRUE lambda from fitted model (sample N individuals from fitted lambda)
    # If N > N_fit, sample with replacement; if N <= N_fit, sample without replacement
    N_fit = lam_fit.shape[0]
    if N <= N_fit:
        # Sample without replacement if we have enough fitted individuals
        idx = np.random.choice(N_fit, size=N, replace=False)
        lambda_true = lam_fit[idx, :, :]  # [N, K, T] - TRUE fitted lambda
    else:
        # Sample with replacement if we need more individuals
        idx = np.random.choice(N_fit, size=N, replace=True)
        lambda_true = lam_fit[idx, :, :]  # [N, K, T] - TRUE fitted lambda
    
    # Use TRUE phi (already subset to selected diseases)
    # phi is [K, D, T] where D = len(selected_diseases)
    
    # Sample G to match lambda (or generate new if needed)
    if gamma is not None:
        P = gamma.shape[0]
        if G_fit is not None and G_fit.shape[1] == P:
            # Use G corresponding to the sampled individuals
            G_new = G_fit[idx]
        else:
            G_new = np.random.randn(N, P)
    else:
        P = 0
        G_new = None

    # Generate Y using TRUE lambda and TRUE phi
    # Softmax over signatures (axis=1)
    x = lambda_true - lambda_true.max(axis=1, keepdims=True)
    ex = np.exp(x)
    theta_true = ex / ex.sum(axis=1, keepdims=True)  # [N, K, T]

    eta = expit(phi)  # [K, D, T] - TRUE fitted phi
    pi = np.einsum("nkt,kdt->ndt", theta_true, eta)  # [N, D, T]

    # Simulate first-event-at-risk
    Y = np.zeros((N, D, T), dtype=np.int8)
    event_times = np.full((N, D), T, dtype=np.int32)
    for n in range(N):
        for d in range(D):
            for t in range(T):
                if Y[n, d, :t].sum() == 0 and np.random.rand() < pi[n, d, t]:
                    Y[n, d, t] = 1
                    event_times[n, d] = t
                    break

    # NOW calculate signature_refs from the generated Y (like production workflow)
    # This is what we'll use for model initialization
    print("Calculating signature_refs from generated Y data (production workflow)...")
    # Use stronger smoothing (frac=0.5) for better smoothing of sparse data
    signature_refs_from_Y = _calculate_signature_refs_from_Y(
        Y, clusters_mapped, K, T, smooth=True, frac=0.5
    )  # [K, T] - calculated from simulated Y
    
    # Verify proportions sum to 1 (debug check)
    props_check = expit(signature_refs_from_Y)
    props_sum = props_check.sum(axis=0)
    if not np.allclose(props_sum, 1.0, atol=0.01):
        print(f"WARNING: Signature proportions don't sum to 1! Sum: {props_sum.mean():.3f} ± {props_sum.std():.3f}")
    
    # Don't add healthy signature - user said to forget it

    return {
        "Y": Y,
        "G": G_new,
        "theta": theta_true,  # TRUE theta from fitted lambda
        "lambda": lambda_true,  # TRUE fitted lambda (sampled individuals)
        "phi": phi,  # TRUE fitted phi
        "psi": psi,  # TRUE initial psi (not fitted) - [K, D] where D is len(selected_diseases)
        "pi": pi,
        "event_times": event_times,
        "signature_refs": signature_refs_from_Y,  # Calculated from Y (like production workflow)
        "selected_diseases": selected_diseases,  # All diseases selected (up to D)
        "sample_diseases": sample_diseases,  # Subset used for signature selection
        "selected_sigs": selected_sigs,  # which signatures from original fit were selected
        "K_fit": K_fit,  # original number of signatures
        "initial_clusters": initial_clusters_subset,  # TRUE cluster assignments for selected diseases
    }






def plot_synthetic_components(data, num_samples=5, save_path=None):
    """
    Visualize the synthetic data components including individual lambda and state trajectories.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the synthetic data components
    num_samples : int
        Number of samples to plot for each component
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed.
    """
    plt.figure(figsize=(20, 12))
    
    # 1. Plot sample phi trajectories for each cluster
    plt.subplot(231)
    for k in range(data['phi'].shape[0]):  # For each cluster
        for d in range(min(num_samples, data['phi'].shape[1])):  # Sample diseases
            plt.plot(data['phi'][k,d,:], alpha=0.5, label=f'Cluster {k}' if d==0 else '')
    plt.title('Sample φ Trajectories by Cluster')
    plt.xlabel('Time')
    plt.ylabel('φ Value')
    plt.legend()
    
    # 2. Plot sample lambda trajectories
    plt.subplot(232)
    for i in range(min(num_samples, data['lambda'].shape[0])):  # Sample individuals
        for k in range(data['lambda'].shape[1]):  # For each cluster
            plt.plot(data['lambda'][i,k,:], alpha=0.5, label=f'Cluster {k}' if i==0 else '')
    plt.title('Sample λ Trajectories')
    plt.xlabel('Time')
    plt.ylabel('λ Value')
    plt.legend()
    
    # 3. Plot psi heatmap
    plt.subplot(233)
    # Use original signature indices if available
    if 'selected_sigs' in data:
        yticklabels = [f'Sig {s}' for s in data['selected_sigs']]
    else:
        yticklabels = [f'Cluster {i}' for i in range(data['psi'].shape[0])]
    # Use selected disease indices for x-axis
    if 'selected_diseases' in data:
        xticklabels = [f'D{d}' for d in data['selected_diseases']]
        # Only show every Nth label to avoid crowding
        step = max(1, len(xticklabels) // 20)
        xticklabels_sparse = [xticklabels[i] if i % step == 0 else '' for i in range(len(xticklabels))]
    else:
        xticklabels_sparse = True  # Auto
    sns.heatmap(data['psi'], cmap='RdBu_r', center=0, 
                yticklabels=yticklabels, xticklabels=xticklabels_sparse)
    plt.title('ψ Values (Cluster-Disease Assignment)')
    plt.xlabel('Disease')
    plt.ylabel('Cluster')
    
    # 4. Plot sample theta (signature weights) as bars
    plt.subplot(234)
    width = 0.15  # Width of bars
    x = np.arange(data['theta'].shape[1])  # Cluster indices
    
    for i in range(min(num_samples, data['theta'].shape[0])):
        plt.bar(x + i*width, data['theta'][i,:,0], 
               width, alpha=0.5, label=f'Individual {i}')
    
    plt.title('Sample θ Values (t=0)')
    plt.xlabel('Cluster')
    plt.ylabel('Weight')
    plt.legend()
    plt.xticks(x + width*2, [f'Cluster {i}' for i in x])
    
    # 5. Plot sample pi trajectories
    plt.subplot(235)
    for i in range(min(num_samples, data['pi'].shape[0])):
        for d in range(min(3, data['pi'].shape[1])):
            plt.plot(data['pi'][i,d,:], alpha=0.5, label=f'Ind {i}, Disease {d}' if i==0 else '')
    plt.title('Sample π Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.yscale('log')
    plt.legend()
    
    # 6. Plot individual state trajectories if available
    if 'state' in data:
        plt.subplot(236)
        for i in range(min(num_samples, data['state'].shape[0])):
            plt.plot(data['state'][i,:], alpha=0.5, label=f'Individual {i}')
        plt.title('Individual State Trajectories')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

