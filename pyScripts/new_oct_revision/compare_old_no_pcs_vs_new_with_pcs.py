"""
Compare OLD model (retrospective, with sex, NO PCs, 10K) vs NEW model (retrospective, with sex, WITH PCs, 400K)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

def softmax_normalize_lambdas(lambdas):
    """Applies softmax normalization to the last dimension of the lambda array."""
    if len(lambdas.shape) == 3:
        # Apply softmax across the signature dimension (axis=1) for each time point
        # Reshape to (N*T, K) for softmax, then reshape back to (N, K, T)
        n_patients, n_signatures, n_timepoints = lambdas.shape
        reshaped_lambdas = lambdas.transpose(0, 2, 1).reshape(-1, n_signatures)
        normalized_lambdas = softmax(reshaped_lambdas, axis=1)
        return normalized_lambdas.reshape(n_patients, n_timepoints, n_signatures).transpose(0, 2, 1)
    elif len(lambdas.shape) == 2:
        return softmax(lambdas, axis=1)
    else:
        raise ValueError("Lambdas must be 2D (patients x signatures) or 3D (patients x signatures x time)")

def compare_old_vs_new_models():
    """
    Compare OLD model (retrospective, with sex, NO PCs, 10K) vs NEW model (retrospective, with sex, WITH PCs, 400K)
    """
    
    print("COMPARING OLD MODEL (NO PCs) vs NEW MODEL (WITH PCs)")
    print("="*60)
    
    # Load OLD model (retrospective, with sex, NO PCs, 10K)
    old_model_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enorlleobecjt/enrollment_model_W0.0001_fulldata_sexspecific.pt"
    print(f"Loading OLD model from: {old_model_path}")
    
    try:
        old_model_raw = torch.load(old_model_path, map_location='cpu')
        old_lambdas = old_model_raw['model_state_dict']['lambda_'].numpy()
        old_thetas = softmax_normalize_lambdas(old_lambdas)
        print(f"✅ OLD model (NO PCs): {old_thetas.shape}")
    except Exception as e:
        print(f"❌ Error loading OLD model: {e}")
        return
    
    # Load NEW model (retrospective, with sex, WITH PCs, 400K)
    new_model_path = "/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt"
    print(f"Loading NEW model from: {new_model_path}")
    
    try:
        new_thetas = torch.load(new_model_path, map_location='cpu').numpy()
        print(f"✅ NEW model (WITH PCs): {new_thetas.shape}")
    except Exception as e:
        print(f"❌ Error loading NEW model: {e}")
        return
    
    # Compare first 10K patients from NEW model with OLD model
    print(f"\nComparing first 10K patients from NEW model with OLD model...")
    
    old_data = old_thetas  # Shape: (10K, 21, 52)
    new_data = new_thetas[:10000, :, :]  # Take first 10K from NEW model
    
    print(f"OLD model (NO PCs): {old_data.shape}")
    print(f"NEW model (WITH PCs, first 10K): {new_data.shape}")
    
    # Ensure same dimensions
    n_patients = min(old_data.shape[0], new_data.shape[0])
    n_signatures = min(old_data.shape[1], new_data.shape[1])
    n_timepoints = min(old_data.shape[2], new_data.shape[2])
    
    old_subset = old_data[:n_patients, :n_signatures, :n_timepoints]
    new_subset = new_data[:n_patients, :n_signatures, :n_timepoints]
    
    print(f"Comparing {n_patients} patients, {n_signatures} signatures, {n_timepoints} timepoints")
    
    # Calculate signature correlations (averaged over time)
    old_avg = np.mean(old_subset, axis=2)  # Average over time
    new_avg = np.mean(new_subset, axis=2)   # Average over time
    
    signature_correlations = []
    for sig_idx in range(n_signatures):
        corr = np.corrcoef(old_avg[:, sig_idx], new_avg[:, sig_idx])[0, 1]
        signature_correlations.append(corr)
    
    signature_correlations = np.array(signature_correlations)
    
    print(f"\nSignature correlations (averaged over time):")
    print(f"Mean correlation: {np.mean(signature_correlations):.6f}")
    print(f"Std correlation: {np.std(signature_correlations):.6f}")
    print(f"Min correlation: {np.min(signature_correlations):.6f}")
    print(f"Max correlation: {np.max(signature_correlations):.6f}")
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Signature Correlations
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(range(n_signatures), signature_correlations, marker='o', linestyle='-', markersize=4)
    ax1.set_title('Signature Correlations\n(OLD model NO PCs vs NEW model WITH PCs)', fontweight='bold')
    ax1.set_xlabel('Signature Index')
    ax1.set_ylabel('Correlation')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(0.7, 1.0)
    
    # Plot 2: Distribution of Signature Correlations
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(signature_correlations, bins=10, kde=False, ax=ax2, edgecolor='black')
    ax2.set_title('Distribution of Signature Correlations\n(OLD model NO PCs vs NEW model WITH PCs)', fontweight='bold')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Count')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: Cross-Correlation Matrix
    cross_corr_matrix = np.zeros((n_signatures, n_signatures))
    for i in range(n_signatures):
        for j in range(n_signatures):
            cross_corr_matrix[i, j] = np.corrcoef(old_avg[:, i], new_avg[:, j])[0, 1]
    
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(cross_corr_matrix, cmap='RdBu_r', vmin=-0.2, vmax=0.8, ax=ax3, 
                cbar_kws={'label': 'Correlation'})
    ax3.set_title('Cross-Correlation Matrix\n(OLD model NO PCs vs NEW model WITH PCs)', fontweight='bold')
    ax3.set_xlabel('NEW model (WITH PCs) Signatures')
    ax3.set_ylabel('OLD model (NO PCs) Signatures')
    ax3.set_aspect('equal')
    
    # Plot 4: Time-Varying Correlations
    ax4 = plt.subplot(2, 2, 4)
    time_correlations = []
    for t in range(n_timepoints):
        # Calculate correlation for each signature at this time point, then average
        corrs_at_t = []
        for sig_idx in range(n_signatures):
            corr = np.corrcoef(old_subset[:, sig_idx, t], new_subset[:, sig_idx, t])[0, 1]
            corrs_at_t.append(corr)
        time_correlations.append(np.mean(corrs_at_t))
    
    ages = np.arange(30, 30 + n_timepoints)
    ax4.plot(ages, time_correlations, marker='o', linestyle='-', markersize=3)
    ax4.set_title('Time-Varying Correlations\n(OLD model NO PCs vs NEW model WITH PCs)', fontweight='bold')
    ax4.set_xlabel('Age (years)')
    ax4.set_ylabel('Average Correlation')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.set_ylim(0.7, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\nSUMMARY:")
    mean_corr = np.mean(signature_correlations)
    if mean_corr > 0.95:
        print("✅ OLD model (NO PCs) and NEW model (WITH PCs) are highly correlated!")
        print("This suggests that adding PCs doesn't dramatically change the signature structure.")
    elif mean_corr > 0.85:
        print("⚠️ OLD model (NO PCs) and NEW model (WITH PCs) are moderately correlated")
        print("This suggests that adding PCs has some impact on signature structure.")
    else:
        print("❌ OLD model (NO PCs) and NEW model (WITH PCs) have low correlation")
        print("This suggests that adding PCs significantly changes the signature structure!")
    
    # Show which signatures are most/least affected by PCs
    print(f"\nSignatures most affected by adding PCs (lowest correlations):")
    sorted_indices = np.argsort(signature_correlations)
    for i in range(3):
        sig_idx = sorted_indices[i]
        print(f"  Signature {sig_idx}: correlation = {signature_correlations[sig_idx]:.6f}")
    
    print(f"\nSignatures least affected by adding PCs (highest correlations):")
    for i in range(3):
        sig_idx = sorted_indices[-(i+1)]
        print(f"  Signature {sig_idx}: correlation = {signature_correlations[sig_idx]:.6f}")

if __name__ == "__main__":
    compare_old_vs_new_models()
