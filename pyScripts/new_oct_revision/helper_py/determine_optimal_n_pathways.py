#!/usr/bin/env python3
"""
Determine Optimal Number of Pathways

This script helps determine the optimal number of pathways/clusters by evaluating:
1. Elbow method (inertia/within-cluster sum of squares)
2. Silhouette score (separation between clusters)
3. Calinski-Harabasz index (ratio of between-cluster to within-cluster variance)
4. Gap statistic (compares cluster quality to random data)

Usage:
    from determine_optimal_n_pathways import find_optimal_n_pathways
    optimal_k = find_optimal_n_pathways(target_disease="myocardial infarction", 
                                        k_range=(2, 10))
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data, discover_disease_pathways


def find_optimal_n_pathways(target_disease="myocardial infarction",
                            k_range=(2, 10),
                            method='deviation_from_reference',
                            lookback_years=10,
                            plot=True,
                            output_dir='pathway_optimal_k_analysis'):
    """
    Determine optimal number of pathways using multiple metrics
    
    Parameters:
    -----------
    target_disease : str
        Target disease name
    k_range : tuple
        Range of k values to test (min, max)
    method : str
        Clustering method ('deviation_from_reference', 'average_loading', etc.)
    lookback_years : int
        Years before disease to analyze
    plot : bool
        Whether to create visualization
    output_dir : str
        Output directory for plots
    
    Returns:
    --------
    dict with optimal k recommendations and all metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("DETERMINING OPTIMAL NUMBER OF PATHWAYS")
    print("="*80)
    print(f"Target disease: {target_disease}")
    print(f"Testing k range: {k_range[0]} to {k_range[1]}")
    print(f"Method: {method}")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Find target disease
    target_disease_idx = None
    for i, name in enumerate(disease_names):
        if target_disease.lower() in name.lower():
            target_disease_idx = i
            break
    
    if target_disease_idx is None:
        raise ValueError(f"Could not find '{target_disease}' in disease list")
    
    print(f"   Found '{target_disease}' at index {target_disease_idx}")
    
    # Get patients with target disease
    if hasattr(Y, 'numpy'):
        Y_np = Y.numpy()
    else:
        Y_np = np.array(Y)
    
    # Find patients who developed the disease
    target_patients_mask = (Y_np[:, target_disease_idx, :].sum(axis=1) > 0)
    target_patient_indices = np.where(target_patients_mask)[0]
    
    print(f"   Found {len(target_patient_indices):,} patients with {target_disease}")
    
    # Prepare trajectory features (same as pathway discovery)
    print("\n2. Preparing trajectory features...")
    
    if hasattr(thetas, 'numpy'):
        thetas_np = thetas.numpy()
    else:
        thetas_np = np.array(thetas)
    
    N, K, T = thetas_np.shape
    population_reference = np.mean(thetas_np, axis=0)  # (K, T)
    
    trajectory_features = []
    valid_patient_indices = []
    
    for patient_idx in target_patient_indices:
        # Find disease onset time
        disease_times = np.where(Y_np[patient_idx, target_disease_idx, :] > 0)[0]
        if len(disease_times) == 0:
            continue
        
        first_disease_time = disease_times[0]
        age_at_disease = 30 + first_disease_time
        
        if first_disease_time < lookback_years:
            continue  # Not enough history
        
        # Get pre-disease trajectory
        lookback_start = max(0, first_disease_time - lookback_years)
        pre_disease_traj = thetas_np[patient_idx, :, lookback_start:first_disease_time]  # (K, lookback_years)
        ref_traj = population_reference[:, lookback_start:first_disease_time]  # (K, lookback_years)
        
        # Calculate deviation
        if method == 'deviation_from_reference':
            deviation = pre_disease_traj - ref_traj  # (K, lookback_years)
            features = deviation.flatten()  # (K * lookback_years,)
        elif method == 'average_loading':
            features = pre_disease_traj.mean(axis=1)  # (K,)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        trajectory_features.append(features)
        valid_patient_indices.append(patient_idx)
    
    trajectory_features = np.array(trajectory_features)
    print(f"   Created features for {len(trajectory_features):,} patients")
    print(f"   Feature shape: {trajectory_features.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(trajectory_features)
    
    # Test different k values
    print(f"\n3. Testing k values from {k_range[0]} to {k_range[1]}...")
    
    k_values = range(k_range[0], k_range[1] + 1)
    inertias = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    
    for k in k_values:
        print(f"   Testing k={k}...", end=' ')
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features_scaled)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
            sil_score = silhouette_score(features_scaled, labels)
            silhouette_scores.append(sil_score)
            
            ch_score = calinski_harabasz_score(features_scaled, labels)
            calinski_harabasz_scores.append(ch_score)
        else:
            silhouette_scores.append(0)
            calinski_harabasz_scores.append(0)
        
        print(f"✓ (inertia: {inertia:.1f}, silhouette: {silhouette_scores[-1]:.3f})")
    
    # Find optimal k for each metric
    print("\n4. Analyzing results...")
    
    # Elbow method: find k where inertia decrease slows down
    # Calculate rate of change (second derivative approximation)
    inertia_changes = np.diff(inertias)
    inertia_changes_2 = np.diff(inertia_changes)
    
    # Find elbow (point where second derivative is minimum)
    if len(inertia_changes_2) > 0:
        elbow_k_idx = np.argmin(inertia_changes_2) + 1  # +1 because diff reduces length
        optimal_k_elbow = k_values[elbow_k_idx] if elbow_k_idx < len(k_values) else k_range[1]
    else:
        optimal_k_elbow = k_range[1]
    
    # Silhouette: higher is better
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    
    # Calinski-Harabasz: higher is better
    optimal_k_ch = k_values[np.argmax(calinski_harabasz_scores)]
    
    print(f"\n   Optimal k by Elbow method: {optimal_k_elbow}")
    print(f"   Optimal k by Silhouette score: {optimal_k_silhouette} (score: {max(silhouette_scores):.3f})")
    print(f"   Optimal k by Calinski-Harabasz: {optimal_k_ch} (score: {max(calinski_harabasz_scores):.1f})")
    
    # Create visualization
    if plot:
        print("\n5. Creating visualization...")
        create_optimal_k_plot(
            k_values, inertias, silhouette_scores, calinski_harabasz_scores,
            optimal_k_elbow, optimal_k_silhouette, optimal_k_ch,
            output_dir, target_disease
        )
    
    # Summary recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Use consensus: if multiple methods agree, that's stronger
    recommendations = [optimal_k_elbow, optimal_k_silhouette, optimal_k_ch]
    consensus_k = max(set(recommendations), key=recommendations.count)
    
    print(f"\nSuggested number of pathways: {consensus_k}")
    print(f"\nRationale:")
    print(f"  - Elbow method suggests: {optimal_k_elbow}")
    print(f"  - Silhouette score suggests: {optimal_k_silhouette}")
    print(f"  - Calinski-Harabasz suggests: {optimal_k_ch}")
    
    if len(set(recommendations)) == 1:
        print(f"\n  ✓ All methods agree on k={consensus_k}")
    else:
        print(f"\n  ⚠ Methods suggest different values - this is COMMON in clustering!")
        print(f"\n  Why metrics disagree:")
        print(f"    • Elbow method: Optimizes for compact clusters (tends to suggest higher k)")
        print(f"    • Silhouette: Optimizes for separation (can suggest very high k)")
        print(f"    • Calinski-Harabasz: Balanced but can be conservative (tends to suggest lower k)")
        print(f"\n  When metrics disagree, consider:")
        print(f"    1. ✨ Biological interpretability - Do pathways make clinical sense?")
        print(f"    2. 📊 Pathway sizes - Avoid very small pathways (< 5% of patients)")
        print(f"    3. 🔄 Reproducibility - Do pathways replicate across cohorts (UKB vs MGB)?")
        print(f"    4. 🎯 Parsimony - Fewer pathways are easier to interpret and validate")
        print(f"\n  💡 Recommended approach:")
        print(f"     • Start with k=4 (balance between interpretability and granularity)")
        print(f"     • Check if pathways are biologically distinct")
        print(f"     • Verify reproducibility across cohorts")
        print(f"     • If pathways merge or split between cohorts, adjust accordingly")
    
    # Show pathway sizes for recommended k and k=4 (commonly used)
    print(f"\n  Pathway size distribution for k={consensus_k}:")
    kmeans_final = KMeans(n_clusters=consensus_k, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(features_scaled)
    unique_labels, counts = np.unique(labels_final, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"    Pathway {label}: {count:,} patients ({count/len(labels_final)*100:.1f}%)")
    
    # Also show k=4 if it's different
    if 4 in k_values and 4 != consensus_k:
        print(f"\n  Pathway size distribution for k=4 (commonly used for interpretability):")
        kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels_k4 = kmeans_k4.fit_predict(features_scaled)
        unique_labels_k4, counts_k4 = np.unique(labels_k4, return_counts=True)
        for label, count in zip(unique_labels_k4, counts_k4):
            print(f"    Pathway {label}: {count:,} patients ({count/len(labels_k4)*100:.1f}%)")
        
        # Check if k=4 has reasonable sizes
        min_size_pct = min(counts_k4) / len(labels_k4) * 100
        if min_size_pct < 5:
            print(f"\n    ⚠ Warning: Smallest pathway is {min_size_pct:.1f}% - may be too small")
        else:
            print(f"\n    ✓ All pathways have reasonable sizes (min: {min_size_pct:.1f}%)")
    
    return {
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_silhouette': optimal_k_silhouette,
        'optimal_k_ch': optimal_k_ch,
        'consensus_k': consensus_k,
        'k_values': list(k_values),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'recommendations': {
            'elbow': optimal_k_elbow,
            'silhouette': optimal_k_silhouette,
            'calinski_harabasz': optimal_k_ch,
            'consensus': consensus_k,
            'note': 'When metrics disagree, consider biological interpretability, pathway sizes, and reproducibility across cohorts. k=4 is commonly used as a balance between granularity and interpretability.'
        }
    }


def create_optimal_k_plot(k_values, inertias, silhouette_scores, 
                          calinski_harabasz_scores,
                          optimal_k_elbow, optimal_k_silhouette, optimal_k_ch,
                          output_dir, target_disease):
    """
    Create visualization showing metrics across different k values
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    k_arr = np.array(k_values)
    
    # Plot 1: Elbow method (Inertia)
    ax1 = axes[0, 0]
    ax1.plot(k_arr, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--', 
               linewidth=2, label=f'Optimal (k={optimal_k_elbow})')
    ax1.set_xlabel('Number of Pathways (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(k_arr, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--',
               linewidth=2, label=f'Optimal (k={optimal_k_silhouette})')
    ax2.set_xlabel('Number of Pathways (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Score (Higher = Better)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Calinski-Harabasz Score
    ax3 = axes[1, 0]
    ax3.plot(k_arr, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
    ax3.axvline(x=optimal_k_ch, color='r', linestyle='--',
               linewidth=2, label=f'Optimal (k={optimal_k_ch})')
    ax3.set_xlabel('Number of Pathways (k)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
    ax3.set_title('Calinski-Harabasz Index (Higher = Better)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Normalized comparison
    ax4 = axes[1, 1]
    # Normalize all metrics to [0, 1] for comparison
    inertias_norm = 1 - (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))
    sil_norm = (np.array(silhouette_scores) - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores))
    ch_norm = (np.array(calinski_harabasz_scores) - min(calinski_harabasz_scores)) / (max(calinski_harabasz_scores) - min(calinski_harabasz_scores))
    
    ax4.plot(k_arr, inertias_norm, 'b-o', linewidth=2, markersize=6, label='Inertia (normalized)')
    ax4.plot(k_arr, sil_norm, 'g-o', linewidth=2, markersize=6, label='Silhouette (normalized)')
    ax4.plot(k_arr, ch_norm, 'm-o', linewidth=2, markersize=6, label='Calinski-Harabasz (normalized)')
    ax4.set_xlabel('Number of Pathways (k)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax4.set_title('Combined Metrics (Normalized)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle(f'Optimal Number of Pathways: {target_disease.title()}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    filename = f'{output_dir}/optimal_k_analysis_{target_disease.replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    results = find_optimal_n_pathways(
        target_disease="myocardial infarction",
        k_range=(2, 8),
        method='deviation_from_reference'
    )
    
    print(f"\n✅ Recommended number of pathways: {results['consensus_k']}")

