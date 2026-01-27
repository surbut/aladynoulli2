#!/usr/bin/env python3
"""
Test if heterogeneity clustering needs to be remade for unregularized fixedgk.

This script:
1. Loads thetas from original model
2. Loads lambda from unregularized fixedgk model checkpoints
3. Converts lambda to theta (softmax)
4. Compares cluster assignments (k-means on time-averaged thetas) for same diseases
5. Reports similarity metrics (adjusted rand index, cluster overlap)

If clusters are highly similar, heterogeneity analysis doesn't need to be remade.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import contingency, pearsonr
import glob
import warnings
warnings.filterwarnings('ignore')

def load_original_thetas():
    """Load thetas from original training run (what heterogeneity analysis uses)"""
    print("Loading original thetas from training run...")
    thetas_path = '/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective_correctE.pt'
    thetas = torch.load(thetas_path, map_location='cpu', weights_only=False)
    if torch.is_tensor(thetas):
        thetas = thetas.numpy()
    print(f"  Original thetas shape: {thetas.shape}")
    print(f"  Note: These are from original training run (what heterogeneity analysis uses)")
    return thetas

def load_unregularized_training_lambda():
    """Load lambda from unregularized TRAINING batches (same context as original) and convert to theta"""
    print("\nLoading lambda from unregularized TRAINING batches...")
    print("  (These are what fixedgk predictions are based on)")
    
    # Load from unregularized training batches (same as compare_regularized_vs_unregularized_phis_lambdas.py)
    base_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr")
    model_pattern = 'enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt'
    model_files = sorted(base_path.glob(model_pattern),
                        key=lambda x: int(x.stem.split('_')[-2]) if x.stem.split('_')[-2].isdigit() else 0)
    
    if not model_files:
        raise FileNotFoundError(f"No unregularized training batches found at {base_path}/{model_pattern}")
    
    print(f"  Found {len(model_files)} model files")
    
    lambda_batches = []
    for model_file in model_files[:40]:  # First 40 batches = 400K
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        lambda_batch = state_dict['lambda_'].detach().cpu().numpy()
        lambda_batches.append(lambda_batch)
        if len(lambda_batches) <= 5:
            print(f"    Loaded {model_file.name}: lambda shape {lambda_batch.shape}")
    
    # Concatenate all batches
    lambda_full = np.concatenate(lambda_batches, axis=0)
    print(f"  Combined lambda shape: {lambda_full.shape}")
    
    # Subset to first 400K to match original
    if lambda_full.shape[0] > 400000:
        print(f"  Subsetting to first 400K patients (from {lambda_full.shape[0]})")
        lambda_full = lambda_full[:400000, :, :]
    
    # Convert lambda to theta using softmax
    print("  Converting lambda to theta (softmax)...")
    from scipy.special import softmax
    theta_full = softmax(lambda_full, axis=1)
    
    print(f"  Theta shape: {theta_full.shape}")
    return theta_full

def compare_clustering(original_thetas, unregularized_thetas, Y, disease_names, target_diseases, n_clusters=3, random_state=42):
    """Compare cluster assignments between original and fixedgk models"""
    print("\n" + "="*80)
    print("COMPARING CLUSTER ASSIGNMENTS")
    print("="*80)
    
    # Calculate time-averaged thetas
    original_time_avg = original_thetas.mean(axis=2)
    unregularized_time_avg = unregularized_thetas.mean(axis=2)
    
    print(f"\nTime-averaged theta shapes:")
    print(f"  Original (regularized): {original_time_avg.shape}")
    print(f"  Unregularized: {unregularized_time_avg.shape}")
    
    # Check correlation of time-averaged thetas (what's actually used for clustering)
    print("\n" + "="*80)
    print("CHECKING CORRELATION OF TIME-AVERAGED THETAS")
    print("="*80)
    
    # Sample for correlation check (use all if < 100K, otherwise sample)
    n_samples = min(100000, original_time_avg.shape[0])
    if n_samples < original_time_avg.shape[0]:
        sample_idx = np.random.choice(original_time_avg.shape[0], n_samples, replace=False)
        original_sample = original_time_avg[sample_idx, :]
        unregularized_sample = unregularized_time_avg[sample_idx, :]
        print(f"  Sampling {n_samples:,} patients for correlation check")
    else:
        original_sample = original_time_avg
        unregularized_sample = unregularized_time_avg
        print(f"  Using all {n_samples:,} patients for correlation check")
    
    # Flatten for correlation
    original_flat = original_sample.flatten()
    unregularized_flat = unregularized_sample.flatten()
    
    # Calculate correlation
    from scipy.stats import pearsonr
    corr, pval = pearsonr(original_flat, unregularized_flat)
    print(f"\n  Overall correlation (all signatures × patients):")
    print(f"    Pearson r = {corr:.6f} (p < {pval:.2e})")
    
    # Per-signature correlations
    print(f"\n  Per-signature correlations:")
    sig_corrs = []
    for sig_idx in range(original_time_avg.shape[1]):
        sig_corr, _ = pearsonr(original_time_avg[:, sig_idx], unregularized_time_avg[:, sig_idx])
        sig_corrs.append(sig_corr)
        if sig_idx < 5 or sig_idx >= original_time_avg.shape[1] - 2:  # Show first 5 and last 2
            print(f"    Signature {sig_idx}: r = {sig_corr:.6f}")
    
    print(f"    Mean per-signature correlation: {np.mean(sig_corrs):.6f}")
    print(f"    Min per-signature correlation: {np.min(sig_corrs):.6f}")
    print(f"    Max per-signature correlation: {np.max(sig_corrs):.6f}")
    
    results = []
    
    for target_disease in target_diseases:
        # Find disease index
        try:
            disease_ix = disease_names.index(target_disease)
        except ValueError:
            print(f"\n⚠️  Disease '{target_disease}' not found, skipping")
            continue
        
        # Identify diseased patients
        diseased = np.where(Y[:, disease_ix, :].sum(axis=1) > 0)[0]
        print(f"\n{'='*60}")
        print(f"Disease: {target_disease}")
        print(f"  Diseased patients: {len(diseased):,}")
        
        if len(diseased) < n_clusters:
            print(f"  ⚠️  Not enough patients for {n_clusters} clusters, skipping")
            continue
        
        # Get time-averaged thetas for diseased patients
        original_theta_diseased = original_time_avg[diseased, :]
        unregularized_theta_diseased = unregularized_time_avg[diseased, :]
        
        # Cluster both
        print(f"  Clustering original (regularized) thetas...")
        kmeans_original = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters_original = kmeans_original.fit_predict(original_theta_diseased)
        
        print(f"  Clustering unregularized thetas...")
        kmeans_unregularized = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters_unregularized = kmeans_unregularized.fit_predict(unregularized_theta_diseased)
        
        # Find best label permutation to match clusters
        # This handles the case where clusters are the same but labels are permuted
        from scipy.optimize import linear_sum_assignment
        from scipy.stats import contingency
        
        # Create contingency matrix
        contingency_matrix = contingency.crosstab(clusters_original, clusters_unregularized)[1]
        
        # Find optimal label matching using Hungarian algorithm
        # We want to maximize the number of patients in matching clusters
        # So we minimize the negative of matches
        cost_matrix = -contingency_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping from unregularized labels to original labels
        label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        clusters_unregularized_remapped = np.array([label_mapping.get(c, c) for c in clusters_unregularized])
        
        # Calculate similarity metrics with remapped labels
        ari = adjusted_rand_score(clusters_original, clusters_unregularized_remapped)
        
        # Calculate cluster overlap (how many patients in same cluster after remapping)
        same_cluster = (clusters_original == clusters_unregularized_remapped).sum()
        overlap_pct = 100 * same_cluster / len(diseased)
        
        # Cluster sizes (original)
        original_sizes = [np.sum(clusters_original == i) for i in range(n_clusters)]
        unregularized_sizes = [np.sum(clusters_unregularized == i) for i in range(n_clusters)]
        unregularized_sizes_remapped = [np.sum(clusters_unregularized_remapped == i) for i in range(n_clusters)]
        
        # Check cluster centroid similarity
        original_centroids = kmeans_original.cluster_centers_
        unregularized_centroids = kmeans_unregularized.cluster_centers_
        
        # Reorder unregularized centroids to match original
        unregularized_centroids_remapped = np.zeros_like(unregularized_centroids)
        for orig_label in range(n_clusters):
            unreg_label = [k for k, v in label_mapping.items() if v == orig_label][0]
            unregularized_centroids_remapped[orig_label, :] = unregularized_centroids[unreg_label, :]
        
        # Calculate centroid distances
        centroid_distances = np.linalg.norm(original_centroids - unregularized_centroids_remapped, axis=1)
        mean_centroid_distance = np.mean(centroid_distances)
        
        print(f"\n  Results:")
        print(f"    Adjusted Rand Index: {ari:.4f}")
        print(f"    Cluster overlap (after label matching): {overlap_pct:.1f}% ({same_cluster}/{len(diseased)} patients)")
        print(f"    Mean centroid distance (after remapping): {mean_centroid_distance:.6f}")
        print(f"    Original (regularized) cluster sizes: {original_sizes}")
        print(f"    Unregularized cluster sizes (original labels): {unregularized_sizes}")
        print(f"    Unregularized cluster sizes (remapped to match): {unregularized_sizes_remapped}")
        print(f"    Label mapping: {label_mapping}")
        
        results.append({
            'disease': target_disease,
            'n_patients': len(diseased),
            'ari': ari,
            'overlap_pct': overlap_pct,
            'overlap_count': same_cluster,
            'mean_centroid_distance': mean_centroid_distance,
            'original_cluster_sizes': original_sizes,
            'unregularized_cluster_sizes': unregularized_sizes
        })
    
    return results

def main():
    print("="*80)
    print("TESTING HETEROGENEITY CLUSTERING SIMILARITY")
    print("="*80)
    print("\nThis script tests if cluster assignments are similar enough")
    print("that heterogeneity analysis doesn't need to be remade.\n")
    
    # Load Y and disease names
    print("Loading Y matrix and disease names...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', 
                       map_location='cpu', weights_only=False)
    Y = Y_full[:400000, :, :].numpy()
    
    disease_names_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv')
    disease_names = disease_names_df['x'].tolist()
    
    print(f"  Y shape: {Y.shape}")
    print(f"  Diseases: {len(disease_names)}")
    
    # Load thetas from both models
    # Original: from regularized training batches
    # Unregularized: from unregularized training batches (what fixedgk predictions are based on)
    original_thetas = load_original_thetas()
    unregularized_thetas = load_unregularized_training_lambda()
    
    # Target diseases from heterogeneity analysis
    target_diseases = [
        'Myocardial infarction',
        'Malignant neoplasm of female breast',
        'Major depressive disorder'
    ]
    
    # Compare clustering
    results = compare_clustering(
        original_thetas, 
        unregularized_thetas, 
        Y, 
        disease_names, 
        target_diseases,
        n_clusters=3,
        random_state=42
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print("\nCluster Similarity Metrics:")
    print(results_df[['disease', 'n_patients', 'ari', 'overlap_pct', 'mean_centroid_distance']].to_string(index=False))
    
    mean_ari = results_df['ari'].mean()
    mean_overlap = results_df['overlap_pct'].mean()
    mean_centroid_dist = results_df['mean_centroid_distance'].mean()
    
    print(f"\nMean Adjusted Rand Index: {mean_ari:.4f}")
    print(f"Mean Cluster Overlap: {mean_overlap:.1f}%")
    print(f"Mean Centroid Distance: {mean_centroid_dist:.6f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\n⚠️  IMPORTANT NOTE:")
    print("   This compares:")
    print("   - Original: Thetas from REGULARIZED training batches")
    print("   - Unregularized: Thetas from UNREGULARIZED training batches")
    print("\n   Both are from TRAINING (same context). Fixedgk predictions use pooled gamma/kappa")
    print("   from unregularized training, so these thetas represent what fixedgk predictions")
    print("   would produce. The key question is: are clusters similar enough that heterogeneity")
    print("   patterns hold?")
    
    print("\nAdjusted Rand Index (ARI):")
    print("  - ARI = 1.0: Perfect agreement")
    print("  - ARI > 0.9: Very similar clusters")
    print("  - ARI > 0.7: Similar clusters")
    print("  - ARI < 0.5: Different clusters")
    
    print(f"\nCluster Overlap (after label matching):")
    print(f"  - {mean_overlap:.1f}% of patients in matching clusters")
    print(f"  - This is more interpretable than ARI when clusters are imbalanced")
    
    if mean_overlap > 80 and mean_centroid_dist < 0.01:
        print(f"\n✅ High overlap ({mean_overlap:.1f}%) + low centroid distance ({mean_centroid_dist:.6f})")
        print("   → Clusters are VERY SIMILAR despite low ARI")
        print("   → Heterogeneity analysis likely doesn't need to be remade")
    elif mean_overlap > 65:
        print(f"\n⚠️  Moderate overlap ({mean_overlap:.1f}%)")
        print("   → Clusters are SIMILAR but not identical")
        print("   → Check if heterogeneity patterns (PRS correlations, deviations) are similar")
    else:
        print(f"\n❌ Low overlap ({mean_overlap:.1f}%)")
        print("   → Clusters are DIFFERENT")
        print("   → Consider remaking heterogeneity analysis")
    
    # Save results
    output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/clustering_similarity_test.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
