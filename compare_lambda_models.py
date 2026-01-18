#!/usr/bin/env python3
"""
Compare lambda values across different model runs:
1. noG vs results_formanhattan (should be similar - both without genetic effects)
2. censor_e_batchrun_vectorized_11726 vs censor_e_batchrun_vectorized (should be similar - both with genetic effects)
"""

import torch
import numpy as np
from pathlib import Path
import glob
from scipy.stats import spearmanr

def load_lambda_from_checkpoint(checkpoint_path):
    """Load lambda from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            lambda_ = checkpoint['model_state_dict']['lambda_']
        elif 'lambda_' in checkpoint:
            lambda_ = checkpoint['lambda_']
        else:
            return None
        
        if torch.is_tensor(lambda_):
            lambda_ = lambda_.numpy()
        return lambda_
    except Exception as e:
        print(f"  Error loading {checkpoint_path}: {e}")
        return None

def compare_lambda_arrays(lambda1, lambda2, name1, name2, tolerance=1e-4, check_gwas_consistency=True):
    """Compare two lambda arrays and print statistics.
    
    For GWAS consistency, checks if relative differences across individuals are preserved.
    """
    if lambda1 is None or lambda2 is None:
        print(f"  Cannot compare: one or both lambdas are None")
        return
    
    # Ensure same shape for comparison
    if lambda1.shape != lambda2.shape:
        print(f"  Shape mismatch: {lambda1.shape} vs {lambda2.shape}")
        return
    
    # Compare absolute values
    max_diff = np.abs(lambda1 - lambda2).max()
    mean_diff = np.abs(lambda1 - lambda2).mean()
    are_similar = np.allclose(lambda1, lambda2, atol=tolerance)
    
    # Statistics
    lambda1_mean_abs = np.abs(lambda1).mean()
    lambda1_max_abs = np.abs(lambda1).max()
    lambda2_mean_abs = np.abs(lambda2).mean()
    lambda2_max_abs = np.abs(lambda2).max()
    
    print(f"\n  {name1} statistics:")
    print(f"    Shape: {lambda1.shape}")
    print(f"    Mean |λ|: {lambda1_mean_abs:.6f}")
    print(f"    Max |λ|: {lambda1_max_abs:.6f}")
    
    print(f"\n  {name2} statistics:")
    print(f"    Shape: {lambda2.shape}")
    print(f"    Mean |λ|: {lambda2_mean_abs:.6f}")
    print(f"    Max |λ|: {lambda2_max_abs:.6f}")
    
    print(f"\n  Absolute value comparison:")
    print(f"    Max difference: {max_diff:.8f}")
    print(f"    Mean difference: {mean_diff:.8f}")
    print(f"    Similar (within {tolerance})? {are_similar}")
    
    if not are_similar:
        relative_diff = max_diff / (max(lambda1_max_abs, lambda2_max_abs) + 1e-10)
        print(f"    Relative max difference: {relative_diff:.6f}")
    
    # Check GWAS consistency: relative differences across individuals
    if check_gwas_consistency:
        print(f"\n  GWAS consistency check (relative differences across individuals):")
        
        # Flatten to (N, K*T) for correlation analysis
        N, K, T = lambda1.shape
        lambda1_flat = lambda1.reshape(N, -1)
        lambda2_flat = lambda2.reshape(N, -1)
        
        # Pearson correlation for each signature-timepoint combination
        correlations = []
        for k in range(K):
            for t in range(T):
                corr = np.corrcoef(lambda1[:, k, t], lambda2[:, k, t])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        min_corr = np.min(correlations)
        print(f"    Mean Pearson correlation across (K×T) combinations: {mean_corr:.6f}")
        print(f"    Min Pearson correlation: {min_corr:.6f}")
        print(f"    % combinations with corr > 0.95: {100*np.mean(np.array(correlations) > 0.95):.1f}%")
        print(f"    % combinations with corr > 0.99: {100*np.mean(np.array(correlations) > 0.99):.1f}%")
        
        # Spearman rank correlation (more relevant for GWAS, which cares about ordering)
        rank_correlations = []
        for k in range(K):
            for t in range(T):
                try:
                    rho, p = spearmanr(lambda1[:, k, t], lambda2[:, k, t])
                    if not np.isnan(rho):
                        rank_correlations.append(rho)
                except:
                    pass
        
        if rank_correlations:
            mean_rank_corr = np.mean(rank_correlations)
            min_rank_corr = np.min(rank_correlations)
            print(f"    Mean Spearman rank correlation: {mean_rank_corr:.6f}")
            print(f"    Min Spearman rank correlation: {min_rank_corr:.6f}")
            print(f"    % combinations with rank corr > 0.95: {100*np.mean(np.array(rank_correlations) > 0.95):.1f}%")
            print(f"    % combinations with rank corr > 0.99: {100*np.mean(np.array(rank_correlations) > 0.99):.1f}%")
        
        # Variance comparison (if variances are similar, GWAS effect sizes should be similar)
        var1 = np.var(lambda1, axis=0)  # (K, T)
        var2 = np.var(lambda2, axis=0)  # (K, T)
        var_ratio = var2 / (var1 + 1e-10)
        mean_var_ratio = np.mean(var_ratio)
        print(f"    Variance ratio (var2/var1): {mean_var_ratio:.6f}")
        print(f"    Variance ratio range: [{np.min(var_ratio):.4f}, {np.max(var_ratio):.4f}]")
        
        # Check top/bottom individuals overlap
        # For a few example signature-timepoints
        example_sigs = [0, 5, 10] if K > 10 else [0, K//2, K-1]
        example_times = [0, T//2, T-1] if T > 3 else [0, T-1]
        top_overlap = []
        bottom_overlap = []
        for k in example_sigs[:min(3, K)]:
            for t in example_times[:min(3, T)]:
                # Top 10% of individuals
                top_n = max(100, N // 10)
                top1_idx = np.argsort(lambda1[:, k, t])[-top_n:]
                top2_idx = np.argsort(lambda2[:, k, t])[-top_n:]
                overlap = len(np.intersect1d(top1_idx, top2_idx)) / top_n
                top_overlap.append(overlap)
                
                # Bottom 10% of individuals
                bottom1_idx = np.argsort(lambda1[:, k, t])[:top_n]
                bottom2_idx = np.argsort(lambda2[:, k, t])[:top_n]
                overlap = len(np.intersect1d(bottom1_idx, bottom2_idx)) / top_n
                bottom_overlap.append(overlap)
        
        print(f"    Top 10% individuals overlap: {np.mean(top_overlap):.4f}")
        print(f"    Bottom 10% individuals overlap: {np.mean(bottom_overlap):.4f}")
        
        print(f"\n  → For GWAS: High correlations (>0.95) suggest similar genetic associations")
    
    return are_similar

def main():
    print("="*80)
    print("COMPARING LAMBDA VALUES ACROSS MODEL RUNS")
    print("="*80)
    
    # Directories - check both Dropbox and Dropbox-Personal
    base_dir_dropbox = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox")
    base_dir_dropbox_personal = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal")
    
    # Find batch files (these are in Dropbox, not Dropbox-Personal)
    noG_dir = base_dir_dropbox / "censor_e_batchrun_vectorized_noG"
    vectorized_11726_dir = base_dir_dropbox / "censor_e_batchrun_vectorized_11726"
    vectorized_dir = base_dir_dropbox / "censor_e_batchrun_vectorized"
    
    # Find results_formanhattan directory - check both Dropbox and Dropbox-Personal
    results_formanhattan_paths = [
        base_dir_dropbox_personal / "results_formanhattan" / "results" / "output_0_10000" / "model.pt",
        base_dir_dropbox / "results_formanhattan" / "results" / "output_0_10000" / "model.pt",
        base_dir_dropbox / "resultshighamp" / "results" / "output_0_10000" / "model.pt",
        base_dir_dropbox_personal / "resultshighamp" / "results" / "output_0_10000" / "model.pt",
        # Also try checking if there's a batch file pattern
        base_dir_dropbox_personal / "results_formanhattan" / "results" / "output_0_10000" / "enrollment_model_W0.0001_batch_0_10000.pt",
        base_dir_dropbox / "results_formanhattan" / "results" / "output_0_10000" / "enrollment_model_W0.0001_batch_0_10000.pt",
        base_dir_dropbox / "resultshighamp" / "results" / "output_0_10000" / "enrollment_model_W0.0001_batch_0_10000.pt",
        base_dir_dropbox_personal / "resultshighamp" / "results" / "output_0_10000" / "enrollment_model_W0.0001_batch_0_10000.pt",
    ]
    
    results_formanhattan_path = None
    for path in results_formanhattan_paths:
        if path.exists():
            results_formanhattan_path = path
            break
    
    # Get first batch file (0_10000) from each directory
    # Note: different directories use slightly different filename patterns
    noG_pattern = str(noG_dir / "enrollment_model_VECTORIZED_W0.0001_nog_batch_0_10000.pt")
    vectorized_11726_pattern = str(vectorized_11726_dir / "enrollment_model_VECTORIZED_W0.0001_batch_0_10000.pt")
    # Original vectorized uses different pattern (no VECTORIZED in filename)
    vectorized_pattern = str(vectorized_dir / "enrollment_model_W0.0001_batch_0_10000.pt")
    
    # Load lambdas
    print("\n" + "="*80)
    print("1. LOADING LAMBDAS FROM CHECKPOINTS")
    print("="*80)
    
    lambda_noG = None
    lambda_vectorized_11726 = None
    lambda_vectorized = None
    lambda_results_formanhattan = None
    
    # Load noG
    if Path(noG_pattern).exists():
        print(f"\nLoading noG checkpoint: {noG_pattern}")
        lambda_noG = load_lambda_from_checkpoint(noG_pattern)
        if lambda_noG is not None:
            print(f"  ✓ Loaded lambda shape: {lambda_noG.shape}")
    else:
        print(f"\n⚠️  NoG checkpoint not found: {noG_pattern}")
    
    # Load vectorized_11726
    if Path(vectorized_11726_pattern).exists():
        print(f"\nLoading vectorized_11726 checkpoint: {vectorized_11726_pattern}")
        lambda_vectorized_11726 = load_lambda_from_checkpoint(vectorized_11726_pattern)
        if lambda_vectorized_11726 is not None:
            print(f"  ✓ Loaded lambda shape: {lambda_vectorized_11726.shape}")
    else:
        print(f"\n⚠️  Vectorized_11726 checkpoint not found: {vectorized_11726_pattern}")
    
    # Load vectorized (original)
    if Path(vectorized_pattern).exists():
        print(f"\nLoading vectorized checkpoint: {vectorized_pattern}")
        lambda_vectorized = load_lambda_from_checkpoint(vectorized_pattern)
        if lambda_vectorized is not None:
            print(f"  ✓ Loaded lambda shape: {lambda_vectorized.shape}")
    else:
        print(f"\n⚠️  Vectorized checkpoint not found: {vectorized_pattern}")
    
    # Load results_formanhattan
    if results_formanhattan_path and results_formanhattan_path.exists():
        print(f"\nLoading results_formanhattan checkpoint: {results_formanhattan_path}")
        lambda_results_formanhattan = load_lambda_from_checkpoint(results_formanhattan_path)
        if lambda_results_formanhattan is not None:
            print(f"  ✓ Loaded lambda shape: {lambda_results_formanhattan.shape}")
    else:
        print(f"\n⚠️  Results_formanhattan checkpoint not found. Tried:")
        for path in results_formanhattan_paths:
            print(f"    {path}")
    
    # Comparisons
    print("\n" + "="*80)
    print("2. COMPARISON 1: noG vs results_formanhattan")
    print("   (Should be similar - both without genetic effects)")
    print("="*80)
    
    if lambda_noG is not None and lambda_results_formanhattan is not None:
        compare_lambda_arrays(lambda_noG, lambda_results_formanhattan, 
                             "noG", "results_formanhattan", tolerance=1e-3)
    else:
        print("\n  ⚠️  Cannot compare: missing one or both lambdas")
        if lambda_noG is None:
            print("    - lambda_noG is None")
        if lambda_results_formanhattan is None:
            print("    - lambda_results_formanhattan is None")
    
    print("\n" + "="*80)
    print("3. COMPARISON 2: vectorized_11726 vs vectorized (original)")
    print("   (Should be similar - both with genetic effects)")
    print("="*80)
    
    if lambda_vectorized_11726 is not None and lambda_vectorized is not None:
        compare_lambda_arrays(lambda_vectorized_11726, lambda_vectorized,
                             "vectorized_11726", "vectorized", tolerance=1e-3)
    else:
        print("\n  ⚠️  Cannot compare: missing one or both lambdas")
        if lambda_vectorized_11726 is None:
            print("    - lambda_vectorized_11726 is None")
        if lambda_vectorized is None:
            print("    - lambda_vectorized is None")
    
    # Additional comparison: noG vs vectorized_11726 (should be DIFFERENT due to genetic effects)
    print("\n" + "="*80)
    print("4. COMPARISON 3: noG vs vectorized_11726")
    print("   (Should be DIFFERENT - one without genetic effects, one with)")
    print("="*80)
    
    if lambda_noG is not None and lambda_vectorized_11726 is not None:
        compare_lambda_arrays(lambda_noG, lambda_vectorized_11726,
                             "noG", "vectorized_11726", tolerance=1e-2)
    else:
        print("\n  ⚠️  Cannot compare: missing one or both lambdas")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✓ Comparisons complete. Check results above.")

if __name__ == "__main__":
    main()
