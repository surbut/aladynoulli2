"""
Compare gamma values for sex vs genetics in LR vs non-LR models.
This will help understand how regularization affects different covariates.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

def load_gamma_from_model(model_path):
    """Load gamma from a saved model file"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        gamma = checkpoint['model_state_dict']['gamma']
    elif 'gamma' in checkpoint:
        gamma = checkpoint['gamma']
    else:
        raise ValueError(f"Could not find gamma in {model_path}")
    
    if torch.is_tensor(gamma):
        gamma = gamma.cpu().numpy()
    
    return gamma

def analyze_gamma(gamma, n_genetic_features, n_pcs=10):
    """
    Analyze gamma values, assuming structure:
    - First n_genetic_features: genetic PRS
    - Next 1: sex
    - Next n_pcs: principal components (if included)
    """
    results = {}
    
    # Genetic effects (first n_genetic_features columns)
    genetic_gamma = gamma[:n_genetic_features, :]
    results['genetic_mean'] = np.mean(np.abs(genetic_gamma))
    results['genetic_std'] = np.std(genetic_gamma)
    results['genetic_max'] = np.max(np.abs(genetic_gamma))
    
    # Sex effect (next column after genetics)
    sex_idx = n_genetic_features
    sex_gamma = gamma[sex_idx, :]
    results['sex_mean'] = np.mean(np.abs(sex_gamma))
    results['sex_std'] = np.std(sex_gamma)
    results['sex_max'] = np.max(np.abs(sex_gamma))
    
    # PC effects (if included)
    if gamma.shape[0] > sex_idx + 1:
        pc_start = sex_idx + 1
        pc_gamma = gamma[pc_start:pc_start+n_pcs, :]
        results['pc_mean'] = np.mean(np.abs(pc_gamma))
        results['pc_std'] = np.std(pc_gamma)
        results['pc_max'] = np.max(np.abs(pc_gamma))
    
    return results, genetic_gamma, sex_gamma

# Paths to model files
# Adjust these paths to your actual model files
# LR version (with lambda_reg=0.01)
lr_model_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_withLR/model_enroll_fixedphi_sex_0_10000.pt")

# Non-LR version (without regularization or with lambda_reg=0)
# Check if there's a non-LR version - might be in a different directory
possible_non_lr_paths = [
    Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/model_enroll_fixedphi_sex_0_10000.pt"),
    Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_withLR/model_enroll_fixedphi_sex_0_10000.pt"),  # Same dir, different lambda_reg
]

# Find the first existing non-LR path, or use the first one as default
non_lr_model_path = None
for path in possible_non_lr_paths:
    if path.exists() and path != lr_model_path:
        non_lr_model_path = path
        break

if non_lr_model_path is None:
    non_lr_model_path = possible_non_lr_paths[0]  # Use first as default, will check existence later

# Number of genetic features (PRS) - adjust based on your actual G matrix
# This is the number of PRS columns in G before sex and PCs
n_genetic_features = 36  # 36 PRS features, then sex at index 36, then 10 PCs

print("="*80)
print("COMPARING GAMMA VALUES: LR vs NON-LR")
print("="*80)

# Print what we're looking for
print(f"\nLooking for model files:")
print(f"  LR model: {lr_model_path}")
print(f"    Exists: {lr_model_path.exists()}")
print(f"  Non-LR model: {non_lr_model_path}")
print(f"    Exists: {non_lr_model_path.exists()}")

# If files don't exist, try to find them
if not lr_model_path.exists() or not non_lr_model_path.exists():
    print(f"\n⚠️  Some model files not found. Searching for alternatives...")
    
    # Search in common directories
    search_dirs = [
        Path("/Users/sarahurbut/Library/CloudStorage/Dropbox"),
        Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal"),
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            # Look for model files
            model_files = list(search_dir.rglob("model_enroll_fixedphi_sex_*_*.pt"))
            if model_files:
                print(f"\nFound {len(model_files)} model files in {search_dir}:")
                for f in model_files[:5]:  # Show first 5
                    print(f"  {f}")
                if len(model_files) > 5:
                    print(f"  ... and {len(model_files) - 5} more")
                break

if lr_model_path.exists() and non_lr_model_path.exists():
    print(f"\nLoading LR model: {lr_model_path}")
    gamma_lr = load_gamma_from_model(lr_model_path)
    print(f"  Gamma shape: {gamma_lr.shape}")
    
    print(f"\nLoading non-LR model: {non_lr_model_path}")
    gamma_nonlr = load_gamma_from_model(non_lr_model_path)
    print(f"  Gamma shape: {gamma_nonlr.shape}")
    
    # Analyze both
    print("\n" + "="*80)
    print("LR MODEL (with regularization):")
    print("="*80)
    results_lr, genetic_lr, sex_lr = analyze_gamma(gamma_lr, n_genetic_features)
    print(f"Genetic effects:")
    print(f"  Mean |gamma|: {results_lr['genetic_mean']:.6f}")
    print(f"  Std |gamma|:  {results_lr['genetic_std']:.6f}")
    print(f"  Max |gamma|:  {results_lr['genetic_max']:.6f}")
    
    print(f"\nSex effect:")
    print(f"  Mean |gamma|: {results_lr['sex_mean']:.6f}")
    print(f"  Std |gamma|:  {results_lr['sex_std']:.6f}")
    print(f"  Max |gamma|:  {results_lr['sex_max']:.6f}")
    print(f"  Sex gamma values per signature: {sex_lr}")
    
    print("\n" + "="*80)
    print("NON-LR MODEL (no regularization):")
    print("="*80)
    results_nonlr, genetic_nonlr, sex_nonlr = analyze_gamma(gamma_nonlr, n_genetic_features)
    print(f"Genetic effects:")
    print(f"  Mean |gamma|: {results_nonlr['genetic_mean']:.6f}")
    print(f"  Std |gamma|:  {results_nonlr['genetic_std']:.6f}")
    print(f"  Max |gamma|:  {results_nonlr['genetic_max']:.6f}")
    
    print(f"\nSex effect:")
    print(f"  Mean |gamma|: {results_nonlr['sex_mean']:.6f}")
    print(f"  Std |gamma|:  {results_nonlr['sex_std']:.6f}")
    print(f"  Max |gamma|:  {results_nonlr['sex_max']:.6f}")
    print(f"  Sex gamma values per signature: {sex_nonlr}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON:")
    print("="*80)
    genetic_shrinkage = (1 - results_lr['genetic_mean'] / results_nonlr['genetic_mean']) * 100
    sex_shrinkage = (1 - results_lr['sex_mean'] / results_nonlr['sex_mean']) * 100
    
    print(f"\nGenetic effects shrinkage: {genetic_shrinkage:.2f}%")
    print(f"Sex effects shrinkage: {sex_shrinkage:.2f}%")
    print(f"\nRatio (sex/genetic) in LR: {results_lr['sex_mean'] / results_lr['genetic_mean']:.4f}")
    print(f"Ratio (sex/genetic) in non-LR: {results_nonlr['sex_mean'] / results_nonlr['genetic_mean']:.4f}")
    
    # Key insight: Even with 97% shrinkage, if original effects were small, 
    # the shrunk values might still be sufficient for prediction
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print(f"Sex effects shrunk by ~{sex_shrinkage:.1f}%, but AUC unchanged.")
    print(f"This suggests:")
    print(f"  1. Original sex effects were relatively small (mean |gamma|: {results_nonlr['sex_mean']:.6f})")
    print(f"  2. Even after shrinkage, effects may still contribute (mean |gamma|: {results_lr['sex_mean']:.6f})")
    print(f"  3. Or sex/genetic effects are small relative to phi/psi signal")
    print(f"  4. The model is robust to this level of regularization")
    print(f"\nNote: The ~97% shrinkage is a large relative change, but if the")
    print(f"absolute values are small, the impact on predictions may be minimal.")
    
    # Show per-signature comparison
    print("\n" + "="*80)
    print("PER-SIGNATURE SEX GAMMA COMPARISON:")
    print("="*80)
    
    # Calculate shrinkage, handling division by zero
    shrinkage_pct = np.zeros_like(sex_lr)
    for i in range(len(sex_lr)):
        if np.abs(sex_nonlr[i]) > 1e-10:  # Avoid division by zero
            shrinkage_pct[i] = ((sex_nonlr[i] - sex_lr[i]) / np.abs(sex_nonlr[i]) * 100)
        else:
            shrinkage_pct[i] = np.nan
    
    comparison_df = pd.DataFrame({
        'Signature': range(len(sex_lr)),
        'LR': sex_lr,
        'Non-LR': sex_nonlr,
        'Difference': sex_lr - sex_nonlr,
        'Shrinkage_%': shrinkage_pct
    })
    print(comparison_df.to_string(index=False))
    
else:
    print(f"\n⚠️  Model files not found!")
    print(f"LR model: {lr_model_path} (exists: {lr_model_path.exists()})")
    print(f"Non-LR model: {non_lr_model_path} (exists: {non_lr_model_path.exists()})")
    print("\nPlease update the paths in the script to point to your actual model files.")
    print("\nNote: If you don't have a non-LR version, you can:")
    print("  1. Run the same script with --lambda_reg 0 to create a non-LR version")
    print("  2. Or compare two LR versions with different lambda_reg values")
    print("  3. Or update the paths above to point to existing model files")

